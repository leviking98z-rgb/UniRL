"""Canonical Qwen3-Omni preprocessing contract.

The codec owns message/media preparation and the Hugging Face processor call.
Both train-side replay and rollout backends consume the same typed result.
Backends may compress placeholder IDs for transport, but must retain the
expanded IDs and replay conditions produced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from unirl.types.conditions import TextTokenCondition

from .conditions import Qwen3OmniARConditions
from .media import extract_audio_from_video_pyav
from .video import limit_video_frames, sample_video_frames_pyav


@dataclass(frozen=True)
class Qwen3OmniPromptBoundary:
    """Half-open prompt span in the expanded token sequence."""

    start: int
    end: int


@dataclass(frozen=True)
class Qwen3OmniProcessedRow:
    """Canonical processor output for one role-aware conversation."""

    expanded_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_boundary: Qwen3OmniPromptBoundary
    video_frames: Optional[Any]
    audio_waveform: Optional[Any]
    effective_video_fps: float
    multimodal_processor_kwargs: Dict[str, Any]
    pixel_values_videos: Optional[Any]
    video_grid_thw: Optional[Any]
    video_second_per_grid: Optional[Any]
    input_features: Optional[Any]
    feature_attention_mask: Optional[Any]

    def __post_init__(self) -> None:
        if self.expanded_input_ids.ndim != 1:
            raise ValueError(
                "Qwen3OmniProcessedRow.expanded_input_ids must be one-dimensional, "
                f"got {tuple(self.expanded_input_ids.shape)}"
            )
        if tuple(self.attention_mask.shape) != tuple(self.expanded_input_ids.shape):
            raise ValueError(
                "Qwen3OmniProcessedRow.attention_mask must align with expanded_input_ids, "
                f"got {tuple(self.attention_mask.shape)} and {tuple(self.expanded_input_ids.shape)}"
            )
        if self.prompt_boundary != Qwen3OmniPromptBoundary(
            start=0,
            end=int(self.expanded_input_ids.shape[0]),
        ):
            raise ValueError("Qwen3OmniProcessedRow.prompt_boundary must span the complete expanded prompt")
        video_fields = (
            self.pixel_values_videos,
            self.video_grid_thw,
            self.video_second_per_grid,
        )
        if self.has_video and not all(value is not None for value in video_fields):
            raise ValueError(
                "Qwen3OmniProcessedRow: a video prompt requires pixel values, video_grid_thw, and video_second_per_grid"
            )
        if (self.input_features is None) != (self.feature_attention_mask is None):
            raise ValueError(
                "Qwen3OmniProcessedRow: input_features and feature_attention_mask must be present together"
            )
        if self.has_audio and self.input_features is None:
            raise ValueError("Qwen3OmniProcessedRow: an audio prompt requires processor input_features")

    @property
    def has_video(self) -> bool:
        return self.video_frames is not None

    @property
    def has_audio(self) -> bool:
        return self.audio_waveform is not None


@dataclass(frozen=True)
class Qwen3OmniProcessorResult:
    """Batch result shared by rollout serialization and train-side replay."""

    rows: Tuple[Qwen3OmniProcessedRow, ...]
    replay_conditions: Qwen3OmniARConditions

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("Qwen3OmniProcessorResult.rows must be non-empty")
        prompt = self.replay_conditions.prompt
        if prompt is None or prompt.input_ids is None:
            raise ValueError("Qwen3OmniProcessorResult requires replay prompt input_ids")
        if int(prompt.input_ids.shape[0]) != len(self.rows):
            raise ValueError(
                "Qwen3OmniProcessorResult row count does not match replay prompt batch: "
                f"{len(self.rows)} != {int(prompt.input_ids.shape[0])}"
            )

    @property
    def prompt_boundaries(self) -> Tuple[Qwen3OmniPromptBoundary, ...]:
        return tuple(row.prompt_boundary for row in self.rows)


class Qwen3OmniProcessorCodec:
    """Encode Qwen3-Omni conversations into one canonical typed contract."""

    def __init__(
        self,
        *,
        processor: Any,
        tokenizer: Any,
        max_prompt_length: int,
        video_fps: float = 1.0,
        video_max_frames: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        use_audio_in_video: bool = False,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        pad_to_max_length: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        owner: str = "Qwen3OmniProcessorCodec",
    ) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_prompt_length = int(max_prompt_length)
        if self.max_prompt_length <= 0:
            raise ValueError(f"{owner}: max_prompt_length must be positive")
        self.video_fps = float(video_fps)
        self.video_max_frames = int(video_max_frames) if video_max_frames is not None else None
        self.video_max_pixels = int(video_max_pixels) if video_max_pixels else None
        self.use_audio_in_video = bool(use_audio_in_video)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.pad_to_max_length = bool(pad_to_max_length)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.owner = str(owner)

    def _multimodal_processor_kwargs(self, *, video_fps: float) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "fps": float(video_fps),
            "do_sample_frames": False,
            "truncation": True,
        }
        if self.video_max_pixels is not None:
            kwargs["size"] = {
                "shortest_edge": int(self.processor.video_processor.size["shortest_edge"]),
                "longest_edge": self.video_max_pixels,
            }
        return kwargs

    def _prepare_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Optional[Any], float, Optional[Any]]:
        """Decode/sample at most one video block without mutating input."""
        prepared: List[Dict[str, Any]] = []
        video_frames: Optional[Any] = None
        effective_fps = self.video_fps
        audio_waveform: Optional[Any] = None
        video_count = 0

        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                prepared.append(dict(message))
                continue
            blocks: List[Dict[str, Any]] = []
            for raw_block in content:
                block = dict(raw_block)
                if block.get("type") == "video":
                    video_count += 1
                    if video_count > 1:
                        raise ValueError(f"{self.owner}: supports at most one source video per conversation")
                    raw_video = block.get("video")
                    if isinstance(raw_video, str):
                        video_frames, effective_fps = sample_video_frames_pyav(
                            raw_video,
                            target_fps=self.video_fps,
                            max_frames=self.video_max_frames,
                        )
                        if self.use_audio_in_video:
                            sample_rate = int(
                                getattr(
                                    getattr(self.processor, "feature_extractor", None),
                                    "sampling_rate",
                                    16000,
                                )
                            )
                            audio_waveform = extract_audio_from_video_pyav(raw_video, sample_rate)
                    else:
                        video_frames, effective_fps = limit_video_frames(
                            raw_video,
                            fps=self.video_fps,
                            max_frames=self.video_max_frames,
                        )
                    block["video"] = video_frames
                blocks.append(block)
            copied = dict(message)
            copied["content"] = blocks
            prepared.append(copied)
        return prepared, video_frames, float(effective_fps), audio_waveform

    def _encode_one(
        self,
        messages: List[Dict[str, Any]],
        template_overrides: Dict[str, Any],
    ) -> Qwen3OmniProcessedRow:
        prepared, video_frames, effective_fps, audio_waveform = self._prepare_messages(messages)
        template_kwargs = dict(self.chat_template_kwargs)
        template_kwargs.update(template_overrides)
        multimodal_kwargs = (
            self._multimodal_processor_kwargs(video_fps=effective_fps) if video_frames is not None else {}
        )

        if audio_waveform is not None:
            template_kwargs.update(add_generation_prompt=True, tokenize=False)
            prompt_text = self.processor.apply_chat_template(prepared, **template_kwargs)
            processor_kwargs = dict(multimodal_kwargs)
            processor_kwargs.update(
                text=[prompt_text],
                videos=[video_frames],
                audio=[audio_waveform],
                use_audio_in_video=True,
                return_tensors="pt",
            )
            encoding = dict(self.processor(**processor_kwargs))
        else:
            template_kwargs.update(multimodal_kwargs)
            template_kwargs.update(
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            encoding = dict(self.processor.apply_chat_template(prepared, **template_kwargs))

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise ValueError(f"{self.owner}: processor input_ids must have shape [1, L], got {tuple(input_ids.shape)}")
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                f"{self.owner}: attention_mask shape {tuple(attention_mask.shape)} "
                f"does not match input_ids {tuple(input_ids.shape)}"
            )
        prompt_length = int(input_ids.shape[-1])
        if prompt_length > self.max_prompt_length:
            if video_frames is not None or audio_waveform is not None:
                raise ValueError(
                    f"{self.owner}: multimodal prompt produced {prompt_length} tokens, "
                    f"exceeding max_prompt_length={self.max_prompt_length}. "
                    "Reduce video_max_frames, video_max_pixels, or video_fps, "
                    "or raise max_prompt_length."
                )
            input_ids = input_ids[..., -self.max_prompt_length :]
            attention_mask = attention_mask[..., -self.max_prompt_length :]
            prompt_length = self.max_prompt_length

        backend_kwargs = dict(multimodal_kwargs)
        if audio_waveform is not None:
            backend_kwargs["use_audio_in_video"] = True
            temporal_patch_size = int(getattr(self.processor.video_processor, "temporal_patch_size", 2))
            backend_kwargs["second_per_grid_ts"] = [temporal_patch_size / effective_fps]

        return Qwen3OmniProcessedRow(
            expanded_input_ids=input_ids.squeeze(0).detach().cpu().to(torch.long),
            attention_mask=attention_mask.squeeze(0).detach().cpu().to(torch.long),
            prompt_boundary=Qwen3OmniPromptBoundary(start=0, end=prompt_length),
            video_frames=video_frames,
            audio_waveform=audio_waveform,
            effective_video_fps=effective_fps,
            multimodal_processor_kwargs=backend_kwargs,
            pixel_values_videos=encoding.get("pixel_values_videos"),
            video_grid_thw=encoding.get("video_grid_thw"),
            video_second_per_grid=encoding.get("video_second_per_grid"),
            input_features=encoding.get("input_features"),
            feature_attention_mask=encoding.get("feature_attention_mask"),
        )

    def _move(self, value: Any, *, dtype: Optional[torch.dtype] = None) -> Any:
        if value is None:
            return None
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        target_dtype = dtype if dtype is not None else tensor.dtype
        return tensor.to(device=self.device, dtype=target_dtype)

    def _build_conditions(
        self,
        rows: Tuple[Qwen3OmniProcessedRow, ...],
    ) -> Qwen3OmniARConditions:
        max_length = (
            self.max_prompt_length
            if self.pad_to_max_length
            else max(int(row.expanded_input_ids.shape[0]) for row in rows)
        )
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_id is None:
            pad_id = 0

        input_ids = torch.full(
            (len(rows), max_length),
            int(pad_id),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros(
            (len(rows), max_length),
            dtype=torch.long,
            device=self.device,
        )
        for index, row in enumerate(rows):
            length = min(int(row.expanded_input_ids.shape[0]), max_length)
            input_ids[index, :length] = row.expanded_input_ids[:length].to(self.device)
            attention_mask[index, :length] = row.attention_mask[:length].to(self.device)

        pixel_values_videos = [self._move(row.pixel_values_videos, dtype=self.dtype) for row in rows]
        video_grid_thw = [self._move(row.video_grid_thw) for row in rows]
        video_second_per_grid = [self._move(row.video_second_per_grid) for row in rows]
        input_features = [self._move(row.input_features, dtype=self.dtype) for row in rows]
        feature_attention_mask = [self._move(row.feature_attention_mask) for row in rows]

        has_video = any(value is not None for value in pixel_values_videos)
        has_audio = any(value is not None for value in input_features)
        return Qwen3OmniARConditions(
            prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask),
            pixel_values_videos=pixel_values_videos if has_video else None,
            video_grid_thw=video_grid_thw if has_video else None,
            video_second_per_grid=video_second_per_grid if has_video else None,
            input_features=input_features if has_audio else None,
            feature_attention_mask=feature_attention_mask if has_audio else None,
        )

    def encode_messages(
        self,
        conversations: List[List[Dict[str, Any]]],
        *,
        template_overrides: Optional[Dict[str, Any]] = None,
    ) -> Qwen3OmniProcessorResult:
        """Encode one role-aware conversation per frontier row."""
        if not conversations:
            raise ValueError(f"{self.owner}: empty conversation batch")
        overrides = dict(template_overrides or {})
        rows = tuple(self._encode_one(messages, overrides) for messages in conversations)
        return Qwen3OmniProcessorResult(
            rows=rows,
            replay_conditions=self._build_conditions(rows),
        )


__all__ = [
    "Qwen3OmniProcessedRow",
    "Qwen3OmniProcessorCodec",
    "Qwen3OmniProcessorResult",
    "Qwen3OmniPromptBoundary",
]
