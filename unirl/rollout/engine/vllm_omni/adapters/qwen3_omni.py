"""Qwen3-Omni Thinker rollout adapters."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.models.qwen3_omni.processor import (
    Qwen3OmniProcessorCodec,
    Qwen3OmniProcessorResult,
)
from unirl.models.types.conversations import build_video_messages
from unirl.rollout.engine.vllm_omni.adapters.base import ModelAdapter, register_adapter
from unirl.rollout.engine.vllm_omni.adapters.hi3 import Hi3TextOutputAdapter
from unirl.rollout.engine.vllm_omni.backends import (
    STAGE_KIND_AR,
    GenerateCall,
    OmniRawResult,
    StageSampling,
)
from unirl.types.sample import Sample
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment
from unirl.types.segments.base import SegmentStatus

logger = logging.getLogger(__name__)


def _compress_qwen3_omni_prompt_ids(
    token_ids: List[int],
    *,
    audio_token_id: int,
    image_token_id: int,
    video_token_id: int,
    vision_bos_token_id: int,
    vision_eos_token_id: int,
    audio_bos_token_id: int,
    audio_eos_token_id: int,
    use_audio_in_video: bool,
) -> List[int]:
    """Undo HF multimodal expansion before vLLM processes the raw media.

    This mirrors vLLM's
    ``Qwen3OmniMoeThinkerMultiModalProcessor._get_raw_input_ids``. Replay
    keeps the original expanded IDs; only the IDs sent to vLLM are compressed.
    """
    result = list(token_ids)
    if use_audio_in_video:
        while True:
            start = next(
                (i for i in range(len(result) - 1) if result[i : i + 2] == [vision_bos_token_id, audio_bos_token_id]),
                None,
            )
            if start is None:
                break
            end = next(
                (
                    i
                    for i in range(start + 2, len(result) - 1)
                    if result[i : i + 2] == [audio_eos_token_id, vision_eos_token_id]
                ),
                None,
            )
            if end is None:
                raise ValueError(
                    "Qwen3OmniThinkerInputAdapter: expanded audio-in-video span "
                    "has no matching audio/vision end tokens."
                )
            result = result[:start] + [vision_bos_token_id, video_token_id, vision_eos_token_id] + result[end + 2 :]

    for mm_token_id in (audio_token_id, image_token_id, video_token_id):
        compressed: List[int] = []
        for token_id in result:
            if token_id != mm_token_id or not compressed or compressed[-1] != mm_token_id:
                compressed.append(token_id)
        result = compressed
    return result


class Qwen3OmniThinkerInputAdapter:
    """Build one batched AR generate call from a role-aware request Sample."""

    def __init__(
        self,
        modality: str,
        *,
        model_path: str,
        video_fps: float = 1.0,
        video_max_frames: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        use_audio_in_video: bool = False,
        max_prompt_length: int = 12288,
        system_instruction: Optional[str] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.modality = modality
        self.model_path = str(model_path)
        self.video_fps = float(video_fps)
        self.video_max_frames = int(video_max_frames) if video_max_frames is not None else None
        self.video_max_pixels = int(video_max_pixels) if video_max_pixels else None
        self.use_audio_in_video = bool(use_audio_in_video)
        self.max_prompt_length = int(max_prompt_length)
        self.system_instruction = system_instruction
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

        # Reuse the processor and tokenizer across requests.
        from transformers import AutoProcessor, AutoTokenizer

        self._processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # Some checkpoints store the chat template separately.
        if getattr(self._tokenizer, "chat_template", None) is None:
            import json
            import os

            path = os.path.join(self.model_path, "chat_template.json")
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                        if data.get("chat_template"):
                            self._tokenizer.chat_template = data["chat_template"]
                except (OSError, json.JSONDecodeError):
                    pass
        processor_tokenizer = getattr(self._processor, "tokenizer", None)
        if (
            processor_tokenizer is not None
            and getattr(processor_tokenizer, "chat_template", None) is None
            and getattr(self._tokenizer, "chat_template", None) is not None
        ):
            processor_tokenizer.chat_template = self._tokenizer.chat_template
        canonical_tokenizer = processor_tokenizer or self._tokenizer

        self._processor_codec = Qwen3OmniProcessorCodec(
            processor=self._processor,
            tokenizer=canonical_tokenizer,
            max_prompt_length=self.max_prompt_length,
            video_fps=self.video_fps,
            video_max_frames=self.video_max_frames,
            video_max_pixels=self.video_max_pixels,
            use_audio_in_video=self.use_audio_in_video,
            chat_template_kwargs=self.chat_template_kwargs,
            owner=type(self).__name__,
        )

        # Cached for replay-condition construction by the output adapter. The
        # engine's _generate_lock encloses build -> generate -> response, so no
        # second caller can replace this request-local batch in between.
        self._last_processor_result: Optional[Qwen3OmniProcessorResult] = None

    def _token_id(self, token: str) -> int:
        tokenizer = self._processor_codec.tokenizer
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or int(token_id) < 0:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Qwen3OmniThinkerInputAdapter: cannot resolve token id for {token!r}")
            token_id = encoded[0]
        return int(token_id)

    def _compress_prompt_ids(self, token_ids: List[int], *, use_audio_in_video: bool) -> List[int]:
        """Compress processor-expanded placeholders for vLLM re-expansion."""
        return _compress_qwen3_omni_prompt_ids(
            token_ids,
            audio_token_id=self._token_id("<|audio_pad|>"),
            image_token_id=self._token_id("<|image_pad|>"),
            video_token_id=self._token_id("<|video_pad|>"),
            vision_bos_token_id=self._token_id("<|vision_start|>"),
            vision_eos_token_id=self._token_id("<|vision_end|>"),
            audio_bos_token_id=self._token_id("<|audio_start|>"),
            audio_eos_token_id=self._token_id("<|audio_end|>"),
            use_audio_in_video=use_audio_in_video,
        )

    def build(self, sample: Sample) -> List[GenerateCall]:
        frontier = sample.frontier_gen_part(ARSamplingParams)
        ar = frontier.sampling_params
        assert isinstance(ar, ARSamplingParams)

        chat_overrides = dict((sample.parts[0].control or {}).get("chat") or {})
        system_instruction = chat_overrides.get("system_instruction", self.system_instruction)
        template_overrides = dict(chat_overrides.get("template_kwargs") or {})
        conversations = build_video_messages(sample.turns(), system_instruction)
        require(
            bool(conversations),
            "Qwen3OmniThinkerInputAdapter: Sample carries no text/video conditioning turns.",
        )
        require(
            len(conversations) == len(frontier.sample_ids),
            f"Qwen3OmniThinkerInputAdapter: conversation count {len(conversations)} "
            f"!= frontier id count {len(frontier.sample_ids)}",
        )

        prompts: List[Dict[str, Any]] = []
        self._last_processor_result = None
        result = self._processor_codec.encode_messages(
            conversations,
            template_overrides=template_overrides,
        )
        # The output adapter consumes this cache after generation.
        self._last_processor_result = result
        for row in result.rows:
            expanded_ids = row.expanded_input_ids.tolist()
            rollout_ids = (
                self._compress_prompt_ids(expanded_ids, use_audio_in_video=row.has_audio)
                if row.has_video or row.has_audio
                else expanded_ids
            )
            entry: Dict[str, Any] = {"prompt_token_ids": rollout_ids}
            media: Dict[str, Any] = {}
            if row.has_video:
                media["video"] = [row.video_frames]
            if row.has_audio:
                media["audio"] = [row.audio_waveform]
            if media:
                entry["multi_modal_data"] = media
            if row.multimodal_processor_kwargs:
                # Keep this request-local: globally enabling audio-in-video
                # breaks vLLM's video-only dummy profiling during engine boot.
                mm_processor_kwargs = dict(row.multimodal_processor_kwargs)
                entry["mm_processor_kwargs"] = mm_processor_kwargs
                logger.debug(
                    "Qwen3-Omni rollout prompt: expanded_tokens=%d compressed_tokens=%d "
                    "modalities=%s mm_processor_kwargs=%s",
                    len(expanded_ids),
                    len(rollout_ids),
                    sorted(media),
                    mm_processor_kwargs,
                )
            prompts.append(entry)

        max_new_tokens = int(ar.max_new_tokens)
        temperature = float(ar.temperature)
        top_p = float(ar.top_p)
        top_k_val = int(ar.top_k)
        top_k = top_k_val if top_k_val > 0 else -1  # vLLM: -1 disables top_k
        stop_token_id = ar.stop_token_id

        base_sampling_kwargs: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_new_tokens,
            "logprobs": 1,
        }
        if stop_token_id is not None:
            base_sampling_kwargs["stop_token_ids"] = [int(stop_token_id)]

        # Keep seed unset: AsyncOmniEngine.add_request receives each prompt as an
        # independent request, and patch_per_request_ar_seed clones the shared
        # SamplingParams with a fresh seed for every request.
        return [
            GenerateCall(
                prompts=prompts,
                sampling=[StageSampling(kind=STAGE_KIND_AR, kwargs=base_sampling_kwargs)],
            )
        ]


class Qwen3OmniThinkerOutputAdapter(Hi3TextOutputAdapter):
    """Build AR responses from the cached canonical processor result."""

    def __init__(self, modality: str, input_adapter: "Qwen3OmniThinkerInputAdapter") -> None:
        super().__init__(modality)
        self._input_adapter = input_adapter

    def build_conditions(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Dict[str, Any]:
        """Return replay conditions from the cached canonical processor result."""
        del per_request
        result = self._input_adapter._last_processor_result
        if result is None:
            raise RuntimeError(
                "Qwen3OmniThinkerOutputAdapter.build_conditions: input adapter "
                "cache is empty — ``build_inputs`` must run before ``build_response``."
            )
        frontier = sample.frontier_gen_part(ARSamplingParams)
        require(
            len(result.rows) == len(frontier.sample_ids),
            f"Qwen3OmniThinkerOutputAdapter: processor row count {len(result.rows)} "
            f"!= frontier id count {len(frontier.sample_ids)}",
        )
        return result.replay_conditions.to_dict()

    @staticmethod
    def _stage0(group: List[OmniRawResult]) -> OmniRawResult:
        for output in group:
            if getattr(output, "stage_id", None) == 0:
                return output
        raise RuntimeError("Qwen3OmniThinkerOutputAdapter: backend result group has no stage-0 AR output.")

    @classmethod
    def _status(cls, per_request: List[List[OmniRawResult]]) -> torch.Tensor:
        mapping = {
            "stop": SegmentStatus.COMPLETED,
            "length": SegmentStatus.TRUNCATED,
            "abort": SegmentStatus.ABORTED,
        }
        values: List[int] = []
        for group in per_request:
            output = cls._stage0(group)
            request_output = getattr(output, "request_output", None)
            completions = getattr(request_output, "outputs", None) or []
            finish_reason = getattr(completions[0], "finish_reason", None) if completions else None
            values.append(int(mapping.get(str(finish_reason), SegmentStatus.PENDING)))
        return torch.tensor(values, dtype=torch.long)

    def build(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        """Fill exactly the existing AR frontier from one result per row."""
        frontier = sample.frontier_gen_part(ARSamplingParams)
        require(
            len(per_request) == len(frontier.sample_ids),
            f"Qwen3OmniThinkerOutputAdapter: result count {len(per_request)} "
            f"!= frontier id count {len(frontier.sample_ids)}",
        )
        # Missing stage outputs are infrastructure failures, not valid empty
        # assistant turns. Validate before any best-effort helper runs.
        for group in per_request:
            self._stage0(group)

        segment = self.build_segment(sample, per_request)
        if segment is None:
            # A present stage may legitimately finish without an emitted token.
            empty = [torch.zeros(0, dtype=torch.long) for _ in per_request]
            empty_logp = [torch.zeros(0, dtype=torch.float32) for _ in per_request]
            segment = TextSegment.pack(tokens=empty, log_probs=empty_logp)
        decoded = self.build_decoded(sample, per_request)
        conditions = self.build_conditions(sample, per_request)
        return sample.replace_frontier(
            frontier.fill(
                segment=segment,
                primitives={"text": decoded},
                conditions=conditions,
                status=self._status(per_request),
            )
        )


@register_adapter("qwen3_omni_thinker")
class Qwen3OmniThinkerAdapter(ModelAdapter):
    """Qwen3-Omni Thinker — text/video → AR text (single stage, TP>1, LoRA)."""

    # TODO: This anchored AR topology is temporarily migrated from unified
    # models; replace these knobs with formal TP/DP/PP support.
    stage_yaml = "qwen3_omni_thinker_only_rl_1x4.yaml"
    stage_yaml_source = "local"
    omni_mode = None
    needs_sigmas = False
    needs_driver_tokenizer = False
    ar_lora_passthrough = True
    clear_cuda_visible = True
    lora_copy_transport = True

    def __init__(
        self,
        config: Any,
        model_config: Any,
        *,
        strategy: Any = None,
        tokenize_fn: Any = None,
    ) -> None:
        super().__init__(config, model_config, strategy=strategy, tokenize_fn=tokenize_fn)

        mc = model_config
        model_path = str(config.model_path)
        video_fps = float(getattr(mc, "video_fps", 1.0)) if mc is not None else 1.0
        video_max_frames = getattr(mc, "video_max_frames", None) if mc is not None else None
        video_max_pixels = getattr(mc, "video_max_pixels", None) if mc is not None else None
        use_audio_in_video = bool(getattr(mc, "use_audio_in_video", False)) if mc is not None else False
        max_prompt_length = int(getattr(mc, "max_prompt_length", 12288)) if mc is not None else 12288
        config_video_fps = getattr(config, "video_fps", None)
        config_video_max_pixels = getattr(config, "video_max_pixels", None)
        config_use_audio_in_video = getattr(config, "use_audio_in_video", None)
        config_max_prompt_length = getattr(config, "max_prompt_length", None)
        if config_video_fps is not None:
            video_fps = float(config_video_fps)
        if config_video_max_pixels is not None:
            video_max_pixels = int(config_video_max_pixels)
        if config_use_audio_in_video is not None:
            use_audio_in_video = bool(config_use_audio_in_video)
        if config_max_prompt_length is not None:
            max_prompt_length = int(config_max_prompt_length)
        system_instruction = getattr(mc, "system_instruction", None) if mc is not None else None
        chat_template_kwargs = dict(getattr(config, "chat_template_kwargs", {}) or {})

        logger.info(
            "Resolved Qwen3-Omni rollout adapter config: model_path=%s video_fps=%s "
            "video_max_frames=%s video_max_pixels=%s use_audio_in_video=%s max_prompt_length=%s "
            "system_instruction_set=%s model_config_available=%s",
            model_path,
            video_fps,
            video_max_frames,
            video_max_pixels,
            use_audio_in_video,
            max_prompt_length,
            system_instruction is not None,
            model_config is not None,
        )

        self.input_adapter = Qwen3OmniThinkerInputAdapter(
            self.modality,
            model_path=model_path,
            video_fps=video_fps,
            video_max_frames=video_max_frames,
            video_max_pixels=video_max_pixels,
            use_audio_in_video=use_audio_in_video,
            max_prompt_length=max_prompt_length,
            system_instruction=system_instruction,
            chat_template_kwargs=chat_template_kwargs,
        )
        self.output_adapter = Qwen3OmniThinkerOutputAdapter(self.modality, self.input_adapter)

    def schedule_policy(self) -> Any:
        """Return no diffusion schedule for the AR-only stage."""
        return None

    def validate_request(self, sample: Sample) -> None:
        sample.frontier_gen_part(ARSamplingParams)
        # Rendering is the single validation source for supported modalities,
        # frontier alignment, and the one-video contract.
        conversations = build_video_messages(sample.turns())
        require(bool(conversations), f"modality={self.modality!r} requires text/video conditioning turns.")

    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        return self.input_adapter.build(sample)

    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        return self.output_adapter.build(sample, per_request)


__all__ = ["Qwen3OmniThinkerAdapter", "Qwen3OmniThinkerInputAdapter"]
