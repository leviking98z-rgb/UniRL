"""Build role-aware token and TMRoPE video conditions for Qwen3-Omni."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from unirl.models.types.conversations import build_video_messages
from unirl.types.primitives import Texts
from unirl.types.sample import Turn

from .bundle import Qwen3OmniBundle
from .conditions import Qwen3OmniARConditions
from .processor import Qwen3OmniProcessorCodec, Qwen3OmniProcessorResult

Qwen3OmniChatInput = Union[List[Turn], Texts]


class Qwen3OmniChatTemplateStage:
    def __init__(
        self,
        bundle: Qwen3OmniBundle,
        *,
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 4096,
        pad_to_max_length: bool = False,
        video_fps: float = 1.0,
        video_max_frames: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        use_audio_in_video: bool = False,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.bundle = bundle
        self.system_instruction = system_instruction
        self.max_prompt_length = int(max_prompt_length)
        # Cross-worker CONCAT requires a common sequence length when enabled.
        self.pad_to_max_length = bool(pad_to_max_length)
        self.video_fps = float(video_fps)
        self.video_max_frames = int(video_max_frames) if video_max_frames is not None else None
        self.video_max_pixels = int(video_max_pixels) if video_max_pixels else None
        self.use_audio_in_video = bool(use_audio_in_video)
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.processor_codec = Qwen3OmniProcessorCodec(
            processor=bundle.processor,
            tokenizer=bundle.tokenizer,
            max_prompt_length=self.max_prompt_length,
            video_fps=self.video_fps,
            video_max_frames=self.video_max_frames,
            video_max_pixels=self.video_max_pixels,
            use_audio_in_video=self.use_audio_in_video,
            chat_template_kwargs=self.chat_template_kwargs,
            pad_to_max_length=self.pad_to_max_length,
            device=bundle.device,
            dtype=bundle.dtype,
            owner=type(self).__name__,
        )

    def embed(
        self,
        value: Qwen3OmniChatInput,
        videos: Optional[List[Optional[Any]]] = None,
    ) -> Qwen3OmniARConditions:
        """Render Sample-native turns or supervised text/video rows.

        ``List[Turn]`` is the rollout path and retains the complete role-aware
        trajectory. ``Texts`` plus optional videos is the supervised path. Both
        normalize to the same processor-message representation so rollout and
        replay share the exact encoding stored on the generated Part.
        """
        if isinstance(value, Texts):
            batch_size = len(value)
            if batch_size == 0:
                raise ValueError("Qwen3OmniChatTemplateStage.embed: expected at least one text row.")
            video_rows = [None] * batch_size if videos is None else list(videos)
            if len(video_rows) != batch_size:
                raise ValueError(
                    f"Qwen3OmniChatTemplateStage.embed: videos length {len(video_rows)} != text batch {batch_size}."
                )
            conversations = []
            for text, video in zip(value.texts, video_rows):
                messages: List[Dict[str, Any]] = []
                if self.system_instruction is not None:
                    messages.append({"role": "system", "content": self.system_instruction})
                content: List[Dict[str, Any]] = []
                if video is not None:
                    content.append({"type": "video", "video": video})
                content.append({"type": "text", "text": text})
                messages.append({"role": "user", "content": content})
                conversations.append(messages)
        else:
            if videos is not None:
                raise ValueError(
                    "Qwen3OmniChatTemplateStage.embed: videos must be carried by Turn content; "
                    "the separate videos argument is only valid with Texts input."
                )
            if not value:
                raise ValueError("Qwen3OmniChatTemplateStage.embed: expected at least one conversation turn.")
            conversations = build_video_messages(value, self.system_instruction)

        return self.embed_messages(conversations)

    def embed_messages(
        self,
        conversations: List[List[Dict[str, Any]]],
    ) -> Qwen3OmniARConditions:
        """Return the replay conditions from the canonical processor result."""
        return self.process_messages(conversations).replay_conditions

    def process_messages(
        self,
        conversations: List[List[Dict[str, Any]]],
    ) -> Qwen3OmniProcessorResult:
        """Expose the canonical typed result for train-side consumers."""
        return self.processor_codec.encode_messages(conversations)


__all__ = ["Qwen3OmniChatInput", "Qwen3OmniChatTemplateStage"]
