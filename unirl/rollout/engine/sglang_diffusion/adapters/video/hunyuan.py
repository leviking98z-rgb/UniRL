"""HunyuanVideo SGLang diffusion adapter."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.video.base import VideoAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.conditions.text import TextEmbedCondition


@register_adapter("hunyuan_video")
class HunyuanVideoAdapter(VideoAdapter):
    """HunyuanVideo-1.0 T2V with video output and dual text conditions."""

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        """Keep HunyuanVideo's LLaMA and pooled-CLIP streams separate."""
        require(bool(results), "HunyuanVideo: cannot build conditions from empty results")

        llama_conditions: List[TextEmbedCondition] = []
        clip_list: List[torch.Tensor] = []
        mask_presence: List[bool] = []
        for result in results:
            prompt_embeds = result.prompt_embeds
            require(
                isinstance(prompt_embeds, (list, tuple)) and len(prompt_embeds) >= 2,
                "HunyuanVideo: expected prompt_embeds=[LLaMA, CLIP-pooled]; got "
                f"{type(prompt_embeds).__name__} with "
                f"{len(prompt_embeds) if isinstance(prompt_embeds, (list, tuple)) else 'n/a'} entries",
            )
            llama, clip = prompt_embeds[:2]
            require(
                torch.is_tensor(llama) and llama.ndim == 3,
                "HunyuanVideo: LLaMA prompt embed must be [B, seq, hidden]",
            )
            require(
                torch.is_tensor(clip) and clip.ndim in (2, 3),
                "HunyuanVideo: pooled CLIP embed must be [B, hidden] or [B, 1, hidden]",
            )
            require(
                int(llama.shape[0]) == int(clip.shape[0]),
                "HunyuanVideo: LLaMA and CLIP prompt embed batch sizes must match",
            )

            attention_mask = None
            encoder_masks = result.encoder_attention_mask
            if isinstance(encoder_masks, (list, tuple)) and encoder_masks and encoder_masks[0] is not None:
                attention_mask = encoder_masks[0]
                require(
                    torch.is_tensor(attention_mask)
                    and attention_mask.ndim == 2
                    and tuple(attention_mask.shape) == tuple(llama.shape[:2]),
                    "HunyuanVideo: LLaMA attention mask must match [B, seq]",
                )

            mask_presence.append(attention_mask is not None)
            llama_conditions.append(
                TextEmbedCondition(
                    embeds=llama.detach().cpu(),
                    attn_mask=attention_mask.detach().cpu() if attention_mask is not None else None,
                )
            )
            clip_list.append(clip.detach().cpu().reshape(int(clip.shape[0]), -1))

        require(
            all(mask_presence) or not any(mask_presence),
            "HunyuanVideo: LLaMA attention masks must be present for every result or none",
        )
        return {
            "text_llama": TextEmbedCondition.concat(llama_conditions),
            "pooled_clip": TextEmbedCondition(embeds=torch.cat(clip_list, dim=0)),
        }


__all__ = ["HunyuanVideoAdapter"]
