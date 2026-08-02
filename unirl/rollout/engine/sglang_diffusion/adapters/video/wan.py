"""WAN-family SGLang diffusion adapters."""

from __future__ import annotations

from typing import Any, Dict

from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.video.base import VideoAdapter
from unirl.types.sample import Sample


@register_adapter("wan22")
class Wan22T2VAdapter(VideoAdapter):
    """WAN 2.2-A14B T2V with secondary low-noise guidance."""

    def build_sampling(self, sample: Sample, *, diffusion: Any) -> Dict[str, Any]:
        kwargs = super().build_sampling(sample, diffusion=diffusion)
        guidance_scale_2 = getattr(diffusion, "guidance_scale_2", None)
        if guidance_scale_2 is not None:
            kwargs["guidance_scale_2"] = float(guidance_scale_2)
        return kwargs


@register_adapter("wan21")
class Wan21T2VAdapter(VideoAdapter):
    """WAN 2.1 T2V with the shared video response contract."""

    pass


__all__ = ["Wan21T2VAdapter", "Wan22T2VAdapter"]
