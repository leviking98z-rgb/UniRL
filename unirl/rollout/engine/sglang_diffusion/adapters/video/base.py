"""Shared response contract for SGLang diffusion video adapters.

``VideoAdapter`` owns the six-dimensional video trajectory and decoded-video
packing shared by WAN and HunyuanVideo. ``MochiAdapter`` intentionally remains
on the legacy image path until that family has a verified video reward baseline.
"""

from __future__ import annotations

from typing import List, Optional

from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import make_video_segment


class VideoAdapter(ImageAdapter):
    """Base for true video-output families (6-D latent trajectory → ``Videos``).

    Reuses ``ImageAdapter``'s request side verbatim and overrides only the
    response-shape variation points: the segment is stamped ``Modality.VIDEO``
    and decoded media is packed as ``Videos``.
    """

    segment_factory = staticmethod(make_video_segment)

    def build_segment(
        self,
        sample: Sample,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """Collect and assemble ``[B, T+1, C, F, H, W]`` trajectories."""
        trajectory = utils.collect_trajectory_latents(results)
        if trajectory.ndim != 6:
            raise ValueError(
                f"{self.model_family}: expected a 6-D video-form trajectory "
                f"[B, T+1, C, F, H, W]; got rank {trajectory.ndim}, "
                f"shape {tuple(trajectory.shape)}."
            )
        return utils.build_latent_segment(
            trajectory,
            results=results,
            expected_sigmas=sample.frontier_gen_part(DiffusionSamplingParams).sampling_params.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
        )

    def build_decoded(self, sample: Sample, results: List[RawResult]):
        del sample
        return utils.stack_decoded_videos(results)


@register_adapter("mochi")
class MochiAdapter(ImageAdapter):
    """Mochi's legacy image-path behavior."""

    # Drop decoded 4-D video instead of squeezing a single frame to an image.
    squeeze_single_frame_4d = False


__all__ = ["VideoAdapter", "MochiAdapter"]
