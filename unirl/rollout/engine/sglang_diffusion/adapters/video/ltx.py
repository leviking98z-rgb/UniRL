"""LTX-2 SGLang diffusion adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.video.base import VideoAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams


@register_adapter("ltx2")
class Ltx2T2VAdapter(VideoAdapter):
    """LTX-2 T2V with coupled packed video and audio trajectories."""

    def schedule_policy(self):
        from unirl.models.ltx2.schedule import build_ltx2_schedule_policy

        return build_ltx2_schedule_policy(float(self.model_config.shift))

    def build_sampling(self, sample: Sample, *, diffusion: Any) -> Dict[str, Any]:
        kwargs = super().build_sampling(sample, diffusion=diffusion)
        kwargs["max_sequence_length"] = int(self.model_config.max_sequence_length)

        from unirl.models.ltx2.diffusion import audio_latent_shape
        from unirl.types.noise_recipe import NoiseRecipe

        audio_noise = NoiseRecipe.from_sample(sample).resolve(
            salt="audio",
            latent_shape=audio_latent_shape(diffusion),
        )
        if audio_noise is not None:
            kwargs["initial_audio_noise"] = audio_noise
        return kwargs

    @staticmethod
    def _fuse_audio_condition(results: List[RawResult], field: str) -> Optional[TextEmbedCondition]:
        tensors = []
        for result in results:
            value = utils.fuse_encoder_outputs(getattr(result, field, None))
            if value is not None:
                tensors.append(value.detach().cpu())
        if not tensors:
            return None
        require(
            len(tensors) == len(results),
            f"LTX-2: {field} must be present for every result or none",
        )
        return TextEmbedCondition(embeds=torch.cat(tensors, dim=0))

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        out = super().build_condition(results)
        text = out.get("text")
        negative_text = out.get("negative_text")

        # The LTX connector replaces padded positions with learned registers.
        # Dropping the pre-connector mask matches SGLang's all-valid mask.
        if text is not None:
            out["text"] = TextEmbedCondition(embeds=text.embeds, pooled=text.pooled)
        if negative_text is not None:
            out["negative_text"] = TextEmbedCondition(
                embeds=negative_text.embeds,
                pooled=negative_text.pooled,
            )

        audio_text = self._fuse_audio_condition(results, "audio_prompt_embeds")
        negative_audio_text = self._fuse_audio_condition(results, "negative_audio_prompt_embeds")
        if audio_text is not None:
            out["audio_text"] = audio_text
        if negative_audio_text is not None:
            out["negative_audio_text"] = negative_audio_text
        return out

    def build_segment(
        self,
        sample: Sample,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """Assemble packed video tokens plus the parallel audio trajectory."""
        trajectory = utils.collect_trajectory_latents(results)
        if trajectory.ndim < 3:
            raise ValueError(
                f"ltx2: expected a packed trajectory [B, T+1, ...]; "
                f"got rank {trajectory.ndim}, shape {tuple(trajectory.shape)}."
            )
        audio_trajectory = utils.collect_aux_trajectory_latents(results)
        return utils.build_latent_segment(
            trajectory,
            results=results,
            expected_sigmas=sample.frontier_gen_part(DiffusionSamplingParams).sampling_params.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
            aux_trajectory=audio_trajectory,
        )


__all__ = ["Ltx2T2VAdapter"]
