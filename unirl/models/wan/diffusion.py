"""Diffusion kernel and stage shared by the WAN model family.

WAN 2.1 and WAN 2.2 use the same latent geometry, CFG batching, optional
I2V condition concat, SDE transition, and single-stream rollout/replay
runner. WAN 2.2 adds one variant hook: select the high- or low-noise
transformer and its guidance scale from the current sigma.

Historical imports from ``unirl.models.wan21.diffusion`` and
``unirl.models.wan22.diffusion`` re-export the concrete classes defined
here for one-release compatibility.
"""

from __future__ import annotations

from functools import partial
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import torch

from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.diffusion_runner import SingleStreamDiffusionRunner
from unirl.models.types.replay_result import ReplayResult
from unirl.models.wan.conditions import WANConditions
from unirl.models.wan.geometry import wan_latent_shape
from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment, make_video_segment
from unirl.utils.dtypes import parse_torch_dtype

_WAN_TIMESTEP_SCALE: float = 1000.0
_MISSING = object()


class WANDiffusionStep(DiffusionStep[Any, WANConditions]):
    """Shared WAN CFG predictor and SDE transition."""

    def _route(
        self,
        model: Any,
        sigma: torch.Tensor,
        guidance_scale: float,
        guidance_scale_2: Optional[float],
    ) -> Tuple[float, Dict[str, Any]]:
        del model, sigma, guidance_scale_2
        return float(guidance_scale), {}

    def predict_noise(
        self,
        model: Any,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: WANConditions,
        *,
        guidance_scale: float,
        guidance_scale_2: Optional[float] = None,
    ) -> torch.Tensor:
        """Run the active WAN transformer with optional batched CFG."""
        owner = type(self).__name__
        if conditions.text is None:
            raise ValueError(f"{owner}.predict_noise: conditions.text is None")
        prompt_embeds = conditions.text.embeds
        if prompt_embeds is None:
            raise ValueError(f"{owner}.predict_noise: conditions.text.embeds is None")

        active_guidance, route_kwargs = self._route(
            model,
            sigma,
            guidance_scale,
            guidance_scale_2,
        )
        batch_size = int(sample.shape[0])
        timestep = sigma * _WAN_TIMESTEP_SCALE
        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)
        elif int(timestep.shape[0]) != batch_size:
            timestep = timestep.expand(batch_size)

        embeds_dtype = prompt_embeds.dtype
        sample_cast = sample.to(dtype=embeds_dtype)
        image_latent = conditions.image_latent
        if image_latent is not None and image_latent.latents is not None:
            sample_cast = torch.cat(
                [
                    sample_cast,
                    image_latent.latents.to(
                        device=sample_cast.device,
                        dtype=embeds_dtype,
                    ),
                ],
                dim=1,
            )

        image_embed = conditions.image_embed
        image_embeds = image_embed.embeds if image_embed is not None else None
        if image_embeds is not None:
            image_embeds = image_embeds.to(
                device=sample_cast.device,
                dtype=embeds_dtype,
            )

        transformer_kwargs = dict(route_kwargs)
        if active_guidance > 1.0:
            negative = conditions.negative_text
            negative_embeds = (
                negative.embeds
                if negative is not None and negative.embeds is not None
                else torch.zeros_like(prompt_embeds)
            )
            if image_embeds is not None:
                transformer_kwargs["encoder_hidden_states_image"] = torch.cat(
                    [image_embeds, image_embeds],
                    dim=0,
                )
            noise_pred = model.transformer(
                hidden_states=torch.cat([sample_cast, sample_cast], dim=0),
                encoder_hidden_states=torch.cat(
                    [negative_embeds, prompt_embeds],
                    dim=0,
                ),
                timestep=torch.cat([timestep, timestep], dim=0),
                return_dict=False,
                **transformer_kwargs,
            )[0]
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            return noise_pred_uncond + active_guidance * (noise_pred_cond - noise_pred_uncond)

        if image_embeds is not None:
            transformer_kwargs["encoder_hidden_states_image"] = image_embeds
        return model.transformer(
            hidden_states=sample_cast,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            return_dict=False,
            **transformer_kwargs,
        )[0]

    def forward(
        self,
        *,
        strategy: StepStrategy,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run one SDE transition from a precomputed noise prediction."""
        return strategy.denoise(
            noise_pred=noise_pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=eta,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            step_index=step_index,
        )

    def step(
        self,
        model: Any,
        conditions: WANConditions,
        *,
        strategy: StepStrategy,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        guidance_scale: float,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
        guidance_scale_2: Any = _MISSING,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the WAN model forward and one SDE transition."""
        predict_kwargs: Dict[str, Any] = {"guidance_scale": guidance_scale}
        if guidance_scale_2 is not _MISSING:
            predict_kwargs["guidance_scale_2"] = guidance_scale_2
        noise_pred = self.predict_noise(
            model,
            sample,
            sigma,
            conditions,
            **predict_kwargs,
        )
        return self.forward(
            strategy=strategy,
            noise_pred=noise_pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            eta=eta,
            step_index=step_index,
        )

    def step_with_logp(
        self,
        model: Any,
        conditions: WANConditions,
        *,
        strategy: StepStrategy,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        guidance_scale: float,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
        guidance_scale_2: Any = _MISSING,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the WAN model forward and one replay-capable transition."""
        return self.step(
            model,
            conditions,
            strategy=strategy,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            guidance_scale=guidance_scale,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            eta=eta,
            step_index=step_index,
            guidance_scale_2=guidance_scale_2,
        )


class WAN21DiffusionStep(WANDiffusionStep):
    """WAN 2.1 single-transformer diffusion step."""


class WAN22DiffusionStep(WANDiffusionStep):
    """WAN 2.2 dual-transformer diffusion step."""

    @staticmethod
    def _select_for_sigma(
        sigma: torch.Tensor,
        guidance_scale: float,
        guidance_scale_2: Optional[float],
        *,
        boundary_ratio: float,
    ) -> Tuple[bool, float]:
        """Return the active transformer branch and guidance scale."""
        sigma_val = float(sigma.item()) if sigma.dim() == 0 else float(sigma.flatten()[0].item())
        if sigma_val >= boundary_ratio:
            return True, float(guidance_scale)
        active = float(guidance_scale_2) if guidance_scale_2 is not None else float(guidance_scale)
        return False, active

    def _route(
        self,
        model: Any,
        sigma: torch.Tensor,
        guidance_scale: float,
        guidance_scale_2: Optional[float],
    ) -> Tuple[float, Dict[str, Any]]:
        use_high_noise, active_guidance = self._select_for_sigma(
            sigma,
            guidance_scale,
            guidance_scale_2,
            boundary_ratio=float(model.boundary_ratio),
        )
        return active_guidance, {"use_high_noise": use_high_noise}


class WANDiffusionStage(DiffusionStage[WANConditions]):
    """Shared WAN latent geometry and single-stream runner wiring."""

    _no_split_modules: ClassVar[Tuple[str, ...]] = ("WanTransformerBlock",)
    _SPATIAL_DOWNSAMPLE: ClassVar[int] = 8
    _TEMPORAL_DOWNSAMPLE: ClassVar[int] = 4
    _DEFAULT_LATENT_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        *,
        model: Any,
        step: WANDiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model
        self.step = step
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(
            autocast_precision,
            field_name="autocast_precision",
        )
        self.trajectory_dtype = parse_torch_dtype(
            trajectory_precision,
            field_name="trajectory_precision",
        )
        self.logprob_dtype = parse_torch_dtype(
            logprob_precision,
            field_name="logprob_precision",
        )
        self.vae_scale_factor = self._SPATIAL_DOWNSAMPLE
        self.temporal_scale_factor = self._TEMPORAL_DOWNSAMPLE
        self.latent_channels = int(
            getattr(
                getattr(model.vae, "config", None),
                "z_dim",
                self._DEFAULT_LATENT_CHANNELS,
            )
        )
        self.runner = SingleStreamDiffusionRunner(
            strategy=strategy,
            autocast_dtype=self.autocast_dtype,
            trajectory_dtype=self.trajectory_dtype,
            logprob_dtype=self.logprob_dtype,
            owner=type(self).__name__,
        )

    def _latent_shape(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
    ) -> Tuple[int, int, int, int]:
        return wan_latent_shape(
            num_frames=num_frames,
            height=height,
            width=width,
            latent_channels=self.latent_channels,
            spatial_downsample=self.vae_scale_factor,
            temporal_downsample=self.temporal_scale_factor,
        )

    def _variant_step_kwargs(
        self,
        params: DiffusionSamplingParams,
    ) -> Dict[str, Any]:
        del params
        return {}

    def _predict_noise_kwargs(
        self,
        params: DiffusionSamplingParams,
    ) -> Dict[str, Any]:
        return self._variant_step_kwargs(params)

    def _transition(
        self,
        conditions: WANConditions,
        params: DiffusionSamplingParams,
    ) -> Any:
        return partial(
            self.step.step_with_logp,
            self.model,
            conditions,
            strategy=self.strategy,
            guidance_scale=float(params.guidance_scale),
            **self._variant_step_kwargs(params),
        )

    def diffuse(
        self,
        conditions: WANConditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run WAN sampling and return its video latent trajectory."""
        owner = type(self).__name__
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError(f"{owner}.diffuse: conditions.text.embeds is None")
        prompt_embeds = conditions.text.embeds
        latent_shape = self._latent_shape(
            num_frames=int(params.num_frames),
            height=int(params.height),
            width=int(params.width),
        )
        return self.runner.sample(
            schedule=schedule,
            params=params,
            batch_size=int(prompt_embeds.shape[0]),
            latent_shape=latent_shape,
            device=prompt_embeds.device,
            initial_latents=initial_latents,
            transition=self._transition(conditions, params),
            segment_factory=make_video_segment,
            shape_description=(
                f"for num_frames={int(params.num_frames)}, height={int(params.height)}, width={int(params.width)}"
            ),
        )

    def replay(
        self,
        conditions: WANConditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Replay WAN SDE transitions through the same variant kernel."""
        return self.runner.replay(
            segment=segment,
            params=params,
            transition=self._transition(conditions, params),
            step_indices=step_indices,
        )

    def predict_noise_at_step(
        self,
        conditions: WANConditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Run one WAN noise prediction without a scheduler transition."""
        return self.step.predict_noise(
            self.model,
            sample,
            sigma,
            conditions,
            guidance_scale=float(params.guidance_scale),
            **self._predict_noise_kwargs(params),
        )

    def trainable_module(self) -> "torch.nn.Module":
        """Return the single trainable transformer surface."""
        return self.model.transformer


class WAN21DiffusionStage(WANDiffusionStage):
    """WAN 2.1 single-transformer rollout and replay stage."""


class WAN22DiffusionStage(WANDiffusionStage):
    """WAN 2.2 dual-transformer rollout and replay stage."""

    def _variant_step_kwargs(
        self,
        params: DiffusionSamplingParams,
    ) -> Dict[str, Any]:
        guidance_scale_2 = (
            params.guidance_scale_2 if params.guidance_scale_2 is not None else self.model.guidance_scale_2
        )
        return {"guidance_scale_2": guidance_scale_2}

    def _predict_noise_kwargs(
        self,
        params: DiffusionSamplingParams,
    ) -> Dict[str, Any]:
        return {"guidance_scale_2": params.guidance_scale_2}


__all__ = [
    "WAN21DiffusionStage",
    "WAN21DiffusionStep",
    "WAN22DiffusionStage",
    "WAN22DiffusionStep",
    "WANDiffusionStage",
    "WANDiffusionStep",
]
