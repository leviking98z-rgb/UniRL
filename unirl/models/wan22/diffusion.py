"""WAN 2.2 diffusion: dual-transformer per-step kernel + rollout-level stage.

WAN 2.2 introduces sigma-boundary-based routing between two transformer
copies:

- ``sigma >= boundary_ratio`` → ``high_noise`` branch (coarse structure)
- ``sigma <  boundary_ratio`` → ``low_noise`` branch (detail
  refinement); optionally with its own ``guidance_scale_2``

The routing is **per-step, per-sigma**, so it belongs in
:class:`WAN22DiffusionStep` (the kernel), not in the stage loop. The
stage layer is responsible for bookkeeping only and is otherwise
identical to WAN 2.1.

CFG batching follows the WAN 2.1 pattern (``[uncond, cond]`` along
batch dim, ``chunk(2)``, interpolate). The transformer call goes
through ``WAN22Bundle.transformer.forward(use_high_noise=..., ...)``
(the :class:`WanDualTransformer` composite) so branch routing stays
behind the stage abstraction. FSDPPolicy does not root-wrap the
composite; it discovers and fully-shards the ``WanTransformerBlock``
instances under both branches.

The replay path uses the same per-step kernel with ``prev_sample`` set;
this means each replay step also routes by sigma, mirroring how the
rollout was produced.

Math derived from ``models/wan22.py::forward_denoiser`` and
``samplers/fsdp/wan22_sampler.py`` (do NOT import legacy code).
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

from .bundle import WAN22Bundle

_WAN_TIMESTEP_SCALE: float = 1000.0


class WAN22DiffusionStep(DiffusionStep[WAN22Bundle, WANConditions]):
    """Per-step WAN 2.2 denoising kernel — stateless, dual-transformer routing.

    For each call, decides whether to route through the high- or
    low-noise sub-transformer based on the current sigma vs the
    bundle's ``boundary_ratio``. The low-noise branch optionally uses
    a separate guidance scale (``guidance_scale_2``); when ``None``,
    the same scale flows through both branches.
    """

    @staticmethod
    def _select_for_sigma(
        sigma: torch.Tensor,
        guidance_scale: float,
        guidance_scale_2: Optional[float],
        *,
        boundary_ratio: float,
    ) -> Tuple[bool, float]:
        """Decide which sub-transformer to use and which guidance to apply.

        Returns ``(use_high_noise, active_guidance)``.

        Boundary policy (matches ``models/wan22.py::_select_guidance_for_sigma``
        in spirit, but only branches on ``sigma`` directly — WAN 2.2's
        sigma schedule lives in ``[0, 1]`` and ``boundary_ratio`` is
        defined in that same domain):

        - ``sigma >= boundary_ratio`` → high_noise + ``guidance_scale``
        - ``sigma <  boundary_ratio`` → low_noise + ``guidance_scale_2``
          (falls back to ``guidance_scale`` when the per-stage scale
          is ``None``)

        Per-sample sigma policy: when ``sigma`` is a 1D tensor of per-
        sample values, we read ``sigma[0]`` to pick the branch for the
        whole batch. This is consistent with how both rollout and replay
        invoke the kernel — every call site here passes a single
        ``schedule[i]`` scalar (or broadcasts it), so all samples share
        the same sigma in any one ``predict_noise`` call. If a future
        consumer ever ships heterogeneous per-sample sigmas through this
        step, this assumption must be revisited (it would require
        per-sample routing — likely two forwards followed by per-sample
        gather).
        """
        sigma_val = float(sigma.item()) if sigma.dim() == 0 else float(sigma.flatten()[0].item())
        if sigma_val >= boundary_ratio:
            return True, float(guidance_scale)
        active = float(guidance_scale_2) if guidance_scale_2 is not None else float(guidance_scale)
        return False, active

    def predict_noise(
        self,
        model: WAN22Bundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: WANConditions,
        *,
        guidance_scale: float,
        guidance_scale_2: Optional[float] = None,
    ) -> torch.Tensor:
        """Run dual-transformer noise prediction with optional CFG.

        Routes by sigma against ``model.boundary_ratio`` then applies CFG
        in the active branch. The call always goes through
        ``model.transformer.forward`` (the :class:`WanDualTransformer`
        composite) so sampling code depends on one stage-level routing
        surface rather than reaching into high/low sub-transformers.
        """
        if conditions.text is None:
            raise ValueError("WAN22DiffusionStep.predict_noise: conditions.text is None")
        text = conditions.text
        prompt_embeds = text.embeds
        if prompt_embeds is None:
            raise ValueError("WAN22DiffusionStep.predict_noise: conditions.text.embeds is None")

        use_high_noise, active_guidance = self._select_for_sigma(
            sigma,
            guidance_scale,
            guidance_scale_2,
            boundary_ratio=model.boundary_ratio,
        )

        batch_size = int(sample.shape[0])
        timestep = sigma * _WAN_TIMESTEP_SCALE
        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)
        elif int(timestep.shape[0]) != batch_size:
            timestep = timestep.expand(batch_size)

        embeds_dtype = prompt_embeds.dtype
        sample_cast = sample.to(dtype=embeds_dtype)

        # I2V channel concat: when an image-condition latent is present,
        # prepend it on the channel axis (16 noise + 20 mask+image →
        # 36 transformer ``in_channels``). Identical across cond/uncond.
        image_latent = conditions.image_latent
        if image_latent is not None and image_latent.latents is not None:
            sample_cat = torch.cat(
                [sample_cast, image_latent.latents.to(device=sample_cast.device, dtype=embeds_dtype)],
                dim=1,
            )
        else:
            sample_cat = sample_cast

        # I2V CLIP-vision: forward ``encoder_hidden_states_image`` only
        # when the slot is populated. WAN 2.2's mainstream checkpoints
        # have ``image_dim == 0`` and never see this kwarg; the
        # composite :class:`WanDualTransformer` transparently routes
        # ``**kwargs`` to both ``high_noise`` and ``low_noise``.
        image_embed = conditions.image_embed
        image_embeds = image_embed.embeds if image_embed is not None and image_embed.embeds is not None else None
        extra: Dict[str, Any] = {}
        if image_embeds is not None:
            image_embeds = image_embeds.to(device=sample_cast.device, dtype=embeds_dtype)

        if active_guidance > 1.0:
            neg = conditions.negative_text
            if neg is not None and neg.embeds is not None:
                negative_prompt_embeds = neg.embeds
            else:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)

            if image_embeds is not None:
                extra["encoder_hidden_states_image"] = torch.cat([image_embeds, image_embeds], dim=0)

            noise_pred = model.transformer(
                use_high_noise=use_high_noise,
                hidden_states=torch.cat([sample_cat, sample_cat], dim=0),
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
                timestep=torch.cat([timestep, timestep], dim=0),
                return_dict=False,
                **extra,
            )[0]
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            return noise_pred_uncond + active_guidance * (noise_pred_cond - noise_pred_uncond)

        if image_embeds is not None:
            extra["encoder_hidden_states_image"] = image_embeds

        return model.transformer(
            use_high_noise=use_high_noise,
            hidden_states=sample_cat,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            return_dict=False,
            **extra,
        )[0]

    # ---- Protocol surface ---------------------------------------------------

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
        """Run one SDE transition given a precomputed ``noise_pred``."""
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
        model: WAN22Bundle,
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
        guidance_scale_2: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run dual-transformer forward + SDE transition. End-to-end one step.

        ``guidance_scale_2`` is the WAN 2.2 extension over the
        Protocol's ``step`` signature (Protocol accepts ``**kwargs``-style
        extension for model-specific knobs; here it's keyword-only with
        a default of ``None`` so we stay backwards compatible with
        callers that don't pass it).
        """
        noise_pred = self.predict_noise(
            model,
            sample,
            sigma,
            conditions,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
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
        model: WAN22Bundle,
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
        guidance_scale_2: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run dual-transformer forward + SDE transition (delegates to ``step``)."""
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


class WAN22DiffusionStage(DiffusionStage[WANConditions]):
    """WAN 2.2 T2V rollout-level diffusion stage with dual-transformer routing.

    Owns the SDE ``strategy`` (stateful strategies require a stable
    instance across the loop) + bundle + kernel + precision policy.
    The kernel routes per-step between high- and low-noise transformers
    based on the bundle's ``boundary_ratio``; the stage loop is
    otherwise identical to WAN 2.1.

    ``replay`` also routes per-sigma, mirroring the rollout exactly.

    ``_no_split_modules`` provides the FSDPPolicy fallback for HF
    auto-discovery; WanTransformerBlock is shared by both
    sub-transformers in :class:`WanDualTransformer`.
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = ("WanTransformerBlock",)

    _SPATIAL_DOWNSAMPLE: ClassVar[int] = 8
    _TEMPORAL_DOWNSAMPLE: ClassVar[int] = 4
    _DEFAULT_LATENT_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        *,
        model: WAN22Bundle,
        step: WAN22DiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model
        self.step = step
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")
        self.vae_scale_factor = self._SPATIAL_DOWNSAMPLE
        self.temporal_scale_factor = self._TEMPORAL_DOWNSAMPLE
        self.latent_channels = int(getattr(getattr(model.vae, "config", None), "z_dim", self._DEFAULT_LATENT_CHANNELS))
        self.runner = SingleStreamDiffusionRunner(
            strategy=strategy,
            autocast_dtype=self.autocast_dtype,
            trajectory_dtype=self.trajectory_dtype,
            logprob_dtype=self.logprob_dtype,
            owner=type(self).__name__,
        )

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    def _latent_shape(self, *, num_frames: int, height: int, width: int) -> Tuple[int, int, int, int]:
        return wan_latent_shape(
            num_frames=num_frames,
            height=height,
            width=width,
            latent_channels=self.latent_channels,
            spatial_downsample=self.vae_scale_factor,
            temporal_downsample=self.temporal_scale_factor,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def diffuse(
        self,
        conditions: WANConditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run full WAN 2.2 T2V sampling. Returns a ``LatentSegment``.

        ``initial_latents`` (optional) — x_T resolved from the request
        ``Sample``'s diffusion generation Part; see
        :class:`SD3DiffusionStage.diffuse` for the contract.
        """
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("WAN22DiffusionStage.diffuse: conditions.text.embeds is None")
        prompt_embeds = conditions.text.embeds
        device = prompt_embeds.device
        batch_size = int(prompt_embeds.shape[0])
        latent_shape = self._latent_shape(
            num_frames=int(params.num_frames),
            height=int(params.height),
            width=int(params.width),
        )

        # Per-stage CFG scale: prefer the request-time value if provided,
        # else fall back to the bundle-time default (which itself defaults
        # to None → reuse the primary guidance scale).
        guidance_scale_2 = (
            params.guidance_scale_2 if params.guidance_scale_2 is not None else self.model.guidance_scale_2
        )
        transition = partial(
            self.step.step_with_logp,
            self.model,
            conditions,
            strategy=self.strategy,
            guidance_scale=float(params.guidance_scale),
            guidance_scale_2=guidance_scale_2,
        )
        return self.runner.sample(
            schedule=schedule,
            params=params,
            batch_size=batch_size,
            latent_shape=latent_shape,
            device=device,
            initial_latents=initial_latents,
            transition=transition,
            segment_factory=make_video_segment,
            shape_description=(
                f"for num_frames={int(params.num_frames)}, height={int(params.height)}, width={int(params.width)}"
            ),
        )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self,
        conditions: WANConditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Segment-based log-prob replay. Routes by sigma exactly as rollout."""
        guidance_scale_2 = (
            params.guidance_scale_2 if params.guidance_scale_2 is not None else self.model.guidance_scale_2
        )
        transition = partial(
            self.step.step_with_logp,
            self.model,
            conditions,
            strategy=self.strategy,
            guidance_scale=float(params.guidance_scale),
            guidance_scale_2=guidance_scale_2,
        )
        return self.runner.replay(
            segment=segment,
            params=params,
            transition=transition,
            step_indices=step_indices,
        )

    # ------------------------------------------------------------------
    # Single-step noise prediction (forward-process algorithms: DiffusionNFT et al.)
    # ------------------------------------------------------------------

    def predict_noise_at_step(
        self,
        conditions: WANConditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward — no scheduler iteration.

        Delegates to ``WAN22DiffusionStep.predict_noise``; routing between
        high-noise / low-noise sub-transformers + ``guidance_scale_2``
        handling are owned by the kernel.
        """
        return self.step.predict_noise(
            self.model,
            sample,
            sigma,
            conditions,
            guidance_scale=float(params.guidance_scale),
            guidance_scale_2=getattr(params, "guidance_scale_2", None),
        )

    # ------------------------------------------------------------------
    # Trainable surface for FSDPPolicy
    # ------------------------------------------------------------------

    def trainable_module(self) -> "torch.nn.Module":
        """Return the composite :class:`WanDualTransformer` for FSDP wrapping.

        **FSDP wrapping policy** (matches the v2 FSDP backend,
        ``unirl/train/backend/fsdp.py``):
        the block-wrap policy does NOT call ``fully_shard`` on the
        composite root. It enumerates the ``WanTransformerBlock`` block
        instances inside ``self.model`` and runs ``fully_shard(layer)``
        on each individually (see ``fsdp_policy.py::_enumerate_block_instances``
        + the ``"No root fully_shard"`` comment on the wrap loop).
        Because ``WanDualTransformer`` is a plain ``nn.Module`` whose
        ``named_modules()`` recurses into both ``high_noise.*`` and
        ``low_noise.*``, the block discovery walks both sub-transformers
        and shards every block in both. The composite root itself stays
        unwrapped, exactly like SD3 / HI3.

        **LoRA-on-composite assumption** (worth a GPU smoke):
        LoRA injection (``unirl.train.lora``) calls
        :func:`peft.inject_adapter_in_model` on whatever
        ``trainable_module()`` returns, with a ``target_modules`` list of
        suffix strings (e.g. ``["attn1.to_q", "attn1.to_k", ...]``).
        peft walks ``named_modules()`` and matches by name suffix, so
        for our composite both ``high_noise.transformer_blocks.*.attn1.to_q``
        and ``low_noise.transformer_blocks.*.attn1.to_q`` should be
        replaced with LoRA wrappers, yielding trainable LoRA params in
        BOTH branches.

        Legacy ``WAN22ModelBundle._add_lora_adapters`` did the injection
        explicitly per sub-transformer to be safe; the code relies
        on peft's standard recursive name matching instead. First-run
        verification check (in GPU smoke):
        ``[n for n, p in policy.model.named_parameters() if p.requires_grad]``
        should list LoRA params under both ``high_noise.`` and
        ``low_noise.`` prefixes; if it lists only one branch, fall back
        to explicit per-sub-transformer injection or split the
        ``trainable_module()`` API.
        """
        return self.model.transformer


__all__ = ["WAN22DiffusionStage", "WAN22DiffusionStep"]
