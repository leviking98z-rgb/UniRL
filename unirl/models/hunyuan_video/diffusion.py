"""HunyuanVideo-1.0 diffusion: per-step kernel + rollout-level stage.

Two classes mirror :mod:`unirl.models.hunyuan_video15.diffusion`:

- :class:`HunyuanVideoDiffusionStep` -- stateless per-step kernel.
  :meth:`predict_noise` passes latents directly (no channel-dim packing),
  builds a ``guidance`` tensor (because ``guidance_embeds=True``), and
  forwards through the transformer; the protocol-matching ``forward`` /
  ``step`` / ``step_with_logp`` ride on top.
- :class:`HunyuanVideoDiffusionStage` -- implements
  ``DiffusionStage[HunyuanVideoConditions]``. Owns the SDE strategy,
  loop bookkeeping, latent shape derivation.

Per-request sampling knobs are read from
:class:`unirl.types.sampling.DiffusionSamplingParams`

Latent geometry
---------------
Video latents are 5D: ``[B, C, T_lat, H_lat, W_lat]`` where
- ``T_lat = (num_frames - 1) // temporal_compression_ratio + 1``
- ``H_lat = height // spatial_compression_ratio``
- ``W_lat = width // spatial_compression_ratio``

The VAE downsamples 8x spatially and 4x temporally on the HunyuanVideo-1.0
checkpoint. ``latent_channels=16``.

No channel-dim packing
----------------------
Unlike HunyuanVideo-1.5 (``in_channels = 2*C+1``), HunyuanVideo-1.0 has
``in_channels=16`` -- latents are passed directly without any packing.

Guidance embedding (no CFG)
---------------------------
The transformer has ``guidance_embeds=True``, which means the guidance
scale is passed as a tensor ``[B]`` via the ``guidance`` kwarg. There is
NO classifier-free guidance (no cond/uncond stacking).

Timestep
--------
The transformer takes ``timestep = sigma * 1000`` (sigma in [0, 1] ->
timestep in [0, 1000]); ``TIMESTEP_SCALE`` is exposed on the step kernel.
"""

from __future__ import annotations

from functools import partial
from typing import ClassVar, Dict, List, Optional, Tuple

import torch

from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.diffusion_runner import SingleStreamDiffusionRunner
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment, make_video_segment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import HunyuanVideoBundle
from .conditions import HunyuanVideoConditions


class HunyuanVideoDiffusionStep(DiffusionStep[HunyuanVideoBundle, HunyuanVideoConditions]):
    """Per-step HunyuanVideo-1.0 denoising kernel -- stateless.

    Extends the :class:`DiffusionStep` protocol with HunyuanVideo-1.0-
    specific per-call kwargs on :meth:`predict_noise`, :meth:`step`, and
    :meth:`step_with_logp`. The protocol surface stays structurally
    compatible because Python protocols are non-strict on extra kwargs.
    """

    # Sigma -> transformer timestep scale (sigma in [0, 1] -> t in [0, 1000]).
    TIMESTEP_SCALE: ClassVar[float] = 1000.0

    def predict_noise(
        self,
        model: HunyuanVideoBundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: HunyuanVideoConditions,
        *,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run the transformer forward. No channel-dim packing, no CFG.

        HunyuanVideo-1.0 uses guidance embedding (``guidance_embeds=True``),
        so ``guidance_scale`` is passed as a ``[B]`` tensor via the
        ``guidance`` kwarg. Returns noise prediction of the same shape as
        ``sample`` (``[B, C, T_lat, H_lat, W_lat]``).
        """
        text_llama = conditions.text_llama
        pooled_clip = conditions.pooled_clip
        if text_llama is None or text_llama.embeds is None:
            raise ValueError("HunyuanVideoDiffusionStep.predict_noise: conditions.text_llama must carry embeds.")
        if pooled_clip is None or pooled_clip.embeds is None:
            raise ValueError("HunyuanVideoDiffusionStep.predict_noise: conditions.pooled_clip must carry embeds.")

        prompt_embeds = text_llama.embeds
        attention_mask = text_llama.attn_mask
        pooled_projections = pooled_clip.embeds

        if sample.ndim != 5:
            raise ValueError(
                f"HunyuanVideoDiffusionStep.predict_noise: expected 5D sample "
                f"[B, C, T, H, W], got {tuple(sample.shape)}"
            )
        batch_size = sample.shape[0]
        device = sample.device
        dtype = prompt_embeds.dtype

        # Sigma -> timestep scaling. Always cast to a [B]-shape tensor on
        # the model's compute dtype.
        if sigma.dim() == 0:
            timestep = sigma.unsqueeze(0).expand(batch_size)
        elif sigma.shape[0] != batch_size:
            timestep = sigma.expand(batch_size)
        else:
            timestep = sigma
        timestep = timestep.to(device=device, dtype=dtype) * self.TIMESTEP_SCALE

        # Guidance embedding: pass guidance_scale as a [B] tensor.
        guidance = torch.full((batch_size,), guidance_scale, device=device, dtype=dtype)

        # No channel-dim packing (in_channels=16, sample is already the
        # correct shape). No CFG (guidance_embeds handles this).
        hidden_states = sample.to(dtype)

        # Build kwargs for the transformer forward.
        kwargs: Dict = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_projections,
            "guidance": guidance,
            "return_dict": False,
        }
        # encoder_attention_mask is optional; only pass if we have it
        # (some prompts may have variable-length sequences that need masking).
        if attention_mask is not None:
            kwargs["encoder_attention_mask"] = attention_mask

        return model.transformer(**kwargs)[0]

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
        model: HunyuanVideoBundle,
        conditions: HunyuanVideoConditions,
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run model forward + SDE transition. End-to-end one diffusion step."""
        noise_pred = self.predict_noise(
            model,
            sample,
            sigma,
            conditions,
            guidance_scale=guidance_scale,
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
        model: HunyuanVideoBundle,
        conditions: HunyuanVideoConditions,
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run model forward + SDE transition.

        Returns ``(prev_sample, log_prob, prev_sample_mean)``. ``log_prob``
        and ``prev_sample_mean`` are ``None`` for deterministic strategies.
        """
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
        )


class HunyuanVideoDiffusionStage(DiffusionStage[HunyuanVideoConditions]):
    """HunyuanVideo-1.0 rollout-level diffusion stage.

    Owns the SDE ``strategy`` (stateful strategies like ``DPM2Strategy``
    require a stable instance across the loop), the bundle, the kernel,
    and the precision policy.

    ``diffuse(conditions, *, schedule, params)`` runs the full sampling
    loop and returns a ``LatentSegment`` carrying the 6D trajectory
    ``[B, K, C, T_lat, H_lat, W_lat]`` plus per-SDE log probs
    (``sde_logp [N, S]`` + ``sde_indices [S]``).

    ``replay(conditions, *, segment, params, step_indices=None)``
    recomputes log-probs for the SDE transitions in a stored
    ``LatentSegment``. Returns a :class:`ReplayResult` with ``log_probs``
    of shape ``[B, S']`` aligned with ``segment.sde_logp`` (or a slice
    when ``step_indices`` selects a subset) and ``prev_sample_means``
    for KL-penalty consumption.

    ``_no_split_modules`` is the model-side fallback used by FSDPPolicy
    when HF auto-discovery yields nothing -- HunyuanVideo-1.0's
    transformer block class is ``HunyuanVideoTransformerBlock``.
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = ("HunyuanVideoTransformerBlock",)

    # VAE downsample defaults from upstream; overridden at construction
    # if the bundle's VAE exposes ``spatial_compression_ratio`` /
    # ``temporal_compression_ratio`` attributes.
    # HunyuanVideo-1.0: spatial=8x, temporal=4x, latent_channels=16.
    DEFAULT_SPATIAL_DOWNSAMPLE: ClassVar[int] = 8
    DEFAULT_TEMPORAL_DOWNSAMPLE: ClassVar[int] = 4
    DEFAULT_LATENT_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        *,
        model: HunyuanVideoBundle,
        step: HunyuanVideoDiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        spatial_compression_ratio: Optional[int] = None,
        temporal_compression_ratio: Optional[int] = None,
        latent_channels: Optional[int] = None,
    ) -> None:
        self.model = model
        self.step = step
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")

        # VAE geometry: prefer attributes on the VAE itself, then the VAE
        # config, then the dataclass-level defaults.
        vae = model.vae
        if spatial_compression_ratio is None:
            spatial_compression_ratio = (
                int(getattr(vae, "spatial_compression_ratio", 0))
                or int(getattr(getattr(vae, "config", None), "spatial_compression_ratio", 0))
                or self.DEFAULT_SPATIAL_DOWNSAMPLE
            )
        if temporal_compression_ratio is None:
            temporal_compression_ratio = (
                int(getattr(vae, "temporal_compression_ratio", 0))
                or int(getattr(getattr(vae, "config", None), "temporal_compression_ratio", 0))
                or self.DEFAULT_TEMPORAL_DOWNSAMPLE
            )
        self.spatial_compression_ratio = int(spatial_compression_ratio)
        self.temporal_compression_ratio = int(temporal_compression_ratio)

        if latent_channels is None:
            cfg = getattr(vae, "config", None)
            ch = int(getattr(cfg, "latent_channels", 0)) if cfg is not None else 0
            if not ch:
                # Fall back to transformer's reported out_channels.
                tx_cfg = getattr(model.transformer, "config", None)
                ch = int(getattr(tx_cfg, "out_channels", self.DEFAULT_LATENT_CHANNELS))
            latent_channels = ch
        self.latent_channels = int(latent_channels)
        self.runner = SingleStreamDiffusionRunner(
            strategy=strategy,
            autocast_dtype=self.autocast_dtype,
            trajectory_dtype=self.trajectory_dtype,
            logprob_dtype=self.logprob_dtype,
            owner=type(self).__name__,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _latent_shape(self, *, height: int, width: int, num_frames: int) -> Tuple[int, int, int]:
        latent_t = (int(num_frames) - 1) // self.temporal_compression_ratio + 1
        latent_h = max(1, int(height) // self.spatial_compression_ratio)
        latent_w = max(1, int(width) // self.spatial_compression_ratio)
        return latent_t, latent_h, latent_w

    def diffuse(
        self,
        conditions: HunyuanVideoConditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run full HunyuanVideo-1.0 T2V sampling. Returns a ``LatentSegment``
        with 6D trajectory storage and ``modality=VIDEO``.

        ``initial_latents`` (optional) -- x_T resolved from the request
        ``Sample``'s diffusion generation Part; see
        :class:`SD3DiffusionStage.diffuse` for the contract.
        """
        if conditions.text_llama is None or conditions.text_llama.embeds is None:
            raise ValueError("HunyuanVideoDiffusionStage.diffuse: conditions.text_llama.embeds is None")
        prompt_embeds = conditions.text_llama.embeds
        device = prompt_embeds.device
        batch_size = int(prompt_embeds.shape[0])
        latent_t, latent_h, latent_w = self._latent_shape(
            height=params.height, width=params.width, num_frames=params.num_frames
        )
        expected_latent_shape = (self.latent_channels, latent_t, latent_h, latent_w)
        transition = partial(
            self.step.step_with_logp,
            self.model,
            conditions,
            strategy=self.strategy,
            guidance_scale=float(params.guidance_scale),
        )
        return self.runner.sample(
            schedule=schedule,
            params=params,
            batch_size=batch_size,
            latent_shape=expected_latent_shape,
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
        conditions: HunyuanVideoConditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Segment-based log-prob replay over the rollout's SDE transitions.

        Caller is responsible for ``.train()`` mode + grad scope; this
        method only manages the autocast scope.
        """
        transition = partial(
            self.step.step_with_logp,
            self.model,
            conditions,
            strategy=self.strategy,
            guidance_scale=float(params.guidance_scale),
        )
        return self.runner.replay(
            segment=segment,
            params=params,
            transition=transition,
            step_indices=step_indices,
            expected_latents_ndim=6,
            latent_layout="[B, K, C, T_lat, H_lat, W_lat]",
        )

    # ------------------------------------------------------------------
    # Single-step noise prediction (forward-process algorithms: DiffusionNFT et al.)
    # ------------------------------------------------------------------

    def predict_noise_at_step(
        self,
        conditions: HunyuanVideoConditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward -- no scheduler iteration."""
        return self.step.predict_noise(
            self.model,
            sample,
            sigma,
            conditions,
            guidance_scale=float(params.guidance_scale),
        )

    # ------------------------------------------------------------------
    # Trainable surface for FSDPPolicy
    # ------------------------------------------------------------------

    def trainable_module(self) -> "torch.nn.Module":
        """Return the FSDP wrap target -- the bundle's transformer."""
        return self.model.transformer


__all__ = [
    "HunyuanVideoDiffusionStage",
    "HunyuanVideoDiffusionStep",
]
