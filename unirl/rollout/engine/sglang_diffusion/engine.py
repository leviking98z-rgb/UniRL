"""``sglang_diffusion`` engine core — wiring + delegation only.

A thin core over the backend seam: it names no concrete model (the adapter, picked
from the registry by ``config.model_family``, owns the ``Sample`` → ``Sample``
conversion) and no concrete backend (the seam owns the runtime). Weight sync is a
:class:`WeightSync` component constructed over the seam; the offload lifecycle (a
single flag) lives directly on the engine. Common tensor/NCCL/LoRA receiver
calls use the shared delegating receiver; this class owns its checksum extension.

One-shot construction: after ``__init__`` returns, the generator is spawned and the
engine is usable. ``generate`` / ``sleep`` / ``wake_up`` re-apply ``@distributed``
(the decorator is not inherited — see ``base.py``).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import torch

from unirl.config.dtypes import parse_torch_dtype
from unirl.config.execution import Capability, ComponentCapabilities
from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseSingleTurnRolloutEngine
from unirl.rollout.engine.sglang_diffusion.adapters import get_adapter
from unirl.rollout.engine.sglang_diffusion.backends import SGLangBackend
from unirl.rollout.engine.sglang_diffusion.config import (
    SGLangDiffusionEngineConfig,
    SGLangDiffusionPorts,
)
from unirl.rollout.engine.sglang_diffusion.weight_sync import WeightSync
from unirl.rollout.engine.weight_receiver import DelegatingTensorNCCLLoraReceiver
from unirl.sde.noise import generate_latents
from unirl.sde.runtime import ensure_sample_sigmas
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams

logger = logging.getLogger(__name__)

#: Memory tags released on sleep / restored on wake.
_OFFLOAD_TAGS = ("transformer", "vae", "text_encoder")
#: Tags backed up to CPU rather than dropped.
_CPU_BACKUP_TAGS = ("vae", "text_encoder")


class SGLangDiffusionRolloutEngine(DelegatingTensorNCCLLoraReceiver, BaseSingleTurnRolloutEngine):
    """Rollout engine backed by ``sglang.multimodal_gen.DiffGenerator`` (v2 layout)."""

    CAPABILITIES: ComponentCapabilities = ComponentCapabilities.of(
        Capability.DEDICATED_ROLLOUT,
        Capability.SINGLE_TURN_GENERATION,
        Capability.QUIESCE,
        Capability.MEMORY_LIFECYCLE,
        Capability.MULTI_GPU_COLOCATE,
        Capability.TENSOR_WEIGHT_RECEIVER,
        Capability.NCCL_WEIGHT_RECEIVER,
        Capability.LORA_WEIGHT_RECEIVER,
    )
    _component_name = "sglang_diffusion"

    def __init__(
        self,
        config: SGLangDiffusionEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
        ports: Optional[SGLangDiffusionPorts] = None,
    ) -> None:
        require(
            isinstance(config, SGLangDiffusionEngineConfig),
            f"SGLangDiffusionRolloutEngine requires SGLangDiffusionEngineConfig; got {type(config).__name__}",
        )
        require(
            model_config is not None and bool(model_config.pretrained_model_ckpt_path),
            "SGLangDiffusionRolloutEngine requires model_config.pretrained_model_ckpt_path",
        )

        self.cfg = config
        self.model_config = model_config
        self.strategy = strategy
        self.rank = rank
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_offloaded = False

        # Adapter (the only read of a model knob) — owns the conversion + schedule.
        self.adapter = get_adapter(config.model_family)(config, model_config, strategy=strategy)
        pipeline_prefix, target_modules = self.adapter.lora_spec()

        logger.info(
            "Initializing sglang_diffusion engine (rank=%s, local_mode=%s, "
            "model_family=%s, target_modules=%s, populate_conditions=%s)",
            rank,
            config.local_mode,
            config.model_family,
            target_modules,
            config.populate_conditions,
        )

        # Ports — engine-reserved on this node at the last moment before the spawn.
        # Tests inject a fixed set; remote mode uses cfg host/port/scheduler_port.
        if config.local_mode and ports is None:
            ports = SGLangDiffusionPorts.reserve()

        # Backend (the seam) — booted from the config-spelled intent (ports overlaid).
        intent = config.server_intent(
            model_config=model_config,
            ports=ports,
            extra=self.adapter.boot_kwargs(),
        )
        self._backend = SGLangBackend.boot(
            intent,
            local_mode=bool(config.local_mode),
        )

        # Weight sync — owns all sync/LoRA state, over the live seam.
        self._weight_sync = WeightSync(
            self._backend,
            pipeline_prefix=pipeline_prefix,
            target_modules=target_modules,
            uses_lora=bool(model_config.use_lora),
        )

        # σ schedule policy comes from the adapter (absorbs the generic-vs-factory branch).
        self.schedule_policy = self.adapter.schedule_policy()

        # The DiffGenerator backend is synchronous and its scheduler client is not
        # request-concurrent-safe; the lock serializes concurrent generate callers.
        self._weight_version = 0
        self._generate_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False
        self._shutdown_complete = False

    # ------------------------------------------------------------------ #
    # Generation — sync entrypoint, serialized internally
    # ------------------------------------------------------------------ #

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        """Generate one whole DP shard synchronously."""
        return self._generate_locked(sample)

    def _generate_locked(self, sample: Sample) -> Sample:
        with self._generate_lock:
            if self._shutdown_requested:
                raise RuntimeError("SGLangDiffusionRolloutEngine.generate called after shutdown")
            return self._stamp_weight_version(self._generate_core(sample))

    def _generate_core(self, sample: Sample) -> Sample:
        """Synchronous generation for one whole ``Sample``."""
        gen = sample.frontier_gen_part(DiffusionSamplingParams)
        require(
            int(gen.batch_size) > 0,
            "SGLangDiffusionRolloutEngine.generate requires a non-empty Sample (gen batch_size > 0)",
        )
        # σ SSOT: pin once onto the gen part's (shared) sampling_params, so every
        # forward-batch chunk sees the same schedule.
        self._ensure_sample_sigmas(sample)

        fbs = self.cfg.forward_batch_size
        bs = int(gen.batch_size)
        if fbs is None or bs <= fbs:
            return self._generate_batch(sample)

        # Slice the gen frontier into chunks; ``replace_frontier`` keeps the input
        # part(s) whole (mirrors trainside; preserves any chained inputs).
        gen_chunks: List[Part] = []
        for start in range(0, bs, fbs):
            end = min(start + fbs, bs)
            chunk = self._generate_batch(sample.replace_frontier(gen.slice(start, end)))
            gen_chunks.append(chunk.frontier_gen_part(DiffusionSamplingParams))
            torch.cuda.empty_cache()
        return sample.replace_frontier(Part.concat(gen_chunks))

    def _ensure_sample_sigmas(self, sample: Sample) -> None:
        """Pin the σ schedule onto the gen part's ``DiffusionSamplingParams.sigmas``.

        σ is the single source of truth, computed from the model-owned schedule
        policy and shared across the part's samples (one params object).
        """
        ensure_sample_sigmas(sample, self.schedule_policy)

    def _generate_batch(self, sample: Sample) -> Sample:
        initial_noise = self._resolve_initial_noise(sample)
        kwargs = self.adapter.build_inputs(sample, initial_noise=initial_noise)
        raw = self._backend.generate(kwargs)
        return self.adapter.build_response(sample, raw)

    def _resolve_initial_noise(self, sample: Sample) -> Optional[torch.Tensor]:
        """Driver-authoritative x_T → init_same_noise fallback → None. Model-agnostic.

        ``disable_driver_xt`` returns ``None`` before every recipe/fallback path.
        Otherwise, the x_T noise key is derived from the lineage path (OD-2): the parent
        (group) id under ``init_same_noise`` so siblings share x_T, else the
        per-sample id. ``initial_latents`` (img2img) rides on the gen part's
        ``LatentSegment`` shell; the regen shape on ``init_noise_latent_shape``.
        """
        gen = sample.frontier_gen_part(DiffusionSamplingParams)
        diffusion = gen.sampling_params
        if diffusion is not None and bool(getattr(diffusion, "disable_driver_xt", False)):
            return None
        recipe = NoiseRecipe.from_sample(sample)
        xt = recipe.resolve()
        if xt is not None:
            return xt
        if not bool(self.cfg.init_same_noise):
            return None

        require(
            diffusion is not None and diffusion.seed is not None,
            "init_same_noise=True requires a diffusion seed",
        )
        batch_size = int(gen.batch_size)
        latent_shape = self._backend.prepare_latent_shape(
            height=int(diffusion.height),
            width=int(diffusion.width),
            num_frames=int(diffusion.num_frames),
            batch_size=batch_size,
        )
        dtype = parse_torch_dtype(diffusion.autocast_precision, field_name="autocast_precision")
        return generate_latents(
            batch_size=batch_size,
            latent_shape=latent_shape,
            device=self._device,
            dtype=dtype,
            init_same_noise=True,
            samples_per_prompt=int(diffusion.samples_per_prompt),
            noise_group_ids=[str(g) for g in gen.group_ids],
            base_seed=int(diffusion.seed),
        )

    # ------------------------------------------------------------------ #
    # Lifecycle — the offload flag lives here; decorators re-applied (base.py footgun)
    # ------------------------------------------------------------------ #

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        # Idempotent, symmetric with ``wake_up``: a second ``sleep()`` while already
        # offloaded would issue ``release_memory_occupation`` to the scheduler twice.
        if self._is_offloaded:
            return
        self._backend.release_memory(tags=_OFFLOAD_TAGS, cpu_backup_tags=_CPU_BACKUP_TAGS)
        self._is_offloaded = True
        # The released tags include the transformer weights → the loaded LoRA pool
        # is gone; the next weight sync must re-push.
        self._weight_sync.mark_weights_released()
        logger.info("sglang_diffusion engine slept (release_memory_occupation).")

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        if not self._is_offloaded:
            return
        self._backend.resume_memory(tags=_OFFLOAD_TAGS)
        self._is_offloaded = False

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def onload_weights(self, *, track_prefix: str = "") -> None:
        # Diffusion release/resume is all-or-nothing on one tag set, so onloading
        # weights == waking.
        del track_prefix
        self.wake_up()

    def health_check(self) -> bool:
        return self._backend.ping()

    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            with self._generate_lock:
                self._shutdown_requested = True
            with self._generate_lock:
                self._backend.shutdown()
            self._shutdown_complete = True

    # ------------------------------------------------------------------ #
    # Weight sync — common tensor/NCCL/LoRA calls come from the delegating
    # receiver. Only the diffusion checksum extension remains here.
    # ------------------------------------------------------------------ #

    def loaded_param_checksums(self, *, names: List[str]) -> Dict[int, List[Dict[str, str]]]:
        return self._weight_sync.loaded_param_checksums(names=names)

    # ``update_weights_from_ipc`` is deliberately absent: SGLang diffusion has
    # no bucketed-IPC receiver and does not declare that capability.


__all__ = ["SGLangDiffusionRolloutEngine"]
