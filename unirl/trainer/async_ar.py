"""Async autoregressive RL trainer — disaggregated train/rollout slabs.

Sibling of :class:`~unirl.trainer.ar.ARTrainer` (synchronous + *colocated*:
rollout engine and FSDP train shard time-share each GPU via ``sleep()/wake_up()``,
and every step runs ``generate → reward → train`` in series). ``AsyncARTrainer``
instead places training and rollout on **disjoint GPU slabs**, keeps the engine
**resident**, pushes weights cross-slab via ``NCCLWeightSync``, and overlaps
generation with training.

ONE single-threaded loop (slime's "one trainer loop; async-depth is a knob"
principle, implemented with UniRL-native non-blocking Ray dispatch instead of
slime's thread+asyncio). The async behavior is set by **two numeric knobs**:

* ``max_inflight`` — how many generations run concurrently (overlap/parallelism
  depth). ``1`` ≈ the classic one-step pipeline; higher fans out more.
* ``buffer_max_staleness`` — how many weight-syncs a buffered group may cross
  before it is evicted. ``0`` (default) = **on-policy**: the launch clamp never
  lets a generation cross a weight sync, so ``ratio≈1`` (the colocate-parity
  regime). ``>0`` = **off-policy continuous buffer**: generations may run ahead
  across syncs, bounded by eviction; the rollout-anchored DRPO ratio absorbs it.

Generation is launched as **non-blocking Ray futures** by
``RayGenerationDispatcher`` and reaped by ``AsyncRolloutScheduler`` on the
single driver thread — no producer thread, no locks. Draining all in-flight
generations before each weight sync is **mandatory** (the engine corrupts an
in-flight generation when weights + KV cache update mid-flight); this is the
single-threaded ``_drain_all`` quiesce.

Subclasses ``ARTrainer`` to reuse ``_build_request_sample``/``evaluate`` and ``BaseTrainer``
plumbing, but ``__init__`` calls ``BaseTrainer.__init__`` **directly** (the parent
opens the colocate ``placement(fraction=1.0)`` block we replace with two slabs).
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.algorithms.advantage import GroupedAdvantageEstimator, estimate_part_advantages
from unirl.config.execution import Capability, LoopKind, PlacementMode
from unirl.distributed.group.placement import placement, remote
from unirl.reward.ops import attach_frontier, materialize_reward
from unirl.rollout.async_runtime import InflightGeneration
from unirl.train.stack import TrainStepResult
from unirl.trainer.ar import ARTrainer
from unirl.trainer.base import BaseTrainer, build_advantage_estimator, build_sampling_dict
from unirl.types.sample import Sample
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


class AsyncARTrainer(ARTrainer):
    """Disaggregated async AR trainer (two slabs, resident engine, NCCL sync)."""

    LOOP_KIND: LoopKind = LoopKind.ASYNC_BATCH_RL
    PLACEMENT_OVERRIDE: PlacementMode = PlacementMode.SEPARATE
    REQUIRED_SYNC_CAPABILITIES = frozenset({Capability.NCCL_RENDEZVOUS})
    DEFAULT_SAVE_MODE = "full"

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: Optional[DictConfig] = None,
        logging_cfg: Optional[DictConfig] = None,
        advantage_cfg: Optional[DictConfig] = None,
        adv_normalization_scope: str = "group",
        normalize_adv_by_std: bool = True,
        balance_shards: bool = False,
        eval_interval: int = 0,
        eval_num_prompts: int = -1,
        eval_batch_size: int = 8,
        eval_samples_per_prompt: int = 16,
        eval_temperature: float = 1.0,
        # ---- async knobs ----
        train_fraction: float = 0.5,
        max_inflight: int = 1,
        buffer_max_staleness: Optional[int] = None,
    ) -> None:
        # Call BaseTrainer.__init__ directly: ARTrainer.__init__ opens the
        # colocate ``placement(fraction=1.0)`` block, which is exactly what we
        # must NOT run. (ARTrainer itself just calls BaseTrainer.__init__ here.)
        BaseTrainer.__init__(self, cfg=cfg, logging_cfg=logging_cfg)

        # ---- scalar/config fields (mirrors ar.py:62-88) ----
        self.batch_size = batch_size
        self.adv_normalization_scope = adv_normalization_scope
        self.normalize_adv_by_std = normalize_adv_by_std
        self.advantage_estimator = build_advantage_estimator(
            advantage_cfg,
            default=GroupedAdvantageEstimator(
                scope=self.adv_normalization_scope,
                normalize=self.normalize_adv_by_std,
            ),
        )
        self.balance_shards = bool(balance_shards)
        self.eval_interval = int(eval_interval)
        _num = int(eval_num_prompts)
        self.eval_num_prompts = -1 if _num < 0 else _num
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.eval_samples_per_prompt = int(eval_samples_per_prompt)
        self.eval_temperature = float(eval_temperature)
        self.data_source = instantiate(data_source_cfg)
        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        self.weight_sync = None

        # ---- async state ----
        self._train_fraction = float(train_fraction)
        self._max_inflight = max(1, int(max_inflight))
        self._buffer_max_staleness = buffer_max_staleness
        self._weight_version = 0  # driver-tracked policy version (# of weight syncs issued)
        # DP size of the TRAIN slab — the divisor for balance_shards (the parent
        # uses self.num_devices because colocate training spans the whole pool;
        # here training only spans the train slab).
        self._train_devices = int(round(self.num_devices * self._train_fraction))
        if self._train_devices <= 0 or self._train_devices >= self.num_devices:
            raise ValueError(
                f"train_fraction={train_fraction} yields {self._train_devices} train "
                f"devices of {self.num_devices}; must leave a non-empty rollout slab."
            )
        # DP_SCATTER divisibility: per-rollout sample count must split evenly over
        # BOTH slabs (training over the train slab, generation over the rollout
        # slab). Fail early with a clear message rather than mid-run in dispatch.
        self._rollout_devices = self.num_devices - self._train_devices
        # DP_SCATTER divisibility differs per slab in the Sample model:
        #   * training shards the gen Part (P*N samples) over the train slab;
        #   * generation shards the REQUEST Sample by its root (P prompts =
        #     Sample.batch_size; each prompt-tree stays whole) over the rollout slab.
        prompts = int(self.batch_size)  # P
        total = prompts * total_samples_per_prompt(self.sampling_params)  # P*N
        if total % self._train_devices != 0:
            raise ValueError(
                f"batch_size * samples_per_prompt = {total} is not divisible by the train "
                f"slab size {self._train_devices}; adjust batch_size / samples_per_prompt / train_fraction."
            )
        if prompts % self._rollout_devices != 0:
            raise ValueError(
                f"batch_size = {prompts} prompts is not divisible by the rollout slab size "
                f"{self._rollout_devices} (each prompt-tree DP-scatters whole); adjust batch_size / train_fraction."
            )

        # ---- two disjoint top-level slabs (diffusion.py:115-129 template) ----
        # The train scope must FULLY EXIT before the rollout scope opens, else a
        # nested placement would carve a sub-slab instead of a disjoint slab.
        with placement(self.pool, fraction=self._train_fraction, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
            self.reward = remote_hydra(reward_cfg)
            self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline)
            self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)
            if sync_cfg is not None:
                # NCCL handler: rollout is cross-slab and wired via the handshake
                # below — it takes only ``backend`` (no rollout sibling).
                self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
        # Rollout slab = the rest (fraction is relative to the WHOLE pool).
        with placement(self.pool, fraction=1.0 - self._train_fraction, shared_workers=True):
            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            if self.execution_plan.engine("rollout").is_direct:
                raise ValueError(
                    "AsyncARTrainer needs a dedicated-rollout engine (vllm/sglang) on the "
                    "separate slab; the trainside direct-sampling engine needs the pipeline "
                    "as a local sibling and cannot live cross-slab."
                )
            self.rollout = remote(**rollout_parsed)

        if self.weight_sync is not None:
            self._connect_separate()

    def _connect_separate(self) -> None:
        """One-time cross-slab handshake (NCCL branch of diffusion.py:191-208).

        Rank 0 picks a rendezvous addr/port, is handed the rollout slab's Worker
        actor handles, then ``connect`` fires each rollout worker's
        ``init_weights_update_group`` non-blocking and joins the broadcast group
        itself. Only ``NCCLWeightSync`` is supported here (always cross-slab
        full-weight); a non-NCCL target is a config error.
        """
        sync = self.execution_plan.sync_for("rollout")
        if not sync.supports(Capability.NCCL_RENDEZVOUS):
            raise ValueError(
                f"AsyncARTrainer (separate slabs) requires a cross-slab weight sync "
                f"with NCCL rendezvous; got {sync.node.target!r}."
            )
        addr, port = self.weight_sync.pick_master()[0]
        self.weight_sync.set_rollout_targets(self.rollout.workers, self.rollout.role_name)
        self.weight_sync.connect(
            master_addr=addr,
            master_port=port,
            num_rollout_gpus=len(self.rollout.workers),
        )

    # ------------------------------------------------------------------
    # Generic async-runtime hooks
    # ------------------------------------------------------------------

    def _build_async_sample(self, gen_id: int) -> Sample:
        """Consume one data batch and build the request Sample for ``gen_id``."""
        return self._build_request_sample(self.data_source.get_samples(self.batch_size), gen_id)

    def _score_completed(
        self,
        job: InflightGeneration,
        completed: Sample,
    ) -> List[Sample]:
        """Score a completed Sample and split it into tree-complete groups.

        Scoring must precede ``_drop_decoded`` (the reward reads the decoded
        primitive). Keyed by ``gen_id`` so media panels behave like the old path.
        The filled ``Sample`` is self-contained (it carries its input Parts), so
        no request handle is kept on the in-flight record.
        """
        scored = attach_frontier(self.reward, completed)
        self._drop_decoded(scored, rollout_id=job.gen_id)
        return scored.split()

    # ------------------------------------------------------------------
    # Train tail (mirrors ar.py:152-182, minus wake/sleep) — reward parity
    # ------------------------------------------------------------------

    def _advantage_and_train(
        self,
        sample: Sample,
        *,
        training_progress: float,
        rollout_id: int,
        t0: Optional[float] = None,
    ) -> Tuple[TrainStepResult, float]:
        """Advantage + optimizer step for a SCORED ``Sample`` (rewards already attached)."""
        if t0 is None:
            t0 = time.perf_counter()
        outcome = materialize_reward(sample)
        sample = outcome.sample
        part = outcome.part
        mean_reward = outcome.mean
        part = estimate_part_advantages(part, self.advantage_estimator)
        sample = sample.with_parts([*sample.parts[:-1], part])
        train_part = part
        if self.balance_shards:
            train_part = part.balance_shards(self._train_devices)  # over the TRAIN slab DP size
        result = self.stack.train_track(train_part, training_progress=float(training_progress))
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            sample,
            step_time_s=time.perf_counter() - t0,
            trunc_len=getattr(self.sampling_params.get("ar"), "max_new_tokens", None),
        )
        # train_step is bypassed, so BaseTrainer's per-step reset hook never
        # fires; reclaim transport buffers here (no-op for colocate_store/gpu).
        self._reset_transport_buffers()
        return result, mean_reward
