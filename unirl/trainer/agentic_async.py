"""Fully-async agentic RL trainer — disaggregated producer/consumer (LIN-531).

Sibling of :class:`~unirl.trainer.agentic.AgenticTrainer` (synchronous + *colocated*:
the agentic engine's ``generate`` barrier and the FSDP train shard time-share each
GPU). ``AsyncAgenticTrainer`` places training and rollout on **disjoint GPU slabs**
(like :class:`~unirl.trainer.async_ar.AsyncARTrainer`), keeps the agentic engine
**resident**, pushes weights cross-slab via ``NCCLWeightSync``, and overlaps
multi-turn generation with training.

Mechanism vs policy (LIN-531): the **engine** exposes a ``submit`` / ``poll`` /
``finalize_if_drained`` / ``abort`` interface over a background drain (see
:class:`~unirl.rollout.engine.agentic.engine.AgenticRolloutEngine`); this **trainer**
owns the *policy* —

* **Producer** — keep the rollout slab saturated: ``submit`` a pool of fresh prompt
  siblings + resumed carried partials, ``poll`` completed trajectories, bucket them by
  root id into complete GRPO groups (:class:`_GroupAssembler`), and push each into a
  staleness-bounded :class:`_GroupBuffer`.
* **Consumer** — drain the freshest ``batch_size`` complete groups (within
  ``buffer_max_staleness``), reward + GRPO advantage + one optimizer step (reusing
  :class:`AgenticTrainer`'s helpers), then **quiesce + sync**: ``abort`` the in-flight
  tail at a turn boundary, apply the configured ``tail_policy`` (carry only when the
  environment can resume from the ``Sample``; otherwise drop), ``weight_sync.sync()``,
  bump the version.

ONE single-threaded loop (the ``AsyncARTrainer`` shape): with disjoint slabs the
rollout slab keeps generating in the background (the engine's per-worker drain) while
the driver polls / trains — concurrency from disaggregation, not driver threads. In
steady state the buffer is refilled *during* the previous train step, so :meth:`_next_batch`
returns without waiting. Staleness bounds how far the producer leads the consumer; the
per-token rollout-anchored ratio corrects the off-policy gap (a carried trajectory whose
turns span weight versions is correct per-token because each gen ``Part`` keeps its own
``weight_version`` + logprobs).

.. note::
   The GPU integration (two-slab placement, NCCL sync, and the train loop) follows
   ``AsyncARTrainer`` and requires an end-to-end GPU run for validation.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.algorithms.advantage import GroupedAdvantageEstimator
from unirl.config.execution import Capability, LoopKind, PlacementMode
from unirl.distributed.group.placement import placement, remote
from unirl.trainer.agentic import AgenticTrainer
from unirl.trainer.base import BaseTrainer, build_advantage_estimator, build_sampling_dict
from unirl.types.sample import Part, Sample
from unirl.types.sampling import BaseSamplingParams
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Producer-side bookkeeping
# --------------------------------------------------------------------------- #


class _GroupAssembler:
    """Bucket a flat stream of terminal trajectories into complete GRPO groups.

    The agentic engine ``poll`` returns variable-depth trajectory ``Sample``s; a
    prompt's ``n`` siblings share its slash-free **root id**
    (``sample.parts[0].sample_ids[0]``). A group is *complete* once all ``n`` of a
    root's siblings are terminal. Variable-depth trajectories are NOT concatenated —
    a group is kept as a ``List[Sample]`` (the trainer flattens their gen Parts at
    train time, like :meth:`AgenticTrainer.train_step`).
    """

    def __init__(self, n: int) -> None:
        self._n = int(n)
        self._by_root: Dict[str, List[Sample]] = {}

    @staticmethod
    def root_of(traj: Sample) -> str:
        return traj.parts[0].sample_ids[0]

    def add_completed(self, trajs: List[Sample]) -> None:
        """Accumulate terminal (poll-completed) trajectories, bucketed by root id."""
        for t in trajs:
            self._by_root.setdefault(self.root_of(t), []).append(t)

    def pop_complete_groups(self) -> List[List[Sample]]:
        """Emit + drop every root that has all ``n`` siblings terminal (each group a
        ``List[Sample]`` of exactly ``n`` trajectories)."""
        ready = [root for root, sibs in self._by_root.items() if len(sibs) >= self._n]
        out: List[List[Sample]] = []
        for root in ready:
            sibs = self._by_root.pop(root)
            out.append(sibs[: self._n])
        return out

    def pending_roots(self) -> Set[str]:
        """Roots with some-but-not-all siblings terminal (their done siblings are held
        here; their unfinished siblings are carried by the trainer for resume)."""
        return set(self._by_root)

    def discard_roots(self, roots: Iterable[str]) -> int:
        """Drop incomplete buckets for abandoned roots.

        Returns the number of terminal sibling trajectories that were being held
        while the other siblings were still in flight.
        """
        discarded = 0
        for root in set(roots):
            discarded += len(self._by_root.pop(root, []))
        return discarded

    def size(self) -> int:
        return len(self._by_root)


class _GroupBuffer:
    """Staleness-bounded buffer of complete GRPO groups (the ``AsyncARTrainer._RolloutBuffer``
    shape, but each item is a ``List[Sample]`` group of variable-depth trajectories).

    A group is stamped with the ``weight_version`` it *completed* under and a monotonic
    ``gen_id`` for freshness ordering. The per-token ratio corrects the within-trajectory
    version spread (a carried trajectory's turns); this buffer only bounds how stale a
    *completed* group may be before the consumer trains it.
    """

    def __init__(self) -> None:
        self._items: List[Tuple[List[Sample], int, int]] = []  # (group, weight_version, gen_id)
        self._evicted: List[List[Sample]] = []

    def put(self, group: List[Sample], *, weight_version: int, gen_id: int) -> None:
        self._items.append((list(group), int(weight_version), int(gen_id)))

    def size(self) -> int:
        return len(self._items)

    def drain_freshest(
        self,
        n: int,
        *,
        current_version: Optional[int] = None,
        max_staleness: Optional[int] = None,
    ) -> Optional[List[List[Sample]]]:
        """Pop the ``n`` freshest complete groups, evicting over-stale ones first.

        Returns ``None`` if fewer than ``n`` groups remain after eviction (the consumer
        then waits for the producer to fill more).
        """
        if max_staleness is not None and current_version is not None:
            kept: List[Tuple[List[Sample], int, int]] = []
            for item in self._items:
                if current_version - item[1] <= max_staleness:
                    kept.append(item)
                else:
                    self._evicted.append(item[0])
            self._items = kept
        if len(self._items) < n:
            return None
        self._items.sort(key=lambda it: it[2], reverse=True)  # freshest gen_id first
        picked, self._items = self._items[:n], self._items[n:]
        return [grp for grp, _, _ in picked]

    def pop_evicted_groups(self) -> List[List[Sample]]:
        """Return and clear groups rejected by the latest staleness checks."""
        evicted, self._evicted = self._evicted, []
        return evicted


# --------------------------------------------------------------------------- #
# The trainer
# --------------------------------------------------------------------------- #


class AsyncAgenticTrainer(AgenticTrainer):
    """Disaggregated fully-async agentic trainer (two slabs, resident engine, NCCL sync)."""

    LOOP_KIND: LoopKind = LoopKind.ASYNC_AGENTIC_RL
    PLACEMENT_OVERRIDE: PlacementMode = PlacementMode.SEPARATE
    REQUIRED_SYNC_CAPABILITIES = frozenset({Capability.NCCL_RENDEZVOUS})
    DEFAULT_SAVE_MODE = "full"

    _POLL_INTERVAL_S = 0.02  # backoff between polls while the in-flight drive fills the buffer
    _MAX_REFILLS = 64  # underflow guard: refills of a drained-but-short buffer before we give up

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        samples_per_prompt: int,
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
        stop: Optional[List[str]] = None,
        # ---- async knobs ----
        train_fraction: float = 0.5,
        oversample_batch_size: Optional[int] = None,
        buffer_max_staleness: Optional[int] = None,
        tail_policy: Literal["carry", "drop"] = "carry",
    ) -> None:
        # Call BaseTrainer.__init__ directly: AgenticTrainer.__init__ → ARTrainer.__init__
        # opens the colocate ``placement(fraction=1.0)`` block we replace with two slabs.
        BaseTrainer.__init__(self, cfg=cfg, logging_cfg=logging_cfg)

        # ---- scalar/config fields (the ARTrainer/AgenticTrainer state we still need) ----
        self.batch_size = int(batch_size)
        self.adv_normalization_scope = adv_normalization_scope
        self.normalize_adv_by_std = normalize_adv_by_std
        self.advantage_estimator = build_advantage_estimator(
            advantage_cfg,
            default=GroupedAdvantageEstimator(
                scope=self.adv_normalization_scope,
                normalize=self.normalize_adv_by_std,
                global_std_unbiased=False,
                exclude_non_finite=True,
                variance_epsilon=False,
            ),
        )
        self.balance_shards = False
        self.eval_interval = 0  # AgenticTrainer.evaluate raises; agentic eval is a follow-up
        self.data_source = instantiate(data_source_cfg)
        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        self.weight_sync = None
        # Per-turn stop (AgenticTrainer.__init__): a tool-call turn ends at ``</tool_call>``.
        self._stop = list(stop) if stop else ["</tool_call>"]

        # ---- async state ----
        self._train_fraction = float(train_fraction)
        self._buffer_max_staleness = buffer_max_staleness
        self._tail_policy = str(tail_policy)
        if self._tail_policy not in ("carry", "drop"):
            raise ValueError(f"tail_policy must be 'carry' or 'drop'; got {self._tail_policy!r}")
        self._weight_version = 0
        # Monotonic per-DRIVE nonce. ``rollout_id`` alone does not make root ids
        # unique: :meth:`_next_batch` refills re-submit under the SAME rollout_id, and a
        # data source may restart its ids on every ``get_samples`` (DefaultDataSource
        # numbers by batch position, not prompt identity). Two drives would then
        # namespace different source rows identically, and :class:`_GroupAssembler`
        # — which buckets purely by root id — would merge siblings of unrelated
        # prompts into one GRPO group and overwrite their ``_gt_by_root`` answers.
        self._drive_seq = 0
        self._carried_tail_trajectories = 0
        self._dropped_tail_trajectories = 0
        self._dropped_tail_roots = 0
        self._discarded_completed_trajectories = 0
        # root id -> ground-truth answer, recorded at submit time so the reward judge
        # never depends on the engine preserving root-Part metadata through a (possibly
        # resumed) trajectory. Carried partials keep the root id they were submitted
        # under, so their answer is already here.
        self._gt_by_root: Dict[str, Optional[str]] = {}
        # GRPO group size n. Must equal the engine's ``episode_sampling.samples_per_prompt``
        # (the assembler needs it to know when a root's siblings are all in).
        self._n = int(samples_per_prompt)
        # How many prompt-groups to feed the rollout slab per drive (>= batch_size). A
        # larger pool keeps the slab busy across more train steps between syncs.
        self._oversample = int(oversample_batch_size) if oversample_batch_size else self.batch_size
        if self._oversample < self.batch_size:
            raise ValueError(f"oversample_batch_size={self._oversample} must be >= batch_size={self.batch_size}")

        self._train_devices = int(round(self.num_devices * self._train_fraction))
        if self._train_devices <= 0 or self._train_devices >= self.num_devices:
            raise ValueError(
                f"train_fraction={train_fraction} yields {self._train_devices} train "
                f"devices of {self.num_devices}; must leave a non-empty rollout slab."
            )
        self._rollout_devices = self.num_devices - self._train_devices

        # ---- two disjoint top-level slabs (AsyncARTrainer template) ----
        with placement(self.pool, fraction=self._train_fraction, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
            self.reward = remote_hydra(reward_cfg)
            self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline)
            self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)
            if sync_cfg is not None:
                self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
        with placement(self.pool, fraction=1.0 - self._train_fraction, shared_workers=True):
            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            if self.execution_plan.engine("rollout").is_direct:
                raise ValueError(
                    "AsyncAgenticTrainer needs a dedicated agentic-rollout engine on the "
                    "separate slab (its inner sglang/vllm engine lives cross-slab)."
                )
            self.rollout = remote(**rollout_parsed)

        # Wire the rank-0 coordinator (AgenticTrainer.__init__ does this too).
        self.rollout.set_workers(self.rollout.workers, self.rollout.role_name)

        if self.weight_sync is not None:
            self._connect_separate()

    def _connect_separate(self) -> None:
        """One-time cross-slab NCCL handshake (identical to AsyncARTrainer)."""
        sync = self.execution_plan.sync_for("rollout")
        if not sync.supports(Capability.NCCL_RENDEZVOUS):
            raise ValueError(
                f"AsyncAgenticTrainer (separate slabs) requires a cross-slab weight sync "
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
    # Producer — build/submit a drive, poll completions into the buffer
    # ------------------------------------------------------------------

    def _build_tasks(self, carried: List[Sample], rollout_id: int) -> List[Sample]:
        """A drive's task list: ``n`` fresh siblings per new prompt + carried partials.

        Fresh prompts (``oversample_batch_size`` of them) are split into per-prompt
        request trees and replicated ``n`` times (the GRPO siblings share a root id);
        carried partials (their root ids + turn history preserved) are appended as-is,
        to be resumed by the engine's ``_run_one`` from ``len(gen_parts())``.

        Fresh roots are namespaced by ``_drive_seq`` as well as ``rollout_id`` so a
        refill can never reuse a drained drive's root ids for different source rows.
        Carried partials keep the ids they were submitted under, so they still rejoin
        their own siblings in the assembler.
        """
        self._drive_seq += 1
        fresh = self._build_request_sample(self.data_source.get_samples(self._oversample), rollout_id)
        fresh = fresh.map_sample_ids(lambda sample_id: f"d{self._drive_seq}:{sample_id}")
        root = fresh.parts[0]
        root_meta = root.metadata or [None] * len(root.sample_ids)
        for sid, md in zip(root.sample_ids, root_meta):
            self._gt_by_root[sid] = (md or {}).get("answer")  # remember the answer for this root
        tasks = [prompt for prompt in fresh.split() for _ in range(self._n)]
        tasks.extend(carried)
        return tasks

    def _submit_drive(self, carried: List[Sample], rollout_id: int) -> None:
        """Submit a fresh over-sampled drive (non-blocking). Call only when the prior
        drive is finalized/``abort``ed (else two drains would double-pull)."""
        self.rollout.submit(self._build_tasks(carried, rollout_id))

    def _ingest_completed(self, completed: List[Sample]) -> int:
        """Ingest terminal trajectories and promote newly complete groups."""
        if completed:
            self._assembler.add_completed(completed)
            for group in self._assembler.pop_complete_groups():
                self._buffer.put(group, weight_version=self._weight_version, gen_id=self._gen_id)
                self._gen_id += 1
        return len(completed)

    def _pump(self) -> int:
        """Poll completed trajectories into the assembler and buffer."""
        return self._ingest_completed(self.rollout.poll()[0])

    def _apply_tail_policy(self, carried: List[Sample], rollout_id: int) -> List[Sample]:
        """Keep resumable tails or purge all state belonging to dropped roots."""
        if self._tail_policy == "carry":
            self._carried_tail_trajectories += len(carried)
            logger.info("rollout %d async: carry tail=%d trajectories", rollout_id, len(carried))
            return carried

        roots = {self._assembler.root_of(sample) for sample in carried}
        discarded_completed = self._assembler.discard_roots(roots)
        for root in roots:
            self._gt_by_root.pop(root, None)
        self._dropped_tail_trajectories += len(carried)
        self._dropped_tail_roots += len(roots)
        self._discarded_completed_trajectories += discarded_completed
        logger.info(
            "rollout %d async: drop tail=%d trajectories across %d roots; discarded %d completed siblings",
            rollout_id,
            len(carried),
            len(roots),
            discarded_completed,
        )
        return []

    def _log_tail_metrics(self, rollout_step: int) -> None:
        """Emit tail counters after abort/policy, including the final step."""
        self.observer.log_rollout(
            rollout_step,
            {
                "async/carried_tail_trajectories": self._carried_tail_trajectories,
                "async/dropped_tail_trajectories": self._dropped_tail_trajectories,
                "async/dropped_tail_roots": self._dropped_tail_roots,
                "async/discarded_completed_trajectories": self._discarded_completed_trajectories,
                "async/assembler_pending_roots": self._assembler.size(),
            },
        )

    def _drain_buffer(self, n: int, *, max_staleness: int) -> Optional[List[List[Sample]]]:
        """Drain fresh groups and forget ground truth for stale evictions."""
        picked = self._buffer.drain_freshest(
            n,
            current_version=self._weight_version,
            max_staleness=max_staleness,
        )
        for group in self._buffer.pop_evicted_groups():
            if group:
                self._gt_by_root.pop(self._assembler.root_of(group[0]), None)
        return picked

    def _next_batch(self, rollout_id: int) -> List[List[Sample]]:
        """Pump the producer until the buffer holds ``batch_size`` complete groups within
        the staleness bound, then drain the freshest ones. If the in-flight drive drains
        without filling the buffer (staleness eviction / failures / small over-sample),
        refill with a fresh drive (resuming any carried partials)."""
        stale = self._buffer_max_staleness if self._buffer_max_staleness is not None else 0
        refills = 0
        while True:
            self._pump()
            picked = self._drain_buffer(self.batch_size, max_staleness=stale)
            if picked is not None:
                return picked
            completed = self.rollout.finalize_if_drained()[0]
            if completed is None:
                time.sleep(self._POLL_INTERVAL_S)  # in-flight drive still generating; back off
                continue

            # Atomic engine finalization joins the drain and returns its last
            # completions before a new submit is allowed to reset worker buffers.
            self._ingest_completed(completed)
            picked = self._drain_buffer(self.batch_size, max_staleness=stale)
            if picked is not None:
                return picked

            refills += 1
            if refills > self._MAX_REFILLS:
                raise RuntimeError(
                    f"async-agentic rollout {rollout_id}: buffer underflow after {refills} refills "
                    f"(buffer={self._buffer.size()} < batch={self.batch_size}); raise "
                    f"oversample_batch_size or buffer_max_staleness."
                )
            self._submit_drive(self._pending_carried, rollout_id)
            self._pending_carried = []

    # ------------------------------------------------------------------
    # Consumer — reward + GRPO advantage + one optimizer step over a group batch
    # ------------------------------------------------------------------

    def _train_on_groups(self, groups: List[List[Sample]], *, training_progress: float, rollout_id: int, t0: float):
        """Reward + GROUP-relative advantage + one step over ``batch_size`` complete
        groups using :class:`AgenticTrainer`'s shared reward/GRPO/train tail."""
        trajs: List[Sample] = [t for group in groups for t in group]
        roots = [tr.parts[0].sample_ids[0] for tr in trajs]
        request = Sample.request(Part.input(roots, metadata=[{"answer": self._gt_by_root.get(r)} for r in roots]))
        rewards, group_ids = self._rewards_and_groups(request, trajs, rollout_id)
        for root in set(roots):
            self._gt_by_root.pop(root, None)
        versions = [gp.weight_version for tr in trajs for gp in tr.gen_parts() if gp.weight_version is not None]
        result = self._advantage_train_and_log(
            trajs,
            rewards,
            group_ids,
            rollout_id=rollout_id,
            training_progress=training_progress,
            t0=t0,
            extra_metrics={
                "async/buffer_groups": self._buffer.size(),
                "async/weight_version": self._weight_version,
                "async/version_span": (max(versions) - min(versions)) if versions else 0,
                "async/assembler_pending_roots": self._assembler.size(),
                "async/carried_tail_trajectories": self._carried_tail_trajectories,
                "async/dropped_tail_trajectories": self._dropped_tail_trajectories,
                "async/dropped_tail_roots": self._dropped_tail_roots,
                "async/discarded_completed_trajectories": self._discarded_completed_trajectories,
            },
        )
        return result
