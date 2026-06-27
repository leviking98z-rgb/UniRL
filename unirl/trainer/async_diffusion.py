"""Async disaggregated diffusion RL trainer — the AsyncARTrainer port to diffusion.

Sibling of :class:`~unirl.trainer.diffusion.DiffusionTrainer` (which already
supports two disjoint slabs via ``layout="separate"``: a train slab + a resident
dedicated-rollout engine on the rollout slab, wired by ``NCCLWeightSync``). This
class reuses that construction wholesale and only replaces the **synchronous**
``train`` loop (one ``generate → reward → advantage → train`` in series) with the
**async** loop from :class:`~unirl.trainer.async_ar.AsyncARTrainer`:

  * generation is launched as **non-blocking** Ray futures (``_generate_async``)
    and reaped on the single driver thread, so the rollout slab generates while
    the train slab trains (DigenRL GAP/TCSS overlap),
  * completed prompt-groups land in a ``_RolloutBuffer`` stamped with the
    ``weight_version`` + ``gen_id`` they were generated under,
  * two knobs: ``max_inflight`` (overlap depth = GAP) and ``buffer_max_staleness``
    (how many weight-syncs a group may cross before eviction = TCSS bounded
    off-policy; ``0`` ⇒ on-policy, the launch clamp never lets a generation cross
    a sync),
  * ``_drain_all`` quiesces every in-flight generation before each weight sync
    (the engine corrupts an in-flight generate when weights update mid-flight).

The diffusion-specific reward/advantage/drop-decoded calls mirror
``DiffusionTrainer.train_step``; everything else is the AsyncARTrainer machinery.
A standalone validation of the same pipeline (real WAN work units, measured
1.36–1.44x over the serial baseline) lives in ``digenrl_bench/async_runtime.py``.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import ray
import torch
from omegaconf import DictConfig

from unirl.distributed.group.dispatch import DISPATCH_MODE_REGISTRY, Dispatch
from unirl.distributed.tensor import WorkerLocalTransport, hydrate
from unirl.distributed.tensor.pytree import infer_batch_size
from unirl.train.stack import TrainStepResult
from unirl.trainer.async_ar import _RolloutBuffer
from unirl.trainer.diffusion import DiffusionTrainer
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack

logger = logging.getLogger(__name__)


class AsyncDiffusionTrainer(DiffusionTrainer):
    """Disaggregated async diffusion trainer (two slabs, resident engine, NCCL sync)."""

    def __init__(
        self,
        *,
        max_inflight: int = 1,
        buffer_max_staleness: Optional[int] = None,
        **diffusion_kwargs: Any,
    ) -> None:
        # Force the two-slab topology — async overlap is meaningless colocated.
        # DiffusionTrainer.__init__ already builds the train slab + the resident
        # cross-slab rollout engine + NCCLWeightSync handshake under layout="separate".
        diffusion_kwargs["layout"] = "separate"
        super().__init__(**diffusion_kwargs)
        if self._rollout_is_trainside:
            raise ValueError(
                "AsyncDiffusionTrainer needs a dedicated-rollout engine (vllm/sglang) on the "
                "separate slab; the trainside direct-sampling engine cannot live cross-slab."
            )
        self._max_inflight = max(1, int(max_inflight))
        self._buffer_max_staleness = buffer_max_staleness
        self._weight_version = 0  # driver-tracked policy version (# of weight syncs issued)

    # ------------------------------------------------------------------
    # Non-blocking generate seam (faithful split of the rollout Handle at ray.get)
    # ------------------------------------------------------------------

    def _generate_async(self, req: RolloutReq):
        """Launch ``generate`` non-blocking; return (refs, worker_local)."""
        r = self.rollout
        dispatch_fn = DISPATCH_MODE_REGISTRY[Dispatch.DP_SCATTER]["dispatch_fn"]
        bs = infer_batch_size((req,), {})
        if bs is not None and bs % r.dp_size != 0:
            raise ValueError(f"req batch_size={bs} not divisible by rollout dp_size={r.dp_size}")
        shards = dispatch_fn(r, (req,), {}, bs)
        worker_local = issubclass(r.pool.transport_cls, WorkerLocalTransport)
        shards = r.pool.transport_cls.localize(shards, r.pool, r.device_ids, r.worker_ids)
        refs = r._execute_all("generate", shards, grad_mode=False, call_id=None)
        return refs, worker_local

    def _collect_resp(self, refs, worker_local) -> RolloutResp:
        """Join a completed generate → full RolloutResp (byte-identical to generate)."""
        r = self.rollout
        collect_fn = DISPATCH_MODE_REGISTRY[Dispatch.DP_SCATTER]["collect_fn"]
        results = ray.get(refs)
        results = [r._rebind_tree(x, r.workers[i], worker_local=worker_local) for i, x in enumerate(results)]
        return collect_fn(r, results)

    @staticmethod
    def _is_ready(refs) -> bool:
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
        return len(ready) == len(refs)

    # ------------------------------------------------------------------
    # In-flight bookkeeping + buffer (mirrors AsyncARTrainer)
    # ------------------------------------------------------------------

    def _launch(self, gen_id: int) -> None:
        req = self._build_req(self.data_source.get_samples(self.batch_size), gen_id)
        refs, worker_local = self._generate_async(req)
        self._inflight.append(
            {"refs": refs, "worker_local": worker_local, "req": req,
             "gen_id": gen_id, "weight_version": self._weight_version}
        )

    def _score_into_buffer(self, rec: Dict[str, Any], resp: RolloutResp) -> None:
        """Score a completed generation (diffusion reward) and split into the buffer."""
        req = rec["req"]
        for name, track in list(resp.tracks.items()):
            if track.segment is not None:
                resp.tracks[name] = self.reward.score_and_attach(req=req, track=track)
        self._drop_decoded(req, resp, rollout_id=rec["gen_id"])
        (track,) = resp.tracks.values()
        for group in track.split():
            self._buffer.put(group, weight_version=rec["weight_version"], gen_id=rec["gen_id"])

    def _reap_ready(self) -> None:
        still: List[Dict[str, Any]] = []
        for rec in self._inflight:
            if self._is_ready(rec["refs"]):
                self._score_into_buffer(rec, self._collect_resp(rec["refs"], rec["worker_local"]))
            else:
                still.append(rec)
        self._inflight = still

    def _drain_all(self) -> None:
        """Finish + buffer EVERY in-flight generation (mandatory before a weight sync)."""
        for rec in self._inflight:
            self._score_into_buffer(rec, self._collect_resp(rec["refs"], rec["worker_local"]))
        self._inflight = []

    # ------------------------------------------------------------------
    # Advantage + train tail (mirrors DiffusionTrainer.train_step, no rollout)
    # ------------------------------------------------------------------

    def _advantage_and_train(
        self, track: RolloutTrack, resp: RolloutResp, *, training_progress: float, rollout_id: int, t0: float
    ) -> Tuple[TrainStepResult, float]:
        mean_reward = 0.0
        if track.rewards is not None:
            track.rewards = hydrate(track.rewards)
            mean_reward = float(track.rewards.to(torch.float32).mean().item())
        track = track.compute_advantages(normalize=True, use_global_std=self._adv_use_global_std)
        (name,) = resp.tracks.keys()
        resp.tracks[name] = track
        result = self.stack.train_track(track, training_progress=float(training_progress))
        self.wandb_logger.log_rollout_step(rollout_id, result, resp, step_time_s=time.perf_counter() - t0)
        return result, mean_reward

    # ------------------------------------------------------------------
    # Async train loop (mirrors AsyncARTrainer.train / _next_batch)
    # ------------------------------------------------------------------

    def train(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "auto",
    ) -> None:
        interval = max(1, weight_sync_interval)
        stale = self._buffer_max_staleness if self._buffer_max_staleness is not None else 0
        M = self._max_inflight

        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={"max_inflight": M, "buffer_max_staleness": stale, "weight_sync_interval": interval},
        )

        self._buffer = _RolloutBuffer()
        self._inflight: List[Dict[str, Any]] = []
        self._launch_id = start_rollout

        if resumed and self.weight_sync is not None:
            self.weight_sync.sync()
        if self.eval_interval > 0:
            self.evaluate(start_rollout)

        try:
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                picked = self._next_batch(rollout_id, interval, M, stale, num_rollouts)
                track = RolloutTrack.concat([p[0] for p in picked])
                resp = RolloutResp(tracks={"diffusion": track})
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self._advantage_and_train(
                    track, resp, training_progress=training_progress, rollout_id=rollout_id, t0=t0
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                step = rollout_id + 1
                if self.eval_interval > 0 and step % self.eval_interval == 0:
                    self._drain_all()
                    self.evaluate(rollout_id + 1)
                if save_interval > 0 and (step % save_interval == 0 or step >= num_rollouts):
                    self._drain_all()
                    self.maybe_save_checkpoint(
                        rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                    )
                if step % interval == 0 and self.weight_sync is not None:
                    self._drain_all()  # MANDATORY: weight/KV update corrupts in-flight generations
                    self.weight_sync.sync()
                    self._weight_version += 1
        finally:
            self._drain_all()
            self._finish_wandb()

    def _next_batch(self, rollout_id: int, interval: int, M: int, stale: int, num_rollouts: int):
        """Top up launches (clamped to ``stale`` weight-syncs ahead), reap, return the
        freshest ``batch_size`` groups (blocking on the oldest in-flight if short)."""
        while True:
            staleness_window = ((rollout_id // interval) + 1 + stale) * interval
            ceiling = min(num_rollouts, staleness_window)
            while self._launch_id < ceiling and len(self._inflight) < M:
                self._launch(self._launch_id)
                self._launch_id += 1

            self._reap_ready()
            picked = self._buffer.drain_freshest(
                self.batch_size, current_version=self._weight_version, max_staleness=stale
            )
            if picked is not None:
                return picked
            if self._inflight:
                ray.get(self._inflight[0]["refs"])
            else:
                raise RuntimeError("async-diffusion: buffer underflow with no in-flight generations")
