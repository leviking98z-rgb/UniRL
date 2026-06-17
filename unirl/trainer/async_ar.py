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

Generation is launched as **non-blocking Ray futures** (`_generate_async`) and
reaped on the single driver thread (`_is_ready` / `_collect_resp`) — no producer
thread, no locks. Draining all in-flight generations before each weight sync is
**mandatory** (the engine corrupts an in-flight generation when weights + KV
cache update mid-flight); this is the single-threaded ``_drain_all`` quiesce.

Subclasses ``ARTrainer`` to reuse ``_build_req``/``evaluate`` and ``BaseTrainer``
plumbing, but ``__init__`` calls ``BaseTrainer.__init__`` **directly** (the parent
opens the colocate ``placement(fraction=1.0)`` block we replace with two slabs).
"""

import inspect
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import ray
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.dispatch import DISPATCH_MODE_REGISTRY, Dispatch
from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import WorkerLocalTransport, hydrate
from unirl.distributed.tensor.pytree import infer_batch_size
from unirl.train.stack import TrainStepResult
from unirl.trainer.ar import ARTrainer
from unirl.trainer._partial_rollout import build_continuation_req, concretize, merge_continuation, unfinished_indices
from unirl.trainer.base import BaseTrainer
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import BaseSamplingParams
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


class _RolloutBuffer:
    """Group-keyed rollout buffer (single-threaded; no lock needed).

    Each entry is one prompt's GRPO group — a ``RolloutTrack`` of
    ``samples_per_prompt`` already-scored samples — stamped with the
    ``weight_version`` it was generated under and a monotonic ``gen_id`` for
    freshness ordering. Groups are always complete (the whole ``generate``
    finished before they are ``put``), so there is no partial-group bookkeeping.
    """

    def __init__(self) -> None:
        self._items: List[Tuple[RolloutTrack, int, int]] = []  # (group, weight_version, gen_id)

    def put(self, track: RolloutTrack, *, weight_version: int, gen_id: int) -> None:
        self._items.append((track, int(weight_version), int(gen_id)))

    def size(self) -> int:
        return len(self._items)

    def drain_freshest(
        self,
        n: int,
        *,
        current_version: Optional[int] = None,
        max_staleness: Optional[int] = None,
    ) -> Optional[List[Tuple[RolloutTrack, int, int]]]:
        """Pop the ``n`` freshest complete groups, carrying leftovers forward.

        Returns ``None`` if fewer than ``n`` groups remain after eviction. When
        ``max_staleness`` is set, groups older than ``current_version -
        max_staleness`` weight versions are evicted first (bounded off-policy).
        """
        if max_staleness is not None and current_version is not None:
            self._items = [it for it in self._items if current_version - it[1] <= max_staleness]
        if len(self._items) < n:
            return None
        self._items.sort(key=lambda it: it[2], reverse=True)  # freshest gen_id first
        picked, self._items = self._items[:n], self._items[n:]
        return picked


class AsyncARTrainer(ARTrainer):
    """Disaggregated async AR trainer (two slabs, resident engine, NCCL sync)."""

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
        adv_normalization_scope: str = "group",
        normalize_adv_by_std: bool = True,
        balance_shards: bool = False,
        eval_interval: int = 0,
        eval_num_prompts: int = 60,
        eval_samples_per_prompt: int = 16,
        eval_temperature: float = 1.0,
        # ---- async knobs ----
        train_fraction: float = 0.5,
        max_inflight: int = 1,
        buffer_max_staleness: Optional[int] = None,
        partial_rollout: bool = False,
        # ---- FP8-rollout drift guard ----
        rollout_drift_warn: Optional[float] = None,
        rollout_drift_abort: Optional[float] = None,
    ) -> None:
        # Call BaseTrainer.__init__ directly: ARTrainer.__init__ opens the
        # colocate ``placement(fraction=1.0)`` block, which is exactly what we
        # must NOT run. (ARTrainer itself just calls BaseTrainer.__init__ here.)
        BaseTrainer.__init__(self, cfg=cfg, logging_cfg=logging_cfg)

        # ---- scalar/config fields (mirrors ar.py:62-88) ----
        self.batch_size = batch_size
        self.adv_normalization_scope = adv_normalization_scope
        self.normalize_adv_by_std = normalize_adv_by_std
        self.balance_shards = bool(balance_shards)
        self.eval_interval = int(eval_interval)
        self.eval_num_prompts = int(eval_num_prompts)
        self.eval_samples_per_prompt = int(eval_samples_per_prompt)
        self.eval_temperature = float(eval_temperature)
        self.data_source = instantiate(data_source_cfg)
        self.sampling_params: BaseSamplingParams = instantiate(sampling_cfg)
        self.weight_sync = None

        # ---- async state ----
        self._train_fraction = float(train_fraction)
        self._max_inflight = max(1, int(max_inflight))
        self._buffer_max_staleness = buffer_max_staleness
        self._weight_version = 0  # driver-tracked policy version (# of weight syncs issued)
        self._partial = bool(partial_rollout)  # slime-style interrupt-at-sync + carry
        # FP8-rollout drift guard (both None = off; the bf16 default). When the
        # rollout engine generates in a lower precision than the BF16 train
        # forward (rollout.config.quantization='fp8'), the rollout-vs-replay
        # |Δlogp| gap widens; old_logp_source='rollout' makes that an
        # importance-sampling correction, but a gap past these thresholds means
        # the IS ratio tail is heavy enough to bias/destabilize the update — warn,
        # or abort rather than train silently through it. See docs/fp8_rollout.md.
        self._drift_warn = None if rollout_drift_warn is None else float(rollout_drift_warn)
        self._drift_abort = None if rollout_drift_abort is None else float(rollout_drift_abort)
        self._carry = []  # incomplete generations awaiting continuation after a sync
        self._server_urls = []  # cached sglang /abort_request targets (filled post-rollout)
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
        total = int(self.batch_size) * int(self.sampling_params.samples_per_prompt)
        for slab_name, slab in (("train", self._train_devices), ("rollout", self._rollout_devices)):
            if total % slab != 0:
                raise ValueError(
                    f"batch_size * samples_per_prompt = {total} is not divisible by the "
                    f"{slab_name} slab size {slab}; adjust batch_size / samples_per_prompt / train_fraction."
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
            if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
                raise ValueError(
                    "AsyncARTrainer needs a dedicated-rollout engine (vllm/sglang) on the "
                    "separate slab; the trainside direct-sampling engine needs the pipeline "
                    "as a local sibling and cannot live cross-slab."
                )
            self.rollout = remote(**rollout_parsed)
            try:
                self._server_urls = [u for u in self.rollout.get_server_url() if u]
            except Exception as _e:
                logger.warning("partial-rollout: could not fetch server urls for abort: %s", _e)
            # sgl-router rollout LB (HTTP backend, multi-worker): point every engine at
            # one router that load-balances by policy (none = static per-rank DP_SCATTER).
            _rcfg = rollout_parsed.get("config")
            if (_rcfg or {}).get("backend", "native") == "http" and (_rcfg or {}).get("router_policy", "none") != "none":
                from unirl.rollout.engine.sglang.backends.router import launch_router, resolve_policy
                _urls = [u for u in self.rollout.get_server_url() if u]
                self._router_process, _rurl = launch_router(_urls, resolve_policy(_rcfg["router_policy"]))
                self.rollout.install_router(_rurl)
                logger.info("sgl-router installed at %s (policy=%s, %d workers)", _rurl, _rcfg["router_policy"], len(_urls))

        if self.weight_sync is not None:
            self._connect_separate(sync_cfg)

    def _connect_separate(self, sync_cfg: DictConfig) -> None:
        """One-time cross-slab handshake (NCCL branch of diffusion.py:191-208).

        Rank 0 picks a rendezvous addr/port, is handed the rollout slab's Worker
        actor handles, then ``connect`` fires each rollout worker's
        ``init_weights_update_group`` non-blocking and joins the broadcast group
        itself. Only ``NCCLWeightSync`` is supported here (always cross-slab
        full-weight); a non-NCCL target is a config error.
        """
        target = str(sync_cfg.get("_target_", ""))
        if not target.endswith("NCCLWeightSync"):
            raise ValueError(
                f"AsyncARTrainer (separate slabs) requires a cross-slab weight sync "
                f"(NCCLWeightSync); got sync._target_={target!r}."
            )
        addr, port = self.weight_sync.pick_master()[0]
        self.weight_sync.set_rollout_targets(self.rollout.workers, self.rollout.role_name)
        self.weight_sync.connect(
            master_addr=addr,
            master_port=port,
            num_rollout_gpus=len(self.rollout.workers),
        )

    # ------------------------------------------------------------------
    # Non-blocking generate seam (faithful split of handle.py:271-300 at ray.get)
    # ------------------------------------------------------------------

    def _generate_async(self, req: RolloutReq):
        """Launch ``generate`` non-blocking; return (refs, worker_local).

        ``refs`` is a flat list of ObjectRefs, one per rollout worker. This runs
        the launch half of ``handle_fn`` (handle.py:271-290) for the rollout
        Handle's ``@distributed(DP_SCATTER)`` ``generate``; no ``ray.get`` here.
        Generation is always grad_mode=False, so the grad-context branch is moot.
        """
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
        """Join a completed generate (handle.py:291,297,300) → full RolloutResp.

        Byte-identical to ``self.rollout.generate(req)``; blocks in ``ray.get``
        (instant for an already-ready generation).
        """
        r = self.rollout
        collect_fn = DISPATCH_MODE_REGISTRY[Dispatch.DP_SCATTER]["collect_fn"]
        results = ray.get(refs)
        results = [r._rebind_tree(x, r.workers[i], worker_local=worker_local) for i, x in enumerate(results)]
        return collect_fn(r, results)

    @staticmethod
    def _is_ready(refs) -> bool:
        """True iff every worker's generate ref is resolved (non-blocking reap)."""
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
        return len(ready) == len(refs)

    # ------------------------------------------------------------------
    # In-flight bookkeeping
    # ------------------------------------------------------------------

    def _launch(self, gen_id: int) -> None:
        """Build a request and launch one non-blocking generation."""
        req = self._build_req(self.data_source.get_samples(self.batch_size), gen_id)
        refs, worker_local = self._generate_async(req)
        self._inflight.append(
            {
                "refs": refs,
                "worker_local": worker_local,
                "req": req,
                "orig_req": req,
                "carried_track": None,
                "cont_idx": None,
                "gen_id": gen_id,
                "weight_version": self._weight_version,
            }
        )

    def _score_into_buffer(self, rec: Dict[str, Any], resp: RolloutResp) -> None:
        """Score a completed generation and split its groups into the buffer.

        Scoring must precede ``_drop_decoded`` (the reward reads ``decoded``).
        Keyed by ``gen_id`` so media panels behave like the old pipeline path.
        """
        req = rec["req"]
        for name, track in list(resp.tracks.items()):
            if track.segment is not None:
                resp.tracks[name] = self.reward.score_and_attach(req=req, track=track)
        self._drop_decoded(req, resp, rollout_id=rec["gen_id"])
        (track,) = resp.tracks.values()
        for group in track.split():
            self._buffer.put(group, weight_version=rec["weight_version"], gen_id=rec["gen_id"])

    def _reap_ready(self) -> None:
        """Move every completed in-flight generation into the buffer (scored)."""
        still: List[Dict[str, Any]] = []
        for rec in self._inflight:
            if self._is_ready(rec["refs"]):
                resp = self._collect_resp(rec["refs"], rec["worker_local"])
                if self._partial:
                    carry = self._ingest_partial(rec, resp)
                    if carry is not None:
                        # incomplete outside a sync (an aborted sample) — continue now
                        cont_req = build_continuation_req(carry["orig_req"], carry["carried_track"], carry["_unfinished_idx"], dp_size=self._rollout_devices)
                        refs, wl = self._generate_async(cont_req)
                        still.append({**carry, "req": cont_req, "cont_idx": carry["_unfinished_idx"],
                                      "refs": refs, "worker_local": wl})
                else:
                    self._score_into_buffer(rec, resp)
            else:
                still.append(rec)
        self._inflight = still

    # ------------------------------------------------------------------
    # Partial rollout (slime-style): interrupt in-flight generations at the
    # weight-sync boundary, carry the unfinished ones, continue after the sync.
    # ------------------------------------------------------------------
    def _abort_servers(self) -> None:
        """Tell every rollout SRT server to abort all in-flight requests. POSTed
        directly from the driver (the ray engine actor is busy inside generate),
        which makes each pending generate() return its partial output."""
        import json as _json
        import urllib.request as _u
        for url in self._server_urls:
            try:
                _r = _u.Request(f"{url}/abort_request", data=_json.dumps({"abort_all": True}).encode(),
                                headers={"Content-Type": "application/json"})
                _u.urlopen(_r, timeout=30)
            except Exception as _e:  # pragma: no cover
                logger.warning("partial-rollout: abort POST to %s failed: %s", url, _e)

    def _ingest_partial(self, rec, resp):
        """Merge a (possibly partial) generation into its carried state. Returns
        None when the whole generation is complete (scored + buffered), else a
        carry rec with the accumulated track + the still-unfinished indices."""
        track_name = list(resp.tracks)[0]
        (track,) = resp.tracks.values()
        carried = rec.get("carried_track")
        merged = track if carried is None else merge_continuation(carried, track, rec["cont_idx"])
        idx = unfinished_indices(merged)
        if not idx:
            self._score_into_buffer({**rec, "req": rec["orig_req"]},
                                    RolloutResp(tracks={track_name: merged}))
            return None
        if carried is None:
            merged = concretize(merged)  # fresh raw track -> concrete so it survives the sync
        return {**rec, "carried_track": merged, "_unfinished_idx": idx}

    def _abort_and_carry(self) -> None:
        """Interrupt-at-sync replacement for _drain_all. A daemon thread RE-POSTs
        abort_all every 0.3s while we collect the in-flight generations: with
        max_inflight>1 the per-generation backends keep submitting requests after a
        one-shot abort, so a single abort leaves residual stragglers that gate the
        collect (measured: one-shot abort quiesce ~= baseline drain ~40s; repeated
        abort drops it to ~6s). Each collected generation is buffered if complete or
        stashed for continuation."""
        import threading as _thr
        _tq = time.perf_counter()
        _stop = _thr.Event()
        _spam = {"n": 0}
        def _keep_aborting():
            while not _stop.is_set():
                self._abort_servers(); _spam["n"] += 1
                _stop.wait(0.3)
        _th = _thr.Thread(target=_keep_aborting, daemon=True); _th.start()
        try:
            for rec in self._inflight:
                carry = self._ingest_partial(rec, self._collect_resp(rec["refs"], rec["worker_local"]))
                if carry is not None:
                    self._carry.append(carry)
        finally:
            _stop.set(); _th.join(timeout=2)
        logger.info("partial-rollout: abort+collect=%.2fs (abort_posts=%d)", time.perf_counter() - _tq, _spam["n"])
        logger.info("partial-rollout: sync boundary — %d generations aborted, %d carried for continuation",
                    len(self._inflight), len(self._carry))
        self._inflight = []

    def _relaunch_carry(self) -> None:
        """After a weight sync, continue each carried generation from its tokens-
        so-far (input_ids = prompt + generated; off-policy across this version)."""
        for rec in self._carry:
            cont_req = build_continuation_req(rec["orig_req"], rec["carried_track"], rec["_unfinished_idx"], dp_size=self._rollout_devices)
            refs, worker_local = self._generate_async(cont_req)
            self._inflight.append({**rec, "req": cont_req, "cont_idx": rec["_unfinished_idx"],
                                   "refs": refs, "worker_local": worker_local,
                                   "weight_version": self._weight_version})
        if self._carry:
            logger.info("partial-rollout: relaunched %d carried generations under weight v%d", len(self._carry), self._weight_version)
        self._carry = []

    def _drain_all(self) -> None:
        """Finish + buffer EVERY in-flight generation (the single-threaded quiesce).

        Mandatory before a weight sync (the engine corrupts an in-flight generate
        when weights + KV cache update mid-flight), before eval/checkpoint (shared
        engine), and in ``finally`` (no leaked ObjectRefs).
        """
        for rec in self._inflight:
            self._score_into_buffer(rec, self._collect_resp(rec["refs"], rec["worker_local"]))
        self._inflight = []

    # ------------------------------------------------------------------
    # Train tail (mirrors ar.py:152-182, minus wake/sleep) — reward parity
    # ------------------------------------------------------------------

    def _advantage_and_train(
        self,
        track: RolloutTrack,
        resp: RolloutResp,
        *,
        training_progress: float,
        rollout_id: int,
        t0: Optional[float] = None,
    ) -> Tuple[TrainStepResult, float]:
        """Advantage + optimizer step for a SCORED track (rewards already attached)."""
        if t0 is None:
            t0 = time.perf_counter()
        mean_reward = 0.0
        if track.rewards is not None:
            track.rewards = hydrate(track.rewards)
            mean_reward = float(track.rewards.to(torch.float32).mean().item())
        track = track.compute_advantages(normalize=self.normalize_adv_by_std, scope=self.adv_normalization_scope)
        (name,) = resp.tracks.keys()  # single-track for now; revisit if multi-track lands
        resp.tracks[name] = track
        if self.balance_shards:
            track = track.balance_shards(self._train_devices)  # over the TRAIN slab DP size
        result = self.stack.train_track(track, training_progress=float(training_progress))
        self._guard_rollout_drift(result, rollout_id=rollout_id)
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            resp,
            step_time_s=time.perf_counter() - t0,
            trunc_len=getattr(self.sampling_params, "max_new_tokens", None),
        )
        # train_step is bypassed, so BaseTrainer's per-step reset hook never
        # fires; reclaim transport buffers here (no-op for colocate_store/gpu).
        self._reset_transport_buffers()
        return result, mean_reward

    def _guard_rollout_drift(self, result: TrainStepResult, *, rollout_id: int) -> None:
        """FP8-rollout safety valve over the rollout↔replay |Δlogp| gap.

        The AR algorithms (GRPO/DRPO/CPPO) already emit
        ``rollout_replay_logp_absdiff_mean`` — the mean per-token |Δlogp| between
        the rollout engine's emitted logprobs (the behaviour policy ``mu``) and
        the BF16 teacher-forced replay (``pi``). With ``old_logp_source='rollout'``
        this gap IS the importance-sampling correction's exponent; a small gap is
        the FP8↔BF16 engine difference being absorbed cleanly. Past
        ``_drift_warn`` it signals a heavy IS ratio tail; past ``_drift_abort``
        it is large enough to bias/destabilize the update, so we refuse to train
        silently through it (the spec's "do NOT silently let a large mismatch
        through"). Both thresholds default off (bf16 rollout), so this is a no-op
        unless a recipe opts in.

        TODO(fp8): when the tail is heavy but bounded, prefer Truncated
        Importance Sampling (clamp ``r_t = exp(new_logp - old_logp)`` at a cap
        ``c``, e.g. r_t <- min(r_t, c)) over aborting — a per-token TIS clamp in
        the AR loss (cppo.py/drpo.py ``_*_loss``) keeps the update unbiased in
        expectation while taming the variance. Until that lands, the abort gate
        is the conservative stand-in. See docs/fp8_rollout.md.
        """
        if self._drift_warn is None and self._drift_abort is None:
            return
        drift = (result.metrics or {}).get("rollout_replay_logp_absdiff_mean")
        if drift is None:
            return
        drift = float(drift)
        if self._drift_abort is not None and drift > self._drift_abort:
            raise RuntimeError(
                f"rollout↔replay drift mean|Δlogp|={drift:.4f} exceeds "
                f"rollout_drift_abort={self._drift_abort:.4f} at rollout {rollout_id}. "
                "The FP8-rollout importance-sampling ratio tail is too heavy to train "
                "through safely (see docs/fp8_rollout.md). Lower the quantization "
                "aggressiveness, raise the threshold deliberately, or add Truncated "
                "Importance Sampling before re-enabling."
            )
        if self._drift_warn is not None and drift > self._drift_warn:
            logger.warning(
                "rollout↔replay drift mean|Δlogp|=%.4f exceeds rollout_drift_warn=%.4f "
                "at rollout %d — FP8 IS ratio tail is widening (still under the abort "
                "gate). Watch ratio_max / approx_kl; consider Truncated Importance "
                "Sampling (see docs/fp8_rollout.md).",
                drift, self._drift_warn, rollout_id,
            )

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def train(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "full",
    ) -> None:
        interval = max(1, weight_sync_interval)
        # Staleness budget: how many weight-syncs a generation may cross before it
        # is evicted. 0 (default) = on-policy (no generation crosses a sync).
        stale = self._buffer_max_staleness if self._buffer_max_staleness is not None else 0
        M = self._max_inflight

        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        # Single-threaded: exactly one get_samples(batch_size) per launch and
        # launches are 1:1 with rollout_id, so replaying start_rollout times
        # restores the exact stream position (deterministic resume).
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "adv_normalization_scope": self.adv_normalization_scope,
                "max_inflight": M,
                "buffer_max_staleness": stale,
                "weight_sync_interval": interval,
            },
        )

        self._buffer = _RolloutBuffer()
        self._inflight: List[Dict[str, Any]] = []
        self._launch_id = start_rollout

        if resumed and self.weight_sync is not None:
            self.weight_sync.sync()  # push restored weights into the fresh engine
        if self.eval_interval > 0:
            self.evaluate(rollout_id=-1)  # baseline; engine quiescent

        try:
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                picked = self._next_batch(rollout_id, interval, M, stale, num_rollouts)
                track = RolloutTrack.concat([p[0] for p in picked])
                resp = RolloutResp(tracks={"ar": track})
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self._advantage_and_train(
                    track, resp, training_progress=training_progress, rollout_id=rollout_id, t0=t0
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                step = rollout_id + 1
                if self.eval_interval > 0 and step % self.eval_interval == 0:
                    self._drain_all()  # eval shares the engine
                    self.evaluate(rollout_id=rollout_id)
                if save_interval > 0 and (step % save_interval == 0 or step >= num_rollouts):
                    self._drain_all()  # consistent engine + deterministic resume
                    self.maybe_save_checkpoint(
                        rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                    )
                if step % interval == 0 and self.weight_sync is not None:
                    _tq = time.perf_counter()
                    if self._partial:
                        self._abort_and_carry()  # interrupt stragglers, carry partials
                    else:
                        self._drain_all()  # MANDATORY: weight/KV update corrupts in-flight generations
                    logger.info("sync-barrier quiesce: %.3fs (mode=%s, rollout=%d, inflight_was=%d)",
                                time.perf_counter() - _tq, "abort" if self._partial else "drain",
                                rollout_id, len(self._carry) if self._partial else 0)
                    self.weight_sync.sync()
                    self._weight_version += 1
                    if self._partial:
                        self._relaunch_carry()  # continue carried generations under new weights
        finally:
            self._drain_all()
            self._finish_wandb()

    def _next_batch(self, rollout_id: int, interval: int, M: int, stale: int, num_rollouts: int):
        """Top up launches, reap completed generations, and return the freshest
        ``batch_size`` groups for ``rollout_id`` (blocking on the oldest in-flight
        generation if the buffer is short).

        The launch clamp is the load-bearing on-policy guarantee: a generation
        launched now is consumed later, so bound how far ahead we launch to
        ``stale`` weight-syncs. ``stale=0`` ⇒ never launch into a future
        sync-window ⇒ no generation crosses a sync ⇒ ``ratio≈1`` (on-policy).
        """
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
                ray.get(self._inflight[0]["refs"])  # block on oldest; next _reap_ready harvests it
            else:
                raise RuntimeError("async-ar: buffer underflow with no in-flight generations")
