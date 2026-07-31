"""Reusable outer-loop programs and trainer lifecycle ownership.

Model/domain trainers own one-step semantics and runtime wiring.  The programs
in this module own the repeated outer-loop protocol: resume, data restoration,
logging, evaluation/checkpoint cadence, quiescence, and ordered teardown.

There are deliberately three program families rather than one callback-heavy
universal loop:

* :class:`BatchRLProgram` for synchronous and buffered AR/diffusion-style RL;
* :class:`AgenticRLProgram` for barrier, partial, and resident agentic drives;
* :class:`SFTProgram` for supervised epoch/cursor semantics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class LoopSpec:
    """Invocation-owned cadence and checkpoint settings for one loop run."""

    total_steps: int
    weight_sync_interval: int = 1
    save_interval: int = 0
    save_dir: Optional[str] = None
    load_dir: Optional[str] = None
    save_mode: str = "auto"

    @property
    def sync_interval(self) -> int:
        return max(1, int(self.weight_sync_interval))

    def save_due(self, completed_step: int) -> bool:
        return self.save_interval > 0 and (
            completed_step % self.save_interval == 0 or completed_step >= self.total_steps
        )


@dataclass
class LoopState:
    """Mutable progress shared by a program and its trainer hooks."""

    spec: LoopSpec
    start_step: int
    resumed: bool
    step: int = -1

    @property
    def completed_step(self) -> int:
        return self.step + 1

    @property
    def training_progress(self) -> float:
        return self.step / max(1, self.spec.total_steps - 1)

    @property
    def is_first_resumed_step(self) -> bool:
        return self.resumed and self.step == self.start_step

    @property
    def sync_before_step(self) -> bool:
        return self.is_first_resumed_step or (self.step > 0 and self.step % self.spec.sync_interval == 0)


class TrainerLifecycle:
    """Prepare and tear down a trainer while preserving the primary failure.

    Cleanup is ordered as ``before_finish`` callbacks (async drain / agentic
    abort), checkpoint and weight-sync flush plus logger finalization, then
    runtime shutdown.  Every callback is attempted.  If training is already
    failing, cleanup failures are logged and suppressed; otherwise the first
    cleanup failure is re-raised after the remaining cleanup has run.
    """

    def __init__(self, trainer: Any, logger: logging.Logger) -> None:
        self.trainer = trainer
        self.logger = logger
        self._before_finish: list[tuple[str, Callable[[], None]]] = []

    def __enter__(self) -> "TrainerLifecycle":
        return self

    def add_before_finish(self, name: str, callback: Callable[[], None]) -> None:
        self._before_finish.append((name, callback))

    def start(
        self,
        spec: LoopSpec,
        *,
        restore_data: Callable[[int], None],
        wandb_extra: Optional[Dict[str, Any]] = None,
    ) -> LoopState:
        start_step = self.trainer.maybe_load_checkpoint(
            spec.load_dir,
            num_rollouts=spec.total_steps,
        )
        state = LoopState(
            spec=spec,
            start_step=start_step,
            resumed=bool(spec.load_dir),
        )
        restore_data(start_step)
        self.trainer._init_wandb(
            num_rollouts=spec.total_steps,
            extra=wandb_extra,
        )
        return state

    def __exit__(self, exc_type, exc, traceback) -> bool:
        primary_active = exc is not None
        first_cleanup_error: Optional[BaseException] = None

        def run(name: str, callback: Callable[[], None]) -> None:
            nonlocal first_cleanup_error
            try:
                callback()
            except BaseException as cleanup_error:  # noqa: BLE001 - teardown must continue
                if primary_active or first_cleanup_error is not None:
                    self.logger.warning("%s failed during trainer teardown", name, exc_info=True)
                else:
                    first_cleanup_error = cleanup_error

        for name, callback in self._before_finish:
            run(name, callback)
        run(
            "trainer finalization",
            lambda: self.trainer._finish_wandb(active_exception=primary_active or first_cleanup_error is not None),
        )
        run("runtime shutdown", self.trainer._shutdown_runtime)

        if not primary_active and first_cleanup_error is not None:
            raise first_cleanup_error
        return False


class LoopProgram:
    """Base for explicit outer-loop families."""

    def __init__(self, trainer: Any, logger: logging.Logger) -> None:
        self.trainer = trainer
        self.logger = logger

    def _save(self, state: LoopState) -> None:
        spec = state.spec
        self.trainer.maybe_save_checkpoint(
            state.step,
            spec.total_steps,
            save_interval=spec.save_interval,
            save_dir=spec.save_dir,
            save_mode=spec.save_mode,
        )


class BatchRLProgram(LoopProgram):
    """Synchronous and buffered asynchronous batch-RL outer loops."""

    def run(self, spec: LoopSpec) -> None:
        """Run a synchronous ``request → train_step`` program."""
        trainer = self.trainer
        with TrainerLifecycle(trainer, self.logger) as lifecycle:
            state = lifecycle.start(
                spec,
                restore_data=trainer._loop_restore_data,
                wandb_extra=trainer._loop_wandb_extra(),
            )
            try:
                if trainer.eval_interval > 0:
                    trainer._loop_evaluate_baseline(state)

                for step in range(state.start_step, spec.total_steps):
                    state.step = step
                    trainer._loop_before_step(state)
                    inputs = trainer.data_source.get_samples(trainer.batch_size)
                    sample = trainer._build_request_sample(inputs, step)
                    sync_weights = state.sync_before_step or trainer._loop_force_sync(state)
                    result, mean_reward = trainer.train_step(
                        sample,
                        training_progress=state.training_progress,
                        sync_weights=sync_weights,
                        rollout_id=step,
                    )
                    trainer.wandb_logger.log_progress(
                        step,
                        spec.total_steps,
                        result,
                        mean_reward,
                        logger=self.logger,
                    )
                    if trainer.eval_interval > 0 and state.completed_step % trainer.eval_interval == 0:
                        trainer._loop_evaluate_periodic(state)
                    self._save(state)
            except Exception:
                trainer._loop_on_error(state)
                raise

    def run_async(self, spec: LoopSpec) -> None:
        """Run the shared buffered AR/diffusion program.

        The domain trainer supplies request/scoring/train-tail hooks; this method
        owns scheduler construction, launch/reap cadence, and the mandatory
        quiescence before eval, checkpoint, or weight update.
        """
        from unirl.rollout.async_runtime import AsyncRolloutScheduler, RayGenerationDispatcher
        from unirl.types.sample import Sample

        trainer = self.trainer
        scheduler_started = False

        def drain() -> None:
            if scheduler_started:
                trainer._async_scheduler.drain_all(trainer._score_completed)

        stale = trainer._buffer_max_staleness if trainer._buffer_max_staleness is not None else 0
        max_inflight = trainer._max_inflight
        extra = dict(trainer._loop_wandb_extra() or {})
        extra.update(
            {
                "max_inflight": max_inflight,
                "buffer_max_staleness": stale,
                "weight_sync_interval": spec.sync_interval,
            }
        )
        if hasattr(trainer, "_train_fraction"):
            extra["train_fraction"] = trainer._train_fraction

        with TrainerLifecycle(trainer, self.logger) as lifecycle:
            lifecycle.add_before_finish("in-flight generation drain", drain)
            state = lifecycle.start(
                spec,
                restore_data=trainer._loop_restore_data,
                wandb_extra=extra,
            )
            trainer._async_scheduler = AsyncRolloutScheduler(
                RayGenerationDispatcher(trainer.rollout),
                groups_per_step=trainer.batch_size,
                reap_before_launch=bool(getattr(trainer, "_ASYNC_REAP_BEFORE_LAUNCH", False)),
            )
            trainer._async_scheduler.reset(state.start_step)
            scheduler_started = True

            if state.resumed and trainer.weight_sync is not None:
                trainer.weight_sync.sync()
            if trainer.eval_interval > 0:
                trainer._loop_evaluate_baseline(state)

            for step in range(state.start_step, spec.total_steps):
                state.step = step
                t0 = time.perf_counter()
                picked = trainer._async_scheduler.next_step(
                    rollout_id=step,
                    sync_interval=spec.sync_interval,
                    max_inflight=max_inflight,
                    max_staleness=stale,
                    num_rollouts=spec.total_steps,
                    current_version=trainer._weight_version,
                    build_sample=trainer._build_async_sample,
                    on_complete=trainer._score_completed,
                )
                sample = Sample.concat([item.sample for item in picked])
                result, mean_reward = trainer._advantage_and_train(
                    sample,
                    training_progress=state.training_progress,
                    rollout_id=step,
                    t0=t0,
                )
                trainer.wandb_logger.log_progress(
                    step,
                    spec.total_steps,
                    result,
                    mean_reward,
                    logger=self.logger,
                )

                eval_due = trainer.eval_interval > 0 and state.completed_step % trainer.eval_interval == 0
                save_due = spec.save_due(state.completed_step)
                sync_due = state.completed_step % spec.sync_interval == 0 and trainer.weight_sync is not None
                if eval_due or save_due or sync_due:
                    drain()
                if eval_due:
                    trainer._loop_evaluate_periodic(state)
                if save_due:
                    self._save(state)
                if sync_due:
                    trainer.weight_sync.sync()
                    trainer._weight_version += 1


class AgenticRLProgram(BatchRLProgram):
    """Barrier, colocated-partial, and disaggregated agentic programs."""

    def run_barrier(self, spec: LoopSpec) -> None:
        """Agentic barrier generation has batch cadence but agentic step semantics."""
        super().run(spec)

    def run_partial(self, spec: LoopSpec) -> None:
        """Run colocated over-sample/commit/abort agentic training."""
        from unirl.trainer.agentic_async import _GroupAssembler, _GroupBuffer

        trainer = self.trainer
        runtime_started = False
        stale = trainer._buffer_max_staleness if trainer._buffer_max_staleness is not None else 0

        def restore_data(start_step: int) -> None:
            for _ in range(start_step):
                trainer.data_source.get_samples(trainer._oversample)

        def stop_drive() -> None:
            if not runtime_started:
                return
            carried = trainer.rollout.abort()[0]
            trainer._pump()
            trainer._apply_tail_policy(carried, spec.total_steps)

        extra = dict(trainer._loop_wandb_extra() or {})
        extra.update(
            {
                "oversample_batch_size": trainer._oversample,
                "buffer_max_staleness": stale,
                "tail_policy": trainer._tail_policy,
                "weight_sync_interval": spec.sync_interval,
            }
        )

        with TrainerLifecycle(trainer, self.logger) as lifecycle:
            lifecycle.add_before_finish("partial agentic drive abort", stop_drive)
            state = lifecycle.start(spec, restore_data=restore_data, wandb_extra=extra)
            trainer._buffer = _GroupBuffer()
            trainer._assembler = _GroupAssembler(trainer._n)
            trainer._carried = []
            trainer._gen_id = state.start_step
            runtime_started = True

            if trainer.eval_interval > 0:
                trainer._loop_evaluate_baseline(state)
            for step in range(state.start_step, spec.total_steps):
                state.step = step
                t0 = time.perf_counter()
                groups = trainer._drive_partial(
                    step,
                    state.sync_before_step,
                    stale,
                )
                result, mean_reward = trainer._train_on_groups(
                    groups,
                    training_progress=state.training_progress,
                    rollout_id=step,
                    t0=t0,
                )
                trainer._reset_transport_buffers()
                trainer.wandb_logger.log_progress(
                    step,
                    spec.total_steps,
                    result,
                    mean_reward,
                    logger=self.logger,
                )
                if trainer.eval_interval > 0 and state.completed_step % trainer.eval_interval == 0:
                    trainer._loop_evaluate_periodic(state)
                self._save(state)

    def run_async(self, spec: LoopSpec) -> None:
        """Run the resident producer/consumer agentic program."""
        from unirl.trainer.agentic_async import _GroupAssembler, _GroupBuffer

        trainer = self.trainer
        runtime_started = False
        stale = trainer._buffer_max_staleness if trainer._buffer_max_staleness is not None else 0

        def restore_data(start_step: int) -> None:
            for _ in range(start_step):
                trainer.data_source.get_samples(trainer._oversample)

        def stop_drive() -> None:
            if not runtime_started:
                return
            checkpointed = trainer.rollout.abort()[0]
            trainer._pump()
            trainer._apply_tail_policy(checkpointed, spec.total_steps)
            if checkpointed:
                trainer._log_tail_metrics(spec.total_steps)

        extra = dict(trainer._loop_wandb_extra() or {})
        extra.update(
            {
                "buffer_max_staleness": stale,
                "oversample_batch_size": trainer._oversample,
                "train_fraction": trainer._train_fraction,
                "tail_policy": trainer._tail_policy,
                "weight_sync_interval": spec.sync_interval,
            }
        )

        with TrainerLifecycle(trainer, self.logger) as lifecycle:
            lifecycle.add_before_finish("async agentic drive abort", stop_drive)
            state = lifecycle.start(spec, restore_data=restore_data, wandb_extra=extra)
            trainer._buffer = _GroupBuffer()
            trainer._assembler = _GroupAssembler(trainer._n)
            trainer._pending_carried = []
            trainer._gen_id = state.start_step
            runtime_started = True

            has_work = state.start_step < spec.total_steps
            if has_work and state.start_step and trainer.weight_sync is not None:
                trainer.weight_sync.sync()
            if has_work:
                trainer._submit_drive(carried=[], rollout_id=state.start_step)

            for step in range(state.start_step, spec.total_steps):
                state.step = step
                t0 = time.perf_counter()
                groups = trainer._next_batch(step)
                result, mean_reward = trainer._train_on_groups(
                    groups,
                    training_progress=state.training_progress,
                    rollout_id=step,
                    t0=t0,
                )
                trainer._reset_transport_buffers()
                trainer.wandb_logger.log_progress(
                    step,
                    spec.total_steps,
                    result,
                    mean_reward,
                    logger=self.logger,
                )

                save_due = spec.save_due(state.completed_step)
                sync_due = state.completed_step % spec.sync_interval == 0 and trainer.weight_sync is not None
                if save_due or sync_due:
                    checkpointed = trainer.rollout.abort()[0]
                    trainer._pump()
                    carried = trainer._apply_tail_policy(checkpointed, step)
                    if checkpointed:
                        trainer._log_tail_metrics(state.completed_step)
                    if save_due:
                        self._save(state)
                    if sync_due:
                        trainer.weight_sync.sync()
                        trainer._weight_version += 1
                    if state.completed_step < spec.total_steps:
                        trainer._submit_drive(carried=carried, rollout_id=state.completed_step)


class SFTProgram(LoopProgram):
    """Supervised loop with exact dataset-cursor checkpoint semantics."""

    def run(self, spec: LoopSpec) -> None:
        trainer = self.trainer

        def restore_data(start_step: int) -> None:
            trainer._load_data_state(spec.load_dir, start_step)

        with TrainerLifecycle(trainer, self.logger) as lifecycle:
            state = lifecycle.start(spec, restore_data=restore_data)
            if trainer.eval_interval > 0:
                trainer.evaluate(step=-1)
            for step in range(state.start_step, spec.total_steps):
                state.step = step
                t0 = time.perf_counter()
                records = trainer.data_source.get_samples(trainer.batch_size)
                result = trainer.train_step(
                    records,
                    training_progress=state.training_progress,
                )
                step_time = time.perf_counter() - t0
                self.logger.info(
                    "step %d/%d  loss=%.5f grad_norm=%.4f lr=%.2e epoch=%.3f  %.1fs",
                    state.completed_step,
                    spec.total_steps,
                    result.loss,
                    result.grad_norm,
                    result.lr,
                    trainer.data_source.epoch,
                    step_time,
                )
                trainer.wandb_logger.log_step(
                    state.completed_step,
                    {
                        "train/loss": result.loss,
                        "train/grad_norm": result.grad_norm,
                        "train/lr": result.lr,
                        "train/epoch": trainer.data_source.epoch,
                        "perf/step_time_s": step_time,
                        **{f"train/{key}": value for key, value in dict(result.metrics).items()},
                    },
                    prefix="",
                )
                if trainer.eval_interval > 0 and state.completed_step % trainer.eval_interval == 0:
                    trainer.evaluate(step=step)
                self._save(state)
                trainer._save_data_state(
                    step,
                    spec.total_steps,
                    save_interval=spec.save_interval,
                    save_dir=spec.save_dir,
                )


__all__ = [
    "AgenticRLProgram",
    "BatchRLProgram",
    "LoopProgram",
    "LoopSpec",
    "LoopState",
    "SFTProgram",
    "TrainerLifecycle",
]
