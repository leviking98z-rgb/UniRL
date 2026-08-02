"""Provider-neutral timing instrumentation for trainer collaborators."""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator


class PhaseTimer:
    """Accumulate named critical-path wall-clock phases for one train step."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self.phases: Dict[str, float] = {}

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.phases[name] = self.phases.get(name, 0.0) + (time.perf_counter() - started)

    def total(self) -> float:
        return time.perf_counter() - self._t0


_STEP_PHASE_SPECS = (
    ("rollout", "wake_up", "wake_up"),
    ("rollout", "generate", "generate"),
    ("rollout", "sleep", "sleep"),
    ("weight_sync", "sync", "weight_sync"),
    ("reward", "score_and_attach", "reward"),
    ("stack", "train_track", "train"),
)


def install_phase_timing(trainer: Any) -> None:
    """Inject ``perf/<phase>_time_s`` at the observer boundary."""

    inner = getattr(trainer, "train_step", None)
    if not callable(inner):
        return

    @functools.wraps(inner)
    def _steady_step(*args, **kwargs):
        trainer._step_timer = PhaseTimer()
        return inner(*args, **kwargs)

    @functools.wraps(inner)
    def _first_step(*args, **kwargs):
        trainer._step_timer = PhaseTimer()
        _wrap_step_collaborators(trainer)
        trainer.train_step = _steady_step
        return inner(*args, **kwargs)

    trainer._step_timer = PhaseTimer()
    trainer.train_step = _first_step


def _timed_call(trainer: Any, fn, phase: str):
    @functools.wraps(fn)
    def _timed(*args, **kwargs):
        with trainer._step_timer.phase(phase):
            return fn(*args, **kwargs)

    return _timed


def _wrap_step_collaborators(trainer: Any) -> None:
    for handle_attr, method, phase in _STEP_PHASE_SPECS:
        handle = getattr(trainer, handle_attr, None)
        fn = getattr(handle, method, None)
        if callable(fn):
            setattr(handle, method, _timed_call(trainer, fn, phase))

    log_inner = trainer.observer.log_rollout_step

    @functools.wraps(log_inner)
    def _log_with_phases(*args, **kwargs):
        if kwargs.get("phase_times") is None and trainer._step_timer.phases:
            kwargs["phase_times"] = dict(trainer._step_timer.phases)
        return log_inner(*args, **kwargs)

    trainer.observer.log_rollout_step = _log_with_phases


__all__ = ["PhaseTimer", "install_phase_timing"]
