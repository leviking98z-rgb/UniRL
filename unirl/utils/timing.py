"""Cumulative phase-time tracker for training-loop driver scripts.

Replaces the ``rollout_start = time.perf_counter() ... rollout_s =
time.perf_counter() - rollout_start`` book-keeping that scatters
through training drivers with one ``with timings.measure("rollout"):``
guard. ``as_perf_dict`` emits the legacy ``perf/*`` key shape consumed
by :meth:`WandbLogger.log_perf`.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Mapping, Sequence

__all__ = ["PhaseTimings", "critical_path_phase_times"]


def critical_path_phase_times(
    worker_phase_times: Mapping[str, Sequence[float]], *, prefix: str = ""
) -> Dict[str, float]:
    """Reduce per-worker phase samples to the slowest worker for each phase.

    Distributed train stacks return one duration per DP worker. Taking the
    maximum preserves the critical-path signal without adding a timing
    collective to the training step. Empty phase samples are ignored.
    """
    return {
        f"{prefix}{name}": max(float(value) for value in values)
        for name, values in worker_phase_times.items()
        if values
    }


class PhaseTimings:
    """Cumulative per-phase elapsed-time tracker.

    Use as ``with timings.measure("rollout"): ...`` — elapsed deltas
    accumulate under ``name`` (re-entry sums, so the same name can
    span multiple guarded blocks within one step).

    ``as_perf_dict(samples=...)`` returns a flat dict ready for
    ``WandbLogger.log_perf``:
    ``{<name>_phase_s, step_time_s, samples_per_rollout, samples_per_s}``.
    """

    def __init__(self) -> None:
        self._timings: Dict[str, float] = {}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._timings[name] = self._timings.get(name, 0.0) + (time.perf_counter() - start)

    def get(self, name: str, default: float = 0.0) -> float:
        return self._timings.get(name, default)

    def total(self) -> float:
        return sum(self._timings.values())

    def as_perf_dict(self, *, samples: int = 0) -> Dict[str, float]:
        out: Dict[str, float] = {f"{name}_phase_s": float(v) for name, v in self._timings.items()}
        total = self.total()
        out["step_time_s"] = float(total)
        out["samples_per_rollout"] = float(samples)
        out["samples_per_s"] = float(samples) / total if total > 0 else 0.0
        return out
