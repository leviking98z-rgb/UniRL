"""Compatibility facade for :mod:`unirl.observability.profiling`."""

from unirl.observability.profiling import (
    TrainStepProfiler,
    maybe_build_train_profiler,
    maybe_profile_update,
    profile_enabled,
    profile_mode,
    profile_scope,
)

__all__ = [
    "TrainStepProfiler",
    "maybe_build_train_profiler",
    "maybe_profile_update",
    "profile_enabled",
    "profile_mode",
    "profile_scope",
]
