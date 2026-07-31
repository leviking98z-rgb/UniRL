"""Compatibility facade for :mod:`unirl.distributed.process`."""

from unirl.distributed.process import (
    GracefulShutdown,
    ShutdownRequested,
    descendants_of,
    run_with_timeout,
    terminate_descendants,
)

__all__ = [
    "GracefulShutdown",
    "ShutdownRequested",
    "descendants_of",
    "run_with_timeout",
    "terminate_descendants",
]
