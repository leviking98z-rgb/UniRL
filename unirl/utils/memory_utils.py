"""Compatibility facade for :mod:`unirl.distributed.memory`."""

from unirl.distributed.memory import (
    MemorySnapshotSampler,
    aggressive_empty_cache,
    get_memory_info,
    get_process_snapshot_sampler,
    init_process_snapshot_sampler,
    log_memory_usage,
    summarize_snapshot,
)

__all__ = [
    "MemorySnapshotSampler",
    "aggressive_empty_cache",
    "get_memory_info",
    "get_process_snapshot_sampler",
    "init_process_snapshot_sampler",
    "log_memory_usage",
    "summarize_snapshot",
]
