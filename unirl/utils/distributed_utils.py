"""Compatibility facade for :mod:`unirl.distributed.collectives`."""

from unirl.distributed.collectives import (
    distributed_masked_whiten,
    get_gloo_group,
    init_gloo_group,
    init_process_group,
)

__all__ = [
    "distributed_masked_whiten",
    "get_gloo_group",
    "init_gloo_group",
    "init_process_group",
]
