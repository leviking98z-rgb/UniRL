"""Compatibility facade for :mod:`unirl.observability.metrics`."""

from unirl.observability.metrics import (
    aggregate_numeric_metrics,
    build_sync_metrics,
    compute_rollout_sample_metrics,
    flatten_dict,
    flatten_numeric_metrics,
)

__all__ = [
    "aggregate_numeric_metrics",
    "build_sync_metrics",
    "compute_rollout_sample_metrics",
    "flatten_dict",
    "flatten_numeric_metrics",
]
