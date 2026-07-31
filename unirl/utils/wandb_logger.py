"""Compatibility facade for :mod:`unirl.observability.wandb`."""

from unirl.observability.wandb import (
    UniRLWandBLogger,
    aggregate_metrics,
    aggregate_stage_results,
    init_logger,
)

__all__ = [
    "UniRLWandBLogger",
    "aggregate_metrics",
    "aggregate_stage_results",
    "init_logger",
]
