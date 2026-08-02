"""Lazy compatibility facade for utility APIs moved to explicit owners.

Framework code must import the owning package directly. These names remain for
out-of-tree callers during the migration and load no torch/provider dependency
until an attribute is requested.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "AllSDEScheduler": ("unirl.sde.scheduler", "AllSDEScheduler"),
    "SCHEDULER_REGISTRY": ("unirl.sde.scheduler", "SCHEDULER_REGISTRY"),
    "TimestepScheduler": ("unirl.sde.scheduler", "TimestepScheduler"),
    "UniRLWandBLogger": ("unirl.observability.wandb", "UniRLWandBLogger"),
    "WindowConfig": ("unirl.sde.scheduler", "WindowConfig"),
    "WindowScheduler": ("unirl.sde.scheduler", "WindowScheduler"),
    "aggregate_metrics": ("unirl.observability.wandb", "aggregate_metrics"),
    "clear_memory": ("unirl.runtime", "clear_memory"),
    "configure_logger": ("unirl.observability.logging", "configure_logger"),
    "create_indices_scheduler": ("unirl.sde.scheduler", "create_indices_scheduler"),
    "flatten_dict": ("unirl.observability.metrics", "flatten_dict"),
    "init_logger": ("unirl.observability.wandb", "init_logger"),
    "load_function": ("unirl.config.imports", "load_function"),
    "normalize_timestep_fraction": ("unirl.sde.scheduler", "normalize_timestep_fraction"),
    "set_seed": ("unirl.runtime", "set_seed"),
    "switch_adapter": ("unirl.models.adapters", "switch_adapter"),
    "tensor_frame_to_pil": ("unirl.types.media_conversion", "tensor_frame_to_pil"),
    "tensor_to_pil": ("unirl.types.media_conversion", "tensor_to_pil"),
}

__all__ = sorted(_LAZY_ATTRS)


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
