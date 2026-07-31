"""Lazy public surface for SDE kernels, schedules, and timestep policies."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "DPM2Strategy": ("unirl.sde.kernels", "DPM2Strategy"),
    "FlowMatchSchedulePolicy": ("unirl.sde.runtime", "FlowMatchSchedulePolicy"),
    "StepStrategy": ("unirl.sde.kernels", "StepStrategy"),
    "calculate_dynamic_mu": ("unirl.sde.runtime", "calculate_dynamic_mu"),
    "ensure_sample_sigmas": ("unirl.sde.runtime", "ensure_sample_sigmas"),
    "get_sigma_schedule": ("unirl.sde.runtime", "get_sigma_schedule"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
