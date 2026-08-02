"""Lazy public surface for execution and recipe contracts."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "Capability": ("unirl.config.execution", "Capability"),
    "CapabilityGraph": ("unirl.config.execution", "CapabilityGraph"),
    "CheckpointWeightReceiver": ("unirl.config.rollout", "CheckpointWeightReceiver"),
    "ComponentCapabilities": ("unirl.config.execution", "ComponentCapabilities"),
    "EngineSelection": ("unirl.config.execution", "EngineSelection"),
    "ExecutionPlan": ("unirl.config.execution", "ExecutionPlan"),
    "IPCWeightReceiver": ("unirl.config.rollout", "IPCWeightReceiver"),
    "LoraWeightReceiver": ("unirl.config.rollout", "LoraWeightReceiver"),
    "LoopKind": ("unirl.config.execution", "LoopKind"),
    "NCCLWeightReceiver": ("unirl.config.rollout", "NCCLWeightReceiver"),
    "PlacementMode": ("unirl.config.execution", "PlacementMode"),
    "PrecisionName": ("unirl.config.validation", "PrecisionName"),
    "RolePlacement": ("unirl.config.execution", "RolePlacement"),
    "RolloutMemoryLifecycle": ("unirl.config.rollout", "RolloutMemoryLifecycle"),
    "SyncSelection": ("unirl.config.execution", "SyncSelection"),
    "TensorWeightReceiver": ("unirl.config.rollout", "TensorWeightReceiver"),
    "is_direct_sampling": ("unirl.config.validation", "is_direct_sampling"),
    "require": ("unirl.config.require", "require"),
    "validate_dynamic_dotpaths": ("unirl.config.validation", "validate_dynamic_dotpaths"),
    "validate_lora_target_modules": ("unirl.config.validation", "validate_lora_target_modules"),
    "validate_offload_contract": ("unirl.config.validation", "validate_offload_contract"),
    "validate_precision_type": ("unirl.config.validation", "validate_precision_type"),
    "validate_rollout_layout": ("unirl.config.validation", "validate_rollout_layout"),
    "validate_training_batch_geometry": (
        "unirl.config.validation",
        "validate_training_batch_geometry",
    ),
    "validate_weight_sync_contract": ("unirl.config.validation", "validate_weight_sync_contract"),
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
