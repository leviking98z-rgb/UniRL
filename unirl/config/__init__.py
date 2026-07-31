"""Config surface.

Public entry points:
  - ``ExecutionPlan`` and capability/placement enums (``execution``): the
    driver-side typed component graph validated before actor creation.
  - ``PrecisionName`` / ``validate_precision_type`` + the ``validate_*``
    cross-component checkers (``validation``): shared helpers used by config
    dataclasses' ``__post_init__`` and driver-side validation.
  - ``require`` (``require``): one-line precondition helper for ``__post_init__``
    and cross-component validators.
"""

from __future__ import annotations

from unirl.config.execution import (
    Capability,
    CapabilityGraph,
    ComponentCapabilities,
    EngineSelection,
    ExecutionPlan,
    LoopKind,
    PlacementMode,
    RolePlacement,
    SyncSelection,
)
from unirl.config.require import require
from unirl.config.validation import (
    PrecisionName,
    is_direct_sampling,
    validate_dynamic_dotpaths,
    validate_lora_target_modules,
    validate_offload_contract,
    validate_precision_type,
    validate_rollout_layout,
    validate_training_batch_geometry,
    validate_weight_sync_contract,
)

__all__ = [
    "Capability",
    "CapabilityGraph",
    "ComponentCapabilities",
    "EngineSelection",
    "ExecutionPlan",
    "LoopKind",
    "PlacementMode",
    "PrecisionName",
    "RolePlacement",
    "SyncSelection",
    "is_direct_sampling",
    "require",
    "validate_dynamic_dotpaths",
    "validate_lora_target_modules",
    "validate_offload_contract",
    "validate_precision_type",
    "validate_rollout_layout",
    "validate_training_batch_geometry",
    "validate_weight_sync_contract",
]
