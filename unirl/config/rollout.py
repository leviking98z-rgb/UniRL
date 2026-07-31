"""Structural rollout capability contracts shared by providers and consumers.

The contracts live in the dependency-light config/kernel layer so rollout
engines can provide them while weight-sync implementations consume them
without introducing a forbidden ``weight_sync -> rollout`` package edge.
"""

from __future__ import annotations

import inspect
from typing import Any, Protocol, runtime_checkable

from unirl.config.execution import Capability, ComponentCapabilities


@runtime_checkable
class RolloutMemoryLifecycle(Protocol):
    """Offload/onload and health surface used by trainer placement policy."""

    def sleep(self) -> None: ...

    def wake_up(self) -> None: ...

    def onload_weights(self, *, track_prefix: str = "") -> None: ...

    @property
    def is_offloaded(self) -> bool: ...

    def health_check(self) -> bool: ...

    def get_memory_info(self) -> dict[str, float]: ...


@runtime_checkable
class TensorWeightReceiver(Protocol):
    """Receive a serialized one-bag tensor payload."""

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: list[str],
        target_modules: list[str] | None = None,
        load_format: str | None = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None: ...


@runtime_checkable
class NCCLWeightReceiver(Protocol):
    """Join, receive on, and destroy a trainer/rollout NCCL group."""

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        track_prefix: str = "",
    ) -> None: ...

    def update_weights_from_distributed(
        self,
        *,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
        target_modules: list[str] | None = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None: ...

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        track_prefix: str = "",
    ) -> None: ...


@runtime_checkable
class IPCWeightReceiver(Protocol):
    """Receive bucketed weights over the CUDA-IPC transport."""

    def update_weights_from_ipc(
        self,
        *,
        peft_config: dict | None = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        track_prefix: str = "",
    ) -> None: ...


@runtime_checkable
class LoraWeightReceiver(Protocol):
    """Receive an in-memory LoRA adapter."""

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: dict[str, Any],
        *,
        peft_config: dict | None = None,
    ) -> None: ...


@runtime_checkable
class CheckpointWeightReceiver(Protocol):
    """Load weights from a checkpoint path published by the trainer."""

    def update_weights_from_path(self, checkpoint_path: str, *, track_prefix: str = "") -> None: ...


_CAPABILITY_METHODS: dict[Capability, frozenset[str]] = {
    Capability.SINGLE_TURN_GENERATION: frozenset({"generate"}),
    Capability.MULTI_TURN_GENERATION: frozenset({"generate"}),
    Capability.PARTIAL_ROLLOUT: frozenset({"submit", "poll", "finalize_if_drained", "abort"}),
    Capability.MEMORY_LIFECYCLE: frozenset(
        {
            "sleep",
            "wake_up",
            "onload_weights",
            "is_offloaded",
            "health_check",
            "get_memory_info",
        }
    ),
    Capability.TENSOR_WEIGHT_RECEIVER: frozenset({"update_weights_from_tensor"}),
    Capability.NCCL_WEIGHT_RECEIVER: frozenset(
        {
            "init_weights_update_group",
            "update_weights_from_distributed",
            "destroy_weights_update_group",
        }
    ),
    Capability.IPC_WEIGHT_RECEIVER: frozenset({"update_weights_from_ipc"}),
    Capability.LORA_WEIGHT_RECEIVER: frozenset({"set_lora_from_tensors"}),
    Capability.CHECKPOINT_WEIGHT_RECEIVER: frozenset({"update_weights_from_path"}),
}


def validate_rollout_capability_surface(
    component: type,
    capabilities: ComponentCapabilities,
    *,
    label: str,
) -> None:
    """Reject a capability declaration whose provider lacks its method surface."""

    required = {method for capability in capabilities.provides for method in _CAPABILITY_METHODS.get(capability, ())}
    missing = sorted(method for method in required if inspect.getattr_static(component, method, None) is None)
    if missing:
        provided = sorted(capability.value for capability in capabilities.provides)
        raise ValueError(
            f"{label} target {component.__module__}.{component.__name__} declares capabilities "
            f"{provided} but lacks methods {missing}."
        )


__all__ = [
    "CheckpointWeightReceiver",
    "IPCWeightReceiver",
    "LoraWeightReceiver",
    "NCCLWeightReceiver",
    "RolloutMemoryLifecycle",
    "TensorWeightReceiver",
    "validate_rollout_capability_surface",
]
