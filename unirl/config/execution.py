"""Typed execution planning for trainer/component composition.

The recipe remains Hydra-native, but ``_target_`` is used only to resolve the
selected component class.  Runtime decisions are made from capability
declarations owned by those classes, never from target-name suffixes or
constructor introspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional


class Capability(str, Enum):
    """Capabilities supplied or required by an execution component."""

    # Rollout ownership and generation shape.
    DIRECT_ROLLOUT = "direct_rollout"
    DEDICATED_ROLLOUT = "dedicated_rollout"
    SINGLE_TURN_GENERATION = "single_turn_generation"
    MULTI_TURN_GENERATION = "multi_turn_generation"
    QUIESCE = "quiesce"
    PARTIAL_ROLLOUT = "partial_rollout"
    MULTI_GPU_COLOCATE = "multi_gpu_colocate"

    # Engine-side weight receivers.
    TENSOR_WEIGHT_RECEIVER = "tensor_weight_receiver"
    NCCL_WEIGHT_RECEIVER = "nccl_weight_receiver"
    IPC_WEIGHT_RECEIVER = "ipc_weight_receiver"
    LORA_WEIGHT_RECEIVER = "lora_weight_receiver"
    CHECKPOINT_WEIGHT_RECEIVER = "checkpoint_weight_receiver"

    # Weight-sync placement and connection protocols.
    COLOCATED_WEIGHT_SYNC = "colocated_weight_sync"
    CROSS_SLAB_WEIGHT_SYNC = "cross_slab_weight_sync"
    NCCL_RENDEZVOUS = "nccl_rendezvous"
    ROLLOUT_TARGET_HANDOFF = "rollout_target_handoff"


@dataclass(frozen=True)
class ComponentCapabilities:
    """The capabilities a component provides and requires from its peer."""

    provides: frozenset[Capability] = frozenset()
    requires: frozenset[Capability] = frozenset()

    @classmethod
    def of(
        cls,
        *provides: Capability,
        requires: Iterable[Capability] = (),
    ) -> "ComponentCapabilities":
        return cls(frozenset(provides), frozenset(requires))

    def supports(self, capability: Capability) -> bool:
        return capability in self.provides


class ComponentKind(str, Enum):
    ROLLOUT = "rollout"
    WEIGHT_SYNC = "weight_sync"


@dataclass(frozen=True)
class ComponentNode:
    """One resolved recipe component in the capability graph."""

    name: str
    target: str
    kind: ComponentKind
    capabilities: ComponentCapabilities

    def supports(self, capability: Capability) -> bool:
        return self.capabilities.supports(capability)


@dataclass(frozen=True)
class CapabilityBinding:
    """A consumer-to-provider edge in the capability graph."""

    consumer: str
    provider: str


@dataclass(frozen=True)
class CapabilityGraph:
    """Resolved component nodes and their requirement edges."""

    nodes: tuple[ComponentNode, ...]
    bindings: tuple[CapabilityBinding, ...]

    def node(self, name: str) -> ComponentNode:
        for node in self.nodes:
            if node.name == name:
                return node
        raise KeyError(f"unknown capability node {name!r}")

    def validate(self) -> None:
        for binding in self.bindings:
            consumer = self.node(binding.consumer)
            provider = self.node(binding.provider)
            missing = consumer.capabilities.requires - provider.capabilities.provides
            if missing:
                names = ", ".join(sorted(capability.value for capability in missing))
                raise ValueError(
                    f"{consumer.name} ({consumer.target}) requires [{names}], but "
                    f"{provider.name} ({provider.target}) does not provide them."
                )


class LoopKind(str, Enum):
    """The outer trainer loop selected by a trainer class."""

    BATCH_RL = "batch_rl"
    ASYNC_BATCH_RL = "async_batch_rl"
    AGENTIC_RL = "agentic_rl"
    ASYNC_AGENTIC_RL = "async_agentic_rl"
    SUPERVISED = "supervised"

    @property
    def is_async(self) -> bool:
        return self in {LoopKind.ASYNC_BATCH_RL, LoopKind.ASYNC_AGENTIC_RL}

    @property
    def is_agentic(self) -> bool:
        return self in {LoopKind.AGENTIC_RL, LoopKind.ASYNC_AGENTIC_RL}


class PlacementMode(str, Enum):
    """High-level role topology, independent of concrete device IDs."""

    TRAIN_ONLY = "train_only"
    COLOCATED = "colocated"
    SEPARATE = "separate"
    ANCHORED = "anchored"


@dataclass(frozen=True)
class EngineSelection:
    """One rollout track and its resolved engine node."""

    track: str
    node: ComponentNode

    @property
    def is_direct(self) -> bool:
        return self.node.supports(Capability.DIRECT_ROLLOUT)

    @property
    def is_dedicated(self) -> bool:
        return self.node.supports(Capability.DEDICATED_ROLLOUT)

    def supports(self, capability: Capability) -> bool:
        return self.node.supports(capability)


@dataclass(frozen=True)
class SyncSelection:
    """One sync component and the rollout tracks it updates."""

    track: str
    node: ComponentNode
    engine_tracks: tuple[str, ...]

    def supports(self, capability: Capability) -> bool:
        return self.node.supports(capability)


@dataclass(frozen=True)
class RolePlacement:
    """Typed role fractions calculated before a ``DevicePool`` is created."""

    mode: PlacementMode
    train_fraction: float
    rollout_fraction: float
    reward_fraction: float

    def validate(self, *, num_devices: int) -> None:
        if num_devices <= 0:
            raise ValueError(f"num_devices must be positive; got {num_devices}.")
        for role, fraction in (
            ("train", self.train_fraction),
            ("rollout", self.rollout_fraction),
            ("reward", self.reward_fraction),
        ):
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError(f"{role}_fraction must be in [0, 1]; got {fraction}.")
            if fraction == 0.0:
                continue
            count = fraction * num_devices
            if abs(count - round(count)) > 1e-9:
                raise ValueError(
                    f"{role}_fraction={fraction} of num_devices={num_devices} "
                    f"requires {count} devices (not an integer)."
                )

        if self.mode is PlacementMode.TRAIN_ONLY:
            if self.train_fraction != 1.0 or self.rollout_fraction != 0.0:
                raise ValueError("train-only placement must assign the full pool to train and none to rollout.")
        elif self.mode is PlacementMode.SEPARATE:
            if self.train_fraction <= 0.0 or self.rollout_fraction <= 0.0:
                raise ValueError("separate placement requires non-empty train and rollout slabs.")
            if self.train_fraction + self.rollout_fraction + self.reward_fraction > 1.0 + 1e-9:
                raise ValueError("separate train/rollout/reward fractions exceed the device pool.")


@dataclass(frozen=True)
class ExecutionPlan:
    """Validated loop, placement, engine and weight-sync selections."""

    loop_kind: LoopKind
    placement: RolePlacement
    engines: tuple[EngineSelection, ...]
    syncs: tuple[SyncSelection, ...]
    graph: CapabilityGraph
    offload_train: bool
    offload_rollout: bool
    required_sync_capabilities: frozenset[Capability]

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        loop_kind: LoopKind = LoopKind.BATCH_RL,
        placement_override: Optional[PlacementMode] = None,
        required_sync_capabilities: Iterable[Capability] = (),
    ) -> "ExecutionPlan":
        engines = _resolve_engines(cfg)
        mode = _resolve_placement(cfg, loop_kind=loop_kind, override=placement_override, engines=engines)
        role_placement = _resolve_role_placement(cfg, mode)
        syncs, sync_nodes, bindings = _resolve_syncs(cfg, engines)
        engine_nodes = tuple(engine.node for engine in engines)
        graph = CapabilityGraph(nodes=(*engine_nodes, *sync_nodes), bindings=bindings)
        offload_train, offload_rollout = _resolve_offload(cfg)
        plan = cls(
            loop_kind=loop_kind,
            placement=role_placement,
            engines=engines,
            syncs=syncs,
            graph=graph,
            offload_train=offload_train,
            offload_rollout=offload_rollout,
            required_sync_capabilities=frozenset(required_sync_capabilities),
        )
        plan.validate(num_devices=int(_get(cfg, "num_devices", 0)))
        return plan

    def validate(self, *, num_devices: int) -> None:
        self.placement.validate(num_devices=num_devices)
        self.graph.validate()

        if self.loop_kind is LoopKind.SUPERVISED:
            if self.engines or self.syncs:
                raise ValueError("supervised execution is train-only and forbids rollout/sync components.")
            if self.placement.mode is not PlacementMode.TRAIN_ONLY:
                raise ValueError("supervised execution requires train-only placement.")
            return
        if not self.engines:
            raise ValueError(f"{self.loop_kind.value} execution requires at least one rollout engine.")

        for engine in self.engines:
            ownership = int(engine.is_direct) + int(engine.is_dedicated)
            if ownership != 1:
                raise ValueError(
                    f"{engine.node.target} must provide exactly one of direct_rollout or dedicated_rollout."
                )
            generation_shapes = sum(
                engine.supports(capability)
                for capability in (
                    Capability.SINGLE_TURN_GENERATION,
                    Capability.MULTI_TURN_GENERATION,
                )
            )
            if generation_shapes != 1:
                raise ValueError(
                    f"{engine.node.target} must provide exactly one of single_turn_generation or multi_turn_generation."
                )

        if self.loop_kind.is_agentic:
            missing = [
                engine.node.target for engine in self.engines if not engine.supports(Capability.MULTI_TURN_GENERATION)
            ]
            if missing:
                raise ValueError(f"agentic execution requires a multi-turn rollout engine; got {missing}.")

        if self.loop_kind.is_async and self.placement.mode is not PlacementMode.SEPARATE:
            raise ValueError(f"{self.loop_kind.value} execution requires separate train/rollout slabs.")

        if self.placement.mode is PlacementMode.SEPARATE:
            direct = [engine.node.target for engine in self.engines if engine.is_direct]
            if direct:
                raise ValueError(f"separate placement cannot use direct-sampling engines: {direct}.")
        if self.placement.mode is PlacementMode.ANCHORED:
            direct = [engine.node.target for engine in self.engines if engine.is_direct]
            if direct:
                raise ValueError(f"anchored placement cannot use direct-sampling engines: {direct}.")

        if any(engine.is_direct for engine in self.engines) and (self.offload_train or self.offload_rollout):
            raise ValueError(
                "direct rollout shares the live train model and is incompatible with explicit train/rollout offload."
            )

        synced_tracks = {engine_track for sync in self.syncs for engine_track in sync.engine_tracks}
        for engine in self.engines:
            if engine.is_direct and engine.track in synced_tracks:
                raise ValueError(
                    f"direct rollout track {engine.track!r} shares live train weights and forbids a sync component."
                )
            if engine.is_dedicated and engine.track not in synced_tracks:
                raise ValueError(f"dedicated rollout track {engine.track!r} requires a sync component.")

        expected_sync_scope = (
            Capability.CROSS_SLAB_WEIGHT_SYNC
            if self.placement.mode in {PlacementMode.SEPARATE, PlacementMode.ANCHORED}
            else Capability.COLOCATED_WEIGHT_SYNC
        )
        for sync in self.syncs:
            if not sync.supports(expected_sync_scope):
                raise ValueError(
                    f"{sync.node.target} is incompatible with {self.placement.mode.value} placement; "
                    f"it must provide {expected_sync_scope.value}."
                )
            if self.placement.mode in {PlacementMode.SEPARATE, PlacementMode.ANCHORED} and not (
                sync.supports(Capability.NCCL_RENDEZVOUS) or sync.supports(Capability.ROLLOUT_TARGET_HANDOFF)
            ):
                raise ValueError(
                    f"{sync.node.target} provides no cross-process connection protocol; "
                    "expected nccl_rendezvous or rollout_target_handoff."
                )

        available_sync_capabilities = {
            capability for sync in self.syncs for capability in sync.node.capabilities.provides
        }
        missing_sync_capabilities = self.required_sync_capabilities - available_sync_capabilities
        if missing_sync_capabilities:
            raise ValueError(
                f"{self.loop_kind.value} requires sync capabilities "
                f"{sorted(capability.value for capability in missing_sync_capabilities)}."
            )

    def engine(self, track: str = "rollout") -> EngineSelection:
        for engine in self.engines:
            if engine.track == track:
                return engine
        if len(self.engines) == 1 and track in {"rollout", "default"}:
            return self.engines[0]
        raise KeyError(f"execution plan has no rollout track {track!r}; tracks={self.engine_tracks}")

    def sync_for(self, track: str = "rollout") -> SyncSelection:
        matches = [sync for sync in self.syncs if track in sync.engine_tracks]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"execution plan has no sync for rollout track {track!r}.")
        raise KeyError(f"rollout track {track!r} has multiple sync components.")

    @property
    def engine_tracks(self) -> tuple[str, ...]:
        return tuple(engine.track for engine in self.engines)


def component_capabilities(config: Any, *, label: str) -> tuple[str, ComponentCapabilities]:
    """Resolve a config target and return its class-owned declaration."""

    target = str(_get(config, "_target_", "") or "")
    if not target:
        raise ValueError(f"{label} config has no _target_.")
    from hydra.utils import get_method

    component = get_method(target)
    declared = getattr(component, "CAPABILITIES", None)
    if not isinstance(declared, ComponentCapabilities):
        raise ValueError(f"{label} target {target!r} must declare CAPABILITIES: ComponentCapabilities.")
    return target, declared


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _target_config(config: Any) -> Any:
    if config is None:
        return None
    if _get(config, "_target_") is not None:
        return config
    nested = _get(config, "engine")
    return nested if nested is not None and _get(nested, "_target_") is not None else config


def _resolve_engines(cfg: Any) -> tuple[EngineSelection, ...]:
    selections: list[EngineSelection] = []
    for key, track in (
        ("rollout", "rollout"),
        ("ar_rollout", "ar"),
        ("dit_rollout", "diffusion"),
    ):
        raw = _get(cfg, key)
        if raw is None:
            continue
        config = _target_config(raw)
        target, capabilities = component_capabilities(config, label=key)
        selections.append(
            EngineSelection(
                track=track,
                node=ComponentNode(
                    name=f"rollout:{track}",
                    target=target,
                    kind=ComponentKind.ROLLOUT,
                    capabilities=capabilities,
                ),
            )
        )
    return tuple(selections)


def _resolve_syncs(
    cfg: Any,
    engines: tuple[EngineSelection, ...],
) -> tuple[tuple[SyncSelection, ...], tuple[ComponentNode, ...], tuple[CapabilityBinding, ...]]:
    raw_syncs: list[tuple[str, Any]] = []
    sync = _get(cfg, "sync")
    if sync is not None:
        if _get(sync, "_target_") is not None:
            raw_syncs.append(("all", sync))
        else:
            for track in ("ar", "diffusion", "dit"):
                child = _get(sync, track)
                if child is not None and _get(child, "_target_") is not None:
                    raw_syncs.append(("diffusion" if track == "dit" else track, child))
    for key, track in (("ar_sync", "ar"), ("dit_sync", "diffusion")):
        child = _get(cfg, key)
        if child is not None:
            raw_syncs.append((track, child))

    selections: list[SyncSelection] = []
    nodes: list[ComponentNode] = []
    bindings: list[CapabilityBinding] = []
    engine_tracks = tuple(engine.track for engine in engines)
    for index, (track, config) in enumerate(raw_syncs):
        target, capabilities = component_capabilities(config, label=f"sync.{track}")
        if track == "all":
            bound_tracks = engine_tracks
        elif len(engines) == 1:
            # A composed engine has per-track sync handlers, all routed through
            # the same outer rollout component.
            bound_tracks = (engines[0].track,)
        elif track in engine_tracks:
            bound_tracks = (track,)
        else:
            raise ValueError(f"sync track {track!r} has no matching rollout engine; available tracks={engine_tracks}.")
        node = ComponentNode(
            name=f"sync:{track}:{index}",
            target=target,
            kind=ComponentKind.WEIGHT_SYNC,
            capabilities=capabilities,
        )
        nodes.append(node)
        selections.append(SyncSelection(track=track, node=node, engine_tracks=bound_tracks))
        for engine_track in bound_tracks:
            bindings.append(
                CapabilityBinding(
                    consumer=node.name,
                    provider=f"rollout:{engine_track}",
                )
            )
    return tuple(selections), tuple(nodes), tuple(bindings)


def _resolve_placement(
    cfg: Any,
    *,
    loop_kind: LoopKind,
    override: Optional[PlacementMode],
    engines: tuple[EngineSelection, ...],
) -> PlacementMode:
    if override is not None:
        return override
    if loop_kind is LoopKind.SUPERVISED:
        return PlacementMode.TRAIN_ONLY
    layout = str(_get(cfg, "layout", "colocate"))
    if layout not in {"colocate", "colocated", "separate"}:
        raise ValueError(f"layout must be 'colocate' or 'separate'; got {layout!r}.")
    if layout == "separate":
        return PlacementMode.SEPARATE
    if _get(cfg, "rollout_anchor_device") is not None or {engine.track for engine in engines} >= {
        "ar",
        "diffusion",
    }:
        return PlacementMode.ANCHORED
    return PlacementMode.COLOCATED


def _resolve_role_placement(cfg: Any, mode: PlacementMode) -> RolePlacement:
    if mode is PlacementMode.TRAIN_ONLY:
        return RolePlacement(mode=mode, train_fraction=1.0, rollout_fraction=0.0, reward_fraction=0.0)

    reward_fraction = float(_get(cfg, "reward_fraction", 0.0))
    if mode is PlacementMode.SEPARATE:
        train_fraction = float(_get(cfg, "train_fraction", 0.5))
        rollout_fraction = 1.0 - train_fraction - reward_fraction
    else:
        train_fraction = 1.0 - reward_fraction
        rollout_fraction = train_fraction
    return RolePlacement(
        mode=mode,
        train_fraction=train_fraction,
        rollout_fraction=rollout_fraction,
        reward_fraction=reward_fraction,
    )


def _resolve_offload(cfg: Any) -> tuple[bool, bool]:
    training = _get(cfg, "training")
    execution = _get(training, "execution") if training is not None else None
    explicit_train = _get(execution, "offload_train")
    explicit_rollout = _get(execution, "offload_rollout")
    return (
        bool(_get(cfg, "enable_fsdp_offload", False) if explicit_train is None else explicit_train),
        bool(False if explicit_rollout is None else explicit_rollout),
    )


__all__ = [
    "Capability",
    "CapabilityBinding",
    "CapabilityGraph",
    "ComponentCapabilities",
    "ComponentKind",
    "ComponentNode",
    "EngineSelection",
    "ExecutionPlan",
    "LoopKind",
    "PlacementMode",
    "RolePlacement",
    "SyncSelection",
    "component_capabilities",
]
