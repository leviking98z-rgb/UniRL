"""Model-package manifests and pre-allocation composition validation.

UniRL keeps model selection Hydra-native: recipes still point directly at a
bundle factory and a pipeline constructor. A package-local ``ModelPluginSpec``
supplies the missing relationship between those independent dotpaths so the
driver can reject a mixed bundle/pipeline/algorithm graph before it creates
workers or claims accelerators. There is deliberately no central registry and
no plugin import side effect.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Optional


class ModelStageKind(str, Enum):
    """Trainable stage families exposed by model pipelines."""

    AR = "ar"
    DIFFUSION = "diffusion"


@dataclass(frozen=True)
class ModelStageSpec:
    """One pipeline stage and the conditions type consumed by replay."""

    name: str
    kind: ModelStageKind
    conditions_type: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ModelStageSpec.name must be non-empty.")
        if not self.conditions_type.startswith("unirl."):
            raise ValueError(f"ModelStageSpec.conditions_type must be a UniRL dotpath; got {self.conditions_type!r}.")

    @classmethod
    def ar(cls, conditions_type: str, *, name: str = "ar") -> "ModelStageSpec":
        return cls(name=name, kind=ModelStageKind.AR, conditions_type=conditions_type)

    @classmethod
    def diffusion(cls, conditions_type: str, *, name: str = "diffusion") -> "ModelStageSpec":
        return cls(name=name, kind=ModelStageKind.DIFFUSION, conditions_type=conditions_type)


@dataclass(frozen=True)
class ModelPluginSpec:
    """Package-local source of truth for one compatible model family.

    ``bundle_targets`` lists the Hydra factories accepted by the pipeline.  The
    targets are compared as strings on the driver so compatibility validation
    does not resolve the bundle factory (or trigger its optional dependencies).
    ``config_types`` lists config-class dotpaths accepted by the pipeline.
    ``stages`` binds each algorithm-facing ``stage_attr`` to its replay
    conditions type. ``trainable_attrs`` lists the bundle attributes a backend
    may select; the common single-transformer contract remains the default.
    """

    name: str
    bundle_targets: tuple[str, ...]
    config_types: tuple[str, ...]
    stages: tuple[ModelStageSpec, ...]
    trainable_attrs: tuple[str, ...] = ("transformer",)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ModelPluginSpec.name must be non-empty.")
        if not self.bundle_targets:
            raise ValueError(f"ModelPluginSpec({self.name!r}) must declare at least one bundle target.")
        if any(not target.startswith("unirl.models.") for target in self.bundle_targets):
            raise ValueError(
                f"ModelPluginSpec({self.name!r}) bundle targets must be model dotpaths; "
                f"got {self.bundle_targets!r}."
            )
        if not self.config_types:
            raise ValueError(f"ModelPluginSpec({self.name!r}) must declare at least one config type.")
        if any(not target.startswith("unirl.models.") for target in self.config_types):
            raise ValueError(
                f"ModelPluginSpec({self.name!r}) config types must be model dotpaths; got {self.config_types!r}."
            )
        stage_names = [stage.name for stage in self.stages]
        if len(stage_names) != len(set(stage_names)):
            raise ValueError(f"ModelPluginSpec({self.name!r}) has duplicate stage names: {stage_names!r}.")
        if not self.stages:
            raise ValueError(f"ModelPluginSpec({self.name!r}) must declare at least one trainable stage.")
        if not self.trainable_attrs or any(not name.strip() for name in self.trainable_attrs):
            raise ValueError(f"ModelPluginSpec({self.name!r}) trainable_attrs must be non-empty names.")

    def stage(self, name: str) -> ModelStageSpec:
        for stage in self.stages:
            if stage.name == name:
                return stage
        available = ", ".join(stage.name for stage in self.stages)
        raise ValueError(f"model plugin {self.name!r} has no stage {name!r}; available: [{available}].")


@dataclass(frozen=True)
class ModelPluginSelection:
    """One resolved bundle/pipeline pair from a recipe."""

    track: str
    plugin: ModelPluginSpec
    bundle_target: str
    pipeline_target: str
    config_target: str


@dataclass(frozen=True)
class ModelPluginPlan:
    """Validated model plugins selected by one trainer recipe."""

    selections: tuple[ModelPluginSelection, ...]

    @classmethod
    def from_config(cls, cfg: Any) -> "ModelPluginPlan":
        selections = tuple(
            _resolve_selection(track, component_cfg) for track, component_cfg in _model_component_configs(cfg)
        )
        if not selections:
            raise ValueError("training recipe must declare a bundle/pipeline model plugin pair.")
        plan = cls(selections=selections)
        plan._validate_rollout_stages(cfg)
        return plan

    def selection(self, track: str = "default") -> ModelPluginSelection:
        for selection in self.selections:
            if selection.track == track:
                return selection
        if len(self.selections) == 1 and track in {"default", "rollout"}:
            return self.selections[0]
        available = tuple(selection.track for selection in self.selections)
        raise KeyError(f"model plugin plan has no track {track!r}; tracks={available}.")

    @property
    def stage_names(self) -> frozenset[str]:
        return frozenset(stage.name for selection in self.selections for stage in selection.plugin.stages)

    def _validate_rollout_stages(self, cfg: Any) -> None:
        rollout = _get(cfg, "rollout")
        if rollout is None:
            return
        stage_attrs = _get(rollout, "stage_attrs")
        if stage_attrs is None:
            return
        unknown = sorted(set(str(name) for name in stage_attrs) - self.stage_names)
        if unknown:
            raise ValueError(
                f"rollout.stage_attrs contains stages not exposed by the selected model plugins: {unknown}; "
                f"available={sorted(self.stage_names)}."
            )


def _model_component_configs(cfg: Any) -> Iterable[tuple[str, Any]]:
    if _get(cfg, "bundle") is not None or _get(cfg, "pipeline") is not None:
        if _get(cfg, "bundle") is None or _get(cfg, "pipeline") is None:
            raise ValueError("root model configuration must declare bundle and pipeline together.")
        yield "default", cfg
        return

    for track in ("ar", "diffusion"):
        component_cfg = _get(cfg, track)
        if component_cfg is None:
            continue
        bundle = _get(component_cfg, "bundle")
        pipeline = _get(component_cfg, "pipeline")
        if bundle is None and pipeline is None:
            continue
        if bundle is None or pipeline is None:
            raise ValueError(f"{track} model configuration must declare bundle and pipeline together.")
        yield track, component_cfg


def _resolve_selection(track: str, component_cfg: Any) -> ModelPluginSelection:
    bundle_cfg = _get(component_cfg, "bundle")
    pipeline_cfg = _get(component_cfg, "pipeline")
    bundle_target = _target(bundle_cfg, label=f"{track}.bundle")
    pipeline_target, pipeline_cls = _resolve_target_owner(pipeline_cfg, label=f"{track}.pipeline")

    plugin = getattr(pipeline_cls, "MODEL_PLUGIN", None)
    if not isinstance(plugin, ModelPluginSpec):
        raise ValueError(f"{track}.pipeline target {pipeline_target!r} must declare MODEL_PLUGIN: ModelPluginSpec.")
    if bundle_target not in plugin.bundle_targets:
        raise ValueError(
            f"{track} model family mismatch: bundle {bundle_target!r} is incompatible with "
            f"pipeline plugin {plugin.name!r}; expected one of {plugin.bundle_targets!r}."
        )

    config = _get(bundle_cfg, "config")
    config_target = str(_get(config, "_target_", "") or "")
    if not config_target:
        raise ValueError(f"{track}.bundle.config must target one of {plugin.config_types!r}.")
    if config_target not in plugin.config_types:
        raise ValueError(
            f"{track} model config mismatch: {config_target!r} is incompatible with plugin "
            f"{plugin.name!r}; expected one of {plugin.config_types!r}."
        )

    backend = _get(component_cfg, "backend")
    if backend is not None:
        trainable_attr = str(_get(backend, "trainable_attr", "transformer"))
        if trainable_attr not in plugin.trainable_attrs:
            raise ValueError(
                f"{track}.backend.trainable_attr={trainable_attr!r} is not exposed by plugin "
                f"{plugin.name!r}; expected one of {plugin.trainable_attrs!r}."
            )

    for algorithm_name, algorithm in _algorithm_configs(component_cfg):
        stage_name = str(_get(algorithm, "stage_attr", "") or "")
        if not stage_name:
            continue
        stage = plugin.stage(stage_name)
        conditions_cfg = _get(algorithm, "conditions_cls")
        conditions_type = str(_get(conditions_cfg, "path", "") or "")
        if conditions_type and conditions_type != stage.conditions_type:
            raise ValueError(
                f"{track}.{algorithm_name}.conditions_cls={conditions_type!r} does not match "
                f"plugin {plugin.name!r} stage {stage_name!r} ({stage.conditions_type!r})."
            )

    return ModelPluginSelection(
        track=track,
        plugin=plugin,
        bundle_target=bundle_target,
        pipeline_target=pipeline_target,
        config_target=config_target,
    )


def _algorithm_configs(component_cfg: Any) -> Iterable[tuple[str, Any]]:
    algorithm = _get(component_cfg, "algorithm")
    if algorithm is None:
        return
    if _get(algorithm, "_target_") is not None:
        yield "algorithm", algorithm
        return
    for name, child in _items(algorithm):
        if _get(child, "_target_") is not None:
            yield f"algorithm.{name}", child


def _resolve_target_owner(config: Any, *, label: str) -> tuple[str, type]:
    target = _target(config, label=label)
    from hydra.utils import get_method

    component = get_method(target)
    owner = component if isinstance(component, type) else getattr(component, "__self__", None)
    if not isinstance(owner, type):
        raise TypeError(f"{label} target {target!r} must resolve to a class or classmethod; got {component!r}.")
    return target, owner


def _target(config: Any, *, label: str) -> str:
    target = str(_get(config, "_target_", "") or "")
    if not target:
        raise ValueError(f"{label} config has no _target_.")
    return target


def _items(config: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(config, Mapping):
        return config.items()
    items = getattr(config, "items", None)
    return items() if callable(items) else ()


def _get(config: Any, key: str, default: Optional[Any] = None) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


__all__ = [
    "ModelPluginPlan",
    "ModelPluginSelection",
    "ModelPluginSpec",
    "ModelStageKind",
    "ModelStageSpec",
]
