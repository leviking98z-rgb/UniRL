#!/usr/bin/env python3
"""Static CPU contract checks for UniRL's principal extension surfaces.

The project deliberately keeps heavyweight runtime dependencies out of lint CI.
This guard therefore validates contracts from Python syntax rather than importing
torch/Ray/engine packages:

* every concrete rollout engine owns ``generate`` and ``shutdown`` instead of
  accidentally inheriting the base abstract methods;
* every model ``Pipeline`` implementation owns or inherits a concrete
  ``generate`` method;
* the FSDP and VeOmni backend leaves implement the shared backend hook surface;
* ``Sample``/``Part`` retain the request/fork/fill and batch-algebra seam used by
  rollout, reward, and train;
* model bundles and pipelines declare package-local plugin ownership so recipes
  cannot silently compose different model families;
* ordinary image/video diffusion stages reuse the shared denoising runner while
  transition-kernel exceptions stay explicit and reviewable;
* advantage estimators own reward normalization/value-target policy instead of
  growing methods on the ``Part`` wire type;
* trainer variants select explicit loop programs instead of copying ``train``;
* every ``train_*.py`` entrypoint exposes ``main`` and selects exactly one trainer
  module.

These are conformance guards, not implementation tests.  Numerical parity remains
the responsibility of model/engine GPU validation.  Run by pre-commit and kept
stdlib-only so the contract runs on every PR.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "__pycache__", "vendor"}


@dataclass(frozen=True)
class ClassInfo:
    path: Path
    module: str
    name: str
    bases: tuple[str, ...]
    methods: frozenset[str]
    fields: frozenset[str]
    capabilities: frozenset[str] = frozenset()

    @property
    def qualified_name(self) -> str:
        return f"{self.module}.{self.name}"


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = rel.parts[:-1] if rel.name == "__init__.py" else (*rel.parts[:-1], rel.stem)
    return ".".join(parts)


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return ast.unparse(node).rsplit(".", 1)[-1]


def _capability_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "Capability":
        return node.attr
    return None


def _declared_capabilities(node: ast.ClassDef) -> frozenset[str]:
    """Read literal ``ComponentCapabilities.of(Capability.X, ...)`` provides."""

    value: ast.expr | None = None
    for child in node.body:
        if (
            isinstance(child, ast.AnnAssign)
            and isinstance(child.target, ast.Name)
            and child.target.id == "CAPABILITIES"
        ):
            value = child.value
            break
        if isinstance(child, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CAPABILITIES" for target in child.targets
        ):
            value = child.value
            break
    if not isinstance(value, ast.Call):
        return frozenset()

    provided: list[ast.expr] = list(value.args)
    for keyword in value.keywords:
        if keyword.arg != "provides":
            continue
        if isinstance(keyword.value, (ast.List, ast.Set, ast.Tuple)):
            provided.extend(keyword.value.elts)
        else:
            provided.append(keyword.value)
    return frozenset(name for item in provided if (name := _capability_name(item)) is not None)


def _classes(path: Path) -> list[ClassInfo]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module = _module_name(path)
    out: list[ClassInfo] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = frozenset(
            child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        fields = frozenset(
            child.target.id
            for child in node.body
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
        )
        out.append(
            ClassInfo(
                path=path,
                module=module,
                name=node.name,
                bases=tuple(_base_name(base) for base in node.bases),
                methods=methods,
                fields=fields,
                capabilities=_declared_capabilities(node),
            )
        )
    return out


def _iter_python(base: Path) -> Iterable[Path]:
    for path in sorted(base.rglob("*.py")):
        if not SKIP_PARTS.intersection(path.relative_to(ROOT).parts):
            yield path


def _class_index() -> tuple[dict[str, ClassInfo], dict[str, list[ClassInfo]]]:
    qualified: dict[str, ClassInfo] = {}
    simple: dict[str, list[ClassInfo]] = {}
    for path in _iter_python(ROOT / "unirl"):
        for info in _classes(path):
            qualified[info.qualified_name] = info
            simple.setdefault(info.name, []).append(info)
    return qualified, simple


def _resolve_base(info: ClassInfo, name: str, simple: dict[str, list[ClassInfo]]) -> ClassInfo | None:
    local = [candidate for candidate in simple.get(name, ()) if candidate.module == info.module]
    if len(local) == 1:
        return local[0]
    candidates = simple.get(name, ())
    return candidates[0] if len(candidates) == 1 else None


def _descends_from(
    info: ClassInfo,
    ancestor: str,
    simple: dict[str, list[ClassInfo]],
    seen: frozenset[str] = frozenset(),
) -> bool:
    if info.qualified_name in seen:
        return False
    if ancestor in info.bases:
        return True
    next_seen = seen | {info.qualified_name}
    return any(
        parent is not None and _descends_from(parent, ancestor, simple, next_seen)
        for base in info.bases
        for parent in (_resolve_base(info, base, simple),)
    )


def _effective_members(
    info: ClassInfo,
    attr: str,
    simple: dict[str, list[ClassInfo]],
    seen: frozenset[str] = frozenset(),
) -> frozenset[str]:
    if info.qualified_name in seen:
        return frozenset()
    own = getattr(info, attr)
    next_seen = seen | {info.qualified_name}
    inherited = (
        _effective_members(parent, attr, simple, next_seen)
        for base in info.bases
        for parent in (_resolve_base(info, base, simple),)
        if parent is not None
    )
    return own.union(*(members for members in inherited))


def _require_members(
    errors: list[str],
    info: ClassInfo,
    *,
    kind: str,
    methods: Iterable[str] = (),
    fields: Iterable[str] = (),
    simple: dict[str, list[ClassInfo]],
) -> None:
    effective_methods = _effective_members(info, "methods", simple)
    effective_fields = _effective_members(info, "fields", simple)
    missing_methods = sorted(set(methods) - effective_methods)
    missing_fields = sorted(set(fields) - effective_fields)
    if missing_methods:
        errors.append(f"{info.path.relative_to(ROOT)}: {kind} {info.name} lacks methods {missing_methods}")
    if missing_fields:
        errors.append(f"{info.path.relative_to(ROOT)}: {kind} {info.name} lacks fields {missing_fields}")


ROLLOUT_CAPABILITY_METHODS = {
    "SINGLE_TURN_GENERATION": frozenset({"generate"}),
    "MULTI_TURN_GENERATION": frozenset({"generate"}),
    "PARTIAL_ROLLOUT": frozenset({"submit", "poll", "finalize_if_drained", "abort"}),
    "MEMORY_LIFECYCLE": frozenset(
        {
            "sleep",
            "wake_up",
            "onload_weights",
            "is_offloaded",
            "health_check",
            "get_memory_info",
        }
    ),
    "TENSOR_WEIGHT_RECEIVER": frozenset({"update_weights_from_tensor"}),
    "NCCL_WEIGHT_RECEIVER": frozenset(
        {
            "init_weights_update_group",
            "update_weights_from_distributed",
            "destroy_weights_update_group",
        }
    ),
    "IPC_WEIGHT_RECEIVER": frozenset({"update_weights_from_ipc"}),
    "LORA_WEIGHT_RECEIVER": frozenset({"set_lora_from_tensors"}),
    "CHECKPOINT_WEIGHT_RECEIVER": frozenset({"update_weights_from_path"}),
}


def check_rollout_engines(errors: list[str], simple: dict[str, list[ClassInfo]]) -> int:
    engines = [
        info
        for infos in simple.values()
        for info in infos
        if info.name not in {"BaseRolloutEngine", "BaseSingleTurnRolloutEngine"}
        and _descends_from(info, "BaseRolloutEngine", simple)
        and "/rollout/engine/" in info.path.as_posix()
    ]
    for info in engines:
        # These two methods are abstract on BaseRolloutEngine. Requiring them on
        # the concrete class body (not merely effective inheritance) catches an
        # accidental "concrete" class that is unusable when Hydra constructs it.
        missing = sorted({"generate", "shutdown"} - info.methods)
        if missing:
            errors.append(
                f"{info.path.relative_to(ROOT)}: rollout engine {info.name} must implement {missing} directly"
            )
        if "CAPABILITIES" not in info.fields:
            errors.append(
                f"{info.path.relative_to(ROOT)}: rollout engine {info.name} must declare CAPABILITIES directly"
            )
            continue
        if not info.capabilities:
            errors.append(
                f"{info.path.relative_to(ROOT)}: rollout engine {info.name} must use literal Capability members"
            )
            continue
        if "MEMORY_LIFECYCLE" not in info.capabilities:
            errors.append(f"{info.path.relative_to(ROOT)}: rollout engine {info.name} must declare MEMORY_LIFECYCLE")
        effective_methods = _effective_members(info, "methods", simple)
        for capability in sorted(info.capabilities):
            required = ROLLOUT_CAPABILITY_METHODS.get(capability, frozenset())
            missing = sorted(required - effective_methods)
            if missing:
                errors.append(
                    f"{info.path.relative_to(ROOT)}: rollout engine {info.name} declares {capability} "
                    f"but lacks methods {missing}"
                )
    return len(engines)


def check_weight_syncs(errors: list[str], simple: dict[str, list[ClassInfo]]) -> int:
    syncs = [
        info
        for infos in simple.values()
        for info in infos
        if info.name not in {"FullWeightSync", "LoraWeightSyncBase"}
        and (_descends_from(info, "FullWeightSync", simple) or _descends_from(info, "LoraWeightSyncBase", simple))
        and "/distributed/weight_sync/" in info.path.as_posix()
    ]
    for info in syncs:
        if "CAPABILITIES" not in info.fields:
            errors.append(f"{info.path.relative_to(ROOT)}: weight sync {info.name} must declare CAPABILITIES directly")
    return len(syncs)


def check_model_pipelines(errors: list[str], simple: dict[str, list[ClassInfo]]) -> int:
    pipelines = [
        info
        for infos in simple.values()
        for info in infos
        if info.name not in {"Pipeline", "LatentShapeProvider"}
        and _descends_from(info, "Pipeline", simple)
        and "/models/" in info.path.as_posix()
        and info.path.name == "pipeline.py"
    ]
    for info in pipelines:
        effective_fields = _effective_members(info, "fields", simple)
        manifest_fields = {"MODEL_PLUGIN", "COMPOSITE_MODEL_PIPELINE"} & effective_fields
        if len(manifest_fields) != 1:
            errors.append(
                f"{info.path.relative_to(ROOT)}: model pipeline {info.name} must declare exactly one of "
                "MODEL_PLUGIN or COMPOSITE_MODEL_PIPELINE"
            )
        _require_members(
            errors,
            info,
            kind="model pipeline",
            methods=("generate",),
            simple=simple,
        )
    return len(pipelines)


def check_model_bundles(errors: list[str], simple: dict[str, list[ClassInfo]]) -> int:
    bundles = [
        info
        for infos in simple.values()
        for info in infos
        if info.name != "Bundle"
        and _descends_from(info, "Bundle", simple)
        and "/models/" in info.path.as_posix()
        and info.path.name == "bundle.py"
    ]
    for info in bundles:
        effective_fields = _effective_members(info, "fields", simple)
        manifest_fields = {"MODEL_FAMILY", "COMPOSITE_MODEL_BUNDLE"} & effective_fields
        if len(manifest_fields) != 1:
            errors.append(
                f"{info.path.relative_to(ROOT)}: model bundle {info.name} must declare exactly one of "
                "MODEL_FAMILY or COMPOSITE_MODEL_BUNDLE"
            )
    return len(bundles)


def check_model_plugin_contracts(errors: list[str], simple: dict[str, list[ClassInfo]]) -> int:
    expected = {
        "ModelStageSpec": {
            "methods": ("ar", "diffusion"),
            "fields": ("name", "kind", "conditions_type"),
        },
        "ModelPluginSpec": {
            "methods": ("stage",),
            "fields": ("name", "bundle_targets", "config_types", "stages", "trainable_attrs"),
        },
        "ModelPluginSelection": {
            "fields": ("track", "plugin", "bundle_target", "pipeline_target", "config_target"),
        },
        "ModelPluginPlan": {
            "methods": ("from_config", "selection", "stage_names"),
            "fields": ("selections",),
        },
    }
    for name, contract in expected.items():
        matches = [info for info in simple.get(name, ()) if info.module == "unirl.models.types.plugin"]
        if len(matches) != 1:
            errors.append(f"unirl/models/types/plugin.py: expected one {name} class, found {len(matches)}")
            continue
        _require_members(
            errors,
            matches[0],
            kind="model plugin contract",
            methods=contract.get("methods", ()),
            fields=contract.get("fields", ()),
            simple=simple,
        )
    return len(expected)


def check_diffusion_runners(errors: list[str], simple: dict[str, list[ClassInfo]]) -> int:
    expected = {
        "DiffusionLatentSpec": {
            "fields": ("device", "batch_size", "shape"),
        },
        "DiffusionRunner": {
            "methods": (
                "_latent_spec",
                "_prepare_initial_latents",
                "_sampling_state",
                "_step_kwargs",
                "_guidance_scale",
                "_make_segment",
                "_validate_replay_segment",
                "_replay_batched",
                "diffuse",
                "replay",
                "predict_noise_at_step",
                "trainable_module",
            ),
        },
        "VideoDiffusionRunner": {
            "methods": ("_validate_replay_segment",),
        },
    }
    for name, contract in expected.items():
        module = (
            "unirl.models.diffusion.contracts" if name == "DiffusionLatentSpec" else "unirl.models.diffusion.runner"
        )
        matches = [info for info in simple.get(name, ()) if info.module == module]
        if len(matches) != 1:
            path = module.replace(".", "/") + ".py"
            errors.append(f"{path}: expected one {name} class, found {len(matches)}")
            continue
        _require_members(
            errors,
            matches[0],
            kind="diffusion runner contract",
            methods=contract.get("methods", ()),
            fields=contract.get("fields", ()),
            simple=simple,
        )

    exceptions = {
        "unirl.models.bagel.diffusion.BagelDiffusionStage",
        "unirl.models.ltx2.diffusion.LTX2DiffusionStage",
    }
    stages = [
        info
        for infos in simple.values()
        for info in infos
        if info.name.endswith("DiffusionStage")
        and info.name != "DiffusionStage"
        and "/models/" in info.path.as_posix()
        and info.path.name == "diffusion.py"
    ]
    discovered_exceptions = {
        info.qualified_name for info in stages if not _descends_from(info, "DiffusionRunner", simple)
    }
    unexpected = sorted(discovered_exceptions - exceptions)
    missing = sorted(exceptions - discovered_exceptions)
    if unexpected:
        errors.append(
            "ordinary diffusion stages must inherit DiffusionRunner; "
            f"unreviewed transition-kernel exceptions: {unexpected}"
        )
    if missing:
        errors.append(f"diffusion runner exception list is stale; remove migrated/deleted entries: {missing}")
    return len(stages)


def check_train_backends(errors: list[str], simple: dict[str, list[ClassInfo]]) -> int:
    required = (
        "weight_sync_dtype",
        "_clip_grad_norm",
        "_gather_optimizer_state",
        "_load_optimizer_state",
        "_onload_model",
        "_offload_model",
    )
    backends = [
        info
        for infos in simple.values()
        for info in infos
        if info.name != "BaseFSDP2Backend"
        and _descends_from(info, "BaseFSDP2Backend", simple)
        and "/train/backend/" in info.path.as_posix()
    ]
    for info in backends:
        _require_members(
            errors,
            info,
            kind="train backend",
            methods=required,
            simple=simple,
        )
    return len(backends)


def check_sample_contract(errors: list[str], simple: dict[str, list[ClassInfo]]) -> int:
    expected = {
        "Part": {
            "methods": ("input", "fork", "fill", "concat"),
            "fields": ("sample_ids", "segment", "primitives", "conditions", "sampling_params"),
        },
        "Sample": {
            "methods": ("request", "fork", "split", "concat", "conditioning", "replace_frontier"),
            "fields": ("parts",),
        },
    }
    for name, contract in expected.items():
        matches = [info for info in simple.get(name, ()) if info.module == "unirl.types.sample"]
        if len(matches) != 1:
            errors.append(f"unirl/types/sample.py: expected one {name} class, found {len(matches)}")
            continue
        _require_members(
            errors,
            matches[0],
            kind="wire contract",
            methods=contract["methods"],
            fields=contract["fields"],
            simple=simple,
        )
    return len(expected)


def check_advantage_estimators(errors: list[str], simple: dict[str, list[ClassInfo]]) -> int:
    expected = {
        "AdvantageBatch": {
            "fields": ("rewards", "group_ids", "component_rewards", "values", "mask"),
        },
        "AdvantageEstimate": {
            "fields": ("advantages", "returns"),
        },
        "AdvantageEstimator": {
            "methods": ("estimate", "requires_group_ids"),
        },
        "GroupedAdvantageEstimator": {
            "methods": ("estimate", "requires_group_ids"),
        },
        "GeneralizedAdvantageEstimator": {
            "methods": ("estimate", "requires_group_ids"),
        },
    }
    for name, contract in expected.items():
        matches = [info for info in simple.get(name, ()) if info.module == "unirl.algorithms.advantage"]
        if len(matches) != 1:
            errors.append(f"unirl/algorithms/advantage.py: expected one {name} class, found {len(matches)}")
            continue
        _require_members(
            errors,
            matches[0],
            kind="advantage contract",
            methods=contract.get("methods", ()),
            fields=contract.get("fields", ()),
            simple=simple,
        )

    parts = [info for info in simple.get("Part", ()) if info.module == "unirl.types.sample"]
    if len(parts) == 1 and "compute_advantages" in parts[0].methods:
        errors.append(
            "unirl/types/sample.py: Part must remain a wire type; advantage policy belongs in unirl.algorithms"
        )
    return len(expected)


def check_loop_programs(errors: list[str], simple: dict[str, list[ClassInfo]]) -> int:
    expected = {
        "TrainerLifecycle": ("start", "__enter__", "__exit__"),
        "BatchRLProgram": ("run", "run_async"),
        "AgenticRLProgram": ("run_barrier", "run_partial", "run_async"),
        "SFTProgram": ("run",),
    }
    for name, methods in expected.items():
        matches = [info for info in simple.get(name, ()) if info.module == "unirl.trainer.program"]
        if len(matches) != 1:
            errors.append(f"unirl/trainer/program.py: expected one {name} class, found {len(matches)}")
            continue
        _require_members(
            errors,
            matches[0],
            kind="loop program",
            methods=methods,
            simple=simple,
        )

    inherited_loop_trainers = {
        "ARTrainer",
        "DiffusionTrainer",
        "PETrainer",
        "UnifiedModelTrainer",
        "AsyncARTrainer",
        "AsyncDiffusionTrainer",
        "AgenticTrainer",
        "AgenticPartialTrainer",
        "AsyncAgenticTrainer",
    }
    for name in inherited_loop_trainers:
        for info in simple.get(name, ()):
            if "train" in info.methods:
                errors.append(
                    f"{info.path.relative_to(ROOT)}: {name} must select a LoopProgram through BaseTrainer, "
                    "not own a copied train() loop"
                )
    return len(expected)


def check_entrypoints(errors: list[str]) -> int:
    paths = sorted(ROOT.glob("unirl/train_*.py"))
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        trainer_modules = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("unirl.trainer.")
        }
        if "main" not in functions:
            errors.append(f"{path.relative_to(ROOT)}: entrypoint does not define main()")
        if len(trainer_modules) != 1:
            errors.append(
                f"{path.relative_to(ROOT)}: entrypoint must select exactly one trainer module; "
                f"found {sorted(trainer_modules)}"
            )
    return len(paths)


def _self_test() -> None:
    """Exercise the member checker against a synthetic missing-method contract."""
    base = ClassInfo(
        path=ROOT / "_base.py",
        module="_base",
        name="Base",
        bases=(),
        methods=frozenset({"present"}),
        fields=frozenset({"payload"}),
    )
    child = ClassInfo(
        path=ROOT / "_child.py",
        module="_child",
        name="Child",
        bases=("Base",),
        methods=frozenset(),
        fields=frozenset(),
    )
    simple = {"Base": [base], "Child": [child]}
    errors: list[str] = []
    _require_members(
        errors,
        child,
        kind="fixture",
        methods=("present", "missing"),
        fields=("payload",),
        simple=simple,
    )
    if len(errors) != 1 or "missing" not in errors[0] or "present" in errors[0]:
        raise AssertionError(f"framework contract self-test failed: {errors}")


def main() -> int:
    _self_test()
    _, simple = _class_index()
    errors: list[str] = []
    counts = {
        "rollout engines": check_rollout_engines(errors, simple),
        "weight syncs": check_weight_syncs(errors, simple),
        "model bundles": check_model_bundles(errors, simple),
        "model pipelines": check_model_pipelines(errors, simple),
        "model plugin contracts": check_model_plugin_contracts(errors, simple),
        "diffusion stages": check_diffusion_runners(errors, simple),
        "train backends": check_train_backends(errors, simple),
        "wire types": check_sample_contract(errors, simple),
        "advantage contracts": check_advantage_estimators(errors, simple),
        "loop programs": check_loop_programs(errors, simple),
        "entrypoints": check_entrypoints(errors),
    }
    if errors:
        print("check-framework-contracts: FAILED")
        for error in errors:
            print(f"  {error}")
        return 1
    summary = ", ".join(f"{count} {name}" for name, count in counts.items())
    print(f"check-framework-contracts: {summary} conform.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
