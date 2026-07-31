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
    return len(engines)


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
        _require_members(
            errors,
            info,
            kind="model pipeline",
            methods=("generate",),
            simple=simple,
        )
    return len(pipelines)


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
        "model pipelines": check_model_pipelines(errors, simple),
        "train backends": check_train_backends(errors, simple),
        "wire types": check_sample_contract(errors, simple),
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
