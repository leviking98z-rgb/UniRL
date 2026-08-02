#!/usr/bin/env python3
"""Enforce UniRL's source-level dependency direction without importing code.

The repository has three logical layers:

* the framework kernel (config, typed wire contracts, SDE primitives, and the
  distributed actor/tensor substrate);
* loop components (data, models, algorithms, reward, train, rollout, and
  weight-sync);
* cross-cutting services (observability);
* orchestration (trainer and the thin ``train_*.py`` entrypoints).

This guard intentionally checks source imports, including imports below
``TYPE_CHECKING`` and inside functions.  Those edges still couple packages for
typing, ownership, and future refactors even when Python does not execute them at
module import time.  It also recognizes literal ``import_module``/``__import__``
calls so a dependency cannot be hidden behind a lazy import.

The rules are deliberately coarse and stable.  They do not prescribe internal
file layout or ban legitimate component dependencies such as rollout adapters
consuming model contracts.  They prevent the damaging reverse arrows: kernel
code knowing a model/engine/trainer, and a loop component reaching into its
orchestrator or an unrelated peer.

Run by the ``check-architecture-boundaries`` pre-commit hook.  The implementation
is stdlib-only so lint CI does not need torch, Ray, Hydra, or engine packages.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "__pycache__", "vendor"}


@dataclass(frozen=True)
class Boundary:
    """One source ownership rule."""

    name: str
    roots: tuple[str, ...]
    forbidden: tuple[str, ...]
    reason: str


_UPPER_PACKAGES = (
    "unirl.algorithms",
    "unirl.models",
    "unirl.reward",
    "unirl.rollout",
    "unirl.train",
    "unirl.trainer",
)

BOUNDARIES = (
    Boundary(
        name="framework kernel",
        roots=(
            "unirl/config",
            "unirl/types",
            "unirl/sde",
            "unirl/distributed/group",
            "unirl/distributed/tensor",
        ),
        forbidden=(*_UPPER_PACKAGES, "unirl.observability"),
        reason="kernel code defines contracts/substrate and cannot know a loop component or trainer",
    ),
    Boundary(
        name="data component",
        roots=("unirl/data",),
        forbidden=(
            "unirl.algorithms",
            "unirl.models",
            "unirl.reward",
            "unirl.rollout",
            "unirl.train",
            "unirl.trainer",
        ),
        reason="data sources produce typed inputs; they do not own execution components",
    ),
    Boundary(
        name="model component",
        roots=("unirl/models",),
        forbidden=("unirl.reward", "unirl.rollout", "unirl.trainer"),
        reason="model packages expose stages/pipelines and cannot depend on consumers or orchestration",
    ),
    Boundary(
        name="algorithm component",
        roots=("unirl/algorithms",),
        forbidden=("unirl.models", "unirl.reward", "unirl.rollout", "unirl.trainer"),
        reason="algorithms consume stage/type contracts and cannot own models, rollout, reward, or orchestration",
    ),
    Boundary(
        name="reward component",
        roots=("unirl/reward",),
        forbidden=(
            "unirl.algorithms",
            "unirl.models",
            "unirl.rollout",
            "unirl.train",
            "unirl.trainer",
        ),
        reason="reward consumes typed samples and remains independent of the other loop components",
    ),
    Boundary(
        name="train component",
        roots=("unirl/train",),
        forbidden=("unirl.reward", "unirl.rollout", "unirl.trainer"),
        reason="the train stack cannot reach sideways into rollout/reward or upward into its trainer",
    ),
    Boundary(
        name="rollout component",
        roots=("unirl/rollout",),
        forbidden=("unirl.algorithms", "unirl.reward", "unirl.train", "unirl.trainer"),
        reason="rollout may adapt model contracts but cannot own reward, optimization, or orchestration",
    ),
    Boundary(
        name="weight-sync component",
        roots=("unirl/distributed/weight_sync",),
        forbidden=(
            "unirl.algorithms",
            "unirl.models",
            "unirl.reward",
            "unirl.rollout",
            "unirl.trainer",
        ),
        reason="weight-sync is a transport component, not a model/engine/trainer integration point",
    ),
    Boundary(
        name="observability service",
        roots=("unirl/observability",),
        forbidden=(
            "unirl.algorithms",
            "unirl.data",
            "unirl.models",
            "unirl.reward",
            "unirl.rollout",
            "unirl.train",
            "unirl.trainer",
        ),
        reason="observability consumes stable contracts and cannot depend on a loop implementation",
    ),
)


def _module_for_path(path: Path) -> tuple[str, ...]:
    rel = path.relative_to(ROOT)
    if rel.name == "__init__.py":
        return rel.parts[:-1]
    return (*rel.parts[:-1], rel.stem)


def _resolve_relative(path: Path, node: ast.ImportFrom) -> str | None:
    package = _module_for_path(path)[:-1]
    if node.level <= 0 or node.level > len(package) + 1:
        return None
    keep = len(package) - (node.level - 1)
    prefix = package[:keep]
    suffix = tuple(node.module.split(".")) if node.module else ()
    parts = (*prefix, *suffix)
    return ".".join(parts) if parts else None


def _literal_dynamic_import(node: ast.Call) -> str | None:
    if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
        return None
    fn = node.func
    if isinstance(fn, ast.Name) and fn.id in {"__import__", "import_module"}:
        return node.args[0].value
    if isinstance(fn, ast.Attribute) and fn.attr == "import_module":
        return node.args[0].value
    return None


def imports_in_source(path: Path, source: str) -> list[tuple[int, str]]:
    """Return every source-level dependency as ``(line, dotted_module)``."""
    tree = ast.parse(source, filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.level == 0 else _resolve_relative(path, node)
            if module:
                found.append((node.lineno, module))
                # ``from unirl import trainer`` semantically reaches
                # ``unirl.trainer`` even though the ImportFrom module is only
                # ``unirl``.
                found.extend((node.lineno, f"{module}.{alias.name}") for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Call):
            module = _literal_dynamic_import(node)
            if module:
                found.append((node.lineno, module))
    return found


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


def _iter_python(root: str) -> Iterable[Path]:
    base = ROOT / root
    if base.is_file():
        yield base
        return
    for path in sorted(base.rglob("*.py")):
        if not SKIP_PARTS.intersection(path.relative_to(ROOT).parts):
            yield path


def _self_test() -> None:
    """Pin parser behavior so the guard cannot silently lose an import spelling."""
    fake = ROOT / "unirl" / "types" / "_fixture.py"
    source = """
import unirl.trainer
from unirl import rollout
from .segments import Segment
import importlib

def lazy():
    return importlib.import_module("unirl.models.fake")
"""
    imports = {module for _, module in imports_in_source(fake, source)}
    expected = {
        "unirl.trainer",
        "unirl",
        "unirl.rollout",
        "unirl.types.segments",
        "unirl.types.segments.Segment",
        "importlib",
        "unirl.models.fake",
    }
    missing = expected - imports
    if missing:
        raise AssertionError(f"architecture guard parser self-test missed imports: {sorted(missing)}")


def main() -> int:
    _self_test()
    errors: list[str] = []
    checked_files: set[Path] = set()
    edge_count = 0
    for boundary in BOUNDARIES:
        for root in boundary.roots:
            for path in _iter_python(root):
                checked_files.add(path)
                try:
                    imports = imports_in_source(path, path.read_text(encoding="utf-8"))
                except (OSError, SyntaxError) as exc:
                    errors.append(f"{path.relative_to(ROOT)}: cannot inspect imports: {exc}")
                    continue
                for lineno, module in imports:
                    if not module.startswith("unirl"):
                        continue
                    edge_count += 1
                    hit = next((prefix for prefix in boundary.forbidden if _matches(module, prefix)), None)
                    if hit is not None:
                        errors.append(
                            f"{path.relative_to(ROOT)}:{lineno}: {boundary.name} imports {module!r}; "
                            f"{boundary.reason} (forbidden prefix: {hit})"
                        )

    if errors:
        print("check-architecture-boundaries: FAILED")
        for error in errors:
            print(f"  {error}")
        return 1
    print(
        "check-architecture-boundaries: "
        f"{len(checked_files)} files / {edge_count} internal import edges satisfy {len(BOUNDARIES)} boundaries."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
