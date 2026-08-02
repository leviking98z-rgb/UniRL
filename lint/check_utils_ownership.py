#!/usr/bin/env python3
"""Keep ``unirl.utils`` a closed compatibility namespace, not an owner."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
UTILS = ROOT / "unirl/utils"
SKIP_PARTS = {"__pycache__", "vendor"}

FACADE_OWNERS: dict[str, frozenset[str]] = {
    "adapter_utils.py": frozenset({"unirl.models.adapters"}),
    "distributed_utils.py": frozenset({"unirl.distributed.collectives"}),
    "dtypes.py": frozenset({"unirl.config.dtypes"}),
    "graceful_shutdown.py": frozenset({"unirl.distributed.process"}),
    "hydra.py": frozenset({"unirl.config.remote"}),
    "media.py": frozenset({"unirl.types.media_conversion"}),
    "memory_monitor.py": frozenset({"unirl.observability.memory"}),
    "memory_utils.py": frozenset({"unirl.distributed.memory"}),
    "misc.py": frozenset(
        {
            "unirl.config.imports",
            "unirl.observability.logging",
            "unirl.observability.metrics",
            "unirl.runtime",
        }
    ),
    "peft_merge.py": frozenset({"unirl.distributed.peft"}),
    "prepare_alfworld.py": frozenset({"unirl.data.prepare.alfworld"}),
    "prepare_arxivqa_mc.py": frozenset({"unirl.data.prepare.arxivqa_mc"}),
    "prepare_asearcher.py": frozenset({"unirl.data.prepare.asearcher"}),
    "prepare_dapo_math.py": frozenset({"unirl.data.prepare.dapo_math"}),
    "prepare_geo3k_mc.py": frozenset({"unirl.data.prepare.geo3k_mc"}),
    "prepare_sft_agent.py": frozenset({"unirl.data.prepare.sft_agent"}),
    "prepare_sft_t2i.py": frozenset({"unirl.data.prepare.sft_t2i"}),
    "prepare_sft_text.py": frozenset({"unirl.data.prepare.sft_text"}),
    "prepare_sft_vlm.py": frozenset({"unirl.data.prepare.sft_vlm"}),
    "profiling.py": frozenset({"unirl.observability.profiling"}),
    "scheduler_utils.py": frozenset({"unirl.sde.scheduler"}),
    "sglang_endpoint.py": frozenset({"unirl.rollout.endpoint"}),
    "shard_balance.py": frozenset({"unirl.types.sharding"}),
    "timing.py": frozenset({"unirl.observability.timing"}),
    "wandb_logger.py": frozenset({"unirl.observability.wandb"}),
    "wandb_metrics.py": frozenset({"unirl.observability.metrics"}),
}


def _iter_python(base: Path) -> Iterable[Path]:
    for path in sorted(base.rglob("*.py")):
        if not SKIP_PARTS.intersection(path.relative_to(ROOT).parts):
            yield path


def _imports(tree: ast.AST) -> list[tuple[int, str]]:
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
        elif (
            isinstance(node, ast.Call)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in {"__import__", "import_module"}:
                imports.append((node.lineno, node.args[0].value))
            elif isinstance(fn, ast.Attribute) and fn.attr == "import_module":
                imports.append((node.lineno, node.args[0].value))
    return imports


def _owner_path(module: str) -> Path:
    return ROOT.joinpath(*module.split(".")).with_suffix(".py")


def check_facades(errors: list[str]) -> None:
    nested = [path for path in _iter_python(UTILS) if path.parent != UTILS]
    for path in nested:
        errors.append(f"{path.relative_to(ROOT)}: nested utility implementations are not allowed")

    expected = {"__init__.py", *FACADE_OWNERS}
    actual = {path.name for path in UTILS.glob("*.py")}
    for name in sorted(actual - expected):
        errors.append(f"unirl/utils/{name}: unexpected implementation; choose an owning package")
    for name in sorted(expected - actual):
        errors.append(f"unirl/utils/{name}: declared compatibility facade is missing")

    for name, owners in FACADE_OWNERS.items():
        path = UTILS / name
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions = [
            node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if definitions:
            errors.append(
                f"unirl/utils/{name}: facades cannot define implementation symbols "
                f"{[node.name for node in definitions]}"
            )
        imported_owners = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("unirl.")
        }
        if imported_owners != owners:
            errors.append(
                f"unirl/utils/{name}: expected owner imports {sorted(owners)}, found {sorted(imported_owners)}"
            )
        for owner in owners:
            if not _owner_path(owner).is_file():
                errors.append(f"unirl/utils/{name}: owner module {owner!r} does not exist")

    init_path = UTILS / "__init__.py"
    if init_path.exists():
        tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
        definitions = {
            node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        unexpected = definitions - {"__dir__", "__getattr__"}
        if unexpected:
            errors.append(
                f"unirl/utils/__init__.py: lazy facade can only define __getattr__/__dir__; found {sorted(unexpected)}"
            )


def check_framework_imports(errors: list[str]) -> None:
    roots = (ROOT / "unirl", ROOT / "experimental")
    for base in roots:
        for path in _iter_python(base):
            if UTILS in path.parents:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for lineno, module in _imports(tree):
                if module == "unirl.utils" or module.startswith("unirl.utils."):
                    errors.append(
                        f"{path.relative_to(ROOT)}:{lineno}: framework code imports legacy {module!r}; "
                        "import the canonical owner"
                    )

    recipe_paths = sorted((ROOT / "examples").rglob("*.yaml")) + sorted((ROOT / "examples").rglob("*.yml"))
    for path in recipe_paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "unirl.utils." in line:
                errors.append(f"{path.relative_to(ROOT)}:{lineno}: recipe targets must use the canonical owner")


def _self_test() -> None:
    tree = ast.parse(
        "from unirl.utils.dtypes import parse_torch_dtype\n"
        "import importlib\n"
        'importlib.import_module("unirl.utils.media")\n'
    )
    modules = {module for _, module in _imports(tree)}
    expected = {"unirl.utils.dtypes", "importlib", "unirl.utils.media"}
    if not expected.issubset(modules):
        raise AssertionError(f"utils ownership parser self-test failed: {sorted(modules)}")


def main() -> int:
    _self_test()
    errors: list[str] = []
    check_facades(errors)
    check_framework_imports(errors)
    if errors:
        print("check-utils-ownership: FAILED")
        for error in errors:
            print(f"  {error}")
        return 1
    print(
        f"check-utils-ownership: {len(FACADE_OWNERS)} compatibility facades "
        "delegate to canonical owners; framework imports are clean."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
