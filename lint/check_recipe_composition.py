#!/usr/bin/env python3
"""Keep public recipe composition shallow, private, and executable.

Public configs may reuse one ``examples/_base`` config and then merge ``_self_``.
Private bases may not inherit again.  This guard checks that bounded graph and
asks Hydra to compose every supported public ``--config-name`` without importing
any trainer, model, or GPU package.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
BASES = EXAMPLES / "_base"
PACKAGE_MARKER = "# @package _global_"


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _load(path: Path, errors: list[str]) -> dict | None:
    text = path.read_text(encoding="utf-8")
    first_line = text.splitlines()[0] if text else ""
    if first_line != PACKAGE_MARKER:
        errors.append(f"{_relative(path)}: line 1 must be exactly '{PACKAGE_MARKER}'")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        errors.append(f"{_relative(path)}: invalid YAML: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{_relative(path)}: recipe root must be a mapping")
        return None
    return value


def _check_graph(public: list[Path], private: list[Path], errors: list[str]) -> None:
    private_set = set(private)
    uses: Counter[Path] = Counter()

    for path in private:
        value = _load(path, errors)
        if value is not None and "defaults" in value:
            errors.append(f"{_relative(path)}: private bases cannot contain 'defaults'")

    for path in public:
        value = _load(path, errors)
        if value is None or "defaults" not in value:
            continue
        defaults = value["defaults"]
        if (
            not isinstance(defaults, list)
            or len(defaults) != 2
            or not isinstance(defaults[0], str)
            or defaults[1] != "_self_"
            or not defaults[0].startswith("/_base/")
        ):
            errors.append(f"{_relative(path)}: defaults must be exactly one '/_base/...' entry followed by '_self_'")
            continue
        target = EXAMPLES / f"{defaults[0].removeprefix('/')}.yaml"
        if target not in private_set:
            errors.append(f"{_relative(path)}: base '{defaults[0]}' does not exist")
            continue
        uses[target] += 1

    for path in private:
        count = uses[path]
        if count < 2:
            errors.append(
                f"{_relative(path)}: private base must be reused by at least two public entries (found {count})"
            )


def _compose_public(public: list[Path], errors: list[str]) -> None:
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base=None, config_dir=str(EXAMPLES.resolve())):
            for path in public:
                name = str(path.relative_to(EXAMPLES).with_suffix(""))
                try:
                    config = compose(config_name=name)
                    OmegaConf.to_container(config, resolve=False)
                except Exception as exc:
                    errors.append(f"{_relative(path)}: Hydra composition failed: {exc}")
    finally:
        GlobalHydra.instance().clear()


def main() -> int:
    public = sorted(path for path in EXAMPLES.rglob("*.yaml") if BASES not in path.parents)
    private = sorted(BASES.rglob("*.yaml"))
    errors: list[str] = []

    _check_graph(public, private, errors)
    if not errors:
        _compose_public(public, errors)

    if errors:
        print("Recipe composition violations:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    reused = sum(1 for path in public if "defaults:" in path.read_text(encoding="utf-8"))
    print(
        f"check-recipe-composition: composed {len(public)} public recipes; "
        f"{reused} reuse {len(private)} one-layer bases."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
