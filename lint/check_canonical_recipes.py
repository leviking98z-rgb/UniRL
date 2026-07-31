#!/usr/bin/env python3
"""Compose and resolve the canonical Hydra entry recipes."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "examples"

# Keep this list intentionally bounded. It is the reviewed set of public entry
# recipes, not a Cartesian product of every internal overlay.
CANONICAL_RECIPES = (
    "diffusion/wan21/wan21_t2v",
    "diffusion/wan21/wan21_t2v_dancegrpo",
    "diffusion/wan21/wan21_t2v_mixgrpo",
)

REQUIRED_ROOT_KEYS = {
    "algorithm",
    "backend",
    "bundle",
    "data_source",
    "pipeline",
    "reward",
    "rollout",
    "sampling",
    "stack",
}


def main() -> int:
    errors: list[str] = []
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        for recipe in CANONICAL_RECIPES:
            try:
                cfg = compose(config_name=recipe)
                OmegaConf.resolve(cfg)
            except Exception as exc:
                errors.append(f"{recipe}: {type(exc).__name__}: {exc}")
                continue

            missing_sections = sorted(REQUIRED_ROOT_KEYS - set(cfg))
            missing_values = sorted(OmegaConf.missing_keys(cfg))
            if missing_sections:
                errors.append(f"{recipe}: missing root sections {missing_sections}")
            if missing_values:
                errors.append(f"{recipe}: unresolved mandatory values {missing_values}")

    if errors:
        print("check-canonical-recipes: FAILED")
        for error in errors:
            print(f"  {error}")
        return 1

    print(f"check-canonical-recipes: {len(CANONICAL_RECIPES)} recipes compose and resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
