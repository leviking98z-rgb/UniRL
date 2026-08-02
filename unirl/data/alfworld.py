"""Stable ALFWorld game discovery shared by data prep and rollout."""

from __future__ import annotations

import os
from glob import glob
from typing import List, Optional


def list_alfworld_games(split: str = "train", data_dir: Optional[str] = None) -> List[str]:
    """Enumerate ALFWorld TextWorld game files with stable sorted indices."""
    root = data_dir or os.environ.get("ALFWORLD_DATA", "")
    if not root:
        return []
    patterns = [
        os.path.join(root, "json_2.1.1", split, "**", "game.tw-pddl"),
        os.path.join(root, "json_2.1.1", split, "**", "*.tw-pddl"),
        os.path.join(root, "**", split, "**", "*.tw-pddl"),
    ]
    for pattern in patterns:
        games = sorted(glob(pattern, recursive=True))
        if games:
            return games
    return []


__all__ = ["list_alfworld_games"]
