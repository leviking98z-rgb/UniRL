"""Dynamic symbol resolution for configuration validation."""

from __future__ import annotations

import importlib
from typing import Any


def load_function(path: str) -> Any:
    """Load a class or function from a fully qualified dotted path."""
    if path is None or path == "":
        raise ValueError("Path cannot be None or empty")

    parts = path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid path format: {path}. Expected 'module.path.ClassName'")

    module_path, symbol_name = parts
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(f"Could not import module '{module_path}': {exc}") from exc

    try:
        return getattr(module, symbol_name)
    except AttributeError:
        raise AttributeError(f"Module '{module_path}' has no attribute '{symbol_name}'")


__all__ = ["load_function"]
