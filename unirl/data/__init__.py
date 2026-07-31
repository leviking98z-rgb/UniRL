"""Lazy public surface for runtime data sources and datasets."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    "DefaultDataSource": ("unirl.data.data_source", "DefaultDataSource"),
    "MultimodalRLDataSource": ("unirl.data.data_source", "MultimodalRLDataSource"),
    "PromptExampleDataset": ("unirl.data.datasets", "PromptExampleDataset"),
    "TextPromptDataset": ("unirl.data.datasets", "TextPromptDataset"),
    "normalize_prompt_example": ("unirl.data.datasets", "normalize_prompt_example"),
}

__all__ = list(_LAZY_ATTRS)


def __getattr__(name: str):
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
