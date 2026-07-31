"""Video adapters split by model family and resolved lazily.

Importing this package does not import WAN, HunyuanVideo, or LTX-2 adapter
implementations. The historical ``adapters.video`` class exports remain
available for compatibility and resolve one family at a time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from unirl.rollout.engine.sglang_diffusion.adapters import get_adapter

_CLASS_TO_FAMILY = {
    "MochiAdapter": "mochi",
    "HunyuanVideoAdapter": "hunyuan_video",
    "Wan21T2VAdapter": "wan21",
    "Wan22T2VAdapter": "wan22",
    "Ltx2T2VAdapter": "ltx2",
}
_BASE_EXPORTS = {
    "VideoAdapter": "unirl.rollout.engine.sglang_diffusion.adapters.video.base:VideoAdapter",
}


def __getattr__(name: str) -> Any:
    family = _CLASS_TO_FAMILY.get(name)
    if family is not None:
        value = get_adapter(family)
    else:
        dotpath = _BASE_EXPORTS.get(name)
        if dotpath is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        module_name, _, attribute = dotpath.partition(":")
        value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_CLASS_TO_FAMILY) | set(_BASE_EXPORTS))


__all__ = [
    "VideoAdapter",
    "MochiAdapter",
    "HunyuanVideoAdapter",
    "Wan21T2VAdapter",
    "Wan22T2VAdapter",
    "Ltx2T2VAdapter",
]
