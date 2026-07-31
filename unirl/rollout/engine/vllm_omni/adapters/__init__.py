"""Lazy adapter registry for the vLLM-Omni engine.

Modality adapters are grouped by model family and composed from input/output
sub-adapters (the binder constructs both in ``__init__`` and delegates the
two conversion verbs):

- ``hi3``  — hi3_t2i, hi3_it2i, hi3_i2t, hi3_t2t, hi3_ar_recaption, hi3_dit_recaption
- ``sd3``  — sd3_t2i
- ``hv15`` — hv15_t2v
- ``qwen_image`` — qwen_image_t2i
- ``bagel`` — bagel_t2i

``dit`` holds the universal single-stage DiT skeletons
(:class:`DitInputAdapter` / :class:`DitOutputAdapter`) the families derive
from; family-specific sub-adapters carry the family prefix and live in the
family file.
"""

from importlib import import_module

from unirl.rollout.engine.vllm_omni.adapters.base import (
    _BUILTIN_ADAPTERS,
    ModelAdapter,
    get_adapter,
    register_adapter,
    registered_adapters,
)
from unirl.rollout.engine.vllm_omni.adapters.dit import DitInputAdapter, DitOutputAdapter

_CLASS_MODULES = {
    class_name: module_name
    for module_name, class_name in (target.split(":", 1) for target in _BUILTIN_ADAPTERS.values())
}
_CLASS_MODULES.update(
    {
        "BagelInputAdapter": "unirl.rollout.engine.vllm_omni.adapters.bagel",
        "BagelOutputAdapter": "unirl.rollout.engine.vllm_omni.adapters.bagel",
        "Hi3ArRecaptionOutputAdapter": "unirl.rollout.engine.vllm_omni.adapters.hi3",
        "Hi3DitRecaptionInputAdapter": "unirl.rollout.engine.vllm_omni.adapters.hi3",
        "Hi3DitRecaptionOutputAdapter": "unirl.rollout.engine.vllm_omni.adapters.hi3",
        "Hi3ImageOutputAdapter": "unirl.rollout.engine.vllm_omni.adapters.hi3",
        "Hi3InputAdapter": "unirl.rollout.engine.vllm_omni.adapters.hi3",
        "Hi3TextOutputAdapter": "unirl.rollout.engine.vllm_omni.adapters.hi3",
        "Hv15InputAdapter": "unirl.rollout.engine.vllm_omni.adapters.hv15",
        "Hv15VideoOutputAdapter": "unirl.rollout.engine.vllm_omni.adapters.hv15",
        "QwenImageInputAdapter": "unirl.rollout.engine.vllm_omni.adapters.qwen_image",
        "QwenImageOutputAdapter": "unirl.rollout.engine.vllm_omni.adapters.qwen_image",
        "Sd3OutputAdapter": "unirl.rollout.engine.vllm_omni.adapters.sd3",
    }
)


def __getattr__(name: str):
    module_name = _CLASS_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


__all__ = [
    "DitInputAdapter",
    "DitOutputAdapter",
    "BagelInputAdapter",
    "BagelOutputAdapter",
    "BagelT2iAdapter",
    "Hi3ArRecaptionAdapter",
    "Hi3ArRecaptionOutputAdapter",
    "Hi3DitRecaptionAdapter",
    "Hi3DitRecaptionInputAdapter",
    "Hi3DitRecaptionOutputAdapter",
    "Hi3I2tAdapter",
    "Hi3ImageOutputAdapter",
    "Hi3InputAdapter",
    "Hi3It2iAdapter",
    "Hi3T2iAdapter",
    "Hi3T2tAdapter",
    "Hi3TextOutputAdapter",
    "Hv15InputAdapter",
    "Hv15T2vAdapter",
    "Hv15VideoOutputAdapter",
    "ModelAdapter",
    "QwenImageInputAdapter",
    "QwenImageOutputAdapter",
    "QwenImageT2iAdapter",
    "Sd3OutputAdapter",
    "Sd3T2iAdapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
