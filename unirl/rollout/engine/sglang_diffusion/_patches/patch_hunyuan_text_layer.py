"""Align the sglang HunyuanVideo LLaMA text-embedding layer with the trainside.

The trainside HunyuanVideo text encoder
(``unirl/models/hunyuan_video/text_embed.py``) takes the LLaMA **final** hidden
state -- ``outputs.last_hidden_state`` (== ``hidden_states[-1]``) -- and its
comment is explicit: *"Use the last hidden state (unlike HV15 which uses
skip_layers)."* The golden trainside curve (o2c8gg0k) was trained on exactly
these embeddings.

Upstream sglang's ``llama_postprocess_text``
(``configs/pipeline_configs/hunyuan.py``) instead hardcodes
``hidden_state_skip_layer = 2`` -> ``hidden_states[-3]`` (third-from-last
layer). That is a materially DIFFERENT LLaMA layer, so the sglang rollout
conditions on different text embeddings than the trainside sampler -> the two
engines generate systematically different videos -> a constant PickScore offset
from the trainside baseline (~+0.02), even though per-step SDE log-prob stays
consistent (ratio≈1, because replay reuses the rollout's stored embeddings).

Env ``UNIRL_SGLANG_LLAMA_SKIP_LAYER`` pins the skip layer (int, read at CALL
time). Unset -> 0 (trainside-aligned: last hidden state). Set to 2 for upstream
behavior.

Picklability: the replacement MUST be a module-level function. The upstream
function is bound into ``HunyuanConfig.postprocess_text_funcs`` (a
``field(default_factory=lambda: (llama_postprocess_text, clip_postprocess_text))``)
and that config is pickled to Ray workers. A ``<locals>`` closure is
unpicklable (``Can't get local object``), so we define the replacement here at
module scope and simply rebind the module attribute; the default_factory lambda
resolves the global lazily at HunyuanConfig-instantiation time (which happens
after the hijack installs, before ``from_pretrained``).

Idempotent + import-safe (sglang imported inside the fns).
"""

from __future__ import annotations

import os

_SENTINEL = "_unirl_llama_skip_layer_patched"


def _unirl_llama_postprocess_text(outputs, _text_inputs):
    """Module-level (picklable) replacement for sglang ``llama_postprocess_text``.

    Trainside-aligned default: ``UNIRL_SGLANG_LLAMA_SKIP_LAYER=0`` ->
    ``hidden_states[-1]`` (== trainside ``outputs.last_hidden_state``).
    """
    import torch
    from sglang.multimodal_gen.configs.pipeline_configs import hunyuan as _hy

    skip = int(os.environ.get("UNIRL_SGLANG_LLAMA_SKIP_LAYER", "0"))
    assert outputs.hidden_states is not None
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-(skip + 1)]
    crop_start = _hy.prompt_template_video.get("crop_start", -1)
    last_hidden_state = last_hidden_state[:, crop_start:]
    attention_mask = _text_inputs.attention_mask.to(
        device=last_hidden_state.device, dtype=torch.bool
    )
    if crop_start < 0:
        attention_mask = attention_mask[:, crop_start:]
    else:
        attention_mask = attention_mask[
            :, crop_start : crop_start + last_hidden_state.shape[1]
        ]
    seq_lens = [int(x) for x in attention_mask.to(torch.int64).sum(dim=1).tolist()]
    if not getattr(_unirl_llama_postprocess_text, "_logged", False):
        print(
            f"[UNIRL_LLAMA_SKIP] hunyuan LLaMA hidden_state_skip_layer -> {skip} "
            f"(hidden_states[-{skip + 1}]); embed shape={tuple(last_hidden_state.shape)}",
            flush=True,
        )
        _unirl_llama_postprocess_text._logged = True  # type: ignore[attr-defined]
    return _hy.TextConditioningOutput(last_hidden_state, attention_mask, seq_lens)


def patch_hunyuan_text_layer() -> None:
    from sglang.multimodal_gen.configs.pipeline_configs import hunyuan as _hy

    if getattr(_hy, _SENTINEL, False):
        return

    # Rebind the module-level function; picked up by the config default_factory
    # lambda (which resolves the global lazily at instantiation) and picklable
    # to Ray workers (module-level qualname).
    _hy.llama_postprocess_text = _unirl_llama_postprocess_text

    # Also patch the arch-config default (harmless; keeps reported config
    # coherent -- the function reads the env, not this field).
    try:
        from sglang.multimodal_gen.configs.models.encoders.llama import (
            LlamaArchConfig,
        )

        LlamaArchConfig.hidden_state_skip_layer = int(
            os.environ.get("UNIRL_SGLANG_LLAMA_SKIP_LAYER", "0")
        )
    except Exception:  # pragma: no cover - defensive
        pass

    setattr(_hy, _SENTINEL, True)
