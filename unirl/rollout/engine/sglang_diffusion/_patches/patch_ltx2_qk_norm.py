"""Coerce ``qk_norm=True`` -> ``"rms_norm_across_heads"`` for LTX-2 attention.

sglang's LTX-2 DiT (``runtime/models/dits/ltx_2.py``) constructs its attention
layers with ``qk_norm=True`` (a bool; comment: "Always True in LTX2"), but the
``LTX2Attention`` classes it/diffusers instantiate only accept the *string*
``"rms_norm_across_heads"`` and hard-raise ``NotImplementedError`` on anything
else:

    if qk_norm != "rms_norm_across_heads":
        raise NotImplementedError("Only 'rms_norm_across_heads' is supported ...")

The two implementations disagree on the ``qk_norm`` type (bool vs str) in the
sglang/diffusers versions pinned in ``.venv-sglang`` (diffusers 0.37.1). The DiT
comment confirms LTX-2 always uses qk-norm, and ``rms_norm_across_heads`` is the
only supported mode, so ``True`` unambiguously means that mode. This wraps the
``LTX2Attention.__init__`` of both the sglang connector and the diffusers
transformer to normalize a bool-``True`` ``qk_norm`` to the string BEFORE the
check. Only the exact bool ``True`` is coerced; ``None``/``False``/strings pass
through untouched (so a genuine "no qk-norm" is never forced on).

Idempotent + import-safe (targets imported inside the fn; missing targets skipped).
"""

from __future__ import annotations


def _wrap_qk_norm(cls) -> None:
    orig = cls.__init__
    if getattr(orig, "_unirl_qknorm_coerce", False):
        return

    def __init__(self, *args, **kwargs):
        if kwargs.get("qk_norm") is True:
            kwargs["qk_norm"] = "rms_norm_across_heads"
        return orig(self, *args, **kwargs)

    __init__._unirl_qknorm_coerce = True  # type: ignore[attr-defined]
    cls.__init__ = __init__


def patch_ltx2_qk_norm() -> None:
    targets = []
    try:
        from sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector import (
            LTX2Attention as _SglAttn,
        )

        targets.append(_SglAttn)
    except Exception:  # pragma: no cover - module layout dependent
        pass
    try:
        from diffusers.models.transformers.transformer_ltx2 import (
            LTX2Attention as _DiffAttn2,
        )

        targets.append(_DiffAttn2)
    except Exception:  # pragma: no cover
        pass
    try:
        from diffusers.models.transformers.transformer_ltx import (
            LTXAttention as _DiffAttn1,
        )

        targets.append(_DiffAttn1)
    except Exception:  # pragma: no cover
        pass

    for cls in targets:
        try:
            _wrap_qk_norm(cls)
        except Exception:  # pragma: no cover - defensive
            pass
