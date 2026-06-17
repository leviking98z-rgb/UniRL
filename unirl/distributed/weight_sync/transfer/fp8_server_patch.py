"""Server-side SGLang patch enabling FP8 *online weight resync* (issue #94).

Applied inside the SGLang server process (via _launch_server_with_compat) only
when the rollout engine runs quantization=fp8. Two patches:

1. Force the FP8 quant config to BLOCK fp8 (weight_block_size=[128,128]) and mark
   it serialized. block-FP8 process_weights_after_loading mutates
   layer.weight.data in place -> the Parameter object (and its weight_loader)
   survives, which the per-tensor path destroys (= the crash root cause).
2. Wrap the model load_weights so every incoming BF16 weight (initial disk load
   AND each online resync) is re-quantized to (fp8_weight, weight_scale_inv)
   block pairs by requantize_named_weights before the normal load path runs.

Net: trainer keeps broadcasting plain BF16 (no trainer-side change); the engine
re-quantizes on receipt; weight_loader is preserved so resync no longer crashes.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCHED = False


def patch_fp8_block_resync(weight_block_size=(128, 128)) -> bool:
    global _PATCHED
    if _PATCHED:
        return True
    wbs = [int(weight_block_size[0]), int(weight_block_size[1])]

    # (1) Force block-FP8 + serialized on every Fp8Config built in this process.
    try:
        from sglang.srt.layers.quantization.fp8 import Fp8Config
    except Exception as e:  # pragma: no cover
        logger.warning("fp8 resync patch: cannot import Fp8Config (%s); skip", e)
        return False

    _orig_init = Fp8Config.__init__

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if getattr(self, "weight_block_size", None) is None:
            self.weight_block_size = list(wbs)
        # serialized => create_weights makes fp8 params w/ weight_scale_inv +
        # block process path keeps the Parameter object (weight_loader).
        self.is_checkpoint_fp8_serialized = True

    Fp8Config.__init__ = _init

    # (2) Wrap model load_weights to re-quantize BF16 -> block-fp8 on the fly.
    from .fp8_requant import requantize_named_weights

    def _wrap_load_weights(cls):
        if getattr(cls, "_fp8_resync_wrapped", False):
            return
        _orig_lw = cls.load_weights

        def load_weights(self, weights, *a, **kw):
            return _orig_lw(
                self,
                requantize_named_weights(weights, weight_block_size=wbs, verify=True),
                *a,
                **kw,
            )

        cls.load_weights = load_weights
        cls._fp8_resync_wrapped = True

    patched_models = []
    try:
        from sglang.srt.models.qwen3 import Qwen3ForCausalLM

        _wrap_load_weights(Qwen3ForCausalLM)
        patched_models.append("Qwen3ForCausalLM")
    except Exception as e:  # pragma: no cover
        logger.warning("fp8 resync patch: cannot wrap Qwen3 load_weights (%s)", e)

    _PATCHED = True
    logger.info("FP8 block-resync patch applied: block_size=%s, models=%s", wbs, patched_models)
    return True
