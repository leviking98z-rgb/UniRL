"""Re-quantize BF16 trainer weights to block-FP8 for the SGLang rollout engine.

Root cause (issue #94): SGLang `quantization=fp8` over a BF16 checkpoint makes
`Fp8LinearMethod.process_weights_after_loading` REPLACE `layer.weight` with a
fresh `Parameter(qweight)` (per-tensor path), dropping `weight_loader`. Online
resync (`update_weights_from_distributed` -> `model.load_weights`) then crashes
in qwen3.py stacked branch (`weight_loader = param.weight_loader`, no default)
with "'Parameter' object has no attribute 'weight_loader'".

Fix (verl SGLangFP8QuantizerHelper approach): run the engine in *block*-FP8
(weight_block_size=[128,128]) whose process_weights_after_loading mutates
`layer.weight.data` in place (preserves Parameter + weight_loader), and
re-quantize each incoming BF16 weight to (fp8_weight, weight_scale_inv) on every
sync. No-op unless caller opts in (FP8 engine only). Block-cast kernel vendored
from verl (Apache-2.0) in fp8_kernel_vendored.py.
"""
from __future__ import annotations

import logging
from typing import Iterable, Iterator, List, Tuple

import torch

from .fp8_kernel_vendored import FP8_DTYPE, scaled_fp8_blockwise

logger = logging.getLogger(__name__)

DEFAULT_WEIGHT_BLOCK_SIZE: List[int] = [128, 128]

_EXCLUDE_SUBSTRINGS = (
    "embed_tokens", "lm_head", "layernorm", "norm", "ln_",
    "embeddings", "mlp.gate.weight",
)
_INCLUDE_SUBSTRINGS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj",
    "gate_proj", "up_proj", "down_proj", "gate_up_proj",
)


def should_quantize_param(name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    low = name.lower()
    if any(p in low for p in _EXCLUDE_SUBSTRINGS):
        return False
    return any(p in low for p in _INCLUDE_SUBSTRINGS)


def scale_inv_name(weight_name: str) -> str:
    return weight_name[: -len("weight")] + "weight_scale_inv"


def requantize_named_weights(weights, *, weight_block_size=DEFAULT_WEIGHT_BLOCK_SIZE,
                             compute_dtype=torch.bfloat16, verify=False):
    bm, bn = int(weight_block_size[0]), int(weight_block_size[1])
    for name, tensor in weights:
        if not should_quantize_param(name) or tensor.dim() != 2:
            yield name, tensor
            continue
        hp = tensor.to(compute_dtype)
        fp8_weight, descale = scaled_fp8_blockwise(hp, [bm, bn])
        descale = descale.squeeze(-1) if descale.dim() > 2 else descale
        if verify:
            rel = _blockwise_dequant_rel_err(hp, fp8_weight, descale, bm, bn)
            if rel > 0.2:
                raise RuntimeError(
                    f"FP8 re-quant numerical check failed for {name}: rel dequant "
                    f"error {rel:.4f} > 0.2 — refusing to push (would corrupt weights).")
        yield name, fp8_weight
        yield scale_inv_name(name), descale
        del fp8_weight, descale


def _blockwise_dequant_rel_err(hp, fp8_weight, descale, bm, bn):
    m, n = hp.shape
    deq = fp8_weight.to(torch.float32)
    sc = descale.to(torch.float32)
    sc = sc.repeat_interleave(bm, dim=0)[:m].repeat_interleave(bn, dim=1)[:, :n]
    deq = deq * sc
    num = (deq - hp.to(torch.float32)).abs().mean()
    den = hp.to(torch.float32).abs().mean().clamp_min(1e-8)
    return float((num / den).item())
