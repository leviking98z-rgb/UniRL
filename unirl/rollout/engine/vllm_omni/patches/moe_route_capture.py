"""Rollout-side MoE routing capture for HunyuanImage3 (vllm-omni).

The train-side (unirl.train.backend.veomni.ep.route_replay) can FORCE a recorded
routing on replay — but it needs the rollout engine to *record* which experts
each token used. The rollout runs in a separate vllm-omni worker process on the
native ``HunyuanImage3SparseMoeBlock`` (vllm 0.20 FusedMoE), NOT the train-side
``FusedHunyuanMoE``. So capture must happen inside the vllm-omni model, and the
recorded routing is shipped back with the rollout output (cross-engine).

This module monkey-patches ``HunyuanImage3SparseMoeBlock.forward`` to, when a
capture session is active, record the ``topk_indices`` it computes (line ~1293
of vllm_omni .../hunyuan_image3/hunyuan_image3.py) per MoE-layer visit, in the
same fixed layer order the train-side replay walks. Inert (zero overhead beyond
one flag check) when no session is active.

Install (worker-side, alongside the other vllm-omni patches)::

    from unirl.rollout.engine.vllm_omni.patches import moe_route_capture
    moe_route_capture.install()

Capture around a generation::

    with moe_route_capture.capture_session() as cap:
        ... engine.generate(...) ...
    routing = cap.stack()   # [n_layer_visits, tokens, top_k] (or None)

Because the vllm forward processes a flat ``[tokens, hidden]`` batch, the
captured indices are [tokens, top_k] per layer visit; the adapter is
responsible for slicing per-request rows (the same way it slices logprobs) and
attaching each request's slice to its TextSegment.routing before ship-back.

Cross-engine caveat (documented, not hidden): the rollout FusedMoE kernel and
the train-side grouped-GEMM differ, and rollout may run a different dtype (fp8)
— that is exactly why route-replay exists. The recorded *indices* are the
ground-truth selection to force on replay; the train side recomputes the gating
weights from its own router (see route_replay._recompute_gating_weights).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import List, Optional

_TLS = threading.local()
_INSTALLED = False


class _CaptureSession:
    def __init__(self) -> None:
        self.per_layer: List["object"] = []  # list of [tokens, top_k] int tensors

    def record(self, topk_indices) -> None:
        # detach + cpu-long; transport ships these back to the trainer.
        import torch

        self.per_layer.append(topk_indices.detach().to(device="cpu", dtype=torch.int64))

    def stack(self):
        if not self.per_layer:
            return None
        import torch

        return torch.stack(self.per_layer, dim=0)  # [n_layer_visits, tokens, top_k]


def current_session() -> Optional[_CaptureSession]:
    return getattr(_TLS, "cap", None)


@contextmanager
def capture_session():
    prev = getattr(_TLS, "cap", None)
    _TLS.cap = _CaptureSession()
    try:
        yield _TLS.cap
    finally:
        _TLS.cap = prev


def note_routing(topk_indices) -> None:
    """Called from the patched MoE forward. Inert unless a session is active."""
    sess = current_session()
    if sess is not None:
        sess.record(topk_indices)


def install() -> None:
    """Monkey-patch HunyuanImage3SparseMoeBlock.forward to capture topk_indices.

    Idempotent. No-op if vllm-omni's HI3 model isn't importable in this process
    (e.g. the trainer process, which never runs the rollout MoE).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from vllm_omni.model_executor.models.hunyuan_image3 import hunyuan_image3 as hi3
    except Exception:
        # not the rollout worker (or vllm-omni layout changed) — nothing to do.
        _INSTALLED = True
        return

    Block = getattr(hi3, "HunyuanImage3SparseMoeBlock", None)
    if Block is None or getattr(Block.forward, "_route_capture", False):
        _INSTALLED = True
        return

    import torch

    orig_forward = Block.forward

    def forward(self, hidden_states: torch.Tensor):  # noqa: ANN001
        # Fast path: no active capture -> call original untouched.
        if current_session() is None:
            return orig_forward(self, hidden_states)
        # Recompute the routing exactly as the model does (cheap: one softmax+topk
        # on the router logits) so we capture WITHOUT depending on the original's
        # internal packing. Mirrors hunyuan_image3.py lines ~1289-1293 and
        # HunyuanTopKGate.easy_topk.
        orig_shape = hidden_states.shape
        hs = hidden_states.view(-1, orig_shape[-1])
        router_logits, _ = self.gate(hs.float())
        gates = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        _, topk_indices = torch.topk(gates, self.top_k, dim=-1)
        note_routing(topk_indices)
        # Then run the real forward for the actual compute (unchanged output).
        return orig_forward(self, hidden_states)

    forward._route_capture = True  # type: ignore[attr-defined]
    Block.forward = forward
    _INSTALLED = True


__all__ = ["install", "capture_session", "current_session", "note_routing"]
