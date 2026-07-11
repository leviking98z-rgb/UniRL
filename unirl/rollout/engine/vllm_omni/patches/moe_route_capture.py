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
        # We accumulate per FORWARD: each forward pass visits all MoE layers once
        # over the same token axis, so one forward contributes a
        # [n_layers, n_tok_this_forward, top_k] block. AR runs max_num_seqs=1 so
        # forwards are strictly serial per request (prefill then decode steps).
        self._cur_forward: List["object"] = []   # layers of the in-progress forward
        self._seen_ids: set = set()               # module ids seen in current forward
        self.forwards: List["object"] = []        # list of [n_layers, n_tok, top_k]
        self.forward_ntok: List[int] = []         # token count of each forward

    def record_layer(self, layer_id: int, topk_indices) -> None:
        """Record one MoE layer's routing. Detect forward boundary: when a layer
        module id repeats, the previous forward finished — seal it first.

        Works because within one forward each MoE layer is visited exactly once;
        the next forward revisits layer-0's module → id repeats."""
        import torch

        if layer_id in self._seen_ids:
            self.end_forward()
        self._seen_ids.add(layer_id)
        self._cur_forward.append(topk_indices.detach().to(device="cpu", dtype=torch.int64))

    def end_forward(self) -> None:
        """Seal the in-progress forward into one [n_layers, n_tok, top_k] block."""
        import torch

        if not self._cur_forward:
            return
        block = torch.stack(self._cur_forward, dim=0)
        self.forwards.append(block)
        self.forward_ntok.append(int(block.shape[1]))
        self._cur_forward = []
        self._seen_ids = set()

    def drain(self):
        """Return (routing[n_layers, total_tok, top_k], forward_ntok) and reset.

        total_tok is the concatenation over all forwards in visit order; the
        driver splits per-request using forward_ntok + each request's known
        token count.
        """
        import torch

        self.end_forward()
        if not self.forwards:
            return None, []
        cat = torch.cat(self.forwards, dim=1)
        ntok = list(self.forward_ntok)
        self.forwards = []
        self.forward_ntok = []
        self._cur_forward = []
        self._seen_ids = set()
        return cat, ntok


def current_session() -> Optional[_CaptureSession]:
    return getattr(_TLS, "cap", None)


@contextmanager
def capture_session():
    """Driver/test-side context manager (used by unit tests + in-process paths)."""
    prev = getattr(_TLS, "cap", None)
    _TLS.cap = _CaptureSession()
    try:
        yield _TLS.cap
    finally:
        _TLS.cap = prev


# --- Worker-process global buffer (cross-process path) ----------------------
# vllm-omni AR runs in a spawned worker subprocess; the driver can't share a
# thread-local session with it. So when capture is env-enabled we keep a
# PROCESS-GLOBAL session that the patched forward always writes to, and the
# driver drains it via a collective_rpc verb (_diffrl_drain_routing) after each
# generate. Gated so it's inert unless UNIRL_MOE_ROUTE_CAPTURE=1.
_GLOBAL: Optional[_CaptureSession] = None


def _global_enabled() -> bool:
    import os

    return os.environ.get("UNIRL_MOE_ROUTE_CAPTURE", "0") == "1"


def global_session() -> Optional[_CaptureSession]:
    global _GLOBAL
    if not _global_enabled():
        return None
    if _GLOBAL is None:
        _GLOBAL = _CaptureSession()
    return _GLOBAL


def drain_global():
    """Worker-side: drain the process-global capture buffer. Returns
    (routing_or_None, forward_ntok). Called via collective_rpc from the driver."""
    g = global_session()
    if g is None:
        return None, []
    return g.drain()


def note_routing(layer_id: int, topk_indices) -> None:
    """Called from the patched MoE forward. Writes to the active thread-local
    session if any, else the process-global buffer (worker path). Inert if
    neither is active."""
    sess = current_session()
    if sess is None:
        sess = global_session()
    if sess is not None:
        sess.record_layer(layer_id, topk_indices)


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
        # Fast path: no active capture (neither thread-local nor process-global)
        # -> call original untouched.
        if current_session() is None and global_session() is None:
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
        note_routing(id(self), topk_indices)
        # Then run the real forward for the actual compute (unchanged output).
        return orig_forward(self, hidden_states)

    forward._route_capture = True  # type: ignore[attr-defined]
    Block.forward = forward
    _INSTALLED = True


__all__ = ["install", "capture_session", "current_session", "note_routing"]
