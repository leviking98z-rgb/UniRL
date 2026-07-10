"""MoE route-replay: freeze the expert *selection* to the rollout's, recompute
the gating *weights* from the current router.

Why this exists
---------------
In on-policy RL the log-prob is computed twice for the same sequence: once by
the rollout engine (``old_logp``) and once by the train-side forward
(``new_logp``). PPO/GRPO's importance ratio ``exp(new_logp - old_logp)`` is only
meaningful if both passes ran the *same* computation.

For a **dense** model the two passes differ only by smooth numerical noise
(~1e-3). For a **MoE** model the router's ``top_k`` is a *discrete* argmax: a
~1e-6 perturbation of the router logits (fp8 rollout vs bf16 train, different
kernels) can flip which experts a token is routed to. A flipped token then runs
a *different* FFN -> its logp diverges by O(1) -> its ratio explodes. The
divergence is silent (no error, shapes all valid) and shows up only as a
bimodal ``|Δlogp|`` (most tokens fine, a few catastrophic).

route-replay removes this: record the experts each token actually used during
rollout, and on the train-side replay **force the same experts**.

The gradient subtlety (the whole point)
---------------------------------------
``out = Σ_{e∈selected} g_e · FFN_e(x)``.

- ``selected`` (which experts) is ``top_k`` — discrete, **not differentiable**,
  never carries gradient even in normal MoE training. Freezing it to the
  recorded value costs **zero** gradient.
- ``g_e`` (the gating softmax weight) **must be recomputed from the train-side
  router's current logits** (restricted to the frozen experts, renormalized) so
  the router keeps learning. Reusing the recorded ``g_e`` as a constant would
  cut the router's gradient -> the router silently stops training.
- ``FFN_e(x)`` uses train-side expert weights -> experts train normally.

So freezing the *selection* but recomputing the *weight* is gradient-free for
the frozen part and gradient-preserving for the learnable part.

Scope of the freeze
-------------------
The freeze is per-batch (this batch's rollout routing), NOT permanent. The
router weights still update via ``g_e``; the *next* rollout uses the new router
and may pick different experts. This is exactly the on-policy requirement:
replay must reproduce what the policy did when it *generated* this batch.

Usage
-----
Record pass (produce the routing to store on the trajectory)::

    with route_replay_session(mode="record") as rec:
        model(...)                      # runs normal top_k, captures per-layer idx
    routing = rec.stack()               # [n_layers, total_tokens, top_k]

Replay pass (force the recorded routing)::

    with route_replay_session(mode="replay", routing=routing):
        new_logp = model(...)           # forces recorded experts, recomputes g

Both are no-ops when no session is active, so the module is inert on the normal
training path (dense models, or MoE runs that don't opt in).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional

import torch

# Thread-local so concurrent Ray workers / dataloader threads don't clobber each
# other's session. Each forward walks the MoE layers in a fixed order, so a
# per-session monotonic layer counter aligns record<->replay by position.
_TLS = threading.local()


@dataclass
class _RouteSession:
    mode: str  # "record" | "replay"
    # replay input: per-layer recorded expert indices, indexed by layer-visit order.
    # each entry is [tokens, top_k] (long).
    routing_in: Optional[List[torch.Tensor]] = None
    # record output: per-layer captured expert indices, appended in visit order.
    routing_out: List[torch.Tensor] = field(default_factory=list)
    # monotonic counter: which MoE layer we're visiting in this forward.
    _cursor: int = 0
    # replay row-offset: the recorded routing covers only the LAST ``rows`` tokens
    # of each layer's [tokens, top_k] (the response). Tokens before the offset
    # (the prompt) keep their live routing. None => the recorded routing covers
    # ALL tokens (offset 0). Set when the replay forward prepends a prompt whose
    # routing was not recorded (single-engine / response-only capture).
    replay_offset: Optional[int] = None

    def next_recorded(self) -> torch.Tensor:
        assert self.routing_in is not None, "replay session has no routing"
        idx = self.routing_in[self._cursor]
        self._cursor += 1
        return idx

    def capture(self, topk_idx: torch.Tensor) -> None:
        self.routing_out.append(topk_idx.detach().to(torch.long))

    def stack(self) -> Optional[torch.Tensor]:
        """Stack captured per-layer indices into [n_layers, tokens, top_k].

        Returns None if nothing was captured (e.g. a dense model). Layers must
        share token/top_k dims (true within one forward of one model)."""
        if not self.routing_out:
            return None
        return torch.stack(self.routing_out, dim=0)


def current_session() -> Optional[_RouteSession]:
    return getattr(_TLS, "session", None)


@contextmanager
def route_replay_session(mode: str, routing: Optional[torch.Tensor] = None, replay_offset: Optional[int] = None):
    """Activate a route-replay session for the enclosed forward pass.

    Args:
        mode: "record" captures top_k indices per MoE layer; "replay" forces the
            provided routing and recomputes gating weights from the live router.
        routing: for "replay", a [n_layers, rows, top_k] long tensor (as
            produced by a prior record session's ``stack()``), or a list of
            per-layer [rows, top_k] tensors.
        replay_offset: if set, the recorded ``rows`` correspond to the LAST
            ``rows`` tokens of each layer's live [tokens, top_k]; the first
            ``tokens-rows`` tokens (offset region, e.g. a prompt whose routing
            was not recorded) keep their live routing. Requires
            ``offset == tokens - rows``; enforced in ``apply_route_replay``.
            None => recorded routing covers all tokens.
    """
    if mode not in ("record", "replay"):
        raise ValueError(f"route_replay_session: mode must be record|replay, got {mode!r}")
    routing_in: Optional[List[torch.Tensor]] = None
    if mode == "replay":
        if routing is None:
            raise ValueError("route_replay_session(mode='replay') requires routing=")
        routing_in = [routing[i] for i in range(routing.shape[0])] if torch.is_tensor(routing) else list(routing)

    prev = getattr(_TLS, "session", None)
    _TLS.session = _RouteSession(mode=mode, routing_in=routing_in, replay_offset=replay_offset)
    try:
        yield _TLS.session
    finally:
        _TLS.session = prev


def apply_route_replay(
    gate: torch.nn.Module,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hook called by a MoE layer right after its router produced (weights, idx).

    - No active session: returns (topk_weights, topk_idx) unchanged (inert).
    - record: captures topk_idx, returns inputs unchanged.
    - replay: replaces topk_idx with the recorded experts, and RECOMPUTES the
      gating weights on those experts from the router's current logits so the
      router keeps its gradient.

    Args:
        gate: the router module. Must expose ``gate.get_logits(hidden_states)``
            OR be callable to obtain full per-expert logits; see
            :func:`_router_logits`.
        hidden_states: [tokens, hidden] or [b, s, hidden] router input.
        topk_weights: [tokens, top_k] gating weights from the normal router call.
        topk_idx: [tokens, top_k] selected expert indices from the normal call.

    Returns:
        (weights, idx) to feed the expert compute. In replay mode ``idx`` is the
        recorded selection and ``weights`` is recomputed & differentiable wrt the
        router; in record mode both are the originals (and idx is captured).
    """
    sess = current_session()
    if sess is None:
        return topk_weights, topk_idx

    if sess.mode == "record":
        sess.capture(topk_idx)
        return topk_weights, topk_idx

    # replay
    forced_idx = sess.next_recorded().to(topk_idx.device)  # [rows, top_k]
    n_live = topk_idx.shape[0]
    n_rec = forced_idx.shape[0]

    if sess.replay_offset is None:
        # full replay: recorded routing must cover ALL tokens exactly.
        if forced_idx.shape != topk_idx.shape:
            raise ValueError(
                f"route-replay: recorded routing {tuple(forced_idx.shape)} != "
                f"live routing {tuple(topk_idx.shape)} (token/topk misalignment); "
                f"pass replay_offset= for intentional partial (prompt-passthrough) replay"
            )
        merged_idx = forced_idx
    else:
        # partial replay: recorded ``rows`` correspond to the LAST rows tokens
        # (response); the leading offset tokens (prompt) keep live routing.
        offset = sess.replay_offset
        if offset < 0 or offset + n_rec != n_live or forced_idx.shape[1:] != topk_idx.shape[1:]:
            raise ValueError(
                f"route-replay: partial-replay misalignment — live {tuple(topk_idx.shape)}, "
                f"recorded {tuple(forced_idx.shape)}, offset {offset} "
                f"(require offset>=0 and offset+rows==tokens and matching top_k)"
            )
        merged_idx = torch.cat([topk_idx[:offset], forced_idx], dim=0)

    weights = _recompute_gating_weights(gate, hidden_states, merged_idx, ref=topk_weights)
    return weights, merged_idx


def _router_logits(gate: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Full per-expert router logits [tokens, n_experts], differentiable.

    HunyuanTopKGate computes ``logits = self.wg(hidden_states)`` where ``wg`` is
    an fp32 ``nn.Linear`` (modeling_hunyuan_image_3.py:1111). We need these
    pre-topk logits to reproduce ``easy_topk`` on the frozen experts. Probe order:
      1) gate.get_logits(hidden_states)     (explicit accessor, if added)
      2) gate.wg(hidden_states)             (HunyuanTopKGate's router linear)
      3) gate.weight linear / gate.gate     (generic fallbacks)
    Kept defensive + fail-loud: a wrong path must raise, not silently mis-route.
    """
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    for attr in ("get_logits", "router_logits", "compute_logits"):
        fn = getattr(gate, attr, None)
        if callable(fn):
            return fn(flat)
    # HunyuanTopKGate.wg is the fp32 router linear (matches modeling exactly).
    wg = getattr(gate, "wg", None)
    if isinstance(wg, torch.nn.Module):
        return wg(flat.float() if getattr(getattr(wg, "weight", None), "dtype", None) == torch.float32 else flat)
    # a plain nn.Linear-style router: gate.weight [n_experts, hidden]
    w = getattr(gate, "weight", None)
    if isinstance(w, torch.Tensor) and w.dim() == 2:
        return torch.nn.functional.linear(flat.to(w.dtype), w)
    inner = getattr(gate, "gate", None)
    if isinstance(inner, torch.nn.Module):
        return inner(flat)
    raise RuntimeError(
        "route-replay: cannot obtain pre-topk router logits from gate of type "
        f"{type(gate).__name__}; add a get_logits() accessor or extend _router_logits()."
    )


def _recompute_gating_weights(
    gate: torch.nn.Module,
    hidden_states: torch.Tensor,
    forced_idx: torch.Tensor,
    ref: torch.Tensor,
) -> torch.Tensor:
    """g_e on the frozen experts, reproducing HunyuanTopKGate.easy_topk EXACTLY,
    but from the router's CURRENT logits (differentiable) so router keeps grad.

    Official easy_topk (modeling_hunyuan_image_3.py:1132):
        gates = softmax(logits, dim=-1)          # over ALL experts
        w1, idx = topk(gates, moe_topk)
        w = w1 / clamp(w1.sum(-1, keepdim=True), min=1e-8)
    So the gating weight of a frozen expert e is:
        softmax(logits)[e], then renormalized over the frozen top_k.
    NOTE (route-replay pitfall #4): softmax is over ALL experts THEN gather —
    NOT softmax over the selected logits. Getting this wrong makes g mismatch
    even when the expert selection is correct.
    """
    logits = _router_logits(gate, hidden_states)          # [tokens, n_experts], grad-enabled
    gates = torch.softmax(logits, dim=-1)                 # over ALL experts (official)
    sel = torch.gather(gates, dim=-1, index=forced_idx.to(logits.device))  # [tokens, top_k]
    weight_sums = torch.clamp(sel.sum(dim=-1, keepdim=True), min=1e-8)
    weights = sel / weight_sums                           # renormalize over frozen top_k
    return weights.to(ref.dtype)


__all__ = [
    "route_replay_session",
    "current_session",
    "apply_route_replay",
]
