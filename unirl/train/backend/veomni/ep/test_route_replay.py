"""CPU unit tests for MoE route-replay (unirl.train.backend.veomni.ep.route_replay).

Self-contained: a tiny MoE that mimics HunyuanMoE's gate contract
(``gate(hidden, topk_impl=...) -> (topk_weights, topk_idx)`` with
softmax-over-selected), so we can verify the mechanism WITHOUT veomni / GPU /
HI3 weights. Run: python -m pytest unirl/train/backend/veomni/ep/test_route_replay.py
or just `python <thisfile>` (has a __main__ runner).

What it proves:
  T1  record captures per-layer top_k indices in visit order.
  T2  WITHOUT replay, a tiny router perturbation FLIPS routing (the bug).
  T3  WITH replay, the same perturbation does NOT flip (forced experts) and the
      per-token output matches the pre-perturbation forward that used those
      experts -> logp aligns.
  T4  replay recomputes gating weights from the CURRENT router -> router.weight
      has NON-ZERO grad (router keeps learning); frozen selection carries none.
  T5  inert when no session active (normal training path unchanged).
  T6  shape-mismatch (token/topk misalignment) fails loud.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from unirl.train.backend.veomni.ep.route_replay import (
    apply_route_replay,
    current_session,
    route_replay_session,
)


class TinyGate(nn.Module):
    """Mimics HunyuanMoE's router: linear logits -> top_k -> softmax-over-selected.

    Exposes get_logits() so route_replay._router_logits finds the differentiable
    pre-topk logits (the accessor path we recommend for real HunyuanMoE)."""

    def __init__(self, hidden: int, n_experts: int, top_k: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_experts, hidden) * 0.02)
        self.top_k = top_k

    def get_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        return F.linear(flat, self.weight)

    def forward(self, hidden_states: torch.Tensor, topk_impl: str = "easy"):
        logits = self.get_logits(hidden_states)
        top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(top_vals, dim=-1)
        return weights, top_idx


class TinyMoE(nn.Module):
    """A minimal MoE that calls the SAME route-replay hook FusedHunyuanMoE calls."""

    def __init__(self, hidden: int, n_experts: int, top_k: int, inter: int = 8):
        super().__init__()
        self.gate = TinyGate(hidden, n_experts, top_k)
        self.n_experts = n_experts
        # per-expert 2-layer FFN
        self.w1 = nn.Parameter(torch.randn(n_experts, inter, hidden) * 0.1)
        self.w2 = nn.Parameter(torch.randn(n_experts, hidden, inter) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [tokens, hidden]
        weights, idx = self.gate(x, topk_impl="easy")
        # ---- the exact hook FusedHunyuanMoE.forward uses ----
        weights, idx = apply_route_replay(self.gate, x, weights, idx)
        weights = weights.to(x.dtype)
        # dense expert compute (kernel-free, CPU): sum_k g_k * FFN_{idx_k}(x)
        out = torch.zeros_like(x)
        for k in range(idx.shape[-1]):
            e = idx[:, k]                             # [tokens]
            w1 = self.w1[e]                           # [tokens, inter, hidden]
            w2 = self.w2[e]                           # [tokens, hidden, inter]
            h = torch.relu(torch.einsum("th,tih->ti", x, w1))
            y = torch.einsum("ti,thi->th", h, w2)
            out = out + weights[:, k : k + 1] * y
        return out


def _mk(seed=0, hidden=16, n_experts=8, top_k=2, tokens=32):
    torch.manual_seed(seed)
    moe = TinyMoE(hidden, n_experts, top_k)
    x = torch.randn(tokens, hidden)
    return moe, x


def test_T1_record_captures():
    moe, x = _mk()
    with route_replay_session(mode="record") as rec:
        moe(x)
        moe(x)  # second layer visit
    routing = rec.stack()
    assert routing is not None and routing.shape[0] == 2, "should capture 2 layer visits"
    assert routing.shape[1:] == (x.shape[0], moe.gate.top_k)
    print("T1 record captures: PASS", tuple(routing.shape))


def test_T2_without_replay_flips():
    """A tiny router perturbation flips routing when NOT replayed (the bug)."""
    moe, x = _mk(seed=1)
    with torch.no_grad():
        _, idx0 = moe.gate(x)
    # perturb router minimally
    with torch.no_grad():
        moe.gate.weight.add_(torch.randn_like(moe.gate.weight) * 0.05)
        _, idx1 = moe.gate(x)
    flips = (idx0 != idx1).any(dim=-1).sum().item()
    assert flips > 0, "expected some tokens to flip experts under perturbation"
    print(f"T2 without replay flips: PASS ({flips}/{x.shape[0]} tokens flipped)")


def test_T3_replay_prevents_flip_and_aligns_output():
    moe, x = _mk(seed=1)
    # record routing at the ORIGINAL router (the "rollout")
    with route_replay_session(mode="record") as rec:
        out_rollout = moe(x)
    routing = rec.stack()
    with torch.no_grad():
        _, idx_recorded = moe.gate(x)

    # perturb the router (fp8-vs-bf16 style drift)
    with torch.no_grad():
        moe.gate.weight.add_(torch.randn_like(moe.gate.weight) * 0.05)

    # WITHOUT replay: routing flips, output diverges
    with torch.no_grad():
        _, idx_live = moe.gate(x)
        out_noreplay = moe(x)
    flips = (idx_recorded != idx_live).any(dim=-1).sum().item()

    # WITH replay: forced to recorded experts (single forward -> single layer visit)
    with route_replay_session(mode="replay", routing=routing):
        out_replay = moe(x)
    # separately confirm the hook forces the recorded idx (fresh 1-visit session)
    with route_replay_session(mode="replay", routing=routing):
        w_f, i_f = apply_route_replay(moe.gate, x, *moe.gate(x))
    assert torch.equal(i_f, idx_recorded), "replay must force recorded experts"
    # replay output must be closer to rollout than the flipped no-replay output
    d_replay = (out_replay - out_rollout).abs().mean().item()
    d_noreplay = (out_noreplay - out_rollout).abs().mean().item()
    assert flips > 0, "test setup: perturbation should have caused flips"
    assert d_replay < d_noreplay, f"replay ({d_replay:.4e}) should beat no-replay ({d_noreplay:.4e})"
    print(f"T3 replay prevents flip: PASS (flips={flips}, "
          f"d_replay={d_replay:.2e} < d_noreplay={d_noreplay:.2e})")


def test_T4_router_keeps_gradient():
    """Replay recomputes g from current router -> router.weight.grad != 0."""
    moe, x = _mk(seed=2)
    with route_replay_session(mode="record") as rec:
        moe(x)
    routing = rec.stack()
    moe.zero_grad(set_to_none=True)
    with route_replay_session(mode="replay", routing=routing):
        out = moe(x)
        loss = out.pow(2).mean()
    loss.backward()
    g = moe.gate.weight.grad
    assert g is not None and g.abs().sum().item() > 0, "router MUST keep gradient under replay"
    assert moe.w1.grad is not None and moe.w1.grad.abs().sum().item() > 0, "experts must have grad"
    print(f"T4 router keeps gradient: PASS (|router.grad|={g.abs().sum().item():.3e}, "
          f"|expert.grad|={moe.w1.grad.abs().sum().item():.3e})")


def test_T5_inert_without_session():
    moe, x = _mk(seed=3)
    assert current_session() is None
    with torch.no_grad():
        w0, i0 = moe.gate(x)
        w1, i1 = apply_route_replay(moe.gate, x, w0, i0)
    assert torch.equal(i0, i1) and torch.equal(w0, w1), "must be identity with no session"
    print("T5 inert without session: PASS")


def test_T6_shape_mismatch_loud():
    moe, x = _mk(seed=4)
    with route_replay_session(mode="record") as rec:
        moe(x)
    routing = rec.stack()
    bad = routing[:, : x.shape[0] // 2, :]  # wrong token count
    raised = False
    try:
        with route_replay_session(mode="replay", routing=bad):
            moe(x)
    except ValueError as e:
        raised = "misalignment" in str(e) or "!=" in str(e)
    assert raised, "shape mismatch must fail loud"
    print("T6 shape mismatch loud: PASS")


def test_T7_partial_replay_prompt_passthrough():
    """Partial replay: recorded routing covers only the LAST `rows` tokens
    (response); leading `offset` tokens (prompt) keep live routing."""
    moe, x = _mk(seed=5, tokens=20)
    pl, rl = 12, 8  # prompt 12 + response 8 = 20
    # record routing for the FULL sequence at original router
    with route_replay_session(mode="record") as rec:
        moe(x)
    full_routing = rec.stack()  # [1, 20, top_k]
    resp_routing = full_routing[:, pl:, :]  # [1, 8, top_k] response only
    with torch.no_grad():
        _, idx_orig = moe.gate(x)

    # perturb router -> would flip
    with torch.no_grad():
        moe.gate.weight.add_(torch.randn_like(moe.gate.weight) * 0.08)
        _, idx_live = moe.gate(x)

    # partial replay with offset=pl
    with route_replay_session(mode="replay", routing=resp_routing, replay_offset=pl):
        w_m, idx_merged = apply_route_replay(moe.gate, x, *moe.gate(x))

    # response region forced to recorded; prompt region = live (flipped)
    assert torch.equal(idx_merged[pl:], idx_orig[pl:]), "response must use recorded experts"
    assert torch.equal(idx_merged[:pl], idx_live[:pl]), "prompt must keep live routing"
    print(f"T7 partial replay passthrough: PASS (prompt {pl} live, response {rl} forced)")


def test_T8_partial_replay_offset_mismatch_loud():
    moe, x = _mk(seed=6, tokens=20)
    with route_replay_session(mode="record") as rec:
        moe(x)
    resp = rec.stack()[:, 12:, :]  # 8 rows
    raised = False
    try:
        # wrong offset (should be 12, give 5): 5 + 8 != 20
        with route_replay_session(mode="replay", routing=resp, replay_offset=5):
            apply_route_replay(moe.gate, x, *moe.gate(x))
    except ValueError as e:
        raised = "misalignment" in str(e)
    assert raised, "offset mismatch must fail loud"
    print("T8 partial replay offset mismatch loud: PASS")


ALL = [
    test_T1_record_captures,
    test_T2_without_replay_flips,
    test_T3_replay_prevents_flip_and_aligns_output,
    test_T4_router_keeps_gradient,
    test_T5_inert_without_session,
    test_T6_shape_mismatch_loud,
    test_T7_partial_replay_prompt_passthrough,
    test_T8_partial_replay_offset_mismatch_loud,
]

if __name__ == "__main__":
    fails = 0
    for t in ALL:
        try:
            t()
        except Exception as e:  # noqa
            fails += 1
            print(f"{t.__name__}: FAIL -> {type(e).__name__}: {e}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'} ({len(ALL) - fails}/{len(ALL)})")
    raise SystemExit(1 if fails else 0)
