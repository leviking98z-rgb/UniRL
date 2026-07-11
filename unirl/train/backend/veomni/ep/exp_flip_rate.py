"""Controlled experiment: MoE expert-flip rate vs rollout/replay divergence,
on the REAL HunyuanImage3 router — and route-replay's effect.

Question this answers: "route-replay 前后 expert 正确率(翻转率)是多少?"

Setup: real HunyuanTopKGate (layer-0 wg weights, 64 experts, top_k from config).
Feed a batch of hidden states. "rollout" = reference routing under router W_ref.
"replay" = routing recomputed under a perturbed router (models weight drift /
precision/kernel difference between the rollout engine and the train forward).

We measure, per scenario:
  - flip_rate      = fraction of tokens whose top_k expert SET differs
  - top1_flip      = fraction whose #1 expert differs
  - logp_proxy_diff= |log g_chosen(replay) - log g_chosen(rollout)| mean/max
                     (a stand-in for the per-token logp divergence a flip causes)
Then WITH route-replay: force rollout's experts on the replay side and recompute
g from the perturbed router -> flip_rate := 0 by construction; report the
resulting logp_proxy_diff (should collapse toward the no-perturb floor).

Scenarios:
  S0  no perturbation                 (ideal floor)
  S1  weight drift eps=0.005,0.01,0.02 (off-policy staleness / small kernel diff)
  S2  bf16 vs fp8-ish rounding of router input (precision path difference)
  S3  route-replay ON (force rollout experts) -> flip=0, measure logp recovery

Usage (node, unirl venv):
  PYTHONPATH=<worktree> HI3=/root/sync/models/HunyuanImage-3-Instruct \
    python unirl/train/backend/veomni/ep/exp_flip_rate.py
"""

from __future__ import annotations

import glob
import json
import os

import torch
import torch.nn.functional as F

HI3 = os.environ.get("HI3", "/root/sync/models/HunyuanImage-3-Instruct")


def load_gate():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from safetensors import safe_open

    Gate = get_class_from_dynamic_module("modeling_hunyuan_image_3.HunyuanTopKGate", HI3)
    Cfg = get_class_from_dynamic_module("configuration_hunyuan_image_3.HunyuanImage3Config", HI3)
    with open(os.path.join(HI3, "config.json")) as f:
        cfg = Cfg(**json.load(f))
    gate = Gate(cfg, layer_idx=0).cuda().eval()
    w = None
    for st in sorted(glob.glob(os.path.join(HI3, "model-*.safetensors"))):
        with safe_open(st, framework="pt") as f:
            for k in f.keys():
                if k.endswith("mlp.gate.wg.weight") and "layers.0." in k:
                    w = f.get_tensor(k)
                    break
        if w is not None:
            break
    with torch.no_grad():
        gate.wg.weight.copy_(w.to(gate.wg.weight.dtype).cuda())
    topk = gate.moe_topk if isinstance(gate.moe_topk, int) else gate.moe_topk[0]
    return gate, int(topk), w.shape


def easy_topk_from_logits(logits, k):
    """Official easy_topk math: softmax over ALL, top_k, renorm."""
    gates = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(gates, k, dim=-1)
    w = vals / vals.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return w, idx


def g_of_chosen(logits, idx):
    """Renormalized gate weight of the chosen experts (for logp proxy)."""
    gates = torch.softmax(logits, dim=-1)
    sel = torch.gather(gates, -1, idx)
    return sel / sel.sum(-1, keepdim=True).clamp(min=1e-8)


def flip_stats(idx_ref, idx_new):
    # set-level flip: any expert in the top_k set differs
    ref_sorted = idx_ref.sort(dim=-1).values
    new_sorted = idx_new.sort(dim=-1).values
    set_flip = (ref_sorted != new_sorted).any(dim=-1).float().mean().item()
    top1_flip = (idx_ref[:, 0] != idx_new[:, 0]).float().mean().item()
    return set_flip, top1_flip


def main():
    torch.manual_seed(0)
    gate, k, wshape = load_gate()
    hidden = wshape[1]
    n_tok = 4096
    # hidden states with a realistic scale (router input is pre-norm-ish); use a
    # spread so a fraction of tokens sit near routing boundaries (as in real seqs).
    x = (torch.randn(n_tok, hidden) * 0.5).cuda()

    with torch.no_grad():
        logits_ref = gate.wg(x.float())          # rollout reference router
        w_ref, idx_ref = easy_topk_from_logits(logits_ref, k)
        logg_ref = torch.log(g_of_chosen(logits_ref, idx_ref).clamp(min=1e-9))

    W0 = gate.wg.weight.detach().clone()
    rows = []

    # Lightweight expert bank (random but FIXED) so we can measure the REAL harm
    # of a flip: routing to a different expert => a materially different FFN
    # output. This is what route-replay actually prevents; the g-only proxy
    # misses it. We use a small per-expert linear as a stand-in for FFN_e
    # (real 80B expert weights aren't loaded here — the POINT is that different
    # experts = different transforms, which holds for any non-degenerate bank).
    torch.manual_seed(123)
    n_exp = wshape[0]
    inter = 256
    E_w = (torch.randn(n_exp, inter, hidden, device="cuda") * (hidden ** -0.5))

    def moe_out(x_in, idx, g):
        """Σ_k g_k · FFN_{idx_k}(x): FFN_e(x)=relu(x·E_w[e]^T)·... (stand-in)."""
        out = torch.zeros(x_in.shape[0], inter, device="cuda")
        for kk in range(idx.shape[-1]):
            e = idx[:, kk]                          # [tok]
            h = torch.relu(torch.einsum("th,tih->ti", x_in, E_w[e]))
            out = out + g[:, kk : kk + 1] * h
        return out

    with torch.no_grad():
        out_ref = moe_out(x, idx_ref, w_ref)        # rollout's true MoE output

    def eval_perturbed(label, logits_new):
        w_new, idx_new = easy_topk_from_logits(logits_new, k)
        set_flip, top1_flip = flip_stats(idx_ref, idx_new)
        # NO route-replay: flipped experts + perturbed g -> DIFFERENT FFNs
        logg_new = torch.log(g_of_chosen(logits_new, idx_new).clamp(min=1e-9))
        d_g_noreplay = (logg_new - logg_ref).abs()
        out_noreplay = moe_out(x, idx_new, w_new)
        d_out_noreplay = (out_noreplay - out_ref).abs().mean(-1)   # per-token output diff
        # WITH route-replay: force rollout's experts, recompute g from perturbed router
        w_forced = g_of_chosen(logits_new, idx_ref)
        logg_replay = torch.log(w_forced.clamp(min=1e-9))
        d_g_replay = (logg_replay - logg_ref).abs()
        out_replay = moe_out(x, idx_ref, w_forced)                 # SAME experts as ref
        d_out_replay = (out_replay - out_ref).abs().mean(-1)
        rows.append({
            "scenario": label,
            "set_flip_%": round(100 * set_flip, 2),
            "top1_flip_%": round(100 * top1_flip, 2),
            "OUT_diff_noreplay_mean": round(d_out_noreplay.mean().item(), 5),
            "OUT_diff_noreplay_max": round(d_out_noreplay.max().item(), 4),
            "OUT_diff_REPLAY_mean": round(d_out_replay.mean().item(), 6),
            "OUT_diff_REPLAY_max": round(d_out_replay.max().item(), 5),
            "g_diff_noreplay_mean": round(d_g_noreplay.mean().item(), 5),
        })

    # S0: no perturbation
    with torch.no_grad():
        eval_perturbed("S0_no_perturb", gate.wg(x.float()))

    # S1: weight drift (off-policy / kernel diff)
    for eps in (0.005, 0.01, 0.02):
        with torch.no_grad():
            gate.wg.weight.copy_(W0 + torch.randn_like(W0) * eps)
            eval_perturbed(f"S1_drift_eps{eps}", gate.wg(x.float()))
            gate.wg.weight.copy_(W0)

    # S2: precision path diff — round router INPUT to bf16 (rollout) vs fp32 (replay)
    with torch.no_grad():
        gate.wg.weight.copy_(W0)
        x_bf16 = x.to(torch.bfloat16).float()
        eval_perturbed("S2_bf16_input", gate.wg(x_bf16))

    # report
    hdr = ["scenario", "set_flip_%", "top1_flip_%",
           "OUT_diff_noreplay_mean", "OUT_diff_noreplay_max",
           "OUT_diff_REPLAY_mean", "OUT_diff_REPLAY_max"]
    print("\n=== MoE expert-flip controlled experiment (real HI3 layer-0 router, "
          f"{n_tok} tokens, top_k={k}, {wshape[0]} experts) ===")
    print("OUT_diff = per-token |MoE output(perturbed) - MoE output(rollout)|, the REAL harm "
          "(different experts = different FFN). REPLAY forces rollout's experts.")
    print(" | ".join(f"{h}" for h in hdr))
    for r in rows:
        print(" | ".join(f"{r[h]}" for h in hdr))
    print("\nKEY: no-replay OUT_diff is large & grows with flip rate (tokens run WRONG "
          "FFNs); route-replay forces rollout's expert set so OUT_diff collapses toward 0 "
          "(only the tiny g-recompute residual remains). This is exactly the logp divergence "
          "route-replay removes.")
    # also dump json for the record
    out = "/root/shared/.clusters/.tmp/flip_rate_results.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
