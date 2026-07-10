"""Verify the native-gate route-replay monkey-patch on the REAL HunyuanTopKGate.

Proves the non-EP path (patch HunyuanTopKGate.forward) gives a full closed loop:
  P1  install() patches HunyuanTopKGate.forward (idempotent).
  P2  record captures routing; a router perturbation flips experts WITHOUT replay.
  P3  replaying the recorded routing forces the recorded experts through the
      PATCHED gate (the real call path a HunyuanMoE.forward would hit), and the
      recomputed g is normalized; router keeps gradient.

Run (node, unirl venv):
  PYTHONPATH=<worktree> HI3=/root/sync/models/HunyuanImage-3-Instruct \
    python unirl/train/backend/veomni/ep/test_hunyuan_moe_route_patch.py
"""

from __future__ import annotations

import glob
import os

import torch


def _load_gate_cls_and_weight():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    HI3 = os.environ.get("HI3", "/root/sync/models/HunyuanImage-3-Instruct")
    Gate = get_class_from_dynamic_module("modeling_hunyuan_image_3.HunyuanTopKGate", HI3)
    from safetensors import safe_open

    w = None
    for st in sorted(glob.glob(os.path.join(HI3, "model-*.safetensors"))):
        with safe_open(st, framework="pt") as f:
            for k in f.keys():
                if k.endswith("mlp.gate.wg.weight") and "layers.0." in k:
                    w = f.get_tensor(k)
                    break
        if w is not None:
            break
    return Gate, w, HI3


def _build_gate(Gate, w, HI3):
    import json

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    Cfg = get_class_from_dynamic_module(
        "configuration_hunyuan_image_3.HunyuanImage3Config", HI3
    )
    with open(os.path.join(HI3, "config.json")) as f:
        cfg = Cfg(**json.load(f))
    gate = Gate(cfg, layer_idx=0).cuda().eval()
    if w is not None:
        with torch.no_grad():
            gate.wg.weight.copy_(w.to(gate.wg.weight.dtype).cuda())
    return gate


def main():
    from unirl.train.backend.veomni.ep.hunyuan_moe_route_patch import install
    from unirl.train.backend.veomni.ep.route_replay import route_replay_session

    Gate, w, HI3 = _load_gate_cls_and_weight()

    # P1: install patches the class
    ok = install(gate_cls=Gate)
    install(gate_cls=Gate)  # idempotent
    p1 = ok and getattr(Gate.forward, "_route_replay", False)
    print(f"P1 install patches HunyuanTopKGate.forward: {'PASS' if p1 else 'FAIL'}")

    gate = _build_gate(Gate, w, HI3)
    hidden = (torch.randn(1, 48, gate.wg.weight.shape[1]) * 0.5).cuda()

    # P2: record via patched gate; perturb; confirm flips without replay
    with route_replay_session(mode="record") as rec:
        w_rec, idx_rec = gate(hidden, topk_impl="easy")   # patched -> records
    routing = rec.stack()
    with torch.no_grad():
        gate.wg.weight.add_(torch.randn_like(gate.wg.weight) * 0.02)
        w_live, idx_live = gate(hidden, topk_impl="easy")  # no session -> normal
    flips = (idx_rec != idx_live).any(dim=-1).sum().item()
    p2 = routing is not None and flips > 0
    print(f"P2 record+capture, perturbation flips (no replay): "
          f"{'PASS' if p2 else 'FAIL'} (flips={flips}/{hidden.shape[1]})")

    # P3: replay forces recorded experts through the PATCHED gate
    with route_replay_session(mode="replay", routing=routing):
        w_rep, idx_rep = gate(hidden, topk_impl="easy")
    forced = torch.equal(idx_rep, idx_rec)
    gnorm = torch.allclose(w_rep.float().sum(-1), torch.ones(hidden.shape[1]).cuda(), atol=1e-4)
    print(f"P3 replay forces recorded experts (patched path): {'PASS' if forced else 'FAIL'}")
    print(f"P3 replay g normalized: {'PASS' if gnorm else 'FAIL'}")

    # P3b: router keeps gradient
    gate.zero_grad(set_to_none=True)
    with route_replay_session(mode="replay", routing=routing):
        w_rep, _ = gate(hidden, topk_impl="easy")
        loss = w_rep.pow(2).mean()
    loss.backward()
    grad_ok = gate.wg.weight.grad is not None and gate.wg.weight.grad.abs().sum().item() > 0
    print(f"P3b router keeps gradient: {'PASS' if grad_ok else 'FAIL'} "
          f"(|wg.grad|={gate.wg.weight.grad.abs().sum().item():.3e})")

    all_ok = p1 and p2 and forced and gnorm and grad_ok
    print(f"\n{'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
