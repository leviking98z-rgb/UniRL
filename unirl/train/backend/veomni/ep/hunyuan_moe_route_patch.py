"""Train-side route-replay for the NATIVE HunyuanImage3 gate (non-EP path).

The EP path swaps HunyuanMoE -> FusedHunyuanMoE (route-replay hook inline). The
non-EP path (FSDPBackend, ``hi3_vllmomni.yaml``) keeps the HF trust_remote_code
``HunyuanTopKGate``, whose ``forward`` we can't edit in place (read-only under
the checkpoint dir). So monkey-patch ``HunyuanTopKGate.forward``: pass its
``(topk_weight, expert_index)`` output through the same ``apply_route_replay``
hook FusedHunyuanMoE uses. Patching the GATE (the single routing producer)
covers both HunyuanMoE forward branches (flashinfer / deepseek) with one wrap.

Inert unless a route_replay session (record/replay) is active.

Install once on the train workers after the model is built (the class is
imported by then)::

    from unirl.train.backend.veomni.ep.hunyuan_moe_route_patch import install
    install()                    # finds HunyuanTopKGate in sys.modules
    install(gate_cls=SomeGate)   # or pass the class explicitly
"""

from __future__ import annotations

_INSTALLED = False


def _find_gate_cls():
    import sys

    for name, mod in list(sys.modules.items()):
        if "hunyuan_image_3" in name and hasattr(mod, "HunyuanTopKGate"):
            return getattr(mod, "HunyuanTopKGate")
    return None


def install(gate_cls=None) -> bool:
    """Patch ``HunyuanTopKGate.forward`` to route through apply_route_replay.

    Returns True if patched (or already), False if the class isn't found.
    Idempotent.
    """
    global _INSTALLED
    if gate_cls is None:
        gate_cls = _find_gate_cls()
    if gate_cls is None:
        return False
    if getattr(gate_cls.forward, "_route_replay", False):
        _INSTALLED = True
        return True

    from unirl.train.backend.veomni.ep.route_replay import apply_route_replay

    orig_forward = gate_cls.forward

    def forward(self, hidden_states, topk_impl="default"):  # noqa: ANN001
        weights, idx = orig_forward(self, hidden_states, topk_impl=topk_impl)
        # apply_route_replay wants [tokens, hidden] + [tokens, top_k]; the gate
        # already flattened hidden internally, and weights/idx are [tokens, top_k].
        hs_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        return apply_route_replay(self, hs_flat, weights, idx)

    forward._route_replay = True  # type: ignore[attr-defined]
    gate_cls.forward = forward
    _INSTALLED = True
    return True


__all__ = ["install"]
