#!/usr/bin/env python3
"""Render the immutable single-node P2 OFF/ON/ON/OFF plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from harness_lib import (
    ARM_ORDER,
    DP_SIZE,
    EXPECTED_PLANNER_TARGET,
    FUNCTIONAL_OVERRIDES,
    PREFETCH_CAPACITY,
    RECIPE,
    SP_SIZE,
    WORLD_SIZE,
    arm_specs,
    sha256_file,
    validate_p0_contract,
    write_json_atomic,
)


def render(campaign_id: str, node: str, p0_contract_path: Path) -> dict:
    p0 = validate_p0_contract(p0_contract_path)
    harness_root = Path(__file__).resolve().parent
    adapter_module = harness_root / "runtime_overlay" / "p2_runtime_receipts.py"
    sitecustomize = harness_root / "runtime_overlay" / "sitecustomize.py"
    if not adapter_module.is_file() or not sitecustomize.is_file():
        raise FileNotFoundError("P2 runtime overlay adapter is incomplete")
    prompt = p0["prompt_manifest"]
    frozen_checkpoint = p0["frozen_checkpoint"]
    arms = []
    for arm in arm_specs(campaign_id):
        overrides = dict(FUNCTIONAL_OVERRIDES)
        overrides["bundle.config.prompt_embedding_prefetch"] = arm["prefetch"]
        overrides["data_source.args.run.data_path"] = prompt["path"]
        overrides["load_dir"] = str(frozen_checkpoint["load_dir"])
        arms.append({**arm, "node": node, "overrides": overrides})
    return {
        "schema": "unirl-minimax-h3-p2-prefetch-abba-plan-v1",
        "campaign_id": campaign_id,
        "source_commit": p0["p2"]["commit"],
        "source_tree": p0["p2"]["tree"],
        "source_parent_commit": p0["pr38"]["commit"],
        "source_parent_tree": p0["pr38"]["tree"],
        "source_patch_sha256": p0["p2"]["patch_sha256"],
        "integration_source": {
            "commit": p0["source"]["integration_commit"],
            "tree": p0["source"]["integration_tree"],
            "full_tree_sha256": p0["source"]["full_tree_sha256"],
            "archive": p0["source_archive"],
            "tree_manifest": p0["source_tree_manifest"],
        },
        "node": node,
        "world_size": WORLD_SIZE,
        "sp_size": SP_SIZE,
        "dp_size": DP_SIZE,
        "order": [treatment for _, treatment in ARM_ORDER],
        "recipe": RECIPE,
        "p0_contract": {"path": p0["path"], "sha256": p0["sha256"]},
        "p1_source_preflight": {
            "path": p0["p1_source_preflight_binding"]["path"],
            "sha256": p0["p1_source_preflight_binding"]["sha256"],
            "schema": p0["p1_source_preflight"]["schema"],
        },
        "planner_target": EXPECTED_PLANNER_TARGET,
        "planner": {
            "contract_sha256": p0["planner_contract_sha256"],
            "runtime_class": p0["planner"]["runtime_class"],
            "implementation_commit": p0["planner"]["implementation_commit"],
            "implementation_tree": p0["planner"]["implementation_tree"],
        },
        "two_update": {
            "contract_sha256": p0["two_update_contract_sha256"],
            "num_updates": p0["two_update"]["num_updates"],
        },
        "prompt_manifest": {
            "path": prompt["path"],
            "sha256": prompt["sha256"],
            "prompt_ids": prompt["prompt_ids"],
        },
        "frozen_checkpoint": {
            "load_dir": frozen_checkpoint["load_dir"],
            "audit_schema": frozen_checkpoint["audit_payload"]["schema"],
            "audit_sha256": frozen_checkpoint["audit_binding"]["sha256"],
            "tree_sha256": frozen_checkpoint["tree_sha256"],
            "model_summary_sha256": frozen_checkpoint["model_summary_sha256"],
            "adapter_key_signature_sha256": frozen_checkpoint["adapter_key_signature_sha256"],
        },
        "runtime_overlay": {
            "event_schema": p0["runtime_overlay"]["event_schema"],
            "base_module_path": p0["runtime_overlay"]["module_binding"]["path"],
            "base_module_sha256": p0["runtime_overlay"]["module_binding"]["sha256"],
            "adapter_module_path": str(adapter_module),
            "adapter_module_sha256": sha256_file(adapter_module),
            "sitecustomize_path": str(sitecustomize),
            "sitecustomize_sha256": sha256_file(sitecustomize),
        },
        "fixed_contract": {
            "canonical_real_reward": "0.5*VideoPickScore(middle)+0.5*CLAP",
            "world_size": WORLD_SIZE,
            "sp_size": SP_SIZE,
            "dp_size": DP_SIZE,
            "prompt_embedding_share_across_sp": True,
            "prompt_embedding_prefetch_capacity": PREFETCH_CAPACITY,
            "persistent_disk_cache": False,
            "text_encoder_onload_for_embed": False,
            "fresh_process_per_arm": True,
            "strict_seed_prompt_lora_identity": True,
            "functional_difference": "bundle.config.prompt_embedding_prefetch only",
        },
        "arms": arms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--p0-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing existing output: {args.output}")
    write_json_atomic(args.output, render(args.campaign_id, args.node, args.p0_contract))
    print(args.output)


if __name__ == "__main__":
    main()
