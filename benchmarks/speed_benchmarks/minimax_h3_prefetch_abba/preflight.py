#!/usr/bin/env python3
"""CPU-only fail-closed preflight for the P2 canonical AB/BA campaign."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from harness_lib import (
    DP_SIZE,
    EXPECTED_PLANNER_TARGET,
    EXPECTED_REMOTE,
    PREFETCH_CAPACITY,
    RECIPE_PATH,
    SP_SIZE,
    WORLD_SIZE,
    GateError,
    arm_specs,
    functional_overrides,
    load_json,
    nested,
    sha256_file,
    validate_p0_contract,
    validate_source,
    write_json_atomic,
)

try:
    import yaml
except ImportError:  # pragma: no cover - deployment environment must provide PyYAML
    yaml = None


def recipe_gate(repo: Path) -> dict:
    if yaml is None:
        raise GateError("PyYAML is required for preflight")
    path = repo / RECIPE_PATH
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "num_devices": 8,
        "batch_size": 8,
        "height": 768,
        "width": 768,
        "num_frames": 124,
        "num_inference_steps": 10,
        "sde_indices": [0, 3, 6],
        "eta": 0.7,
        "samples_per_prompt": 4,
        "num_updates_per_batch": 2,
        "micro_batch_size": 1,
        "reward_target": "unirl.reward.local.t2av_composite.T2AVCompositeScorer",
        "reward_weights": {"videopickscore": 0.5, "clap": 0.5},
        "cache_dir": None,
        "cache_read_only": False,
        "prefetch_default": False,
        "prefetch_capacity": 8,
    }
    observed = {
        "num_devices": nested(payload, "num_devices"),
        "batch_size": nested(payload, "batch_size"),
        "height": nested(payload, "sampling.height"),
        "width": nested(payload, "sampling.width"),
        "num_frames": nested(payload, "sampling.num_frames"),
        "num_inference_steps": nested(payload, "sampling.num_inference_steps"),
        "sde_indices": nested(payload, "sampling.sde_indices"),
        "eta": nested(payload, "sampling.eta"),
        "samples_per_prompt": nested(payload, "sampling.samples_per_prompt"),
        "num_updates_per_batch": nested(payload, "stack.num_updates_per_batch"),
        "micro_batch_size": nested(payload, "stack.micro_batch_size"),
        "reward_target": nested(payload, "reward.backend._target_"),
        "reward_weights": nested(payload, "reward.backend.config.weights"),
        "cache_dir": nested(payload, "bundle.config.prompt_embedding_cache_dir"),
        "cache_read_only": nested(payload, "bundle.config.prompt_embedding_cache_read_only"),
        "prefetch_default": nested(payload, "bundle.config.prompt_embedding_prefetch"),
        "prefetch_capacity": nested(payload, "bundle.config.prompt_embedding_prefetch_capacity"),
    }
    if observed != expected:
        raise GateError(f"canonical recipe contract mismatch: {observed}")
    return {"path": os.fspath(path), "sha256": sha256_file(path), "observed": observed}


def plan_gate(plan_path: Path, *, p0: dict, campaign_id: str | None) -> dict:
    plan = load_json(plan_path)
    if plan.get("schema") != "unirl-minimax-h3-p2-prefetch-abba-plan-v1":
        raise GateError("invalid P2 plan schema")
    plan_campaign = plan.get("campaign_id")
    if not isinstance(plan_campaign, str) or not plan_campaign:
        raise GateError("plan campaign ID is missing")
    if campaign_id is not None and plan_campaign != campaign_id:
        raise GateError("campaign ID differs between CLI and plan")
    node = plan.get("node")
    if not isinstance(node, str) or not node:
        raise GateError("plan node is missing")
    expected_arms = []
    for arm in arm_specs(plan_campaign):
        expected_arms.append(
            {
                **arm,
                "node": node,
                "overrides": functional_overrides(
                    arm["prefetch"],
                    prompt_manifest_path=p0["prompt_manifest"]["path"],
                    load_dir=str(p0["frozen_checkpoint"]["load_dir"]),
                ),
            }
        )
    expected_fixed = {
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
    }
    runtime_overlay = plan.get("runtime_overlay") or {}
    adapter_path = Path(str(runtime_overlay.get("adapter_module_path", "")))
    sitecustomize_path = Path(str(runtime_overlay.get("sitecustomize_path", "")))
    checks = {
        "source": plan.get("source_commit") == p0["p2"]["commit"]
        and plan.get("source_tree") == p0["p2"]["tree"]
        and plan.get("source_parent_commit") == p0["pr38"]["commit"]
        and plan.get("source_parent_tree") == p0["pr38"]["tree"]
        and plan.get("source_patch_sha256") == p0["p2"]["patch_sha256"],
        "integration_source": plan.get("integration_source")
        == {
            "commit": p0["source"]["integration_commit"],
            "tree": p0["source"]["integration_tree"],
            "full_tree_sha256": p0["source"]["full_tree_sha256"],
            "archive": p0["source_archive"],
            "tree_manifest": p0["source_tree_manifest"],
        },
        "topology": plan.get("world_size") == WORLD_SIZE
        and plan.get("sp_size") == SP_SIZE
        and plan.get("dp_size") == DP_SIZE,
        "order": plan.get("order") == ["off", "on", "on", "off"],
        "arms": plan.get("arms") == expected_arms,
        "fixed_contract": plan.get("fixed_contract") == expected_fixed,
        "prompt_manifest": plan.get("prompt_manifest")
        == {
            "path": p0["prompt_manifest"]["path"],
            "sha256": p0["prompt_manifest"]["sha256"],
            "prompt_ids": p0["prompt_manifest"]["prompt_ids"],
        },
        "planner": plan.get("planner_target") == EXPECTED_PLANNER_TARGET,
        "planner_contract": nested(plan, "planner.contract_sha256") == p0["planner_contract_sha256"],
        "two_update": nested(plan, "two_update.contract_sha256") == p0["two_update_contract_sha256"],
        "frozen_checkpoint": plan.get("frozen_checkpoint")
        == {
            "load_dir": p0["frozen_checkpoint"]["load_dir"],
            "audit_schema": p0["frozen_checkpoint"]["audit_payload"]["schema"],
            "audit_sha256": p0["frozen_checkpoint"]["audit_binding"]["sha256"],
            "tree_sha256": p0["frozen_checkpoint"]["tree_sha256"],
            "model_summary_sha256": p0["frozen_checkpoint"]["model_summary_sha256"],
            "adapter_key_signature_sha256": p0["frozen_checkpoint"]["adapter_key_signature_sha256"],
        },
        "runtime_overlay_base": nested(plan, "runtime_overlay.event_schema") == p0["runtime_overlay"]["event_schema"]
        and nested(plan, "runtime_overlay.base_module_sha256") == p0["runtime_overlay"]["module_binding"]["sha256"],
        "runtime_overlay_adapter": adapter_path.is_file()
        and nested(plan, "runtime_overlay.adapter_module_sha256") == sha256_file(adapter_path),
        "runtime_overlay_sitecustomize": sitecustomize_path.is_file()
        and nested(plan, "runtime_overlay.sitecustomize_sha256") == sha256_file(sitecustomize_path),
        "p0": nested(plan, "p0_contract.sha256") == p0["sha256"],
        "p1_source_preflight": nested(plan, "p1_source_preflight.sha256")
        == p0["p1_source_preflight_binding"]["sha256"],
        "fork_only": EXPECTED_REMOTE == "https://github.com/leviking98z-rgb/UniRL.git",
    }
    bad = [name for name, passed in checks.items() if not passed]
    if bad:
        raise GateError(f"plan gate failed: {bad}")
    return plan


def allocation_receipt_gate(path: Path) -> dict:
    receipt = load_json(path)
    if receipt.get("schema") != "clusterbridge-allocation-receipt-v1":
        raise GateError("allocation receipt schema mismatch")
    nodes = receipt.get("nodes")
    if receipt.get("resource_kind") != "gpu" or receipt.get("status") != "active":
        raise GateError("allocation receipt is not an active GPU allocation")
    if receipt.get("gpu_count") != 8 or not isinstance(nodes, list) or len(nodes) != 1:
        raise GateError("P2 requires exactly one 8-GPU node")
    detail = (receipt.get("node_details") or [{}])[0]
    if detail.get("ip") != nodes[0] or detail.get("gpus") != 8 or detail.get("gpu_type") != "H20":
        raise GateError("allocation receipt is not one 8xH20 node")
    if not all(detail.get(key) is True for key in ("alive", "fresh", "healthy")):
        raise GateError("allocation node health gate failed")
    if not re.fullmatch(r"[0-9a-f-]{36}", str(receipt.get("owner", ""))):
        raise GateError("allocation receipt owner is not a session UUID")
    return {"path": os.fspath(path.resolve()), "sha256": sha256_file(path), "payload": receipt}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--p0-contract", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--allocation-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-harness-changes", action="store_true")
    args = parser.parse_args()
    try:
        p0 = validate_p0_contract(args.p0_contract.resolve())
        source = validate_source(
            args.repo.resolve(),
            p0=p0,
            allow_harness_changes=args.allow_harness_changes,
        )
        recipe = recipe_gate(args.repo.resolve())
        plan = plan_gate(args.plan.resolve(), p0=p0, campaign_id=args.campaign_id) if args.plan else None
        allocation = allocation_receipt_gate(args.allocation_receipt.resolve()) if args.allocation_receipt else None
        payload = {
            "schema": "unirl-minimax-h3-p2-prefetch-preflight-v1",
            "completed": True,
            "source": source,
            "recipe": recipe,
            "p0_contract": p0,
            "plan": plan,
            "allocation": allocation,
        }
        write_json_atomic(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
    except GateError as exc:
        payload = {"schema": "unirl-minimax-h3-p2-prefetch-preflight-v1", "completed": False, "error": str(exc)}
        write_json_atomic(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
