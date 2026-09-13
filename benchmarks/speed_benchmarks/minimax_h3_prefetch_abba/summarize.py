#!/usr/bin/env python3
"""Fail-closed summarizer for the canonical single-node P2 OFF/ON/ON/OFF run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from harness_lib import (
    FROZEN_CHECKPOINT_LOADED_SCHEMA,
    FUNCTIONAL_OVERRIDES,
    P2_RUNTIME_SUMMARY_SCHEMA,
    GateError,
    arm_specs,
    embedding_gate,
    expected_sample_ids,
    load_json,
    load_p0_events,
    metrics_gate,
    paired_effects,
    parse_embedding_telemetry,
    parse_jsonl,
    parse_nvml,
    process_identity_gate,
    read_kv,
    runtime_event_tree_digest,
    sample_ledger_gate,
    scan_failures,
    sha256_bytes,
    sha256_file,
    update_membership_gate,
    validate_p0_contract,
    write_json_atomic,
)


def normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
    def nested(dotted: str) -> Any:
        value: Any = config
        for key in dotted.split("."):
            if not isinstance(value, Mapping) or key not in value:
                return None
            value = value[key]
        return value

    keys = set(FUNCTIONAL_OVERRIDES) | {
        "bundle.config.prompt_embedding_prefetch",
        "data_source.args.run.data_path",
        "load_dir",
    }
    return {key: nested(key) for key in sorted(keys)}


def config_gate(path: Path, *, expected: Mapping[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    if yaml is None:
        return False, ["PyYAML unavailable"], {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, yaml.YAMLError) as exc:
        return False, [f"resolved config unavailable: {exc}"], {}
    observed = normalized_config(payload)
    issues = [
        f"{key}: observed={observed.get(key)!r} expected={value!r}"
        for key, value in expected.items()
        if observed.get(key) != value
    ]
    return not issues, issues, observed


def _runtime_artifact_hashes(artifact: Path, run: str) -> dict[str, str]:
    names = {
        "runtime_summary_sha256": f"runtime_summary_{run}.json",
        "sample_ledger_sha256": f"sample_ledger_{run}.jsonl",
        "update_membership_sha256": f"update_membership_{run}.json",
        "frozen_checkpoint_loaded_sha256": f"frozen_checkpoint_loaded_{run}.json",
    }
    return {key: sha256_file(artifact / name) for key, name in names.items()}


def receipt_gate(
    artifact: Path,
    arm: Mapping[str, Any],
    p0: Mapping[str, Any],
) -> tuple[bool, list[str], dict[str, Any]]:
    run = str(arm["run_name"])
    issues: list[str] = []
    try:
        status = read_kv(artifact / f"{run}.status")
        identity = read_kv(artifact / f"{run}.identity")
        contract = load_json(artifact / f"{run}.contract.json")
        receipt = load_json(artifact / f"{run}.receipt.json")
        started = artifact / f"{run}.started"
        runtime_hashes = _runtime_artifact_hashes(artifact, run)
    except GateError as exc:
        return False, [str(exc)], {}
    expected_common = {
        "campaign_id": arm["run_name"].rsplit(f"-{arm['arm_id']}", 1)[0],
        "run_name": run,
        "period": arm["period"],
        "treatment": arm["treatment"],
    }
    for key, value in expected_common.items():
        if contract.get(key) != value or receipt.get(key) != value:
            issues.append(f"contract/receipt {key} mismatch")
    if status.get("exit_code") != "0" or receipt.get("exit_code") != 0:
        issues.append("arm exit code is nonzero")
    bindings = {
        "contract_sha256": sha256_file(artifact / f"{run}.contract.json"),
        "identity_sha256": sha256_file(artifact / f"{run}.identity"),
        "started_sha256": sha256_file(started),
        "status_sha256": sha256_file(artifact / f"{run}.status"),
        **runtime_hashes,
    }
    for key, value in bindings.items():
        if receipt.get(key) != value:
            issues.append(f"receipt {key} mismatch")
    if identity.get("contract_sha256") != bindings["contract_sha256"]:
        issues.append("identity contract hash mismatch")
    fixed = {
        "p0_contract_sha256": p0["sha256"],
        "prompt_manifest_sha256": p0["prompt_manifest"]["sha256"],
        "frozen_checkpoint_audit_sha256": p0["frozen_checkpoint"]["audit_binding"]["sha256"],
        "frozen_checkpoint_tree_sha256": p0["frozen_checkpoint"]["tree_sha256"],
        "frozen_checkpoint_model_summary_sha256": p0["frozen_checkpoint"]["model_summary_sha256"],
        "frozen_checkpoint_adapter_key_signature_sha256": p0["frozen_checkpoint"]["adapter_key_signature_sha256"],
        "planner_contract_sha256": p0["planner_contract_sha256"],
        "two_update_contract_sha256": p0["two_update_contract_sha256"],
    }
    for key, value in fixed.items():
        if contract.get(key) != value or receipt.get(key) != value or identity.get(key) != value:
            issues.append(f"{key} differs from integration contract")
    if (
        contract.get("p2_implementation_commit") != p0["p2"]["commit"]
        or contract.get("p2_implementation_tree") != p0["p2"]["tree"]
        or contract.get("p2_parent_commit") != p0["pr38"]["commit"]
        or contract.get("p2_parent_tree") != p0["pr38"]["tree"]
        or contract.get("p2_patch_sha256") != p0["p2"]["patch_sha256"]
        or contract.get("integration_source_commit") != p0["source"]["integration_commit"]
        or contract.get("integration_source_tree") != p0["source"]["integration_tree"]
        or contract.get("integration_full_tree_sha256") != p0["source"]["full_tree_sha256"]
    ):
        issues.append("arm source identity mismatch")
    if contract.get("prefetch") is not arm["prefetch"] or identity.get("prefetch") != str(arm["prefetch"]).lower():
        issues.append("prefetch treatment mismatch")
    return (
        not issues,
        issues,
        {
            "status": status,
            "identity": identity,
            "contract": contract,
            "receipt": receipt,
        },
    )


def runtime_summary_gate(
    path: Path,
    *,
    run: str,
    treatment: str,
    p0: Mapping[str, Any],
    event_root: Path | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    try:
        payload = load_json(path)
    except GateError as exc:
        return False, [str(exc)], {}
    conditioner = payload.get("conditioner")
    expected = {
        "schema": P2_RUNTIME_SUMMARY_SCHEMA,
        "run_name": run,
        "treatment": treatment,
        "completed": True,
        "event_schema": "unirl-minimax-h3-p0-runtime-event-v2",
        "planner_contract_sha256": p0["planner_contract_sha256"],
        "two_update_contract_sha256": p0["two_update_contract_sha256"],
        "topology_ranks": list(range(8)),
    }
    issues = [f"runtime summary {key} mismatch" for key, value in expected.items() if payload.get(key) != value]
    if not isinstance(conditioner, Mapping):
        issues.append("runtime summary lacks conditioner evidence")
    else:
        expected_background = 4 if treatment == "on" else 0
        expected_hits = 4 if treatment == "on" else 0
        expected_overlap = 4 if treatment == "on" else 0
        if conditioner.get("encode_calls") != 8:
            issues.append("runtime summary conditioner encode count is not eight")
        if conditioner.get("source_ranks") != [0, 2, 4, 6]:
            issues.append("runtime summary conditioner source ranks are not SP2 leaders")
        if conditioner.get("background_encode_calls") != expected_background:
            issues.append("runtime summary background conditioner count mismatch")
        if conditioner.get("prefetch_consumed_hits") != expected_hits:
            issues.append("runtime summary prefetch consumed-hit count mismatch")
        if conditioner.get("shutdown_ranks") != list(range(8)):
            issues.append("runtime summary does not prove shutdown on all ranks")
        overlap = conditioner.get("overlap")
        if not isinstance(overlap, list) or len(overlap) != expected_overlap:
            issues.append("runtime summary overlap-row count mismatch")
        elif treatment == "on" and (
            any(row.get("overlapped") is not True for row in overlap if isinstance(row, Mapping))
            or conditioner.get("all_background_calls_overlapped") is not True
        ):
            issues.append("runtime summary does not prove conditioner/denoise overlap")
        elif treatment == "off" and conditioner.get("all_background_calls_overlapped") is not True:
            issues.append("OFF runtime summary has inconsistent empty-overlap state")
    cleanup = payload.get("cleanup")
    if not isinstance(cleanup, Mapping) or cleanup.get("pool_shutdown") is not True:
        issues.append("runtime summary does not prove driver/pool cleanup")
    if event_root is not None:
        try:
            events = load_p0_events(event_root)
            if payload.get("event_count") != len(events):
                issues.append("runtime summary event_count mismatch")
            if payload.get("event_tree_sha256") != runtime_event_tree_digest(events):
                issues.append("runtime summary event-tree digest mismatch")
        except GateError as exc:
            issues.append(str(exc))
    expected_hashes = {
        "sample_ledger_sha256": sha256_file(path.parent / f"sample_ledger_{run}.jsonl"),
        "update_membership_sha256": sha256_file(path.parent / f"update_membership_{run}.json"),
        "frozen_checkpoint_loaded_sha256": sha256_file(path.parent / f"frozen_checkpoint_loaded_{run}.json"),
    }
    for key, value in expected_hashes.items():
        if payload.get(key) != value:
            issues.append(f"runtime summary {key} mismatch")
    return not issues, issues, payload


def frozen_checkpoint_gate(path: Path, p0: Mapping[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    try:
        payload = load_json(path)
    except GateError as exc:
        return False, [str(exc)], {}
    expected = {
        "schema": FROZEN_CHECKPOINT_LOADED_SCHEMA,
        "audit_sha256": p0["frozen_checkpoint"]["audit_binding"]["sha256"],
        "tree_sha256": p0["frozen_checkpoint"]["tree_sha256"],
        "model_summary_sha256": p0["frozen_checkpoint"]["model_summary_sha256"],
        "adapter_key_signature_sha256": p0["frozen_checkpoint"]["adapter_key_signature_sha256"],
        "loaded_rollout_step": 0,
        "optimizer_step": 0,
        "loaded_before_optimizer_step": True,
        "loaded_ranks": list(range(8)),
    }
    issues = [f"{key} mismatch" for key, value in expected.items() if payload.get(key) != value]
    if payload.get("load_event_count") != 8:
        issues.append("frozen checkpoint load event count is not eight")
    return not issues, issues, payload


def source_verification_gate(path: Path, p0: Mapping[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    try:
        payload = load_json(path)
    except GateError as exc:
        return False, [str(exc)], {}
    expected = {
        "schema": "unirl-minimax-h3-p2-prefetch-source-identity-v1",
        "completed": True,
        "head": p0["source"]["integration_commit"],
        "tree": p0["source"]["integration_tree"],
        "full_tree_sha256": p0["source"]["full_tree_sha256"],
        "p0_contract_sha256": p0["sha256"],
    }
    issues = [f"{key} mismatch" for key, value in expected.items() if payload.get(key) != value]
    return not issues, issues, payload


def summarize(artifact: Path, campaign_id: str, p0_contract: Path) -> dict[str, Any]:
    p0 = validate_p0_contract(p0_contract)
    plan = load_json(artifact / "plan.json")
    manifest = load_json(artifact / "manifest.json")
    plan_arms = plan.get("arms") or []
    expected_arms = arm_specs(campaign_id)
    plan_ok = bool(
        plan.get("schema") == "unirl-minimax-h3-p2-prefetch-abba-plan-v1"
        and plan.get("campaign_id") == campaign_id
        and plan.get("source_commit") == p0["p2"]["commit"]
        and plan.get("source_tree") == p0["p2"]["tree"]
        and plan.get("source_parent_commit") == p0["pr38"]["commit"]
        and plan.get("source_parent_tree") == p0["pr38"]["tree"]
        and plan.get("source_patch_sha256") == p0["p2"]["patch_sha256"]
        and (plan.get("planner") or {}).get("contract_sha256") == p0["planner_contract_sha256"]
        and (plan.get("two_update") or {}).get("contract_sha256") == p0["two_update_contract_sha256"]
        and (plan.get("frozen_checkpoint") or {}).get("audit_sha256")
        == p0["frozen_checkpoint"]["audit_binding"]["sha256"]
        and (plan.get("frozen_checkpoint") or {}).get("tree_sha256") == p0["frozen_checkpoint"]["tree_sha256"]
        and (plan.get("runtime_overlay") or {}).get("base_module_sha256")
        == p0["runtime_overlay"]["module_binding"]["sha256"]
        and (plan.get("p1_source_preflight") or {}).get("sha256") == p0["p1_source_preflight_binding"]["sha256"]
        and plan.get("order") == ["off", "on", "on", "off"]
        and len(plan_arms) == 4
        and [
            (
                arm.get("period"),
                arm.get("treatment"),
                arm.get("run_name"),
                arm.get("prefetch"),
            )
            for arm in plan_arms
        ]
        == [
            (
                arm["period"],
                arm["treatment"],
                arm["run_name"],
                arm["prefetch"],
            )
            for arm in expected_arms
        ]
        and (plan.get("p0_contract") or {}).get("sha256") == p0["sha256"]
    )
    manifest_ok = bool(
        manifest.get("schema") == "unirl-minimax-h3-p2-prefetch-abba-manifest-v1"
        and manifest.get("campaign_id") == campaign_id
        and manifest.get("plan_sha256") == sha256_file(artifact / "plan.json")
        and manifest.get("p0_contract_sha256") == p0["sha256"]
    )
    harness_hashes = manifest.get("harness_sha256") or {}
    harness_ok = isinstance(harness_hashes, dict) and bool(harness_hashes)
    harness_root = artifact / "harness"
    for relative, expected in harness_hashes.items() if isinstance(harness_hashes, dict) else []:
        path = harness_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            harness_ok = False
    prepared_source_ok, prepared_source_issues, prepared_source = source_verification_gate(
        artifact / "source-prepared.json", p0
    )

    records = []
    prompt_ids = p0["prompt_manifest"]["prompt_ids"]
    expected_ids = expected_sample_ids(prompt_ids)
    source_ranks = [group[0] for group in p0["topology"]["sp_groups"]]
    for arm in arm_specs(campaign_id):
        run = arm["run_name"]
        record: dict[str, Any] = dict(arm)
        issues: list[str] = []
        log_path = artifact / f"{run}.log"
        try:
            log = log_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            log = ""
            issues.append("missing arm log")
        failures = scan_failures(log, treatment=arm["treatment"])
        if failures:
            issues.append(f"fatal log signatures: {failures}")
        source_post_ok, source_post_issues, source_post = source_verification_gate(
            artifact / f"source-post-{run}.json", p0
        )
        issues.extend(source_post_issues)

        receipt_ok, receipt_issues, receipt_detail = receipt_gate(artifact, arm, p0)
        issues.extend(receipt_issues)
        contract = receipt_detail.get("contract") or {}
        expected_config = contract.get("overrides") if isinstance(contract.get("overrides"), Mapping) else {}
        config_ok, config_issues, config_observed = config_gate(
            artifact / f"hydra_{run}_config.yaml", expected=expected_config
        )
        issues.extend(config_issues)

        telemetry = parse_embedding_telemetry(log)
        telemetry_ok, telemetry_issues = embedding_gate(
            telemetry,
            treatment=arm["treatment"],
            source_ranks=source_ranks,
        )
        issues.extend(telemetry_issues)

        try:
            ledger_rows = parse_jsonl(artifact / f"sample_ledger_{run}.jsonl")
            ledger_ok, ledger_issues, fingerprints = sample_ledger_gate(ledger_rows, expected_ids=expected_ids)
        except GateError as exc:
            ledger_rows, fingerprints = [], {}
            ledger_ok, ledger_issues = False, [str(exc)]
        issues.extend(ledger_issues)

        try:
            updates = load_json(artifact / f"update_membership_{run}.json")
            update_ok, update_issues = update_membership_gate(
                updates,
                prompt_ids=prompt_ids,
                planner_contract_sha256=p0["planner_contract_sha256"],
                two_update_contract_sha256=p0["two_update_contract_sha256"],
                frozen_checkpoint_audit_sha256=p0["frozen_checkpoint"]["audit_binding"]["sha256"],
            )
        except GateError as exc:
            updates = {}
            update_ok, update_issues = False, [str(exc)]
        issues.extend(update_issues)

        try:
            metrics = load_json(artifact / f"metrics_{run}.json")
            metrics_ok, metric_issues = metrics_gate(metrics)
        except GateError as exc:
            metrics = {}
            metrics_ok, metric_issues = False, [str(exc)]
        issues.extend(metric_issues)

        frozen_ok, frozen_issues, frozen_loaded = frozen_checkpoint_gate(
            artifact / f"frozen_checkpoint_loaded_{run}.json", p0
        )
        issues.extend(frozen_issues)
        runtime_ok, runtime_issues, runtime_summary = runtime_summary_gate(
            artifact / f"runtime_summary_{run}.json",
            run=run,
            treatment=arm["treatment"],
            p0=p0,
            event_root=(
                artifact / f"runtime_receipts_{run}" if (artifact / f"runtime_receipts_{run}").is_dir() else None
            ),
        )
        issues.extend(runtime_issues)
        nvml = parse_nvml(artifact / f"nvml_{run}_{plan.get('node')}.csv")
        if not nvml["complete"]:
            issues.append("NVML coverage incomplete")
        stderr_path = artifact / f"nvml_{run}_{plan.get('node')}.stderr"
        if not stderr_path.is_file() or stderr_path.read_text(encoding="utf-8").strip():
            issues.append("NVML stderr missing or non-empty")

        record.update(
            {
                "passed": not issues,
                "issues": issues,
                "fatal_log_signatures": failures,
                "source_post_complete": source_post_ok,
                "source_post": source_post,
                "receipt_complete": receipt_ok,
                "receipt": receipt_detail,
                "config_complete": config_ok,
                "resolved_functional_config": config_observed,
                "embedding_telemetry_complete": telemetry_ok,
                "embedding_telemetry": telemetry,
                "sample_ledger_complete": ledger_ok,
                "sample_fingerprints": fingerprints,
                "update_membership_complete": update_ok,
                "update_membership": updates,
                "metrics_complete": metrics_ok,
                "metrics": metrics,
                "frozen_checkpoint_complete": frozen_ok,
                "frozen_checkpoint_loaded": frozen_loaded,
                "runtime_summary_complete": runtime_ok,
                "runtime_summary": runtime_summary,
                "nvml": nvml,
            }
        )
        records.append(record)

    process_ok, process_issues = process_identity_gate(
        [(record.get("receipt") or {}).get("identity") or {} for record in records]
    )

    cross_arm_issues: list[str] = list(process_issues)
    fixed_contracts = []
    for record in records:
        contract = ((record.get("receipt") or {}).get("contract") or {}).copy()
        for key in (
            "period",
            "arm_id",
            "run_name",
            "treatment",
            "prefetch",
            "required_runtime_artifacts",
            "created_utc",
        ):
            contract.pop(key, None)
        overrides = dict(contract.get("overrides") or {})
        overrides.pop("bundle.config.prompt_embedding_prefetch", None)
        contract["overrides"] = overrides
        fixed_contracts.append(sha256_bytes(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()))
    if len(set(fixed_contracts)) != 1:
        cross_arm_issues.append("arms differ in functional contract beyond the prefetch flag")

    sample_maps = [record.get("sample_fingerprints") or {} for record in records]
    if not sample_maps or any(sample_map != sample_maps[0] for sample_map in sample_maps[1:]):
        cross_arm_issues.append("sample-level reward/component/output fingerprints differ across arms")
    update_digests = [
        sha256_bytes(
            json.dumps(
                {
                    "planner_target": (record.get("update_membership") or {}).get("planner_target"),
                    "updates": (record.get("update_membership") or {}).get("updates"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for record in records
    ]
    if len(set(update_digests)) != 1:
        cross_arm_issues.append("update membership differs across arms")
    frozen_identity = [
        (
            item.get("audit_sha256"),
            item.get("tree_sha256"),
            item.get("model_summary_sha256"),
            item.get("adapter_key_signature_sha256"),
        )
        for item in [record.get("frozen_checkpoint_loaded") or {} for record in records]
    ]
    if len(set(frozen_identity)) != 1:
        cross_arm_issues.append("frozen checkpoint identity differs across arms")

    effects = paired_effects(records)
    all_records = len(records) == 4 and all(record["passed"] for record in records)
    completed = bool(
        plan_ok
        and manifest_ok
        and harness_ok
        and prepared_source_ok
        and process_ok
        and not cross_arm_issues
        and all_records
        and effects["complete"]
    )
    return {
        "schema": "unirl-minimax-h3-p2-prefetch-abba-summary-v1",
        "campaign_id": campaign_id,
        "completed": completed,
        "plan_complete": plan_ok,
        "manifest_complete": manifest_ok,
        "harness_snapshot_complete": harness_ok,
        "prepared_source_complete": prepared_source_ok,
        "prepared_source_issues": prepared_source_issues,
        "prepared_source": prepared_source,
        "p0_contract_sha256": p0["sha256"],
        "source_commit": p0["p2"]["commit"],
        "source_tree": p0["p2"]["tree"],
        "source_parent_commit": p0["pr38"]["commit"],
        "source_parent_tree": p0["pr38"]["tree"],
        "process_identity_complete": process_ok,
        "cross_arm_complete": not cross_arm_issues,
        "cross_arm_issues": cross_arm_issues,
        "records": records,
        "classification_counts": dict(Counter("passed" if record["passed"] else "failed" for record in records)),
        "paired_effects": effects,
        "interpretation": (
            "P2 primary estimate is the geometric mean of two within-node OFF/ON "
            "ratios: P1/P2 and P4/P3. One AB/BA campaign is diagnostic; repeat "
            "campaigns are required for stable confidence intervals."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--p0-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        payload = summarize(
            args.artifact.resolve(),
            args.campaign_id,
            args.p0_contract.resolve(),
        )
    except GateError as exc:
        payload = {
            "schema": "unirl-minimax-h3-p2-prefetch-abba-summary-v1",
            "campaign_id": args.campaign_id,
            "completed": False,
            "fatal_error": str(exc),
        }
    if args.output:
        write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload.get("completed") is True else 2)


if __name__ == "__main__":
    main()
