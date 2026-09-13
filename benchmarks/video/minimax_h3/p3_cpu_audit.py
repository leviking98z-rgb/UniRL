"""Run the complete CPU-only MiniMax-H3 P3 trace and reordering audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.video.minimax_h3 import grouped_reordering_ab as ab  # noqa: E402
from benchmarks.video.minimax_h3 import matrix_analyzer as fixed_analyzer  # noqa: E402
from benchmarks.video.minimax_h3 import matrix_driver, matrix_proxy  # noqa: E402
from benchmarks.video.minimax_h3 import mixed_trace_analyzer as mixed  # noqa: E402

AUDIT_SCHEMA = "unirl:minimax-h3:p3-cpu-audit:v1"
PREDICTORS = ("packed_rows", "padded_rows", "attention_rows2")
PLACEMENTS = ("contiguous", "cost-clustered")
DEFAULT_PROMPTS = (
    "A red kite over a quiet beach.",
    "A ceramic robot pouring tea.",
    "Snow falling through a pine forest.",
    "A train crossing a desert at sunset.",
    "Bioluminescent fish in a dark ocean.",
    "A paper boat on a rainy street.",
    "A dancer in a mirrored studio.",
    "Clouds moving around a mountain peak.",
    "A fox walking through autumn leaves.",
    "A glass marble rolling on wood.",
    "Steam rising from a bowl of noodles.",
    "A lighthouse during a storm.",
    "An astronaut tending a small garden.",
    "A hummingbird near purple flowers.",
    "City reflections in a night puddle.",
    "A clockwork bird opening its wings.",
)


class AuditError(ValueError):
    """Fail-closed CPU audit error."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_inputs(output_dir: Path, num_prompts: int) -> tuple[Path, Path]:
    if num_prompts < 1 or num_prompts > len(DEFAULT_PROMPTS):
        raise AuditError(f"num_prompts must be in [1, {len(DEFAULT_PROMPTS)}], got {num_prompts}")
    prompts = output_dir / "prompts.jsonl"
    with prompts.open("w", encoding="utf-8") as destination:
        for index, prompt in enumerate(DEFAULT_PROMPTS[:num_prompts]):
            destination.write(
                json.dumps(
                    {"prompt_id": f"p{index:02d}", "prompt": prompt},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    cfg = matrix_driver._compose_config(matrix_driver._script_repo(), matrix_driver.DEFAULT_CONFIG_NAME)
    checkpoint = output_dir / "checkpoint-0"
    checkpoint.mkdir()
    metadata = {
        "step": 0,
        "save_mode": "adapter",
        "lora_config": matrix_driver._expected_lora_metadata(cfg),
        "policy_state_dict": {},
        "optimizer_state_dict": {},
        "optimizer_step_count": 0,
    }
    try:
        import torch
    except ImportError as exc:
        raise AuditError("p3_cpu_audit.py requires the normal UniRL torch dependency") from exc
    torch.save(metadata, checkpoint / "checkpoint.pt")
    _write_json(checkpoint / "trainer_state.json", {"optimizer_step": 0, "wandb_run_id": None})
    return prompts, checkpoint


def _prepare_matrix(
    output_dir: Path,
    prompts: Path,
    checkpoint: Path,
    *,
    num_devices: int,
    sp_size: int,
    group_size: int,
    num_prompts: int,
) -> Path:
    matrix_dir = output_dir / "matrix"
    args = argparse.Namespace(
        source_config=matrix_driver.DEFAULT_SOURCE_CONFIG,
        config_name=matrix_driver.DEFAULT_CONFIG_NAME,
        prompts=str(prompts),
        lora_checkpoint=str(checkpoint),
        output_dir=str(matrix_dir),
        python_executable=sys.executable,
        num_devices=num_devices,
        sp_size=sp_size,
        group_size=group_size,
        num_prompts=num_prompts,
        cost="packed_rows",
        tail_ratio_threshold=1.05,
        minimum_predicted_speedup=1.05,
    )
    manifest = matrix_driver.build_manifest(args)
    matrix_dir.mkdir()
    manifest_path = matrix_dir / "matrix.json"
    _write_json(manifest_path, manifest)
    matrix_proxy.materialize_proxy(manifest_path)
    return manifest_path


def _predictor_summary(report: dict[str, Any], predictor: str) -> dict[str, Any]:
    aggregate = report["simulation"]["aggregate"][predictor]
    return {
        "decision": report["decision"]["decision"],
        "baseline_tail_ratio": aggregate["baseline_tail_ratio"],
        "grouped_tail_ratio": aggregate["grouped_tail_ratio"],
        "predicted_speedup": aggregate["predicted_speedup"],
    }


def _write_ab_contract(plan_path: Path, output_dir: Path) -> dict[str, Any]:
    contract = ab.build_contract(
        plan_path,
        repetitions=3,
        warmup_repetitions=1,
        measurement="generation_s",
        minimum_speedup=1.05,
        minimum_win_fraction=0.75,
        maximum_reward_delta=0.01,
        maximum_trace_overhead=0.005,
    )
    contract_path = output_dir / "contract.json"
    _write_json(contract_path, contract)
    templates = output_dir / "templates"
    for arm in ab.ARMS:
        _write_json(templates / f"{arm}.template.json", ab.result_template(contract, arm))
    if ab.load_contract(contract_path) != contract:
        raise AuditError(f"A/B contract did not round-trip: {contract_path}")
    return {
        "contract": str(contract_path),
        "contract_id": contract["contract_id"],
        "runs_per_arm": int(contract["repetitions"]) * len(contract["waves"]),
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize and validate every CPU/synthetic P3 gate."""
    repo = matrix_driver._script_repo()
    matrix_driver._assert_clean_worktree(repo)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.is_relative_to(repo):
        raise AuditError(f"output_dir must be outside the source checkout: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AuditError(f"output directory must not exist or must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    previous_model = os.environ.get("PRETRAINED_MODEL")
    os.environ["PRETRAINED_MODEL"] = args.pretrained_model_locator
    try:
        prompts, checkpoint = _prepare_inputs(output_dir, args.num_prompts)
        manifest_path = _prepare_matrix(
            output_dir,
            prompts,
            checkpoint,
            num_devices=args.num_devices,
            sp_size=args.sp_size,
            group_size=args.group_size,
            num_prompts=args.num_prompts,
        )
        fixed_control = fixed_analyzer.analyze_matrix(manifest_path)
        _write_json(output_dir / "matrix" / "fixed-control.json", fixed_control)
        fixed_replay, fixed_plan = mixed.analyze_fixed_replay(
            manifest_path,
            predictors=PREDICTORS,
            trials=args.trials,
            waves=args.waves,
            seed=args.seed,
        )
        fixed_report_path = output_dir / "matrix" / "fixed-replay.json"
        fixed_plan_path = output_dir / "matrix" / "fixed-replay-plan.json"
        _write_json(fixed_report_path, fixed_replay)
        _write_json(fixed_plan_path, fixed_plan)
        mixed.validate_plan(fixed_plan_path)

        synthetic_results: dict[str, Any] = {}
        for placement in PLACEMENTS:
            placement_dir = output_dir / f"mixed-{placement}"
            trace_set = mixed.materialize_synthetic_mixed_traces(
                manifest_path,
                placement_dir / "traces",
                seed=args.seed,
                trial=args.synthetic_trial,
                waves=args.waves,
                text_token_min=args.text_token_min,
                text_token_max=args.text_token_max,
                baseline_placement=placement,
            )
            predictor_results = {}
            for predictor in PREDICTORS:
                report, plan = mixed.analyze_mixed_traces(
                    manifest_path,
                    (),
                    predictor_cost=predictor,
                    outcome_cost=predictor,
                    require_mixed_length=True,
                    trace_set_path=placement_dir / "traces" / "trace-set.json",
                )
                report_path = placement_dir / f"analysis-{predictor}.json"
                plan_path = placement_dir / f"plan-{predictor}.json"
                _write_json(report_path, report)
                _write_json(plan_path, plan)
                mixed.validate_plan(plan_path)
                contract = _write_ab_contract(plan_path, placement_dir / f"ab-{predictor}")
                predictor_results[predictor] = {
                    **_predictor_summary(report, predictor),
                    **contract,
                }
            token_values = list(trace_set["text_tokens"]["by_root"].values())
            synthetic_results[placement] = {
                "trace_set": str(placement_dir / "traces" / "trace-set.json"),
                "trace_set_id": trace_set["trace_set_id"],
                "text_tokens": {
                    "minimum": min(token_values),
                    "maximum": max(token_values),
                    "unique": len(set(token_values)),
                },
                "predictors": predictor_results,
            }
    finally:
        if previous_model is None:
            os.environ.pop("PRETRAINED_MODEL", None)
        else:
            os.environ["PRETRAINED_MODEL"] = previous_model

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conservative_predictors = []
    for result in fixed_replay["predictors"]:
        conservative_predictors.append(
            {
                "predictor": result["predictor_cost"],
                "decision": result["decision"]["decision"],
                "tail_ratio": result["baseline_tail_ratio"],
                "predicted_speedup": result["predicted_speedup"],
                "trial_go_count": result["trial_go_count"],
                "trial_count": result["trial_count"],
            }
        )
    production_reasons = [
        "no measured paired mixed-workload GPU A/B result exists",
        "the current H3 runtime has no per-root geometry plus shape-aware pre-DP-scatter scheduler",
    ]
    if not fixed_replay["decision"]["go"]:
        production_reasons.insert(0, "the conservative fixed-profile proxy gate is NO-GO")

    summary = {
        "schema": AUDIT_SCHEMA,
        "source": {
            "branch": matrix_driver._run_git(repo, "branch", "--show-current"),
            "commit": manifest["binding"]["source_commit"],
            "tree": manifest["binding"]["source_tree"],
            "matrix_id": manifest["manifest_id"],
        },
        "constraints": {
            "gpu_used": False,
            "cluster_training_started": False,
            "evidence": "analytical CPU proxy and deterministic synthetic traces",
        },
        "fixed_control": fixed_control["decision"],
        "fixed_replay": {
            "decision": fixed_replay["decision"],
            "trials": args.trials,
            "waves_per_trial": args.waves,
            "predictors": conservative_predictors,
        },
        "synthetic_mixed_length": synthetic_results,
        "gates": {
            "tooling_and_contracts": {
                "decision": "GO",
                "reason": "all content-addressed plans and A/B contracts validated",
            },
            "production_grouped_reordering_pr": {
                "decision": "NO-GO",
                "reasons": production_reasons,
            },
        },
    }
    summary["audit_id"] = hashlib.sha256(_canonical_json(summary).encode()).hexdigest()
    summary_path = output_dir / "AUDIT_SUMMARY.json"
    _write_json(summary_path, summary)
    checksum_lines = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS":
            continue
        checksum_lines.append(f"{_sha256_file(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-devices", type=_positive_int, default=16)
    parser.add_argument("--sp-size", type=_positive_int, default=8)
    parser.add_argument("--group-size", type=_positive_int, default=2)
    parser.add_argument("--num-prompts", type=_positive_int, default=16)
    parser.add_argument("--trials", type=_positive_int, default=64)
    parser.add_argument("--waves", type=_positive_int, default=4)
    parser.add_argument("--seed", default="p3-balanced-v1")
    parser.add_argument("--synthetic-trial", type=_non_negative_int, default=0)
    parser.add_argument("--text-token-min", type=_positive_int, default=32)
    parser.add_argument("--text-token-max", type=_positive_int, default=512)
    parser.add_argument(
        "--pretrained-model-locator",
        default="p3-cpu-audit://weights-are-not-loaded",
        help="binding-only locator; this CPU audit never opens model weights",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        summary = run_audit(args)
    except (AuditError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"audit_id={summary['audit_id']}")
    print(f"source_commit={summary['source']['commit']}")
    print(f"fixed_control={summary['fixed_control']['decision']}")
    print(f"fixed_replay={summary['fixed_replay']['decision']['decision']}")
    for placement, result in summary["synthetic_mixed_length"].items():
        for predictor, metrics in result["predictors"].items():
            print(
                f"synthetic={placement} predictor={predictor} decision={metrics['decision']} "
                f"tail={metrics['baseline_tail_ratio']:.6f} "
                f"predicted_speedup={metrics['predicted_speedup']:.6f}x"
            )
    print(f"production_grouped_reordering_pr={summary['gates']['production_grouped_reordering_pr']['decision']}")


if __name__ == "__main__":
    main()
