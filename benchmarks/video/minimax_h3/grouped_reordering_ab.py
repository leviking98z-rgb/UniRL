"""Create and validate a fail-closed paired GPU A/B contract for grouped reordering."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.video.minimax_h3.mixed_trace_analyzer import validate_plan  # noqa: E402

CONTRACT_SCHEMA = "unirl:minimax-h3:grouped-reordering-ab-contract:v1"
RESULT_SCHEMA = "unirl:minimax-h3:grouped-reordering-ab-result:v1"
ANALYSIS_SCHEMA = "unirl:minimax-h3:grouped-reordering-ab-analysis:v1"
ARMS = ("baseline", "grouped_lpt")


class ABError(ValueError):
    """Fail-closed paired A/B contract or result error."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ABError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ABError(f"expected a JSON object at {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _content_id(value: dict[str, Any], field: str) -> str:
    payload = dict(value)
    actual = payload.pop(field, None)
    expected = _sha256_json(payload)
    if actual != expected:
        raise ABError(f"{field} does not match the document contents")
    return str(actual)


def _workload_payload(wave: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "root_id": row["root_id"],
            "geometry": row["geometry"],
            "height": row["height"],
            "width": row["width"],
            "num_frames": row["num_frames"],
            "text_tokens": row["text_tokens"],
            "sample_ids": row["sample_ids"],
        }
        for row in wave["request_rows"]
    ]


def _sample_set_sha256(wave: dict[str, Any]) -> str:
    sample_ids = sorted(sample_id for row in wave["request_rows"] for sample_id in row["sample_ids"])
    return _sha256_json(sample_ids)


def _validate_rank_assignment(rank_root_ids: Any, root_ids: list[str], ranks: int, *, label: str) -> None:
    if not isinstance(rank_root_ids, list) or len(rank_root_ids) != ranks:
        raise ABError(f"{label} must contain exactly {ranks} rank shards")
    expected_capacity = len(root_ids) // ranks
    if any(not isinstance(shard, list) or len(shard) != expected_capacity for shard in rank_root_ids):
        raise ABError(f"{label} must keep equal root capacity {expected_capacity} on every rank")
    flattened = [root_id for shard in rank_root_ids for root_id in shard]
    if sorted(flattened) != sorted(root_ids) or len(set(flattened)) != len(flattened):
        raise ABError(f"{label} must contain every workload root exactly once")


def build_contract(
    plan_path: Path,
    *,
    repetitions: int,
    warmup_repetitions: int,
    measurement: str,
    minimum_speedup: float,
    minimum_win_fraction: float,
    maximum_reward_delta: float,
    maximum_trace_overhead: float,
) -> dict[str, Any]:
    """Bind an immutable mixed workload plan to paired A/B success criteria."""
    plan_path = plan_path.expanduser().resolve(strict=True)
    plan = validate_plan(plan_path)
    if repetitions < 2:
        raise ABError("paired A/B requires at least two measured repetitions")
    if warmup_repetitions < 1:
        raise ABError("paired A/B requires at least one unmeasured warmup repetition")
    if measurement not in {"generation_s", "end_to_end_step_s"}:
        raise ABError("measurement must be generation_s or end_to_end_step_s")
    if not math.isfinite(minimum_speedup) or minimum_speedup <= 1.0:
        raise ABError("minimum_speedup must be finite and greater than 1.0")
    if not math.isfinite(minimum_win_fraction) or not 0.5 <= minimum_win_fraction <= 1.0:
        raise ABError("minimum_win_fraction must be in [0.5, 1.0]")
    if not math.isfinite(maximum_reward_delta) or maximum_reward_delta < 0.0:
        raise ABError("maximum_reward_delta must be finite and non-negative")
    if not math.isfinite(maximum_trace_overhead) or not 0.0 <= maximum_trace_overhead < 1.0:
        raise ABError("maximum_trace_overhead must be in [0.0, 1.0)")

    waves = []
    for wave in plan["waves"]:
        if set(wave["arms"]) != set(ARMS):
            raise ABError(f"plan wave {wave.get('wave')} lacks the two required A/B arms")
        payload = _workload_payload(wave)
        root_ids = [row["root_id"] for row in wave["request_rows"]]
        ranks = int(plan["topology"]["dp_groups"])
        for arm in ARMS:
            _validate_rank_assignment(
                wave["arms"][arm]["rank_root_ids"],
                root_ids,
                ranks,
                label=f"plan wave {wave.get('wave')} arm {arm}",
            )
        waves.append(
            {
                "wave": int(wave["wave"]),
                "workload_sha256": _sha256_json(payload),
                "sample_set_sha256": _sample_set_sha256(wave),
                "expected_samples": sum(len(row["sample_ids"]) for row in wave["request_rows"]),
                "arms": {arm: {"rank_root_ids": wave["arms"][arm]["rank_root_ids"]} for arm in ARMS},
            }
        )
    if [wave["wave"] for wave in waves] != list(range(len(waves))):
        raise ABError("plan wave indices must be consecutive from zero")

    contract = {
        "schema": CONTRACT_SCHEMA,
        "plan": str(plan_path),
        "plan_id": plan["plan_id"],
        "plan_sha256": _sha256_file(plan_path),
        "measurement": measurement,
        "repetitions": repetitions,
        "warmup_repetitions": warmup_repetitions,
        "topology": plan["topology"],
        "criteria": {
            "minimum_paired_speedup": minimum_speedup,
            "minimum_win_fraction": minimum_win_fraction,
            "maximum_absolute_reward_delta": maximum_reward_delta,
            "maximum_trace_overhead_fraction": maximum_trace_overhead,
        },
        "waves": waves,
    }
    contract["contract_id"] = _sha256_json(contract)
    return contract


def load_contract(path: Path) -> dict[str, Any]:
    """Validate a content-addressed contract and its plan/source chain."""
    resolved = path.expanduser().resolve(strict=True)
    contract = _read_json(resolved)
    required = {
        "contract_id",
        "criteria",
        "measurement",
        "plan",
        "plan_id",
        "plan_sha256",
        "repetitions",
        "schema",
        "topology",
        "warmup_repetitions",
        "waves",
    }
    if set(contract) != required:
        raise ABError(
            f"contract fields mismatch: missing={sorted(required - set(contract))} "
            f"extra={sorted(set(contract) - required)}"
        )
    if contract["schema"] != CONTRACT_SCHEMA:
        raise ABError(f"unsupported contract schema {contract['schema']!r}")
    _content_id(contract, "contract_id")
    plan_path = Path(contract["plan"]).expanduser().resolve(strict=True)
    if _sha256_file(plan_path) != contract["plan_sha256"]:
        raise ABError("bound A/B plan digest changed")
    plan = validate_plan(plan_path)
    if plan["plan_id"] != contract["plan_id"]:
        raise ABError("contract plan_id differs from the validated plan")
    if contract["topology"] != plan["topology"]:
        raise ABError("contract topology differs from the validated plan")
    if int(contract["repetitions"]) < 2 or int(contract["warmup_repetitions"]) < 1:
        raise ABError("contract has invalid repetition counts")
    if contract["measurement"] not in {"generation_s", "end_to_end_step_s"}:
        raise ABError("contract has an unsupported measurement")
    if len(contract["waves"]) != len(plan["waves"]):
        raise ABError("contract wave count differs from the validated plan")
    expected = build_contract(
        plan_path,
        repetitions=int(contract["repetitions"]),
        warmup_repetitions=int(contract["warmup_repetitions"]),
        measurement=str(contract["measurement"]),
        minimum_speedup=float(contract["criteria"]["minimum_paired_speedup"]),
        minimum_win_fraction=float(contract["criteria"]["minimum_win_fraction"]),
        maximum_reward_delta=float(contract["criteria"]["maximum_absolute_reward_delta"]),
        maximum_trace_overhead=float(contract["criteria"]["maximum_trace_overhead_fraction"]),
    )
    if expected != contract:
        raise ABError("contract semantics differ from the validated A/B plan")
    return contract


def result_template(contract: dict[str, Any], arm: str) -> dict[str, Any]:
    """Return the exact result shape a future GPU runner must populate."""
    if arm not in ARMS:
        raise ABError(f"arm must be one of {ARMS}")
    runs = []
    ranks = int(contract["topology"]["dp_groups"])
    for repetition in range(int(contract["repetitions"])):
        for wave in contract["waves"]:
            runs.append(
                {
                    "repetition": repetition,
                    "wave": wave["wave"],
                    "workload_sha256": wave["workload_sha256"],
                    "sample_set_sha256": wave["sample_set_sha256"],
                    "rank_root_ids": wave["arms"][arm]["rank_root_ids"],
                    "rank_elapsed_s": [None] * ranks,
                    "successful_samples": None,
                    "reward_mean": None,
                    "output_digest": None,
                }
            )
    return {
        "schema": RESULT_SCHEMA,
        "contract_id": contract["contract_id"],
        "plan_id": contract["plan_id"],
        "arm": arm,
        "measurement": contract["measurement"],
        "trace_overhead_fraction": None,
        "runs": runs,
    }


def _load_result(path: Path, contract: dict[str, Any], arm: str) -> dict[str, Any]:
    result = _read_json(path.expanduser().resolve(strict=True))
    required = {
        "arm",
        "contract_id",
        "measurement",
        "plan_id",
        "runs",
        "schema",
        "trace_overhead_fraction",
    }
    if set(result) != required:
        raise ABError(
            f"result fields mismatch in {path}: missing={sorted(required - set(result))} "
            f"extra={sorted(set(result) - required)}"
        )
    if result["schema"] != RESULT_SCHEMA or result["arm"] != arm:
        raise ABError(f"result {path} has the wrong schema or arm")
    if result["contract_id"] != contract["contract_id"] or result["plan_id"] != contract["plan_id"]:
        raise ABError(f"result {path} belongs to a different contract or plan")
    if result["measurement"] != contract["measurement"]:
        raise ABError(f"result {path} uses another measurement")
    overhead = float(result["trace_overhead_fraction"])
    if not math.isfinite(overhead) or overhead < 0.0:
        raise ABError(f"result {path} has invalid trace_overhead_fraction")

    expected_runs = int(contract["repetitions"]) * len(contract["waves"])
    if not isinstance(result["runs"], list) or len(result["runs"]) != expected_runs:
        raise ABError(f"result {path} has {len(result.get('runs', []))} runs, expected {expected_runs}")
    wave_contracts = {int(wave["wave"]): wave for wave in contract["waves"]}
    seen = set()
    for run in result["runs"]:
        required_run = {
            "output_digest",
            "rank_elapsed_s",
            "rank_root_ids",
            "repetition",
            "reward_mean",
            "sample_set_sha256",
            "successful_samples",
            "wave",
            "workload_sha256",
        }
        if not isinstance(run, dict) or set(run) != required_run:
            raise ABError(f"result {path} has malformed run fields")
        key = (int(run["repetition"]), int(run["wave"]))
        if key in seen:
            raise ABError(f"result {path} duplicates run {key}")
        seen.add(key)
        if key[0] < 0 or key[0] >= int(contract["repetitions"]) or key[1] not in wave_contracts:
            raise ABError(f"result {path} has out-of-range run {key}")
        wave = wave_contracts[key[1]]
        if run["workload_sha256"] != wave["workload_sha256"]:
            raise ABError(f"result {path} run {key} has the wrong workload digest")
        if run["sample_set_sha256"] != wave["sample_set_sha256"]:
            raise ABError(f"result {path} run {key} has the wrong sample-set digest")
        if run["rank_root_ids"] != wave["arms"][arm]["rank_root_ids"]:
            raise ABError(f"result {path} run {key} does not execute the contracted rank assignment")
        if int(run["successful_samples"]) != int(wave["expected_samples"]):
            raise ABError(f"result {path} run {key} has missing or extra samples")
        elapsed = run["rank_elapsed_s"]
        ranks = int(contract["topology"]["dp_groups"])
        if (
            not isinstance(elapsed, list)
            or len(elapsed) != ranks
            or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in elapsed)
        ):
            raise ABError(f"result {path} run {key} has invalid rank elapsed times")
        reward = float(run["reward_mean"])
        if not math.isfinite(reward):
            raise ABError(f"result {path} run {key} has a non-finite reward mean")
        output_digest = run["output_digest"]
        if not isinstance(output_digest, str) or not output_digest:
            raise ABError(f"result {path} run {key} must contain an order-independent output digest")
    expected_keys = {
        (repetition, int(wave["wave"]))
        for repetition in range(int(contract["repetitions"]))
        for wave in contract["waves"]
    }
    if seen != expected_keys:
        raise ABError(f"result {path} does not contain the exact contracted run set")
    return result


def analyze_results(contract_path: Path, baseline_path: Path, grouped_path: Path) -> dict[str, Any]:
    """Validate paired result files and calculate the measured go/no-go gate."""
    contract_path = contract_path.expanduser().resolve(strict=True)
    contract = load_contract(contract_path)
    baseline = _load_result(baseline_path, contract, "baseline")
    grouped = _load_result(grouped_path, contract, "grouped_lpt")
    baseline_by_key = {(int(run["repetition"]), int(run["wave"])): run for run in baseline["runs"]}
    grouped_by_key = {(int(run["repetition"]), int(run["wave"])): run for run in grouped["runs"]}
    pairs = []
    output_checked = 0
    output_mismatches = 0
    for key in sorted(baseline_by_key):
        control = baseline_by_key[key]
        treatment = grouped_by_key[key]
        control_makespan = max(float(value) for value in control["rank_elapsed_s"])
        treatment_makespan = max(float(value) for value in treatment["rank_elapsed_s"])
        reward_delta = float(treatment["reward_mean"]) - float(control["reward_mean"])
        output_checked += 1
        output_mismatches += control["output_digest"] != treatment["output_digest"]
        pairs.append(
            {
                "repetition": key[0],
                "wave": key[1],
                "baseline_makespan_s": control_makespan,
                "grouped_makespan_s": treatment_makespan,
                "speedup": control_makespan / treatment_makespan,
                "reward_delta": reward_delta,
                "output_digest_match": control["output_digest"] == treatment["output_digest"],
            }
        )

    speedups = [row["speedup"] for row in pairs]
    aggregate_speedup = sum(row["baseline_makespan_s"] for row in pairs) / sum(
        row["grouped_makespan_s"] for row in pairs
    )
    win_fraction = sum(value > 1.0 for value in speedups) / len(speedups)
    max_reward_delta = max(abs(row["reward_delta"]) for row in pairs)
    max_overhead = max(float(baseline["trace_overhead_fraction"]), float(grouped["trace_overhead_fraction"]))
    criteria = contract["criteria"]
    reasons = []
    if aggregate_speedup < float(criteria["minimum_paired_speedup"]):
        reasons.append(f"aggregate speedup {aggregate_speedup:.6f}x < {float(criteria['minimum_paired_speedup']):.6f}x")
    if statistics.median(speedups) < float(criteria["minimum_paired_speedup"]):
        reasons.append(
            f"median paired speedup {statistics.median(speedups):.6f}x < "
            f"{float(criteria['minimum_paired_speedup']):.6f}x"
        )
    if win_fraction < float(criteria["minimum_win_fraction"]):
        reasons.append(f"paired win fraction {win_fraction:.6f} < {float(criteria['minimum_win_fraction']):.6f}")
    if max_reward_delta > float(criteria["maximum_absolute_reward_delta"]):
        reasons.append(
            f"maximum absolute reward delta {max_reward_delta:.6f} > "
            f"{float(criteria['maximum_absolute_reward_delta']):.6f}"
        )
    if max_overhead > float(criteria["maximum_trace_overhead_fraction"]):
        reasons.append(f"trace overhead {max_overhead:.6f} > {float(criteria['maximum_trace_overhead_fraction']):.6f}")
    if output_mismatches:
        reasons.append(f"{output_mismatches}/{output_checked} paired output digests differ")
    decision = "NO-GO" if reasons else "GO"
    result = {
        "schema": ANALYSIS_SCHEMA,
        "contract": str(contract_path),
        "contract_id": contract["contract_id"],
        "contract_sha256": _sha256_file(contract_path),
        "result_sha256": {
            "baseline": _sha256_file(baseline_path.expanduser().resolve(strict=True)),
            "grouped_lpt": _sha256_file(grouped_path.expanduser().resolve(strict=True)),
        },
        "measurement": contract["measurement"],
        "decision": {
            "decision": decision,
            "go": decision == "GO",
            "reasons": reasons or ["performance, stability, reward, and trace-overhead gates passed"],
        },
        "metrics": {
            "pairs": len(pairs),
            "aggregate_speedup": aggregate_speedup,
            "median_paired_speedup": statistics.median(speedups),
            "minimum_paired_speedup": min(speedups),
            "maximum_paired_speedup": max(speedups),
            "paired_win_fraction": win_fraction,
            "maximum_absolute_reward_delta": max_reward_delta,
            "maximum_trace_overhead_fraction": max_overhead,
            "output_digest_pairs_checked": output_checked,
            "output_digest_mismatches": output_mismatches,
        },
        "pairs": pairs,
    }
    result["analysis_id"] = _sha256_json(result)
    return result


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


def _ratio_above_one(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a finite ratio greater than 1.0")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a finite fraction in [0, 1]")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("expected a finite non-negative number")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="create a content-addressed paired A/B contract")
    prepare.add_argument("--plan", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--templates-dir", type=Path)
    prepare.add_argument("--measurement", choices=("generation_s", "end_to_end_step_s"), default="generation_s")
    prepare.add_argument("--repetitions", type=_positive_int, default=3)
    prepare.add_argument("--warmup-repetitions", type=_non_negative_int, default=1)
    prepare.add_argument("--minimum-speedup", type=_ratio_above_one, default=1.05)
    prepare.add_argument("--minimum-win-fraction", type=_fraction, default=0.75)
    prepare.add_argument("--maximum-reward-delta", type=_non_negative_float, default=0.01)
    prepare.add_argument("--maximum-trace-overhead", type=_fraction, default=0.005)

    validate = subparsers.add_parser("validate", help="validate a contract and its complete source chain")
    validate.add_argument("--contract", required=True, type=Path)

    analyze = subparsers.add_parser("analyze", help="validate paired results and emit measured go/no-go")
    analyze.add_argument("--contract", required=True, type=Path)
    analyze.add_argument("--baseline", required=True, type=Path)
    analyze.add_argument("--grouped", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--require-go", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        if args.command == "prepare":
            contract = build_contract(
                args.plan,
                repetitions=args.repetitions,
                warmup_repetitions=args.warmup_repetitions,
                measurement=args.measurement,
                minimum_speedup=args.minimum_speedup,
                minimum_win_fraction=args.minimum_win_fraction,
                maximum_reward_delta=args.maximum_reward_delta,
                maximum_trace_overhead=args.maximum_trace_overhead,
            )
            output = args.output.expanduser().resolve()
            _write_json(output, contract)
            if args.templates_dir is not None:
                directory = args.templates_dir.expanduser().resolve()
                for arm in ARMS:
                    _write_json(directory / f"{arm}.template.json", result_template(contract, arm))
            print(f"contract_id={contract['contract_id']}")
            print(f"runs_per_arm={contract['repetitions'] * len(contract['waves'])}")
            return
        if args.command == "validate":
            contract = load_contract(args.contract)
            print(f"contract_id={contract['contract_id']}")
            print(f"runs_per_arm={contract['repetitions'] * len(contract['waves'])}")
            return
        result = analyze_results(args.contract, args.baseline, args.grouped)
        _write_json(args.output.expanduser().resolve(), result)
        print(f"decision={result['decision']['decision']}")
        print(f"aggregate_speedup={result['metrics']['aggregate_speedup']:.6f}x")
        print(f"median_paired_speedup={result['metrics']['median_paired_speedup']:.6f}x")
        print(f"paired_win_fraction={result['metrics']['paired_win_fraction']:.6f}")
        for reason in result["decision"]["reasons"]:
            print(f"reason={reason}")
        if args.require_go and not result["decision"]["go"]:
            raise SystemExit(3)
    except (ABError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
