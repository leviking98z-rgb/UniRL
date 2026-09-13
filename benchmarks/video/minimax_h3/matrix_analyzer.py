"""Validate a four-trace matrix and emit a fail-closed grouped-reordering decision."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.video.minimax_h3.grouped_reordering import simulate  # noqa: E402
from unirl.utils.minimax_h3_workload import read_workload_records  # noqa: E402

ANALYSIS_SCHEMA = "unirl:minimax-h3:fixed-geometry-analysis:v1"
_MATRIX_DRIVER = Path(__file__).with_name("matrix_driver.py")


class AnalysisError(ValueError):
    """Fail-closed trace analysis error."""


def _load_matrix_driver():
    spec = importlib.util.spec_from_file_location("minimax_h3_matrix_driver", _MATRIX_DRIVER)
    if spec is None or spec.loader is None:
        raise AnalysisError(f"cannot import matrix driver from {_MATRIX_DRIVER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"expected JSON object at {path}")
    return value


def _validate_receipt(
    *,
    driver: Any,
    manifest: dict[str, Any],
    run: dict[str, Any],
    geometry: dict[str, Any],
) -> tuple[Path, Path, dict[str, str]]:
    trace = Path(run["trace"])
    receipt_path = Path(run["receipt"])
    receipt = _read_json(receipt_path)
    expected_fields = {
        "binding",
        "command",
        "evidence",
        "geometry",
        "manifest_id",
        "records",
        "roots",
        "schema",
        "trace",
        "trace_sha256",
    }
    if set(receipt) != expected_fields:
        raise AnalysisError(
            f"receipt fields mismatch in {receipt_path}: "
            f"missing={sorted(expected_fields - set(receipt))} extra={sorted(set(receipt) - expected_fields)}"
        )
    if receipt.get("schema") != driver.RUN_SCHEMA:
        raise AnalysisError(f"unsupported receipt schema in {receipt_path}")
    if receipt.get("manifest_id") != manifest["manifest_id"]:
        raise AnalysisError(f"receipt {receipt_path} belongs to a different manifest")
    if receipt.get("binding") != manifest["binding"]:
        raise AnalysisError(f"receipt {receipt_path} binding differs from matrix")
    if receipt.get("geometry") != geometry:
        raise AnalysisError(f"receipt {receipt_path} geometry differs from matrix")
    evidence = receipt.get("evidence")
    if evidence not in (driver.MEASURED_EVIDENCE, driver.PROXY_EVIDENCE):
        raise AnalysisError(f"receipt {receipt_path} has unsupported evidence metadata")
    if Path(receipt.get("trace", "")) != trace:
        raise AnalysisError(f"receipt {receipt_path} points at another trace")
    expected_command = driver.expected_command(manifest, geometry["name"])
    if receipt.get("command") != expected_command:
        raise AnalysisError(f"receipt {receipt_path} train command differs from the matrix contract")
    if not trace.is_file():
        raise AnalysisError(f"missing trace {trace}")
    actual_sha = _sha256_file(trace)
    if receipt.get("trace_sha256") != actual_sha:
        raise AnalysisError(f"trace digest changed after receipt creation: {trace}")
    records = read_workload_records([trace])
    counts = driver.validate_trace_records(manifest, geometry, records)
    if evidence == driver.PROXY_EVIDENCE:
        if manifest["settings"]["cost"] not in {"packed_rows", "padded_rows", "attention_rows2"}:
            raise AnalysisError("analytical proxy evidence cannot support measured timing costs")
        if any(
            record.text_tokens != 0
            or record.text_embed_s != 0.0
            or record.denoise_s != 0.0
            or record.decode_s != 0.0
            or record.total_s != 0.0
            for record in records
        ):
            raise AnalysisError(f"analytical proxy fields differ from their declared zero model in {trace}")
    if receipt.get("records") != counts["records"] or receipt.get("roots") != counts["roots"]:
        raise AnalysisError(f"receipt {receipt_path} record/root counts differ from the validated trace")
    return trace, receipt_path, evidence


def _validate_cross_trace_prompts(paths: Sequence[Path], expected_roots: Sequence[str]) -> None:
    tokens_by_trace = []
    for path in paths:
        records = read_workload_records([path])
        tokens = {}
        for record in records:
            tokens.setdefault(record.root_id, record.text_tokens)
        tokens_by_trace.append([tokens[root_id] for root_id in expected_roots])
    expected_tokens = tokens_by_trace[0]
    for path, tokens in zip(paths[1:], tokens_by_trace[1:]):
        if tokens != expected_tokens:
            raise AnalysisError(f"root text-token counts differ between {paths[0]} and {path}")


def _decision(summary: dict[str, Any], *, tail_threshold: float, speedup_threshold: float) -> dict[str, Any]:
    tail_ratio = float(summary["baseline"]["tail_ratio"])
    predicted_speedup = float(summary["grouped_lpt"]["predicted_speedup"])
    reasons = []
    if tail_ratio < tail_threshold:
        reasons.append(f"baseline tail_ratio {tail_ratio:.6f} < {tail_threshold:.6f}")
    if predicted_speedup < speedup_threshold:
        reasons.append(f"predicted speedup {predicted_speedup:.6f}x < {speedup_threshold:.6f}x")
    decision = "NO-GO" if reasons else "GO"
    return {
        "decision": decision,
        "go": decision == "GO",
        "baseline_tail_ratio": tail_ratio,
        "tail_ratio_threshold": tail_threshold,
        "predicted_speedup": predicted_speedup,
        "minimum_predicted_speedup": speedup_threshold,
        "reasons": reasons or ["both imbalance and predicted-speedup thresholds passed"],
    }


def analyze_matrix(manifest_path: Path) -> dict[str, Any]:
    driver = _load_matrix_driver()
    manifest = driver._load_manifest(manifest_path)
    driver._revalidate_binding(driver._script_repo(), manifest)
    geometries = {row["name"]: row for row in manifest["geometries"]}
    expected_names = [row[0] for row in driver.DEFAULT_GEOMETRIES]
    if list(geometries) != expected_names:
        raise AnalysisError(f"matrix must contain exactly {expected_names}, got {list(geometries)}")
    runs = {row["geometry"]: row for row in manifest["runs"]}
    if list(runs) != expected_names:
        raise AnalysisError(f"run order must be exactly {expected_names}, got {list(runs)}")
    validated = [
        _validate_receipt(driver=driver, manifest=manifest, run=runs[name], geometry=geometries[name])
        for name in expected_names
    ]
    paths = [item[0] for item in validated]
    receipts = [item[1] for item in validated]
    evidence = [item[2] for item in validated]
    if any(item != evidence[0] for item in evidence[1:]):
        raise AnalysisError("matrix mixes measured and proxy evidence")
    _validate_cross_trace_prompts(paths, manifest["settings"]["expected_root_ids"])
    settings = manifest["settings"]
    summary = simulate(
        paths,
        ranks=int(settings["dp_groups"]),
        group_size=int(settings["group_size"]),
        cost_name=str(settings["cost"]),
        merge_order="round-robin",
        source_order="dp-rank",
    )
    decision = _decision(
        summary,
        tail_threshold=float(settings["tail_ratio_threshold"]),
        speedup_threshold=float(settings["minimum_predicted_speedup"]),
    )
    return {
        "schema": ANALYSIS_SCHEMA,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": _sha256_file(manifest_path),
        "binding": manifest["binding"],
        "evidence": evidence[0],
        "trace_sha256": {name: _sha256_file(paths[index]) for index, name in enumerate(expected_names)},
        "receipt_sha256": {name: _sha256_file(receipts[index]) for index, name in enumerate(expected_names)},
        "decision": decision,
        "simulation": summary,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_summary(result: dict[str, Any]) -> None:
    decision = result["decision"]
    simulation = result["simulation"]
    print(f"evidence={result['evidence']['kind']}")
    print(f"decision={decision['decision']}")
    print(f"baseline_tail_ratio={decision['baseline_tail_ratio']:.6f}")
    print(f"predicted_speedup={decision['predicted_speedup']:.6f}x")
    for reason in decision["reasons"]:
        print(f"reason={reason}")
    print(
        f"roots={simulation['roots']} samples={simulation['samples']} dp_ranks={simulation['ranks']} "
        f"group_size={simulation['group_size']} cost={simulation['cost']}"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-go", action="store_true", help="exit 3 when the validated decision is NO-GO")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        result = analyze_matrix(args.manifest.expanduser().resolve(strict=True))
    except (AnalysisError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if args.output is not None:
        _write_json(args.output.expanduser().resolve(), result)
    _print_summary(result)
    if args.require_go and not result["decision"]["go"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
