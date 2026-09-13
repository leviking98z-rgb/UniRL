"""Analyze mixed MiniMax-H3 geometry/length waves and build an auditable reorder decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.video.minimax_h3 import matrix_analyzer as fixed_analyzer  # noqa: E402
from benchmarks.video.minimax_h3.grouped_reordering import (  # noqa: E402
    RootWork,
    capacity_lpt,
    contiguous_assignment,
)
from unirl.utils.minimax_h3_workload import (  # noqa: E402
    MiniMaxH3WorkloadRecord,
    append_workload_record,
    read_workload_records,
)

ANALYSIS_SCHEMA = "unirl:minimax-h3:mixed-geometry-length-analysis:v1"
PLAN_SCHEMA = "unirl:minimax-h3:grouped-reordering-ab-plan:v1"
SCHEDULE_SCHEMA = "unirl:minimax-h3:balanced-mixed-schedule:v1"
SYNTHETIC_TRACE_SET_SCHEMA = "unirl:minimax-h3:synthetic-mixed-trace-set:v1"
STRUCTURAL_COSTS = ("packed_rows", "padded_rows", "attention_rows2")
MEASURED_COSTS = ("denoise_s", "total_s")
ALL_COSTS = (*STRUCTURAL_COSTS, *MEASURED_COSTS)
SYNTHETIC_TEXT_METHOD = "deterministic sha256(root_id) projection; synthetic, not conditioner measured"
SYNTHETIC_EVIDENCE = {
    "kind": "analytical_cpu_proxy",
    "provenance": "derived from content-addressed fixed traces with deterministic synthetic text lengths",
    "measured_roi_eligible": False,
}


class MixedAnalysisError(ValueError):
    """Fail-closed mixed-workload analysis error."""


@dataclass(frozen=True)
class FixedProfiles:
    """Validated fixed-profile records and their immutable source matrix."""

    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    fixed_analysis: dict[str, Any]
    geometry_names: tuple[str, ...]
    geometry_rows: Mapping[str, dict[str, Any]]
    root_ids: tuple[str, ...]
    records: Mapping[str, Mapping[str, tuple[MiniMaxH3WorkloadRecord, ...]]]
    trace_sha256: Mapping[str, str]
    evidence: Mapping[str, str]

    @property
    def ranks(self) -> int:
        return int(self.manifest["settings"]["dp_groups"])

    @property
    def group_size(self) -> int:
        return int(self.manifest["settings"]["group_size"])

    @property
    def samples_per_prompt(self) -> int:
        return int(self.manifest["settings"]["samples_per_prompt"])

    @property
    def sp_size(self) -> int:
        return int(self.manifest["settings"]["sp_size"])


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record_cost(record: MiniMaxH3WorkloadRecord, cost_name: str) -> float:
    if cost_name == "packed_rows":
        return float(record.packed_rows)
    if cost_name == "padded_rows":
        return float(record.padded_rows)
    if cost_name == "attention_rows2":
        return float(record.padded_rows) ** 2
    if cost_name not in MEASURED_COSTS:
        raise MixedAnalysisError(f"unsupported cost {cost_name!r}; choose from {ALL_COSTS}")
    value = float(getattr(record, cost_name))
    if not math.isfinite(value) or value <= 0.0:
        raise MixedAnalysisError(
            f"measured cost {cost_name!r} must be positive; sample {record.sample_id!r} has {value}"
        )
    return value


def _records_by_root(
    records: Sequence[MiniMaxH3WorkloadRecord],
) -> dict[str, tuple[MiniMaxH3WorkloadRecord, ...]]:
    grouped: dict[str, list[MiniMaxH3WorkloadRecord]] = {}
    seen_samples: set[str] = set()
    for record in records:
        if record.sample_id in seen_samples:
            raise MixedAnalysisError(f"duplicate sample_id {record.sample_id!r}")
        seen_samples.add(record.sample_id)
        grouped.setdefault(record.root_id, []).append(record)
    return {root_id: tuple(rows) for root_id, rows in grouped.items()}


def load_fixed_profiles(manifest_path: Path) -> FixedProfiles:
    """Validate a fixed matrix and index every prompt/geometry profile."""
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    fixed_analysis = fixed_analyzer.analyze_matrix(manifest_path)
    driver = fixed_analyzer._load_matrix_driver()
    manifest = driver._load_manifest(manifest_path)
    geometry_names = tuple(row["name"] for row in manifest["geometries"])
    geometry_rows = {row["name"]: row for row in manifest["geometries"]}
    root_ids = tuple(manifest["settings"]["expected_root_ids"])
    runs = {row["geometry"]: row for row in manifest["runs"]}
    indexed: dict[str, dict[str, tuple[MiniMaxH3WorkloadRecord, ...]]] = {}
    trace_sha256: dict[str, str] = {}
    for geometry_name in geometry_names:
        trace = Path(runs[geometry_name]["trace"])
        trace_sha256[geometry_name] = _sha256_file(trace)
        indexed[geometry_name] = _records_by_root(read_workload_records([trace]))
        if set(indexed[geometry_name]) != set(root_ids):
            raise MixedAnalysisError(f"fixed trace {trace} root set differs from the source-bound prompt set")

    for root_id in root_ids:
        expected_samples: tuple[str, ...] | None = None
        expected_tokens: int | None = None
        for geometry_name in geometry_names:
            rows = indexed[geometry_name][root_id]
            sample_ids = tuple(row.sample_id for row in rows)
            tokens = {row.text_tokens for row in rows}
            if len(tokens) != 1:
                raise MixedAnalysisError(f"root {root_id!r} has inconsistent text lengths in {geometry_name}")
            if expected_samples is None:
                expected_samples = sample_ids
                expected_tokens = next(iter(tokens))
            elif sample_ids != expected_samples or next(iter(tokens)) != expected_tokens:
                raise MixedAnalysisError(f"root {root_id!r} differs across fixed geometry profiles")

    return FixedProfiles(
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        manifest=manifest,
        fixed_analysis=fixed_analysis,
        geometry_names=geometry_names,
        geometry_rows=geometry_rows,
        root_ids=root_ids,
        records=indexed,
        trace_sha256=trace_sha256,
        evidence=fixed_analysis["evidence"],
    )


def balanced_geometry_schedule(
    root_ids: Sequence[str],
    geometry_names: Sequence[str],
    *,
    source_id: str,
    seed: str,
    trial: int,
    waves: int,
) -> list[dict[str, str]]:
    """Build balanced waves in which every root sees every geometry once per cycle."""
    roots = tuple(root_ids)
    geometries = tuple(geometry_names)
    if len(geometries) < 2:
        raise MixedAnalysisError("mixed scheduling requires at least two geometries")
    if len(roots) % len(geometries):
        raise MixedAnalysisError(f"root count {len(roots)} must be divisible by geometry count {len(geometries)}")
    if waves < 1 or waves % len(geometries):
        raise MixedAnalysisError(f"waves={waves} must be a positive multiple of geometry count {len(geometries)}")
    if trial < 0:
        raise MixedAnalysisError(f"trial must be non-negative, got {trial}")

    result: list[dict[str, str]] = []
    for cycle in range(waves // len(geometries)):
        prefix = f"{SCHEDULE_SCHEMA}\0{source_id}\0{seed}\0{trial}\0{cycle}\0"
        ordered = sorted(roots, key=lambda root_id: (hashlib.sha256((prefix + root_id).encode()).digest(), root_id))
        positions = {root_id: index for index, root_id in enumerate(ordered)}
        for phase in range(len(geometries)):
            assignment = {root_id: geometries[(positions[root_id] + phase) % len(geometries)] for root_id in roots}
            counts = {name: sum(value == name for value in assignment.values()) for name in geometries}
            if len(set(counts.values())) != 1:
                raise AssertionError(f"internal schedule imbalance: {counts}")
            result.append(assignment)
    return result


def _summary_from_loads(loads: Sequence[float]) -> dict[str, Any]:
    values = [float(value) for value in loads]
    if not values:
        raise MixedAnalysisError("cannot summarize an empty rank-load vector")
    total = sum(values)
    mean = total / len(values)
    maximum = max(values)
    minimum = min(values)
    return {
        "rank_loads": values,
        "min": minimum,
        "mean": mean,
        "max": maximum,
        "tail_ratio": maximum / mean if mean else 0.0,
        "imbalance": (maximum - minimum) / mean if mean else 0.0,
        "efficiency": total / (len(values) * maximum) if maximum else 1.0,
    }


def _assignment_cost_summary(
    assignments: Sequence[Sequence[RootWork]],
    root_costs: Mapping[str, float],
) -> dict[str, Any]:
    loads = [sum(root_costs[item.root_id] for item in assignment) for assignment in assignments]
    summary = _summary_from_loads(loads)
    summary["assignments"] = [
        {
            "rank": rank,
            "load": loads[rank],
            "root_ids": [item.root_id for item in assignment],
        }
        for rank, assignment in enumerate(assignments)
    ]
    return summary


def _recorded_assignment(items: Sequence[RootWork], ranks: int) -> list[list[RootWork]]:
    if len(items) < ranks or len(items) % ranks:
        raise MixedAnalysisError(f"root count {len(items)} must be divisible by ranks={ranks}")
    assignments: list[list[RootWork]] = [[] for _ in range(ranks)]
    for item in items:
        if item.source_dp_rank < 0 or item.source_dp_rank >= ranks:
            raise MixedAnalysisError(f"root {item.root_id!r} has invalid recorded DP rank {item.source_dp_rank}")
        assignments[item.source_dp_rank].append(item)
    capacity = len(items) // ranks
    counts = [len(assignment) for assignment in assignments]
    if counts != [capacity] * ranks:
        raise MixedAnalysisError(f"recorded DP placement is not equal-capacity: {counts}")
    return assignments


def _grouped_lpt_from_baseline(
    baseline: Sequence[Sequence[RootWork]],
    group_size: int,
) -> list[list[RootWork]]:
    ranks = len(baseline)
    if group_size < 1 or ranks % group_size:
        raise MixedAnalysisError(f"group_size={group_size} must be a positive divisor of ranks={ranks}")
    assignments: list[list[RootWork]] = [[] for _ in range(ranks)]
    for start in range(0, ranks, group_size):
        local_items = [item for rank in range(start, start + group_size) for item in baseline[rank]]
        local_assignments = capacity_lpt(local_items, group_size)
        for local_rank, assigned in enumerate(local_assignments):
            assignments[start + local_rank] = assigned
    return assignments


def _wave_root_work(
    root_ids: Sequence[str],
    wave: Mapping[str, Sequence[MiniMaxH3WorkloadRecord]],
    *,
    predictor_cost: str,
) -> list[RootWork]:
    items = []
    for index, root_id in enumerate(root_ids):
        rows = tuple(wave[root_id])
        items.append(
            RootWork(
                root_id=root_id,
                source_root_id=root_id,
                source=f"{rows[0].height}x{rows[0].width}x{rows[0].num_frames}",
                source_index=index,
                source_dp_rank=rows[0].dp_rank,
                first_record_index=index,
                sample_ids=tuple(row.sample_id for row in rows),
                sample_costs=tuple(_record_cost(row, predictor_cost) for row in rows),
            )
        )
    return items


def _simulate_waves(
    root_ids: Sequence[str],
    waves: Sequence[Mapping[str, Sequence[MiniMaxH3WorkloadRecord]]],
    *,
    ranks: int,
    group_size: int,
    predictor_cost: str,
    outcome_costs: Sequence[str],
    baseline_placement: str,
    include_wave_details: bool,
) -> dict[str, Any]:
    if predictor_cost not in STRUCTURAL_COSTS:
        raise MixedAnalysisError(f"scheduling predictor must be structural; choose from {STRUCTURAL_COSTS}")
    if not waves:
        raise MixedAnalysisError("at least one mixed workload wave is required")

    totals = {
        cost: {"baseline_makespan": 0.0, "grouped_makespan": 0.0, "ideal_makespan": 0.0} for cost in outcome_costs
    }
    wave_results = []
    for wave_index, wave in enumerate(waves):
        items = _wave_root_work(root_ids, wave, predictor_cost=predictor_cost)
        if baseline_placement == "contiguous":
            baseline_assignments = contiguous_assignment(items, ranks)
        elif baseline_placement == "recorded":
            baseline_assignments = _recorded_assignment(items, ranks)
        else:
            raise MixedAnalysisError(f"unsupported baseline placement {baseline_placement!r}")
        grouped_assignments = _grouped_lpt_from_baseline(baseline_assignments, group_size)
        geometries = {
            root_id: f"{wave[root_id][0].height}x{wave[root_id][0].width}x{wave[root_id][0].num_frames}"
            for root_id in root_ids
        }
        geometry_counts = {name: list(geometries.values()).count(name) for name in sorted(set(geometries.values()))}
        result: dict[str, Any] = {
            "wave": wave_index,
            "geometry_counts": geometry_counts,
            "outcomes": {},
        }
        for cost_name in outcome_costs:
            root_costs = {
                root_id: sum(_record_cost(record, cost_name) for record in wave[root_id]) for root_id in root_ids
            }
            baseline = _assignment_cost_summary(baseline_assignments, root_costs)
            grouped = _assignment_cost_summary(grouped_assignments, root_costs)
            speedup = baseline["max"] / grouped["max"] if grouped["max"] else 1.0
            grouped["predicted_speedup"] = speedup
            totals[cost_name]["baseline_makespan"] += baseline["max"]
            totals[cost_name]["grouped_makespan"] += grouped["max"]
            totals[cost_name]["ideal_makespan"] += baseline["mean"]
            result["outcomes"][cost_name] = {
                "baseline": baseline,
                "grouped_lpt": grouped,
            }
        if include_wave_details:
            result["request_rows"] = [
                {
                    "root_id": root_id,
                    "geometry": geometries[root_id],
                    "height": wave[root_id][0].height,
                    "width": wave[root_id][0].width,
                    "num_frames": wave[root_id][0].num_frames,
                    "text_tokens": wave[root_id][0].text_tokens,
                    "sample_ids": [row.sample_id for row in wave[root_id]],
                    "predictor_sample_costs": [_record_cost(row, predictor_cost) for row in wave[root_id]],
                    "predictor_root_cost": sum(_record_cost(row, predictor_cost) for row in wave[root_id]),
                }
                for root_id in root_ids
            ]
            result["arms"] = {
                "baseline": {
                    "rank_root_ids": [[item.root_id for item in assignment] for assignment in baseline_assignments],
                    "flat_root_order": [item.root_id for assignment in baseline_assignments for item in assignment],
                },
                "grouped_lpt": {
                    "rank_root_ids": [[item.root_id for item in assignment] for assignment in grouped_assignments],
                    "flat_root_order": [item.root_id for assignment in grouped_assignments for item in assignment],
                },
            }
        wave_results.append(result)

    aggregate = {}
    for cost_name, values in totals.items():
        baseline = values["baseline_makespan"]
        grouped = values["grouped_makespan"]
        ideal = values["ideal_makespan"]
        aggregate[cost_name] = {
            **values,
            "baseline_tail_ratio": baseline / ideal if ideal else 0.0,
            "grouped_tail_ratio": grouped / ideal if ideal else 0.0,
            "predicted_speedup": baseline / grouped if grouped else 1.0,
        }
    return {
        "predictor_cost": predictor_cost,
        "ranks": ranks,
        "group_size": group_size,
        "baseline_placement": baseline_placement,
        "waves": len(waves),
        "aggregate": aggregate,
        "wave_results": wave_results if include_wave_details else None,
    }


def _fixed_wave_records(
    profiles: FixedProfiles,
    assignments: Sequence[Mapping[str, str]],
) -> list[dict[str, tuple[MiniMaxH3WorkloadRecord, ...]]]:
    waves = []
    for assignment in assignments:
        if tuple(assignment) != profiles.root_ids:
            raise MixedAnalysisError("schedule root order differs from the matrix prompt order")
        waves.append({root_id: profiles.records[assignment[root_id]][root_id] for root_id in profiles.root_ids})
    return waves


def _replace_record(
    record: MiniMaxH3WorkloadRecord,
    *,
    text_tokens: int | None = None,
    dp_rank: int | None = None,
) -> MiniMaxH3WorkloadRecord:
    """Rebuild one trace row after changing only synthetic workload fields."""
    return MiniMaxH3WorkloadRecord.build(
        sample_id=record.sample_id,
        root_id=record.root_id,
        height=record.height,
        width=record.width,
        num_frames=record.num_frames,
        text_tokens=record.text_tokens if text_tokens is None else int(text_tokens),
        sp_size=record.sp_size,
        dp_rank=record.dp_rank if dp_rank is None else int(dp_rank),
        sp_rank=record.sp_rank,
        text_embed_s=record.text_embed_s,
        denoise_s=record.denoise_s,
        decode_s=record.decode_s,
        total_s=record.total_s,
    )


def _synthetic_text_tokens(
    profiles: FixedProfiles,
    *,
    seed: str,
    minimum: int,
    maximum: int,
) -> dict[str, int]:
    if minimum < 1 or maximum <= minimum:
        raise MixedAnalysisError(f"synthetic mixed lengths require 1 <= minimum < maximum, got {minimum}..{maximum}")
    span = maximum - minimum + 1
    values = {}
    for root_id in profiles.root_ids:
        digest = hashlib.sha256(f"{SYNTHETIC_TRACE_SET_SCHEMA}\0{seed}\0{root_id}".encode()).digest()
        values[root_id] = minimum + int.from_bytes(digest[:8], "big") % span
    if len(values) > 1 and len(set(values.values())) == 1:
        values[profiles.root_ids[0]] = minimum
        values[profiles.root_ids[-1]] = maximum
    return values


def _baseline_rank_assignments(
    profiles: FixedProfiles,
    wave: Mapping[str, Sequence[MiniMaxH3WorkloadRecord]],
    *,
    mode: str,
) -> dict[str, int]:
    roots_per_rank = len(profiles.root_ids) // profiles.ranks
    if mode == "contiguous":
        return {root_id: index // roots_per_rank for index, root_id in enumerate(profiles.root_ids)}
    if mode != "cost-clustered":
        raise MixedAnalysisError(f"unsupported synthetic baseline placement {mode!r}")

    rank_ids = []
    for rank in range(profiles.ranks):
        rank_ids.extend([rank] * roots_per_rank)
    ordered = sorted(
        profiles.root_ids,
        key=lambda root_id: (
            -sum(record.padded_rows**2 for record in wave[root_id]),
            root_id,
        ),
    )
    return {root_id: rank_ids[index] for index, root_id in enumerate(ordered)}


def materialize_synthetic_mixed_traces(
    manifest_path: Path,
    output_dir: Path,
    *,
    seed: str,
    trial: int,
    waves: int,
    text_token_min: int,
    text_token_max: int,
    baseline_placement: str,
) -> dict[str, Any]:
    """Write auditable CPU-only mixed geometry/length waves bound to fixed traces."""
    profiles = load_fixed_profiles(manifest_path)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.is_relative_to(fixed_analyzer._load_matrix_driver()._script_repo()):
        raise MixedAnalysisError(f"synthetic output_dir must be outside the source checkout: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MixedAnalysisError(f"synthetic output directory must not exist or must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    assignments = balanced_geometry_schedule(
        profiles.root_ids,
        profiles.geometry_names,
        source_id=profiles.manifest["manifest_id"],
        seed=seed,
        trial=trial,
        waves=waves,
    )
    max_sequence_length = int(profiles.manifest["frozen_config"]["bundle.config"].get("max_sequence_length", 512))
    if text_token_max > max_sequence_length:
        raise MixedAnalysisError(
            f"synthetic text_token_max={text_token_max} exceeds bound conditioner limit {max_sequence_length}"
        )
    fixed_waves = _fixed_wave_records(profiles, assignments)
    token_counts = _synthetic_text_tokens(
        profiles,
        seed=seed,
        minimum=text_token_min,
        maximum=text_token_max,
    )
    traces = []
    for wave_index, wave in enumerate(fixed_waves):
        tokenized_wave = {
            root_id: tuple(_replace_record(row, text_tokens=token_counts[root_id]) for row in wave[root_id])
            for root_id in profiles.root_ids
        }
        ranks = _baseline_rank_assignments(profiles, tokenized_wave, mode=baseline_placement)
        trace = output_dir / f"wave-{wave_index:02d}.jsonl"
        for root_id in profiles.root_ids:
            for row in tokenized_wave[root_id]:
                append_workload_record(
                    trace,
                    _replace_record(
                        row,
                        dp_rank=ranks[root_id],
                    ),
                )
        traces.append(
            {
                "wave": wave_index,
                "path": str(trace),
                "sha256": _sha256_file(trace),
            }
        )

    document = {
        "schema": SYNTHETIC_TRACE_SET_SCHEMA,
        "source": {
            "matrix": str(profiles.manifest_path),
            "matrix_id": profiles.manifest["manifest_id"],
            "matrix_sha256": profiles.manifest_sha256,
            "trace_sha256": dict(profiles.trace_sha256),
            "source_binding": profiles.manifest["binding"],
        },
        "schedule": {
            "schema": SCHEDULE_SCHEMA,
            "seed": seed,
            "trial": trial,
            "waves": waves,
            "geometry_names": list(profiles.geometry_names),
            "balanced_geometry_per_wave": True,
            "full_geometry_cross_over_per_cycle": True,
        },
        "topology": {
            "physical_devices": int(profiles.manifest["settings"]["num_devices"]),
            "sp_size": profiles.sp_size,
            "dp_groups": profiles.ranks,
            "group_size": profiles.group_size,
            "samples_per_prompt": profiles.samples_per_prompt,
        },
        "text_tokens": {
            "method": SYNTHETIC_TEXT_METHOD,
            "minimum": text_token_min,
            "maximum": text_token_max,
            "conditioner_limit": max_sequence_length,
            "by_root": token_counts,
        },
        "baseline_placement": baseline_placement,
        "evidence": SYNTHETIC_EVIDENCE,
        "traces": traces,
    }
    document["trace_set_id"] = _sha256_json(document)
    _write_json(output_dir / "trace-set.json", document)
    return document


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise MixedAnalysisError("cannot calculate a quantile over no values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    return {
        "min": min(numbers),
        "p10": _quantile(numbers, 0.10),
        "median": statistics.median(numbers),
        "mean": statistics.fmean(numbers),
        "p90": _quantile(numbers, 0.90),
        "max": max(numbers),
    }


def _gate(
    *,
    tail_ratio: float,
    speedup: float,
    tail_threshold: float,
    speedup_threshold: float,
) -> dict[str, Any]:
    reasons = []
    if tail_ratio < tail_threshold:
        reasons.append(f"baseline tail_ratio {tail_ratio:.6f} < {tail_threshold:.6f}")
    if speedup < speedup_threshold:
        reasons.append(f"predicted speedup {speedup:.6f}x < {speedup_threshold:.6f}x")
    decision = "NO-GO" if reasons else "GO"
    return {
        "decision": decision,
        "go": decision == "GO",
        "baseline_tail_ratio": tail_ratio,
        "tail_ratio_threshold": tail_threshold,
        "predicted_speedup": speedup,
        "minimum_predicted_speedup": speedup_threshold,
        "reasons": reasons or ["both imbalance and predicted-speedup thresholds passed"],
    }


def _text_length_summary(
    waves: Sequence[Mapping[str, Sequence[MiniMaxH3WorkloadRecord]]],
    root_ids: Sequence[str],
) -> dict[str, Any]:
    values = [int(waves[0][root_id][0].text_tokens) for root_id in root_ids]
    for wave in waves[1:]:
        actual = [int(wave[root_id][0].text_tokens) for root_id in root_ids]
        if actual != values:
            raise MixedAnalysisError("prompt text-token counts differ across mixed workload waves")
    ordered = sorted(values)
    return {
        "available": any(value > 0 for value in values),
        "varies": len(set(values)) > 1,
        "unique": len(set(values)),
        "min": min(values),
        "median": statistics.median(ordered),
        "max": max(values),
    }


def _primary_outcome(profiles: FixedProfiles, predictor_cost: str) -> str:
    if profiles.evidence.get("kind") != "measured_gpu_trace":
        return predictor_cost
    for geometry_name in profiles.geometry_names:
        for rows in profiles.records[geometry_name].values():
            for row in rows:
                if row.total_s <= 0.0:
                    raise MixedAnalysisError("measured fixed-profile evidence contains a non-positive total_s")
    return "total_s"


def _build_fixed_plan(
    profiles: FixedProfiles,
    *,
    predictor_cost: str,
    seed: str,
    trial: int,
    assignments: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    waves = _fixed_wave_records(profiles, assignments)
    simulation = _simulate_waves(
        profiles.root_ids,
        waves,
        ranks=profiles.ranks,
        group_size=profiles.group_size,
        predictor_cost=predictor_cost,
        outcome_costs=(predictor_cost,),
        baseline_placement="contiguous",
        include_wave_details=True,
    )
    plan = {
        "schema": PLAN_SCHEMA,
        "source": {
            "mode": "fixed_profile_replay",
            "matrix": str(profiles.manifest_path),
            "matrix_id": profiles.manifest["manifest_id"],
            "matrix_sha256": profiles.manifest_sha256,
            "trace_sha256": dict(profiles.trace_sha256),
            "source_binding": profiles.manifest["binding"],
            "evidence": dict(profiles.evidence),
        },
        "schedule": {
            "schema": SCHEDULE_SCHEMA,
            "seed": seed,
            "trial": trial,
            "waves": len(assignments),
            "geometry_names": list(profiles.geometry_names),
        },
        "topology": {
            "physical_devices": int(profiles.manifest["settings"]["num_devices"]),
            "sp_size": profiles.sp_size,
            "dp_groups": profiles.ranks,
            "group_size": profiles.group_size,
            "samples_per_prompt": profiles.samples_per_prompt,
        },
        "predictor_cost": predictor_cost,
        "waves": simulation["wave_results"],
        "execution_contract": {
            "paired_arms": ["baseline", "grouped_lpt"],
            "same_roots_geometry_text_and_samples": True,
            "equal_roots_per_rank": True,
            "siblings_are_atomic": True,
            "directly_executable_on_current_unirl": False,
            "blocker": (
                "MiniMax-H3 currently carries one shared DiffusionSamplingParams and one dense latent shape per "
                "rollout Sample; request-level mixed geometry needs a shape-aware scheduler/worker pool first"
            ),
        },
    }
    plan["plan_id"] = _sha256_json(plan)
    return plan


def _build_observed_plan(
    profiles: FixedProfiles,
    paths: Sequence[Path],
    waves: Sequence[Mapping[str, Sequence[MiniMaxH3WorkloadRecord]]],
    *,
    predictor_cost: str,
    synthetic_trace_set: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    simulation = _simulate_waves(
        profiles.root_ids,
        waves,
        ranks=profiles.ranks,
        group_size=profiles.group_size,
        predictor_cost=predictor_cost,
        outcome_costs=(predictor_cost,),
        baseline_placement="recorded",
        include_wave_details=True,
    )
    source: dict[str, Any] = {
        "mode": "observed_mixed_trace",
        "matrix": str(profiles.manifest_path),
        "matrix_id": profiles.manifest["manifest_id"],
        "matrix_sha256": profiles.manifest_sha256,
        "mixed_traces": [{"path": str(path), "sha256": _sha256_file(path)} for path in paths],
        "source_binding": profiles.manifest["binding"],
        "evidence": {
            "kind": "externally_supplied_mixed_trace",
            "provenance": "content-addressed after collection; no runtime receipt",
        },
    }
    if synthetic_trace_set is not None:
        document = synthetic_trace_set["document"]
        source.update(
            {
                "mode": "synthetic_mixed_trace_set",
                "trace_set": {
                    "path": str(synthetic_trace_set["path"]),
                    "sha256": _sha256_file(synthetic_trace_set["path"]),
                    "trace_set_id": document["trace_set_id"],
                },
                "evidence": document["evidence"],
            }
        )

    plan = {
        "schema": PLAN_SCHEMA,
        "source": source,
        "schedule": {
            "schema": "unirl:minimax-h3:observed-mixed-schedule:v1",
            "waves": len(waves),
        },
        "topology": {
            "physical_devices": int(profiles.manifest["settings"]["num_devices"]),
            "sp_size": profiles.sp_size,
            "dp_groups": profiles.ranks,
            "group_size": profiles.group_size,
            "samples_per_prompt": profiles.samples_per_prompt,
        },
        "predictor_cost": predictor_cost,
        "waves": simulation["wave_results"],
        "execution_contract": {
            "paired_arms": ["baseline", "grouped_lpt"],
            "same_roots_geometry_text_and_samples": True,
            "equal_roots_per_rank": True,
            "siblings_are_atomic": True,
            "directly_executable_on_current_unirl": False,
            "blocker": (
                "request-level mixed geometry still needs a shape-aware scheduler/worker pool; this plan only "
                "defines the exact paired workload and rank assignments"
            ),
        },
    }
    plan["plan_id"] = _sha256_json(plan)
    return plan


def analyze_fixed_replay(
    manifest_path: Path,
    *,
    predictors: Sequence[str],
    trials: int,
    waves: int,
    seed: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay balanced mixed waves from validated fixed profiles."""
    profiles = load_fixed_profiles(manifest_path)
    if trials < 1:
        raise MixedAnalysisError(f"trials must be positive, got {trials}")
    predictor_names = tuple(dict.fromkeys(predictors))
    if not predictor_names or any(name not in STRUCTURAL_COSTS for name in predictor_names):
        raise MixedAnalysisError(f"predictors must be one or more of {STRUCTURAL_COSTS}")

    schedules = [
        balanced_geometry_schedule(
            profiles.root_ids,
            profiles.geometry_names,
            source_id=profiles.manifest["manifest_id"],
            seed=seed,
            trial=trial,
            waves=waves,
        )
        for trial in range(trials)
    ]
    settings = profiles.manifest["settings"]
    tail_threshold = float(settings["tail_ratio_threshold"])
    speedup_threshold = float(settings["minimum_predicted_speedup"])
    predictor_results = []
    decisions = []
    for predictor_cost in predictor_names:
        outcome_cost = _primary_outcome(profiles, predictor_cost)
        trial_results = []
        for trial, assignments in enumerate(schedules):
            wave_records = _fixed_wave_records(profiles, assignments)
            simulation = _simulate_waves(
                profiles.root_ids,
                wave_records,
                ranks=profiles.ranks,
                group_size=profiles.group_size,
                predictor_cost=predictor_cost,
                outcome_costs=(outcome_cost,),
                baseline_placement="contiguous",
                include_wave_details=False,
            )
            aggregate = simulation["aggregate"][outcome_cost]
            trial_results.append(
                {
                    "trial": trial,
                    "baseline_tail_ratio": aggregate["baseline_tail_ratio"],
                    "grouped_tail_ratio": aggregate["grouped_tail_ratio"],
                    "predicted_speedup": aggregate["predicted_speedup"],
                    "baseline_makespan": aggregate["baseline_makespan"],
                    "grouped_makespan": aggregate["grouped_makespan"],
                }
            )
        tail_distribution = _distribution([row["baseline_tail_ratio"] for row in trial_results])
        speedup_distribution = _distribution([row["predicted_speedup"] for row in trial_results])
        decision = _gate(
            tail_ratio=tail_distribution["p10"],
            speedup=speedup_distribution["p10"],
            tail_threshold=tail_threshold,
            speedup_threshold=speedup_threshold,
        )
        decisions.append(decision)
        predictor_results.append(
            {
                "predictor_cost": predictor_cost,
                "outcome_cost": outcome_cost,
                "gate_statistic": "p10 across deterministic balanced schedule trials",
                "baseline_tail_ratio": tail_distribution,
                "predicted_speedup": speedup_distribution,
                "trial_go_count": sum(
                    row["baseline_tail_ratio"] >= tail_threshold and row["predicted_speedup"] >= speedup_threshold
                    for row in trial_results
                ),
                "trial_count": trials,
                "trial_go_fraction": sum(
                    row["baseline_tail_ratio"] >= tail_threshold and row["predicted_speedup"] >= speedup_threshold
                    for row in trial_results
                )
                / trials,
                "decision": decision,
                "trials": trial_results,
            }
        )

    overall_reasons = []
    if any(not decision["go"] for decision in decisions):
        overall_reasons.append("at least one declared structural predictor fails the p10 ROI gate")
    if len({decision["decision"] for decision in decisions}) > 1:
        overall_reasons.append("structural predictors disagree across the 5% threshold")
    overall = {
        "decision": "NO-GO" if overall_reasons else "GO",
        "go": not overall_reasons,
        "scope": "proxy implementation gate; not a paired mixed-workload GPU result",
        "reasons": overall_reasons or ["all declared structural predictors pass the p10 ROI gate"],
    }
    primary_trials = predictor_results[0]["trials"]
    primary_median = float(predictor_results[0]["predicted_speedup"]["median"])
    plan_trial = min(
        range(trials),
        key=lambda index: (
            abs(float(primary_trials[index]["predicted_speedup"]) - primary_median),
            index,
        ),
    )
    primary_assignments = schedules[plan_trial]
    plan = _build_fixed_plan(
        profiles,
        predictor_cost=predictor_names[0],
        seed=seed,
        trial=plan_trial,
        assignments=primary_assignments,
    )
    primary_waves = _fixed_wave_records(profiles, primary_assignments)
    report = {
        "schema": ANALYSIS_SCHEMA,
        "mode": "counterfactual_fixed_profile_replay",
        "source": {
            "matrix": str(profiles.manifest_path),
            "matrix_id": profiles.manifest["manifest_id"],
            "matrix_sha256": profiles.manifest_sha256,
            "trace_sha256": dict(profiles.trace_sha256),
            "evidence": dict(profiles.evidence),
        },
        "schedule": {
            "schema": SCHEDULE_SCHEMA,
            "seed": seed,
            "trials": trials,
            "waves_per_trial": waves,
            "balanced_geometry_per_wave": True,
            "full_geometry_cross_over_per_cycle": True,
            "plan_trial": plan_trial,
            "plan_trial_policy": f"closest to median {predictor_names[0]} predicted speedup",
        },
        "topology": {
            "physical_devices": int(settings["num_devices"]),
            "sp_size": profiles.sp_size,
            "dp_groups": profiles.ranks,
            "group_size": profiles.group_size,
            "roots": len(profiles.root_ids),
            "samples_per_prompt": profiles.samples_per_prompt,
        },
        "text_length": _text_length_summary(primary_waves, profiles.root_ids),
        "predictors": predictor_results,
        "decision": overall,
        "ab_plan_id": plan["plan_id"],
        "limitations": [
            "fixed-profile replay composes one geometry per prompt from separate runs; it is not a concurrent mixed run",
            "analytical_cpu_proxy evidence has text_tokens=0 and no measured phase timings",
            "the current UniRL MiniMax-H3 request carries one shared geometry, so the plan is not directly executable",
        ],
    }
    report["analysis_id"] = _sha256_json(report)
    return report, plan


def _validate_mixed_wave(
    profiles: FixedProfiles,
    trace: Path,
    *,
    expected_text_tokens: Mapping[str, int] | None = None,
) -> dict[str, tuple[MiniMaxH3WorkloadRecord, ...]]:
    records = read_workload_records([trace])
    expected_count = len(profiles.root_ids) * profiles.samples_per_prompt
    if len(records) != expected_count:
        raise MixedAnalysisError(f"mixed trace {trace} has {len(records)} records, expected {expected_count}")
    by_root = _records_by_root(records)
    if set(by_root) != set(profiles.root_ids):
        raise MixedAnalysisError(f"mixed trace {trace} root IDs differ from the source-bound prompt set")
    allowed = {
        (int(row["height"]), int(row["width"]), int(row["num_frames"])) for row in profiles.geometry_rows.values()
    }
    for root_id in profiles.root_ids:
        rows = by_root[root_id]
        expected_samples = tuple(f"{root_id}/{index}" for index in range(profiles.samples_per_prompt))
        if tuple(row.sample_id for row in rows) != expected_samples:
            raise MixedAnalysisError(f"mixed trace root {root_id!r} does not contain its ordered sibling set")
        geometries = {(row.height, row.width, row.num_frames) for row in rows}
        if len(geometries) != 1 or next(iter(geometries)) not in allowed:
            raise MixedAnalysisError(f"mixed trace root {root_id!r} has an invalid or non-atomic geometry")
        if len({row.text_tokens for row in rows}) != 1:
            raise MixedAnalysisError(f"mixed trace root {root_id!r} has inconsistent sibling text lengths")
        bound_text_tokens = (
            int(expected_text_tokens[root_id])
            if expected_text_tokens is not None
            else profiles.records[profiles.geometry_names[0]][root_id][0].text_tokens
        )
        if rows[0].text_tokens != bound_text_tokens:
            raise MixedAnalysisError(
                f"mixed trace root {root_id!r} text length {rows[0].text_tokens} differs from "
                f"the bound trace contract ({bound_text_tokens})"
            )
        if len({row.dp_rank for row in rows}) != 1:
            raise MixedAnalysisError(f"mixed trace root {root_id!r} spans DP ranks")
        if {row.sp_size for row in rows} != {profiles.sp_size} or {row.sp_rank for row in rows} != {0}:
            raise MixedAnalysisError(f"mixed trace root {root_id!r} has the wrong SP placement")
        rank = rows[0].dp_rank
        if rank < 0 or rank >= profiles.ranks:
            raise MixedAnalysisError(f"mixed trace root {root_id!r} has invalid DP rank {rank}")
    return {root_id: by_root[root_id] for root_id in profiles.root_ids}


def _load_synthetic_trace_set(path: Path, profiles: FixedProfiles) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MixedAnalysisError(f"cannot read synthetic trace set {resolved}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != SYNTHETIC_TRACE_SET_SCHEMA:
        raise MixedAnalysisError(f"unsupported synthetic trace-set schema in {resolved}")
    payload = dict(document)
    trace_set_id = payload.pop("trace_set_id", None)
    if trace_set_id != _sha256_json(payload):
        raise MixedAnalysisError("synthetic trace_set_id does not match its contents")

    expected_source = {
        "matrix": str(profiles.manifest_path),
        "matrix_id": profiles.manifest["manifest_id"],
        "matrix_sha256": profiles.manifest_sha256,
        "trace_sha256": dict(profiles.trace_sha256),
        "source_binding": profiles.manifest["binding"],
    }
    if document.get("source") != expected_source:
        raise MixedAnalysisError("synthetic trace set source differs from the validated fixed matrix")
    if document.get("evidence") != SYNTHETIC_EVIDENCE:
        raise MixedAnalysisError("synthetic trace set evidence declaration differs from the CPU-only contract")

    schedule = document.get("schedule")
    if not isinstance(schedule, dict):
        raise MixedAnalysisError("synthetic trace set has no schedule")
    expected_assignments = balanced_geometry_schedule(
        profiles.root_ids,
        profiles.geometry_names,
        source_id=profiles.manifest["manifest_id"],
        seed=str(schedule.get("seed", "")),
        trial=int(schedule.get("trial", -1)),
        waves=int(schedule.get("waves", 0)),
    )
    expected_schedule = {
        "schema": SCHEDULE_SCHEMA,
        "seed": str(schedule.get("seed", "")),
        "trial": int(schedule.get("trial", -1)),
        "waves": len(expected_assignments),
        "geometry_names": list(profiles.geometry_names),
        "balanced_geometry_per_wave": True,
        "full_geometry_cross_over_per_cycle": True,
    }
    if schedule != expected_schedule:
        raise MixedAnalysisError("synthetic trace set schedule differs from deterministic regeneration")

    expected_topology = {
        "physical_devices": int(profiles.manifest["settings"]["num_devices"]),
        "sp_size": profiles.sp_size,
        "dp_groups": profiles.ranks,
        "group_size": profiles.group_size,
        "samples_per_prompt": profiles.samples_per_prompt,
    }
    if document.get("topology") != expected_topology:
        raise MixedAnalysisError("synthetic trace set topology differs from the fixed matrix")

    text = document.get("text_tokens")
    if not isinstance(text, dict):
        raise MixedAnalysisError("synthetic trace set has no text-token contract")
    minimum = int(text.get("minimum", 0))
    maximum = int(text.get("maximum", 0))
    expected_tokens = _synthetic_text_tokens(
        profiles,
        seed=str(schedule["seed"]),
        minimum=minimum,
        maximum=maximum,
    )
    max_sequence_length = int(profiles.manifest["frozen_config"]["bundle.config"].get("max_sequence_length", 512))
    expected_text = {
        "method": SYNTHETIC_TEXT_METHOD,
        "minimum": minimum,
        "maximum": maximum,
        "conditioner_limit": max_sequence_length,
        "by_root": expected_tokens,
    }
    if text != expected_text:
        raise MixedAnalysisError("synthetic text-token mapping differs from deterministic regeneration")

    baseline_placement = str(document.get("baseline_placement", ""))
    if baseline_placement not in {"contiguous", "cost-clustered"}:
        raise MixedAnalysisError(f"unsupported synthetic baseline placement {baseline_placement!r}")
    trace_entries = document.get("traces")
    if not isinstance(trace_entries, list) or len(trace_entries) != len(expected_assignments):
        raise MixedAnalysisError("synthetic trace set does not bind every scheduled wave")
    paths = []
    waves = []
    for wave_index, (entry, assignment) in enumerate(zip(trace_entries, expected_assignments)):
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "wave"}:
            raise MixedAnalysisError("synthetic trace set has malformed trace entries")
        if int(entry["wave"]) != wave_index:
            raise MixedAnalysisError("synthetic trace wave indices must be consecutive from zero")
        trace_path = Path(str(entry["path"])).expanduser().resolve(strict=True)
        if _sha256_file(trace_path) != entry["sha256"]:
            raise MixedAnalysisError(f"synthetic mixed trace digest changed: {trace_path}")
        wave = _validate_mixed_wave(profiles, trace_path, expected_text_tokens=expected_tokens)
        actual_geometry = {
            root_id: next(
                name
                for name, row in profiles.geometry_rows.items()
                if (
                    int(row["height"]),
                    int(row["width"]),
                    int(row["num_frames"]),
                )
                == (
                    wave[root_id][0].height,
                    wave[root_id][0].width,
                    wave[root_id][0].num_frames,
                )
            )
            for root_id in profiles.root_ids
        }
        if actual_geometry != assignment:
            raise MixedAnalysisError(f"synthetic mixed trace wave {wave_index} differs from its geometry schedule")
        expected_ranks = _baseline_rank_assignments(profiles, wave, mode=baseline_placement)
        if any(wave[root_id][0].dp_rank != expected_ranks[root_id] for root_id in profiles.root_ids):
            raise MixedAnalysisError(f"synthetic mixed trace wave {wave_index} differs from its baseline placement")
        paths.append(trace_path)
        waves.append(wave)
    return {
        "document": document,
        "path": resolved,
        "paths": paths,
        "waves": waves,
        "text_tokens": expected_tokens,
    }


def analyze_mixed_traces(
    manifest_path: Path,
    trace_paths: Sequence[Path],
    *,
    predictor_cost: str,
    outcome_cost: str,
    require_mixed_length: bool,
    trace_set_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze already mixed geometry/length waves against the fixed source binding."""
    profiles = load_fixed_profiles(manifest_path)
    if predictor_cost not in STRUCTURAL_COSTS:
        raise MixedAnalysisError(f"predictor_cost must be one of {STRUCTURAL_COSTS}")
    if outcome_cost not in ALL_COSTS:
        raise MixedAnalysisError(f"outcome_cost must be one of {ALL_COSTS}")
    synthetic = None
    if trace_set_path is not None:
        if trace_paths:
            raise MixedAnalysisError("--trace-set is mutually exclusive with --trace")
        synthetic = _load_synthetic_trace_set(trace_set_path, profiles)
        paths = synthetic["paths"]
        waves = synthetic["waves"]
    else:
        paths = [path.expanduser().resolve(strict=True) for path in trace_paths]
        if not paths:
            raise MixedAnalysisError("at least one --trace or --trace-set is required")
        if len(set(paths)) != len(paths):
            raise MixedAnalysisError("mixed trace paths must be unique")
        waves = [_validate_mixed_wave(profiles, path) for path in paths]
    length_summary = _text_length_summary(waves, profiles.root_ids)
    geometry_counts = []
    for wave in waves:
        geometries = {(rows[0].height, rows[0].width, rows[0].num_frames) for rows in wave.values()}
        if len(geometries) < 2:
            raise MixedAnalysisError("each mixed trace wave must contain at least two geometries")
        geometry_counts.append(len(geometries))
    if require_mixed_length and not length_summary["varies"]:
        raise MixedAnalysisError("mixed-length analysis requested, but every root has the same text token count")

    simulation = _simulate_waves(
        profiles.root_ids,
        waves,
        ranks=profiles.ranks,
        group_size=profiles.group_size,
        predictor_cost=predictor_cost,
        outcome_costs=(outcome_cost,),
        baseline_placement="recorded",
        include_wave_details=True,
    )
    aggregate = simulation["aggregate"][outcome_cost]
    settings = profiles.manifest["settings"]
    decision = _gate(
        tail_ratio=aggregate["baseline_tail_ratio"],
        speedup=aggregate["predicted_speedup"],
        tail_threshold=float(settings["tail_ratio_threshold"]),
        speedup_threshold=float(settings["minimum_predicted_speedup"]),
    )
    plan = _build_observed_plan(
        profiles,
        paths,
        waves,
        predictor_cost=predictor_cost,
        synthetic_trace_set=synthetic,
    )
    report = {
        "schema": ANALYSIS_SCHEMA,
        "mode": "observed_mixed_trace",
        "source": plan["source"],
        "topology": plan["topology"],
        "text_length": length_summary,
        "geometry_kinds_per_wave": geometry_counts,
        "predictor_cost": predictor_cost,
        "outcome_cost": outcome_cost,
        "baseline_placement": "recorded DP ranks from each trace",
        "simulation": simulation,
        "decision": {
            **decision,
            "scope": (
                "synthetic CPU proxy; not eligible for a measured ROI claim"
                if synthetic is not None
                else "provisional offline replay; a paired baseline/treatment GPU run is still required"
            ),
        },
        "ab_plan_id": plan["plan_id"],
    }
    if synthetic is not None:
        report["synthetic_trace_set"] = {
            "path": str(synthetic["path"]),
            "trace_set_id": synthetic["document"]["trace_set_id"],
            "sha256": _sha256_file(synthetic["path"]),
            "baseline_placement": synthetic["document"]["baseline_placement"],
            "evidence": synthetic["document"]["evidence"],
        }
    report["analysis_id"] = _sha256_json(report)
    return report, plan


def validate_plan(plan_path: Path) -> dict[str, Any]:
    """Validate a content-addressed A/B plan and its bound fixed source."""
    try:
        plan = json.loads(plan_path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MixedAnalysisError(f"cannot read A/B plan {plan_path}: {exc}") from exc
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise MixedAnalysisError(f"unsupported A/B plan schema in {plan_path}")
    payload = dict(plan)
    plan_id = payload.pop("plan_id", None)
    expected = _sha256_json(payload)
    if plan_id != expected:
        raise MixedAnalysisError("A/B plan_id does not match its contents")
    source = plan.get("source", {})
    manifest_path = Path(str(source.get("matrix", "")))
    profiles = load_fixed_profiles(manifest_path)
    mode = source.get("mode")
    if mode == "fixed_profile_replay":
        schedule = plan.get("schedule", {})
        assignments = balanced_geometry_schedule(
            profiles.root_ids,
            profiles.geometry_names,
            source_id=profiles.manifest["manifest_id"],
            seed=str(schedule.get("seed", "")),
            trial=int(schedule.get("trial", -1)),
            waves=int(schedule.get("waves", 0)),
        )
        regenerated = _build_fixed_plan(
            profiles,
            predictor_cost=str(plan.get("predictor_cost", "")),
            seed=str(schedule.get("seed", "")),
            trial=int(schedule.get("trial", -1)),
            assignments=assignments,
        )
    elif mode in {"observed_mixed_trace", "synthetic_mixed_trace_set"}:
        trace_entries = source.get("mixed_traces")
        if not isinstance(trace_entries, list) or not trace_entries:
            raise MixedAnalysisError("observed A/B plan must bind at least one mixed trace")
        synthetic = None
        if mode == "synthetic_mixed_trace_set":
            trace_set = source.get("trace_set")
            if not isinstance(trace_set, dict) or set(trace_set) != {"path", "sha256", "trace_set_id"}:
                raise MixedAnalysisError("synthetic A/B plan has malformed trace-set binding")
            trace_set_path = Path(str(trace_set["path"])).expanduser().resolve(strict=True)
            if _sha256_file(trace_set_path) != trace_set["sha256"]:
                raise MixedAnalysisError(f"synthetic trace-set digest changed: {trace_set_path}")
            synthetic = _load_synthetic_trace_set(trace_set_path, profiles)
            if synthetic["document"]["trace_set_id"] != trace_set["trace_set_id"]:
                raise MixedAnalysisError("synthetic A/B plan binds another trace_set_id")
            paths = synthetic["paths"]
            waves = synthetic["waves"]
        else:
            paths = []
            for entry in trace_entries:
                if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                    raise MixedAnalysisError("observed A/B plan has malformed mixed trace binding")
                path = Path(str(entry["path"])).expanduser().resolve(strict=True)
                if _sha256_file(path) != entry["sha256"]:
                    raise MixedAnalysisError(f"mixed trace digest changed: {path}")
                paths.append(path)
            waves = [_validate_mixed_wave(profiles, path) for path in paths]
        expected_trace_entries = [{"path": str(path), "sha256": _sha256_file(path)} for path in paths]
        if trace_entries != expected_trace_entries:
            raise MixedAnalysisError("A/B plan mixed trace bindings differ from validated trace sources")
        regenerated = _build_observed_plan(
            profiles,
            paths,
            waves,
            predictor_cost=str(plan.get("predictor_cost", "")),
            synthetic_trace_set=synthetic,
        )
    else:
        raise MixedAnalysisError(f"unsupported A/B plan source mode {mode!r}")
    if regenerated != plan:
        raise MixedAnalysisError("A/B plan semantics differ from its validated source traces")
    return plan


def _print_fixed_summary(report: dict[str, Any]) -> None:
    print(f"mode={report['mode']}")
    print(f"evidence={report['source']['evidence']['kind']}")
    print(f"decision={report['decision']['decision']}")
    for row in report["predictors"]:
        print(
            f"predictor={row['predictor_cost']} outcome={row['outcome_cost']} "
            f"p10_tail={row['baseline_tail_ratio']['p10']:.6f} "
            f"median_speedup={row['predicted_speedup']['median']:.6f}x "
            f"p10_speedup={row['predicted_speedup']['p10']:.6f}x "
            f"go_fraction={row['trial_go_fraction']:.3f} decision={row['decision']['decision']}"
        )
    print(
        f"text_length_available={str(report['text_length']['available']).lower()} "
        f"text_length_varies={str(report['text_length']['varies']).lower()}"
    )
    for reason in report["decision"]["reasons"]:
        print(f"reason={reason}")


def _print_mixed_summary(report: dict[str, Any]) -> None:
    decision = report["decision"]
    aggregate = report["simulation"]["aggregate"][report["outcome_cost"]]
    print(f"mode={report['mode']}")
    print(f"decision={decision['decision']}")
    print(f"baseline_tail_ratio={aggregate['baseline_tail_ratio']:.6f}")
    print(f"predicted_speedup={aggregate['predicted_speedup']:.6f}x")
    print(
        f"text_length_available={str(report['text_length']['available']).lower()} "
        f"text_length_varies={str(report['text_length']['varies']).lower()}"
    )
    for reason in decision["reasons"]:
        print(f"reason={reason}")


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
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser(
        "synthesize-mixed",
        help="materialize content-addressed CPU-only mixed geometry/length waves",
    )
    synthetic.add_argument("--manifest", required=True, type=Path)
    synthetic.add_argument("--output-dir", required=True, type=Path)
    synthetic.add_argument("--seed", default="p3-balanced-v1")
    synthetic.add_argument("--trial", type=_non_negative_int, default=0)
    synthetic.add_argument("--waves", type=_positive_int, default=4)
    synthetic.add_argument("--text-token-min", type=_positive_int, default=32)
    synthetic.add_argument("--text-token-max", type=_positive_int, default=512)
    synthetic.add_argument(
        "--baseline-placement",
        choices=("contiguous", "cost-clustered"),
        default="cost-clustered",
    )

    fixed = subparsers.add_parser("fixed-replay", help="compose balanced mixed waves from four fixed traces")
    fixed.add_argument("--manifest", required=True, type=Path)
    fixed.add_argument("--predictor", action="append", choices=STRUCTURAL_COSTS)
    fixed.add_argument("--trials", type=_positive_int, default=64)
    fixed.add_argument("--waves", type=_positive_int, default=4)
    fixed.add_argument("--seed", default="p3-balanced-v1")
    fixed.add_argument("--output", required=True, type=Path)
    fixed.add_argument("--plan-output", required=True, type=Path)
    fixed.add_argument("--require-go", action="store_true")

    mixed = subparsers.add_parser("mixed", help="analyze one or more already-mixed workload traces")
    mixed.add_argument("--manifest", required=True, type=Path)
    mixed_source = mixed.add_mutually_exclusive_group(required=True)
    mixed_source.add_argument("--trace", action="append", type=Path)
    mixed_source.add_argument("--trace-set", type=Path)
    mixed.add_argument("--predictor-cost", choices=STRUCTURAL_COSTS, default="packed_rows")
    mixed.add_argument("--outcome-cost", choices=ALL_COSTS, default="total_s")
    mixed.add_argument("--require-mixed-length", action="store_true")
    mixed.add_argument("--output", required=True, type=Path)
    mixed.add_argument("--plan-output", required=True, type=Path)
    mixed.add_argument("--require-go", action="store_true")

    validate = subparsers.add_parser("validate-plan", help="revalidate an immutable A/B plan and source matrix")
    validate.add_argument("--plan", required=True, type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        if args.command == "validate-plan":
            plan = validate_plan(args.plan)
            print(f"plan_id={plan['plan_id']}")
            print(f"waves={len(plan['waves'])}")
            return
        if args.command == "synthesize-mixed":
            trace_set = materialize_synthetic_mixed_traces(
                args.manifest,
                args.output_dir,
                seed=args.seed,
                trial=args.trial,
                waves=args.waves,
                text_token_min=args.text_token_min,
                text_token_max=args.text_token_max,
                baseline_placement=args.baseline_placement,
            )
            print(f"trace_set_id={trace_set['trace_set_id']}")
            print(f"waves={len(trace_set['traces'])}")
            print(f"baseline_placement={trace_set['baseline_placement']}")
            print(f"text_tokens={trace_set['text_tokens']['minimum']}..{trace_set['text_tokens']['maximum']}")
            return
        if args.command == "fixed-replay":
            predictors = args.predictor or list(STRUCTURAL_COSTS)
            report, plan = analyze_fixed_replay(
                args.manifest,
                predictors=predictors,
                trials=args.trials,
                waves=args.waves,
                seed=args.seed,
            )
            _write_json(args.output.expanduser().resolve(), report)
            _write_json(args.plan_output.expanduser().resolve(), plan)
            _print_fixed_summary(report)
        else:
            report, plan = analyze_mixed_traces(
                args.manifest,
                args.trace or (),
                predictor_cost=args.predictor_cost,
                outcome_cost=args.outcome_cost,
                require_mixed_length=args.require_mixed_length,
                trace_set_path=args.trace_set,
            )
            _write_json(args.output.expanduser().resolve(), report)
            _write_json(args.plan_output.expanduser().resolve(), plan)
            _print_mixed_summary(report)
        if args.require_go and not report["decision"]["go"]:
            raise SystemExit(3)
    except (MixedAnalysisError, ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
