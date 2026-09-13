"""Simulate equal-capacity grouped reordering for MiniMax-H3 workload traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from unirl.utils.minimax_h3_workload import (  # noqa: E402
    MiniMaxH3WorkloadGeometry,
    MiniMaxH3WorkloadRecord,
    append_workload_record,
    read_workload_records,
)

_COST_CHOICES = ("packed_rows", "padded_rows", "attention_rows2", "total_s", "denoise_s")


@dataclass(frozen=True)
class RootWork:
    """One indivisible root prompt and all of its generated siblings."""

    root_id: str
    source_root_id: str
    source: str
    source_index: int
    source_dp_rank: int
    first_record_index: int
    sample_ids: tuple[str, ...]
    sample_costs: tuple[float, ...]

    @property
    def cost(self) -> float:
        return sum(self.sample_costs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "source_root_id": self.source_root_id,
            "source": self.source,
            "source_index": self.source_index,
            "source_dp_rank": self.source_dp_rank,
            "first_record_index": self.first_record_index,
            "sample_ids": list(self.sample_ids),
            "sample_costs": list(self.sample_costs),
            "cost": self.cost,
        }


def _sample_cost(record: MiniMaxH3WorkloadRecord, cost_name: str) -> float:
    if cost_name == "packed_rows":
        return float(record.packed_rows)
    if cost_name == "padded_rows":
        return float(record.padded_rows)
    if cost_name == "attention_rows2":
        return float(record.padded_rows) ** 2
    value = float(getattr(record, cost_name))
    if value <= 0:
        raise ValueError(f"cost {cost_name!r} must be positive; sample {record.sample_id!r} has {value}")
    return value


def load_root_work(
    paths: Sequence[Path],
    cost_name: str,
    merge_order: str = "round-robin",
    source_order: str = "input",
) -> list[RootWork]:
    """Read traces and aggregate sibling samples by source-local root id."""
    roots_by_source: list[OrderedDict[str, dict[str, Any]]] = []
    seen_samples: set[tuple[int, str]] = set()
    for source_index, path in enumerate(paths):
        roots: OrderedDict[str, dict[str, Any]] = OrderedDict()
        record_index = 0
        for record in read_workload_records([path]):
            sample_key = (source_index, record.sample_id)
            if sample_key in seen_samples:
                raise ValueError(f"duplicate sample_id {record.sample_id!r} in {path}")
            seen_samples.add(sample_key)
            entry = roots.setdefault(
                record.root_id,
                {
                    "root_id": f"source:{source_index}:{record.root_id}",
                    "source_root_id": record.root_id,
                    "source": str(path),
                    "source_index": source_index,
                    "source_dp_rank": record.dp_rank,
                    "first_record_index": record_index,
                    "sample_ids": [],
                    "sample_costs": [],
                },
            )
            if entry["source_dp_rank"] != record.dp_rank:
                raise ValueError(f"root {record.root_id!r} spans DP ranks in {path}")
            entry["sample_ids"].append(record.sample_id)
            entry["sample_costs"].append(_sample_cost(record, cost_name))
            record_index += 1
        roots_by_source.append(roots)

    if source_order == "input":
        entries_by_source = [list(roots.values()) for roots in roots_by_source]
    elif source_order == "dp-rank":
        entries_by_source = [
            sorted(roots.values(), key=lambda entry: (entry["source_dp_rank"], entry["first_record_index"]))
            for roots in roots_by_source
        ]
    else:
        raise ValueError(f"unsupported source_order {source_order!r}")
    if merge_order == "input":
        ordered_entries = [entry for source in entries_by_source for entry in source]
    elif merge_order == "round-robin":
        ordered_entries = []
        for index in range(max((len(source) for source in entries_by_source), default=0)):
            ordered_entries.extend(source[index] for source in entries_by_source if index < len(source))
    else:
        raise ValueError(f"unsupported merge_order {merge_order!r}")

    items = []
    for order, entry in enumerate(ordered_entries):
        items.append(
            RootWork(
                root_id=entry["root_id"],
                source_root_id=entry["source_root_id"],
                source=entry["source"],
                source_index=entry["source_index"],
                source_dp_rank=entry["source_dp_rank"],
                first_record_index=order,
                sample_ids=tuple(entry["sample_ids"]),
                sample_costs=tuple(entry["sample_costs"]),
            )
        )
    return items


def contiguous_assignment(items: Sequence[RootWork], ranks: int) -> list[list[RootWork]]:
    """Assign the ordered roots to equal contiguous rank shards."""
    _validate_capacity(len(items), ranks)
    per_rank = len(items) // ranks
    return [list(items[rank * per_rank : (rank + 1) * per_rank]) for rank in range(ranks)]


def capacity_lpt(items: Sequence[RootWork], ranks: int) -> list[list[RootWork]]:
    """Assign longest roots first while keeping an equal root count per rank."""
    _validate_capacity(len(items), ranks)
    capacity = len(items) // ranks
    assignments: list[list[RootWork]] = [[] for _ in range(ranks)]
    loads = [0.0] * ranks
    ordered = sorted(items, key=lambda item: (-item.cost, item.first_record_index, item.root_id))
    for item in ordered:
        candidates = [rank for rank in range(ranks) if len(assignments[rank]) < capacity]
        rank = min(candidates, key=lambda candidate: (loads[candidate], len(assignments[candidate]), candidate))
        assignments[rank].append(item)
        loads[rank] += item.cost
    for assignment in assignments:
        assignment.sort(key=lambda item: item.first_record_index)
    return assignments


def grouped_lpt_assignment(items: Sequence[RootWork], ranks: int, group_size: int) -> list[list[RootWork]]:
    """Run equal-capacity LPT independently inside contiguous rank groups."""
    baseline = contiguous_assignment(items, ranks)
    size = int(group_size)
    if size < 1 or ranks % size:
        raise ValueError(f"group_size={size} must be a positive divisor of ranks={ranks}")
    assignments: list[list[RootWork]] = [[] for _ in range(ranks)]
    for start in range(0, ranks, size):
        local_items = [item for rank in range(start, start + size) for item in baseline[rank]]
        local_assignments = capacity_lpt(local_items, size)
        for local_rank, assigned in enumerate(local_assignments):
            assignments[start + local_rank] = assigned
    return assignments


def _validate_capacity(num_items: int, ranks: int) -> None:
    if ranks < 1:
        raise ValueError(f"ranks must be >= 1, got {ranks}")
    if num_items < ranks:
        raise ValueError(f"need at least one root per rank, got roots={num_items} ranks={ranks}")
    if num_items % ranks:
        raise ValueError(f"root count {num_items} must be divisible by ranks={ranks} for equal-capacity simulation")


def assignment_summary(assignments: Sequence[Sequence[RootWork]]) -> dict[str, Any]:
    """Calculate load, tail, efficiency, and assignment details."""
    loads = [sum(item.cost for item in assignment) for assignment in assignments]
    total = sum(loads)
    mean = total / len(loads)
    maximum = max(loads)
    minimum = min(loads)
    return {
        "rank_loads": loads,
        "min": minimum,
        "mean": mean,
        "max": maximum,
        "tail_ratio": maximum / mean if mean else 0.0,
        "imbalance": (maximum - minimum) / mean if mean else 0.0,
        "efficiency": total / (len(loads) * maximum) if maximum else 1.0,
        "assignments": [
            {
                "rank": rank,
                "load": loads[rank],
                "root_ids": [item.root_id for item in assignment],
                "source_root_ids": [item.source_root_id for item in assignment],
            }
            for rank, assignment in enumerate(assignments)
        ],
        "permutation": [item.first_record_index for assignment in assignments for item in assignment],
    }


def simulate(
    paths: Sequence[Path],
    *,
    ranks: int,
    group_size: int,
    cost_name: str,
    merge_order: str = "round-robin",
    source_order: str = "input",
) -> dict[str, Any]:
    """Compare contiguous dispatch with local equal-capacity LPT."""
    if cost_name not in _COST_CHOICES:
        raise ValueError(f"unsupported cost {cost_name!r}; choose from {_COST_CHOICES}")
    items = load_root_work(paths, cost_name, merge_order, source_order)
    baseline = assignment_summary(contiguous_assignment(items, ranks))
    reordered = assignment_summary(grouped_lpt_assignment(items, ranks, group_size))
    reordered["predicted_speedup"] = baseline["max"] / reordered["max"] if reordered["max"] else 1.0
    return {
        "schema": "unirl:minimax-h3:grouped-reordering:v1",
        "inputs": [str(path) for path in paths],
        "cost": cost_name,
        "merge_order": merge_order,
        "source_order": source_order,
        "cost_definition": (
            "sum of packed rows per root"
            if cost_name == "packed_rows"
            else "sum of SP-padded packed rows per root"
            if cost_name == "padded_rows"
            else "sum of squared SP-padded packed rows per root"
            if cost_name == "attention_rows2"
            else f"sum of measured {cost_name} per root"
        ),
        "ranks": ranks,
        "group_size": group_size,
        "roots": len(items),
        "samples": sum(len(item.sample_ids) for item in items),
        "roots_per_rank": len(items) // ranks,
        "root_work": [item.to_dict() for item in items],
        "baseline": baseline,
        "grouped_lpt": reordered,
    }


def _parse_geometry(value: str) -> tuple[int, int, int, int]:
    try:
        geometry, count_text = value.rsplit(":", 1)
        dimensions = geometry.lower().split("x")
        if len(dimensions) != 3:
            raise ValueError
        height, width, frames = (int(part) for part in dimensions)
        count = int(count_text)
        if count < 1:
            raise ValueError
        MiniMaxH3WorkloadGeometry.resolve(height=height, width=width, num_frames=frames)
        return height, width, frames, count
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected HEIGHTxWIDTHxFRAMES:ROOTS with legal positive values") from exc


def _parse_text_tokens(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in value.split(","))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected comma-separated non-negative integers") from exc
    if not values or any(token < 0 for token in values):
        raise argparse.ArgumentTypeError("expected comma-separated non-negative integers")
    return values


def synthesize(
    geometries: Sequence[tuple[int, int, int, int]],
    *,
    output: Path,
    text_tokens: Sequence[int],
    samples_per_prompt: int,
    sp_size: int,
) -> int:
    """Write a deterministic mixed-geometry trace for simulator validation."""
    if output.exists():
        output.unlink()
    if samples_per_prompt < 1:
        raise ValueError(f"samples_per_prompt must be >= 1, got {samples_per_prompt}")
    if sp_size < 1:
        raise ValueError(f"sp_size must be >= 1, got {sp_size}")
    root_index = 0
    records = 0
    for geometry_index in range(max(count for _, _, _, count in geometries)):
        for height, width, frames, count in geometries:
            if geometry_index >= count:
                continue
            root_id = f"synthetic-root-{root_index:06d}"
            tokens = int(text_tokens[root_index % len(text_tokens)])
            for sibling in range(samples_per_prompt):
                append_workload_record(
                    output,
                    MiniMaxH3WorkloadRecord.build(
                        sample_id=f"{root_id}/{sibling}",
                        root_id=root_id,
                        height=height,
                        width=width,
                        num_frames=frames,
                        text_tokens=tokens,
                        sp_size=sp_size,
                    ),
                )
                records += 1
            root_index += 1
    return records


def _positive_int(value: str) -> int:
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return integer


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_summary(summary: dict[str, Any]) -> None:
    baseline = summary["baseline"]
    reordered = summary["grouped_lpt"]
    print(
        f"roots={summary['roots']} samples={summary['samples']} dp_ranks={summary['ranks']} "
        f"group_size={summary['group_size']} roots/rank={summary['roots_per_rank']} cost={summary['cost']}"
    )
    print("| strategy | min | mean | max | tail | imbalance | efficiency | speedup |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    print(
        f"| contiguous | {baseline['min']:.3f} | {baseline['mean']:.3f} | {baseline['max']:.3f} | "
        f"{baseline['tail_ratio']:.4f} | {baseline['imbalance']:.4f} | {baseline['efficiency']:.4f} | 1.0000 |"
    )
    print(
        f"| grouped-lpt | {reordered['min']:.3f} | {reordered['mean']:.3f} | {reordered['max']:.3f} | "
        f"{reordered['tail_ratio']:.4f} | {reordered['imbalance']:.4f} | {reordered['efficiency']:.4f} | "
        f"{reordered['predicted_speedup']:.4f} |"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="compare contiguous dispatch with grouped equal-capacity LPT")
    analyze.add_argument("traces", nargs="+", type=Path)
    analyze.add_argument("--ranks", required=True, type=_positive_int, help="number of DP groups, not physical GPUs")
    analyze.add_argument(
        "--group-size",
        type=_positive_int,
        help="DP ranks per local reorder group (default: all ranks)",
    )
    analyze.add_argument("--cost", choices=_COST_CHOICES, default="packed_rows")
    analyze.add_argument(
        "--merge-order",
        choices=("round-robin", "input"),
        default="round-robin",
        help="interleave roots from multiple trace files or concatenate files in argument order",
    )
    analyze.add_argument(
        "--source-order",
        choices=("input", "dp-rank"),
        default="input",
        help="preserve file order or reconstruct deterministic DP-shard order within each trace",
    )
    analyze.add_argument("--output", type=Path, help="optional JSON summary path")

    synthetic = subparsers.add_parser("synthesize", help="create a deterministic mixed-geometry JSONL trace")
    synthetic.add_argument(
        "--geometry",
        action="append",
        required=True,
        type=_parse_geometry,
        help="HEIGHTxWIDTHxFRAMES:ROOTS; repeat for each geometry",
    )
    synthetic.add_argument("--text-tokens", type=_parse_text_tokens, default=(128,))
    synthetic.add_argument("--samples-per-prompt", type=_positive_int, default=1)
    synthetic.add_argument("--sp-size", type=_positive_int, default=1)
    synthetic.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "synthesize":
        count = synthesize(
            args.geometry,
            output=args.output,
            text_tokens=args.text_tokens,
            samples_per_prompt=args.samples_per_prompt,
            sp_size=args.sp_size,
        )
        print(f"wrote {count} records to {args.output}")
        return

    group_size = args.group_size or args.ranks
    summary = simulate(
        args.traces,
        ranks=args.ranks,
        group_size=group_size,
        cost_name=args.cost,
        merge_order=args.merge_order,
        source_order=args.source_order,
    )
    if args.output is not None:
        _write_json(args.output, summary)
    _print_summary(summary)


if __name__ == "__main__":
    main()
