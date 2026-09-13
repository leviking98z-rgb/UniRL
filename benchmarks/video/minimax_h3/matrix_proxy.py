"""Materialize an auditable CPU-only proxy for a bound MiniMax-H3 matrix."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from unirl.utils.minimax_h3_workload import (  # noqa: E402
    MiniMaxH3WorkloadRecord,
    append_workload_record,
)

_MATRIX_DRIVER = Path(__file__).with_name("matrix_driver.py")
_STRUCTURAL_COSTS = {"packed_rows", "padded_rows", "attention_rows2"}


def _load_matrix_driver():
    spec = importlib.util.spec_from_file_location("minimax_h3_matrix_driver", _MATRIX_DRIVER)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import matrix driver from {_MATRIX_DRIVER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def materialize_proxy(manifest_path: Path) -> dict[str, Any]:
    """Write geometry-only traces and receipts without launching model code."""
    driver = _load_matrix_driver()
    manifest = driver._load_manifest(manifest_path)
    driver._revalidate_binding(driver._script_repo(), manifest)
    settings = manifest["settings"]
    if settings["cost"] not in _STRUCTURAL_COSTS:
        raise driver.MatrixError(
            f"CPU proxy supports only {sorted(_STRUCTURAL_COSTS)}; "
            f"manifest requests measured timing cost {settings['cost']!r}"
        )

    output_slots = {}
    for geometry in manifest["geometries"]:
        output_slots[geometry["name"]] = driver._assert_output_slots(manifest, geometry["name"])

    roots = list(settings["expected_root_ids"])
    siblings = int(settings["samples_per_prompt"])
    dp_groups = int(settings["dp_groups"])
    roots_per_dp = len(roots) // dp_groups
    results = []
    for geometry in manifest["geometries"]:
        trace, receipt = output_slots[geometry["name"]]
        trace.parent.mkdir(parents=True, exist_ok=True)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        for root_index, root_id in enumerate(roots):
            dp_rank = root_index // roots_per_dp
            for sibling in range(siblings):
                append_workload_record(
                    trace,
                    MiniMaxH3WorkloadRecord.build(
                        sample_id=f"{root_id}/{sibling}",
                        root_id=root_id,
                        height=int(geometry["height"]),
                        width=int(geometry["width"]),
                        num_frames=int(geometry["num_frames"]),
                        text_tokens=0,
                        sp_size=int(settings["sp_size"]),
                        dp_rank=dp_rank,
                        sp_rank=0,
                    ),
                )
        driver._write_receipt(
            receipt,
            manifest=manifest,
            geometry=geometry,
            trace=trace,
            command=driver.expected_command(manifest, geometry["name"]),
            evidence=driver.PROXY_EVIDENCE,
        )
        results.append(
            {
                "geometry": geometry["name"],
                "trace": str(trace),
                "trace_sha256": driver._sha256_file(trace),
                "receipt": str(receipt),
                "receipt_sha256": driver._sha256_file(receipt),
            }
        )
    return {
        "manifest_id": manifest["manifest_id"],
        "evidence": driver.PROXY_EVIDENCE,
        "runs": results,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        result = materialize_proxy(args.manifest.expanduser().resolve(strict=True))
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"manifest_id={result['manifest_id']}")
    print(f"evidence={result['evidence']['kind']}")
    for run in result["runs"]:
        print(f"{run['geometry']} trace_sha256={run['trace_sha256']} receipt_sha256={run['receipt_sha256']}")


if __name__ == "__main__":
    main()
