#!/usr/bin/env python3
"""Aggregate canonical P0 runtime events into P2 fail-closed receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness_lib import (
    GateError,
    aggregate_p0_runtime_events,
    validate_p0_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--p0-contract", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--treatment", choices=("off", "on"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        contract = validate_p0_contract(args.p0_contract.resolve())
        payload = aggregate_p0_runtime_events(
            args.event_root.resolve(),
            campaign_id=args.campaign_id,
            run_name=args.run_name,
            treatment=args.treatment,
            prompt_ids=contract["prompt_manifest"]["prompt_ids"],
            prompt_sha256=contract["prompt_manifest"]["prompt_sha256"],
            planner_contract_sha256=contract["planner_contract_sha256"],
            two_update_contract_sha256=contract["two_update_contract_sha256"],
            frozen_checkpoint=contract["frozen_checkpoint"],
            output_dir=args.output_dir.resolve(),
        )
    except GateError as exc:
        print(json.dumps({"completed": False, "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
