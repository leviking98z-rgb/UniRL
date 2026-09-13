"""CPU tests for MiniMax-H3 mixed trace analysis and grouped-reordering A/B contracts."""

from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any

from unirl.utils.minimax_h3_workload import MiniMaxH3WorkloadRecord, append_workload_record

HERE = Path(__file__).resolve().parent


def _load_module(name: str) -> ModuleType:
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"p3_test_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MIXED = _load_module("mixed_trace_analyzer")
AB = _load_module("grouped_reordering_ab")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fake_profiles(root: Path):
    root_ids = tuple(f"root-{index}" for index in range(8))
    geometry_rows = {
        "g0": {"name": "g0", "height": 768, "width": 768, "num_frames": 124},
        "g1": {"name": "g1", "height": 768, "width": 1024, "num_frames": 124},
        "g2": {"name": "g2", "height": 768, "width": 768, "num_frames": 175},
        "g3": {"name": "g3", "height": 768, "width": 1024, "num_frames": 175},
    }
    records = {}
    traces = {}
    trace_sha256 = {}
    for name, geometry in geometry_rows.items():
        path = root / f"{name}.jsonl"
        by_root = {}
        for root_index, root_id in enumerate(root_ids):
            rows = []
            for sibling in range(2):
                row = MiniMaxH3WorkloadRecord.build(
                    sample_id=f"{root_id}/{sibling}",
                    root_id=root_id,
                    height=geometry["height"],
                    width=geometry["width"],
                    num_frames=geometry["num_frames"],
                    text_tokens=(root_index + 1) * 17,
                    sp_size=2,
                    dp_rank=root_index // 4,
                    sp_rank=0,
                )
                append_workload_record(path, row)
                rows.append(row)
            by_root[root_id] = tuple(rows)
        records[name] = by_root
        traces[name] = path
        trace_sha256[name] = MIXED._sha256_file(path)
    matrix = root / "matrix.json"
    matrix.write_text("{}\n", encoding="utf-8")
    manifest = {
        "manifest_id": "fake-matrix",
        "binding": {"source_commit": "deadbeef"},
        "frozen_config": {"bundle.config": {"max_sequence_length": 512}},
        "settings": {
            "num_devices": 4,
            "sp_size": 2,
            "dp_groups": 2,
            "group_size": 2,
            "samples_per_prompt": 2,
            "tail_ratio_threshold": 1.05,
            "minimum_predicted_speedup": 1.05,
        },
    }
    profiles = MIXED.FixedProfiles(
        manifest_path=matrix,
        manifest_sha256=MIXED._sha256_file(matrix),
        manifest=manifest,
        fixed_analysis={"evidence": {"kind": "measured_gpu_trace"}},
        geometry_names=tuple(geometry_rows),
        geometry_rows=geometry_rows,
        root_ids=root_ids,
        records=records,
        trace_sha256=trace_sha256,
        evidence={"kind": "measured_gpu_trace"},
    )
    return profiles, traces


class MixedAnalyzerTests(unittest.TestCase):
    """Exercise deterministic scheduling, mixed lengths, and placement semantics."""

    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="unirl-p3-test."))
        self.addCleanup(shutil.rmtree, self.temp, True)
        self.profiles, self.traces = _fake_profiles(self.temp)

    def test_balanced_schedule_is_deterministic_and_crosses_over(self) -> None:
        first = MIXED.balanced_geometry_schedule(
            self.profiles.root_ids,
            self.profiles.geometry_names,
            source_id="source",
            seed="seed",
            trial=7,
            waves=4,
        )
        second = MIXED.balanced_geometry_schedule(
            self.profiles.root_ids,
            self.profiles.geometry_names,
            source_id="source",
            seed="seed",
            trial=7,
            waves=4,
        )
        self.assertEqual(first, second)
        for wave in first:
            self.assertEqual(
                {name: list(wave.values()).count(name) for name in self.profiles.geometry_names},
                {name: 2 for name in self.profiles.geometry_names},
            )
        for root_id in self.profiles.root_ids:
            self.assertEqual({wave[root_id] for wave in first}, set(self.profiles.geometry_names))

    def test_mixed_lengths_change_structural_cost(self) -> None:
        assignments = MIXED.balanced_geometry_schedule(
            self.profiles.root_ids,
            self.profiles.geometry_names,
            source_id="source",
            seed="seed",
            trial=0,
            waves=4,
        )
        waves = MIXED._fixed_wave_records(self.profiles, assignments)
        self.assertTrue(MIXED._text_length_summary(waves, self.profiles.root_ids)["varies"])
        result = MIXED._simulate_waves(
            self.profiles.root_ids,
            waves,
            ranks=2,
            group_size=2,
            predictor_cost="packed_rows",
            outcome_costs=("packed_rows",),
            baseline_placement="contiguous",
            include_wave_details=True,
        )
        request_rows = result["wave_results"][0]["request_rows"]
        self.assertEqual(len({row["text_tokens"] for row in request_rows}), len(self.profiles.root_ids))
        for row in request_rows:
            expected = next(
                records
                for records in self.profiles.records.values()
                if f"{records[row['root_id']][0].height}x{records[row['root_id']][0].width}x"
                f"{records[row['root_id']][0].num_frames}" == row["geometry"]
            )[row["root_id"]]
            self.assertEqual(row["predictor_root_cost"], sum(record.packed_rows for record in expected))

    def test_recorded_baseline_and_grouped_assignment_are_distinct(self) -> None:
        assignments = MIXED.balanced_geometry_schedule(
            self.profiles.root_ids,
            self.profiles.geometry_names,
            source_id="source",
            seed="seed",
            trial=1,
            waves=4,
        )
        waves = MIXED._fixed_wave_records(self.profiles, assignments)
        first = waves[0]
        moved = {}
        for root_index, root_id in enumerate(self.profiles.root_ids):
            target_rank = 1 if root_index < 4 else 0
            moved[root_id] = tuple(
                MiniMaxH3WorkloadRecord.build(
                    sample_id=row.sample_id,
                    root_id=row.root_id,
                    height=row.height,
                    width=row.width,
                    num_frames=row.num_frames,
                    text_tokens=row.text_tokens,
                    sp_size=row.sp_size,
                    dp_rank=target_rank,
                    sp_rank=0,
                )
                for row in first[root_id]
            )
        result = MIXED._simulate_waves(
            self.profiles.root_ids,
            [moved],
            ranks=2,
            group_size=2,
            predictor_cost="attention_rows2",
            outcome_costs=("packed_rows",),
            baseline_placement="recorded",
            include_wave_details=True,
        )
        observed = result["wave_results"][0]["arms"]["baseline"]["rank_root_ids"]
        self.assertEqual(observed[0], list(self.profiles.root_ids[4:]))
        self.assertEqual(observed[1], list(self.profiles.root_ids[:4]))

    def test_fixed_plan_validation_rebuilds_semantics(self) -> None:
        assignments = MIXED.balanced_geometry_schedule(
            self.profiles.root_ids,
            self.profiles.geometry_names,
            source_id=self.profiles.manifest["manifest_id"],
            seed="seed",
            trial=0,
            waves=4,
        )
        plan = MIXED._build_fixed_plan(
            self.profiles,
            predictor_cost="packed_rows",
            seed="seed",
            trial=0,
            assignments=assignments,
        )
        path = self.temp / "plan.json"
        _write_json(path, plan)
        original_loader = MIXED.load_fixed_profiles
        self.addCleanup(setattr, MIXED, "load_fixed_profiles", original_loader)
        MIXED.load_fixed_profiles = lambda _: self.profiles
        self.assertEqual(MIXED.validate_plan(path)["plan_id"], plan["plan_id"])

        tampered = copy.deepcopy(plan)
        tampered["waves"][0]["arms"]["grouped_lpt"]["rank_root_ids"][0][0] = self.profiles.root_ids[-1]
        tampered["plan_id"] = MIXED._sha256_json({key: value for key, value in tampered.items() if key != "plan_id"})
        _write_json(path, tampered)
        with self.assertRaisesRegex(MIXED.MixedAnalysisError, "semantics differ"):
            MIXED.validate_plan(path)

    def test_synthetic_mixed_trace_set_is_reproducible_and_fail_closed(self) -> None:
        source = self.temp / "synthetic-a"
        repeated = self.temp / "synthetic-b"
        self.profiles.evidence["kind"] = "analytical_cpu_proxy"
        original_loader = MIXED.load_fixed_profiles
        self.addCleanup(setattr, MIXED, "load_fixed_profiles", original_loader)
        MIXED.load_fixed_profiles = lambda _: self.profiles

        first = MIXED.materialize_synthetic_mixed_traces(
            self.profiles.manifest_path,
            source,
            seed="synthetic-seed",
            trial=3,
            waves=4,
            text_token_min=32,
            text_token_max=512,
            baseline_placement="cost-clustered",
        )
        second = MIXED.materialize_synthetic_mixed_traces(
            self.profiles.manifest_path,
            repeated,
            seed="synthetic-seed",
            trial=3,
            waves=4,
            text_token_min=32,
            text_token_max=512,
            baseline_placement="cost-clustered",
        )
        self.assertEqual(first["text_tokens"], second["text_tokens"])
        first_trace_bytes = [Path(row["path"]).read_bytes() for row in first["traces"]]
        second_trace_bytes = [Path(row["path"]).read_bytes() for row in second["traces"]]
        self.assertEqual(first_trace_bytes, second_trace_bytes)

        report, plan = MIXED.analyze_mixed_traces(
            self.profiles.manifest_path,
            (),
            predictor_cost="packed_rows",
            outcome_cost="packed_rows",
            require_mixed_length=True,
            trace_set_path=source / "trace-set.json",
        )
        self.assertTrue(report["text_length"]["varies"])
        self.assertEqual(plan["source"]["mode"], "synthetic_mixed_trace_set")
        self.assertFalse(plan["source"]["evidence"]["measured_roi_eligible"])

        plan_path = self.temp / "synthetic-plan.json"
        _write_json(plan_path, plan)
        self.assertEqual(MIXED.validate_plan(plan_path)["plan_id"], plan["plan_id"])

        trace = Path(first["traces"][0]["path"])
        with trace.open("a", encoding="utf-8") as output:
            output.write("{}\n")
        with self.assertRaisesRegex(MIXED.MixedAnalysisError, "digest changed"):
            MIXED.analyze_mixed_traces(
                self.profiles.manifest_path,
                (),
                predictor_cost="packed_rows",
                outcome_cost="packed_rows",
                require_mixed_length=True,
                trace_set_path=source / "trace-set.json",
            )


class ABHarnessTests(unittest.TestCase):
    """Exercise the A/B contract and measured gate without launching a workload."""

    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="unirl-p3-ab-test."))
        self.addCleanup(shutil.rmtree, self.temp, True)
        self.profiles, _ = _fake_profiles(self.temp)
        assignments = MIXED.balanced_geometry_schedule(
            self.profiles.root_ids,
            self.profiles.geometry_names,
            source_id=self.profiles.manifest["manifest_id"],
            seed="seed",
            trial=0,
            waves=4,
        )
        self.plan = MIXED._build_fixed_plan(
            self.profiles,
            predictor_cost="packed_rows",
            seed="seed",
            trial=0,
            assignments=assignments,
        )
        self.plan_path = self.temp / "plan.json"
        _write_json(self.plan_path, self.plan)
        self.mixed_loader = MIXED.load_fixed_profiles
        self.ab_validate = AB.validate_plan
        MIXED.load_fixed_profiles = lambda _: self.profiles
        AB.validate_plan = MIXED.validate_plan
        self.addCleanup(setattr, MIXED, "load_fixed_profiles", self.mixed_loader)
        self.addCleanup(setattr, AB, "validate_plan", self.ab_validate)

    def _contract(self) -> tuple[Path, dict[str, Any]]:
        contract = AB.build_contract(
            self.plan_path,
            repetitions=3,
            warmup_repetitions=1,
            measurement="generation_s",
            minimum_speedup=1.05,
            minimum_win_fraction=0.75,
            maximum_reward_delta=0.01,
            maximum_trace_overhead=0.005,
        )
        path = self.temp / "contract.json"
        _write_json(path, contract)
        return path, contract

    def _results(self, contract: dict[str, Any], grouped_s: float) -> tuple[Path, Path]:
        baseline = AB.result_template(contract, "baseline")
        grouped = AB.result_template(contract, "grouped_lpt")
        baseline["trace_overhead_fraction"] = 0.001
        grouped["trace_overhead_fraction"] = 0.001
        for result, elapsed in ((baseline, 10.0), (grouped, grouped_s)):
            for run in result["runs"]:
                run["rank_elapsed_s"] = [elapsed, elapsed - 0.1]
                run["successful_samples"] = next(
                    wave["expected_samples"] for wave in contract["waves"] if wave["wave"] == run["wave"]
                )
                run["reward_mean"] = 0.5
                run["output_digest"] = "same-output"
        baseline_path = self.temp / "baseline.json"
        grouped_path = self.temp / "grouped.json"
        _write_json(baseline_path, baseline)
        _write_json(grouped_path, grouped)
        return baseline_path, grouped_path

    def test_contract_and_result_gates(self) -> None:
        contract_path, contract = self._contract()
        self.assertEqual(AB.load_contract(contract_path)["contract_id"], contract["contract_id"])
        baseline, grouped = self._results(contract, 9.0)
        go = AB.analyze_results(contract_path, baseline, grouped)
        self.assertEqual(go["decision"]["decision"], "GO")
        baseline, grouped = self._results(contract, 9.8)
        no_go = AB.analyze_results(contract_path, baseline, grouped)
        self.assertEqual(no_go["decision"]["decision"], "NO-GO")

    def test_assignment_tamper_is_rejected(self) -> None:
        contract_path, contract = self._contract()
        baseline, grouped = self._results(contract, 9.0)
        result = json.loads(grouped.read_text())
        result["runs"][0]["rank_root_ids"][0][0] = self.profiles.root_ids[-1]
        _write_json(grouped, result)
        with self.assertRaisesRegex(AB.ABError, "contracted rank assignment"):
            AB.analyze_results(contract_path, baseline, grouped)


if __name__ == "__main__":
    unittest.main()
