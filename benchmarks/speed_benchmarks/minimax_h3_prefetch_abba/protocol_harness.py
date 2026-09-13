#!/usr/bin/env python3
"""CPU-only protocol tests for P2 prefetch, source sealing, and AB/BA gates."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from harness_lib import (
    EXPECTED_P0_PLANNER_COMMIT,
    EXPECTED_P0_PLANNER_TREE,
    EXPECTED_PLANNER_CLASS,
    EXPECTED_REBASED_PR38_COMMIT,
    EXPECTED_REBASED_PR38_TREE,
    FROZEN_CHECKPOINT_LOADED_SCHEMA,
    GENERATED_SAMPLES,
    P0_RUNTIME_EVENT_SCHEMA,
    P2_RUNTIME_SUMMARY_SCHEMA,
    SAMPLE_LEDGER_SCHEMA,
    UPDATE_MEMBERSHIP_SCHEMA,
    GateError,
    arm_specs,
    expected_sample_ids,
    expected_update_membership,
    load_p0_events,
    load_prompt_manifest,
    metrics_gate,
    paired_effects,
    runtime_event_tree_digest,
    sample_ledger_gate,
    sha256_bytes,
    sha256_file,
    update_membership_gate,
)
from partial_summary import write_partial_summary
from seal_source import seal_repository, source_chain_gate
from summarize import runtime_summary_gate
from verify_source_bundle import extract_verified, verify_archive, verify_directory

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PREFETCH_PATH = REPO / "unirl/models/minimax_h3/prefetch.py"

Test = Callable[[], None]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_raises(
    exception_type: type[BaseException],
    function: Callable[[], Any],
    *,
    contains: str | None = None,
) -> BaseException:
    try:
        function()
    except exception_type as exc:
        if contains is not None and contains not in str(exc):
            raise AssertionError(f"exception {exc!r} does not contain {contains!r}") from exc
        return exc
    raise AssertionError(f"expected {exception_type.__name__}")


def load_prefetch_module() -> Any:
    spec = importlib.util.spec_from_file_location("p2_prefetch_protocol_under_test", PREFETCH_PATH)
    require(spec is not None and spec.loader is not None, "cannot load prefetch module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PREFETCH = load_prefetch_module()


def test_prefetch_capacity_dedup_and_exception() -> None:
    prefetcher = PREFETCH.BoundedPrefetcher[str, str](2, thread_name="p2-test-prefetch")
    active_started = threading.Event()
    active_release = threading.Event()
    calls = 0

    def slow() -> str:
        nonlocal calls
        calls += 1
        active_started.set()
        require(active_release.wait(2), "test producer release timed out")
        return "value-a"

    require(prefetcher.submit("a", slow), "first submission rejected")
    require(active_started.wait(2), "prefetch worker did not start")
    require(prefetcher.submit("a", lambda: "duplicate"), "deduplicated submission rejected")
    require(prefetcher.submit("b", lambda: "value-b"), "second retained entry rejected")
    require(not prefetcher.submit("c", lambda: "value-c"), "capacity overflow was accepted")
    active_release.set()
    require(prefetcher.take("a") == (True, "value-a"), "active value mismatch")
    require(prefetcher.take("b") == (True, "value-b"), "queued value mismatch")
    require(calls == 1, "deduplication executed the producer more than once")

    require(
        prefetcher.submit("boom", lambda: (_ for _ in ()).throw(ValueError("producer-boom"))),
        "exception producer was rejected",
    )
    expect_raises(ValueError, lambda: prefetcher.take("boom"), contains="producer-boom")
    prefetcher.shutdown()
    require(prefetcher.closed and prefetcher.retained == 0, "shutdown did not clear retained state")
    require(not prefetcher.submit("late", lambda: "late"), "submit-after-close was accepted")


def test_prefetch_shutdown_cancels_and_joins() -> None:
    prefetcher = PREFETCH.BoundedPrefetcher[str, str](2, thread_name="p2-test-shutdown")
    active_started = threading.Event()
    active_release = threading.Event()
    active_finished = threading.Event()

    def slow() -> str:
        active_started.set()
        require(active_release.wait(2), "test producer release timed out")
        active_finished.set()
        return "active"

    require(prefetcher.submit("active", slow), "active producer rejected")
    require(active_started.wait(2), "active producer did not start")
    require(prefetcher.submit("queued", lambda: "queued"), "queued producer rejected")

    shutdown_thread = threading.Thread(target=prefetcher.shutdown, name="p2-test-shutdown-caller")
    shutdown_thread.start()
    deadline = time.monotonic() + 2
    while not prefetcher.closed and time.monotonic() < deadline:
        time.sleep(0.005)
    require(prefetcher.closed, "shutdown did not close submissions")
    expect_raises(
        PREFETCH.PrefetchCancelledError,
        lambda: prefetcher.take("queued"),
        contains="cancelled during shutdown",
    )
    active_release.set()
    shutdown_thread.join(2)
    require(not shutdown_thread.is_alive(), "shutdown did not join the active producer")
    require(active_finished.is_set(), "shutdown did not finish the active producer")
    prefetcher.shutdown()
    require(prefetcher.retained == 0, "idempotent shutdown retained entries")


def test_prompt_manifest_json_and_jsonl() -> None:
    rows = [
        {
            "prompt_id": f"p{index}",
            "prompt": f"prompt {index}",
            "prompt_sha256": sha256_bytes(f"prompt {index}".encode()),
        }
        for index in range(8)
    ]
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        json_path = root / "prompts.json"
        jsonl_path = root / "prompts.jsonl"
        json_path.write_text(json.dumps({"prompts": rows}), encoding="utf-8")
        jsonl_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        require(load_prompt_manifest(json_path) == {"prompts": rows}, "JSON prompt load mismatch")
        require(load_prompt_manifest(jsonl_path) == {"prompts": rows}, "JSONL prompt load mismatch")
        jsonl_path.write_text('{"prompt_id":"broken"\\n', encoding="utf-8")
        expect_raises(GateError, lambda: load_prompt_manifest(jsonl_path), contains="invalid prompt JSONL")


def _git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", os.fspath(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout.strip()


def test_corrected_source_chain_gate() -> None:
    result = source_chain_gate(REPO)
    require(result["planner_commit"] == EXPECTED_P0_PLANNER_COMMIT, "planner commit mismatch")
    require(result["planner_tree"] == EXPECTED_P0_PLANNER_TREE, "planner tree mismatch")
    require(result["pr38_commit"] == EXPECTED_REBASED_PR38_COMMIT, "PR38 commit mismatch")
    require(result["pr38_tree"] == EXPECTED_REBASED_PR38_TREE, "PR38 tree mismatch")
    require(result["p2_commit"] == "48225e5ec732a3c7a06d2e0cfc61a98286a969cd", "P2 commit mismatch")

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        repo = root / "repo"
        repo.mkdir()
        _git(repo, "init", "-q")
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "P2 Test",
            "GIT_AUTHOR_EMAIL": "p2@example.invalid",
            "GIT_COMMITTER_NAME": "P2 Test",
            "GIT_COMMITTER_EMAIL": "p2@example.invalid",
        }
        path = repo / "chain.txt"
        commits: list[str] = []
        trees: list[str] = []
        for value in ("planner", "pr38", "p2"):
            path.write_text(value + "\n", encoding="utf-8")
            _git(repo, "add", "chain.txt", env=env)
            if value == "p2":
                _git(
                    repo,
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    "160000",
                    commits[0],
                    "vendor/upstream",
                    env=env,
                )
            _git(repo, "commit", "-q", "-m", value, env=env)
            commits.append(_git(repo, "rev-parse", "HEAD"))
            trees.append(_git(repo, "rev-parse", "HEAD^{tree}"))
        patch = subprocess.run(
            ["git", "-C", os.fspath(repo), "diff", "--binary", "--full-index", commits[1], commits[2]],
            check=True,
            capture_output=True,
        ).stdout
        kwargs = {
            "planner_commit": commits[0],
            "planner_tree": trees[0],
            "pr38_commit": commits[1],
            "pr38_tree": trees[1],
            "p2_commit": commits[2],
            "p2_tree": trees[2],
            "patch_sha256": sha256_bytes(patch),
            "require_fork": False,
            "expected_paths": {"chain.txt", "vendor/upstream"},
        }
        result = source_chain_gate(repo, **kwargs)
        require(
            result["production_paths"] == ["chain.txt", "vendor/upstream"],
            "source-chain path set mismatch",
        )
        seal = seal_repository(repo, root / "seal", **kwargs)
        manifest_path = Path(seal["tree_manifest"]["path"])
        archive_path = Path(seal["archive"]["path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require(
            manifest["gitlinks"] == [{"path": "vendor/upstream", "commit": commits[0]}],
            "source seal did not bind the gitlink",
        )
        archive = verify_archive(archive_path, manifest)
        require(archive["gitlink_count"] == 1, "archive did not retain the gitlink")
        extraction = root / "extracted"
        extract_verified(archive_path, extraction, manifest)
        directory = verify_directory(extraction, manifest)
        require(directory["full_tree_sha256"] == seal["full_tree_sha256"], "sealed tree mismatch")
        tampered = copy.deepcopy(manifest)
        tampered["gitlinks"][0]["commit"] = "0" * 40
        expect_raises(
            GateError,
            lambda: verify_archive(archive_path, tampered),
            contains="full_tree_sha256 mismatch",
        )
        stale_pr38 = dict(kwargs)
        stale_pr38["pr38_commit"] = commits[0]
        stale_pr38["pr38_tree"] = trees[0]
        expect_raises(
            GateError,
            lambda: source_chain_gate(repo, **stale_pr38),
            contains="PR38 is not a single commit",
        )
        stale_p2 = dict(kwargs)
        stale_p2["p2_commit"] = commits[1]
        stale_p2["p2_tree"] = trees[1]
        expect_raises(
            GateError,
            lambda: source_chain_gate(repo, **stale_p2),
            contains="P2 is not a single commit",
        )


def test_abba_order_and_fail_closed_effects() -> None:
    campaign = "p2-prefetch-abba-protocol"
    arms = arm_specs(campaign)
    require(
        [(arm["period"], arm["treatment"], arm["prefetch"]) for arm in arms]
        == [(1, "off", False), (2, "on", True), (3, "on", True), (4, "off", False)],
        "AB/BA arm order is not OFF/ON/ON/OFF",
    )
    durations = {
        1: (10.0, 6.0, 2.0, 2.0),
        2: (8.0, 4.8, 1.6, 1.6),
        3: (9.0, 5.4, 1.8, 1.8),
        4: (12.0, 7.2, 2.4, 2.4),
    }
    metric_names = (
        "perf/step_time_s",
        "perf/generate_time_s",
        "perf/reward_time_s",
        "perf/train_time_s",
    )
    records = []
    for arm in arms:
        metrics = dict(zip(metric_names, durations[arm["period"]]))
        records.append({**arm, "passed": True, "metrics": metrics})
    effects = paired_effects(records)
    require(effects["complete"] is True, "golden AB/BA effects were incomplete")
    require(
        [(pair["off_period"], pair["on_period"]) for pair in effects["pairs"]] == [(1, 2), (4, 3)],
        "AB/BA pairing direction changed",
    )
    expected_speedup = math.sqrt((10.0 / 8.0) * (12.0 / 9.0))
    require(
        math.isclose(
            effects["perf/step_time_s"]["geometric_mean_speedup"],
            expected_speedup,
        ),
        "AB/BA geometric-mean speedup mismatch",
    )
    failed = copy.deepcopy(records)
    failed[2]["passed"] = False
    failed_effects = paired_effects(failed)
    require(failed_effects["complete"] is False, "failed ON arm remained performance-usable")
    require(
        failed_effects["perf/step_time_s"]["pair_count"] == 1,
        "failed AB/BA pair leaked into the primary estimate",
    )


def _context(rank: int, campaign: str, run: str) -> dict[str, Any]:
    if rank < 0:
        return {
            "campaign_id": campaign,
            "run_name": run,
            "requested_sp_size": 2,
            "rank": -1,
        }
    return {
        "campaign_id": campaign,
        "run_name": run,
        "requested_sp_size": 2,
        "rank": rank,
        "world_size": 8,
        "sp_size": 2,
        "dp_size": 4,
        "sp_rank": rank % 2,
        "dp_rank": rank // 2,
        "tp_size": 1,
        "pp_size": 1,
        "ep_size": 1,
    }


def _fingerprint(label: str) -> dict[str, Any]:
    return {
        "mode": "deterministic_sample_v1",
        "shape": [1, 2, 3],
        "dtype": "torch.float32",
        "numel": 6,
        "sample_count": 6,
        "finite": True,
        "projection": 1.0,
        "sample_l2": 2.0,
        "sample_absmax": 3.0,
        "quantized_sha256": sha256_bytes(label.encode()),
    }


def _planner_payload(local_ids: list[str]) -> dict[str, Any]:
    rows = [{"sample_id": sample_id} for sample_id in local_ids]
    first = [row for row in rows if int(row["sample_id"].rsplit("/", 1)[1]) < 2]
    second = [row for row in rows if int(row["sample_id"].rsplit("/", 1)[1]) >= 2]
    after = [*first, *second]
    return {
        "planner_class": EXPECTED_PLANNER_CLASS,
        "invocation_index": 0,
        "num_updates": 2,
        "micro_batch_size": 1,
        "before": rows,
        "after": after,
        "permutation": [rows.index(row) for row in after],
        "updates": [
            {"update_index": 0, "rows": first, "micros": []},
            {"update_index": 1, "rows": second, "micros": []},
        ],
    }


def _emit_events(root: Path, *, treatment: str, campaign: str, run: str) -> list[dict[str, Any]]:
    prompts = [f"p{index}" for index in range(8)]
    prompt_sha = [sha256_bytes(f"prompt {index}".encode()) for index in range(8)]
    sample_ids = expected_sample_ids(prompts)
    by_rank = {
        rank: [
            sample_id
            for prompt_index, prompt_id in enumerate(prompts)
            if prompt_index // 2 == rank // 2
            for sample_id in expected_sample_ids([prompt_id])
        ]
        for rank in range(0, 8, 2)
    }
    events: list[dict[str, Any]] = []
    monotonic_ns = 1_000_000_000

    def add(name: str, rank: int, payload: dict[str, Any], *, at: int | None = None) -> None:
        nonlocal monotonic_ns
        monotonic_ns = monotonic_ns + 10_000_000 if at is None else at
        events.append(
            {
                "schema": P0_RUNTIME_EVENT_SCHEMA,
                "event": name,
                "sequence": len(events),
                "utc": "2026-09-14T00:00:00Z",
                "monotonic_ns": monotonic_ns,
                "context": _context(rank, campaign, run),
                "payload": payload,
            }
        )

    add(
        "p2_overlay_installed",
        -1,
        {
            "schema": P0_RUNTIME_EVENT_SCHEMA,
            "persistent_disk_cache": False,
            "conditioner_device": "cpu",
            "prefetch_expected": treatment == "on",
        },
    )
    add(
        "trainer_contract",
        -1,
        {
            "num_devices": 8,
            "batch_size": 8,
            "samples_per_prompt": 4,
            "num_inference_steps": 10,
            "height": 768,
            "width": 768,
            "num_frames": 124,
            "eta": 0.7,
            "sde_indices": [0, 3, 6],
            "data_shuffle": False,
            "data_source_runtime_shuffle": False,
            "offload_train_during_reward": True,
            "aux_components_on_cpu": True,
            "prompt_embedding_cache_dir": None,
            "prompt_embedding_cache_read_only": False,
            "prompt_embedding_share_across_sp": True,
            "prompt_embedding_prefetch": treatment == "on",
            "prompt_embedding_prefetch_capacity": 8,
            "text_encoder_onload_for_embed": False,
            "remote_handle_dereferenced": False,
        },
    )
    add(
        "data_batch",
        -1,
        {
            "batch_size": 8,
            "shuffle": False,
            "ordered_root_sample_ids": [f"prompt:{prompt_id}:sample:0" for prompt_id in prompts],
            "ordered_prompt_sha256": prompt_sha,
        },
    )
    for rank in range(8):
        add(
            "planner_contract",
            rank,
            {
                "planner_class": EXPECTED_PLANNER_CLASS,
                "num_updates_per_batch": 2,
                "micro_batch_size": 1,
                "algorithm_supports_multi_update": True,
                "expected_num_updates_per_batch": 2,
            },
        )
        source_rank = rank // 2 * 2
        add("planner_arrange", rank, copy.deepcopy(_planner_payload(by_rank[source_rank])))
        add(
            "sp_topology",
            rank,
            {
                "world_size": 8,
                "observed_sp_size": 2,
                "observed_dp_size": 4,
                "sp_group_ranks": [source_rank, source_rank + 1],
                "sp_installed": True,
                "sp_processor_count": 50,
                "attention_block_count": 50,
            },
        )
        add(
            "checkpoint_load",
            rank,
            {
                "success": True,
                "path": "/tmp/frozen-lora",
                "loaded_rollout_step": 0,
                "backend_optimizer_step": 0,
                "audit": {
                    "rollout_step": 0,
                    "optimizer_step_count": 0,
                    "trainer_optimizer_step": 0,
                    "save_mode": "adapter",
                },
            },
        )
        for update_index in range(2):
            add(
                "optimizer_step",
                rank,
                {
                    "before": update_index,
                    "after": update_index + 1,
                    "update_index": update_index,
                    "grad_norm": 1.0 + update_index,
                    "success": True,
                    "committed": True,
                    "finite_nonzero_grad": True,
                },
            )
            add(
                "train_update",
                rank,
                {
                    "success": True,
                    "optimizer_updates": 1,
                    "before_optimizer_step": update_index,
                    "after_optimizer_step": update_index + 1,
                },
            )
        add("run_updates", rank, {"optimizer_updates": 2, "per_update_count": 2})
        add(
            "prefetch_shutdown",
            rank,
            {
                "configured": treatment == "on",
                "closed": treatment == "on",
                "retained_before": 0,
                "retained_after": 0,
                "success": True,
                "error_type": None,
            },
        )
    add(
        "train_step",
        -1,
        {
            "success": True,
            "optimizer_updates": 2,
            "per_update_count": 2,
            "has_backward": True,
        },
    )

    denoise_end_by_rank: dict[int, int] = {}
    for source_index, rank in enumerate((0, 2, 4, 6)):
        denoise_end = 3_000_000_000 + source_index * 100_000_000
        denoise_end_by_rank[rank] = denoise_end
        add(
            "phase_timer",
            rank,
            {"timer": "denoise_s", "success": True, "duration_s": 0.2},
            at=denoise_end,
        )
        for prompt_offset in range(2):
            index = source_index + prompt_offset * 4
            background = treatment == "on" and prompt_offset == 1
            started = denoise_end - 150_000_000 if background else denoise_end + 10_000_000
            add(
                "conditioner_encode_call",
                rank,
                {
                    "prompt_sha256": prompt_sha[index],
                    "started_monotonic_ns": started,
                    "finished_monotonic_ns": started + 50_000_000,
                    "duration_s": 0.05,
                    "thread_name": ("minimax-h3-prompt-prefetch" if background else "MainThread"),
                    "background_prefetch": background,
                    "success": True,
                    "error_type": None,
                    "contract_violation": False,
                },
                at=started + 50_000_000,
            )

    for index, sample_id in enumerate(sample_ids):
        prompt_index = index // 4
        sibling = index % 4
        prompt_id = prompts[prompt_index]
        source_rank = (prompt_index // 2) * 2
        receiver_rank = source_rank + 1
        prompt_digest = prompt_sha[prompt_index]
        root_id = f"r0:prompt:{prompt_id}:sample:0"
        group_id = f"prompt:{prompt_id}"
        segment = {
            "indices": [0, 1, 2],
            "sde_indices": [0, 3, 6],
            "sigmas": [1.0, 0.5, 0.0],
            "video_trajectory": _fingerprint(f"{sample_id}:video-trajectory"),
            "audio_trajectory": _fingerprint(f"{sample_id}:audio-trajectory"),
            "initial_video_source": "initial_latents",
            "initial_video": _fingerprint(f"{sample_id}:initial-video"),
            "initial_audio": _fingerprint(f"{sample_id}:initial-audio"),
            "sde_logp": {"present": False},
            "sde_means": {"present": False},
            "final_video": _fingerprint(f"{sample_id}:final-video"),
            "final_audio": _fingerprint(f"{sample_id}:final-audio"),
        }
        generation = {
            "authoritative": True,
            "sample_id": sample_id,
            "root_id": root_id,
            "group_id": group_id,
            "sibling_ordinal": sibling,
            "prompt_sha256": prompt_digest,
            "segment": segment,
            "decoded_video": _fingerprint(f"{sample_id}:decoded-video"),
            "decoded_audio": _fingerprint(f"{sample_id}:decoded-audio"),
        }
        add("sample_generation", source_rank, generation)
        add(
            "generation_manifest",
            source_rank,
            {
                "authoritative": True,
                "sample_count": 1,
                "ordered_sample_ids": [sample_id],
                "entrypoint": "MiniMaxH3Pipeline._generate",
                "next_prompt_supplied": treatment == "on" and not (prompt_index % 2 == 1 and sibling == 3),
            },
        )
        add(
            "generation_replica_presence",
            receiver_rank,
            {
                "authoritative": False,
                "sample_count": 1,
                "entrypoint": "MiniMaxH3Pipeline._generate",
            },
        )
        video_reward = index / 100
        clap_reward = index / 200
        add(
            "sample_score",
            -1,
            {
                "rollout_id": 0,
                "sample_id": sample_id,
                "root_id": root_id,
                "group_id": group_id,
                "sibling_ordinal": sibling,
                "prompt_sha256": prompt_digest,
                "reward": 0.5 * video_reward + 0.5 * clap_reward,
                "videopickscore": video_reward,
                "clap": clap_reward,
                "expected_weighted_total": 0.5 * video_reward + 0.5 * clap_reward,
                "advantage": float(index - 16) / 16,
            },
        )
        add(
            "prefetch_consume",
            source_rank,
            {
                "requested": 1,
                "hits": int(treatment == "on" and prompt_index % 2 == 1 and sibling == 0),
                "enabled": treatment == "on",
                "success": True,
                "error_type": None,
            },
        )
        if treatment == "on" and not (prompt_index % 2 == 1 and sibling == 3):
            add(
                "prefetch_submit",
                source_rank,
                {
                    "prompt_count": 1,
                    "prompt_sha256": [prompt_digest],
                    "accepted": True,
                    "enabled": True,
                    "success": True,
                    "error_type": None,
                },
            )
            add(
                "prefetch_submit",
                receiver_rank,
                {
                    "prompt_count": 1,
                    "prompt_sha256": [prompt_digest],
                    "accepted": True,
                    "enabled": True,
                    "success": True,
                    "error_type": None,
                },
            )
    add(
        "p2_driver_cleanup",
        -1,
        {
            "pool_shutdown": True,
            "training_succeeded": True,
            "error_type": None,
            "duration_s": 0.01,
        },
    )

    event_dir = root / "events"
    event_dir.mkdir(parents=True)
    for index, event in enumerate(events):
        path = event_dir / f"{index:06d}-{event['event']}.json"
        path.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
        path.with_name(path.name + ".sha256").write_text(
            f"{sha256_file(path)}  {path.name}\n",
            encoding="utf-8",
        )
    return events


def _aggregate(
    root: Path,
    *,
    treatment: str,
    run: str,
    output: Path,
) -> dict[str, Any]:
    from harness_lib import aggregate_p0_runtime_events

    prompts = [f"p{index}" for index in range(8)]
    prompt_sha = [sha256_bytes(f"prompt {index}".encode()) for index in range(8)]
    return aggregate_p0_runtime_events(
        root,
        campaign_id="p2-prefetch-abba-protocol",
        run_name=run,
        treatment=treatment,
        prompt_ids=prompts,
        prompt_sha256=prompt_sha,
        planner_contract_sha256="1" * 64,
        two_update_contract_sha256="2" * 64,
        frozen_checkpoint={
            "load_dir": "/tmp/frozen-lora",
            "audit_binding": {"sha256": "3" * 64},
            "tree_sha256": "4" * 64,
            "model_summary_sha256": "5" * 64,
            "adapter_key_signature_sha256": "6" * 64,
        },
        output_dir=output,
    )


def test_runtime_golden_off_on_and_fail_closed_mutations() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        for treatment in ("off", "on"):
            event_root = root / treatment
            output = root / f"{treatment}-out"
            run = f"protocol-{treatment}"
            _emit_events(
                event_root,
                treatment=treatment,
                campaign="p2-prefetch-abba-protocol",
                run=run,
            )
            summary = _aggregate(event_root, treatment=treatment, run=run, output=output)
            require(summary["completed"] is True, f"{treatment} golden aggregation failed")
            require(
                summary["event_tree_sha256"] == runtime_event_tree_digest(load_p0_events(event_root)),
                "event-tree digest mismatch",
            )
            ledger = [json.loads(line) for line in (output / f"sample_ledger_{run}.jsonl").read_text().splitlines()]
            ok, issues, _ = sample_ledger_gate(
                ledger,
                expected_ids=expected_sample_ids([f"p{index}" for index in range(8)]),
            )
            require(ok, f"{treatment} sample ledger failed: {issues}")

        cases: list[tuple[str, str, Callable[[Path], None], str]] = []

        def remove_sidecar(event_root: Path) -> None:
            next((event_root / "events").glob("*.json.sha256")).unlink()

        def add_orphan_sidecar(event_root: Path) -> None:
            (event_root / "events" / "orphan.json.sha256").write_text(
                f"{'0' * 64}  orphan.json\n",
                encoding="utf-8",
            )

        def mutate_event(
            event_root: Path,
            event_name: str,
            mutate: Callable[[dict[str, Any]], None],
        ) -> None:
            for path in sorted((event_root / "events").glob(f"*-{event_name}.json")):
                payload = json.loads(path.read_text())
                mutate(payload)
                path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
                path.with_name(path.name + ".sha256").write_text(
                    f"{sha256_file(path)}  {path.name}\n",
                    encoding="utf-8",
                )
                return
            raise AssertionError(f"missing synthetic event {event_name}")

        cases.extend(
            [
                ("missing-sidecar", "on", remove_sidecar, "sidecar set mismatch"),
                ("orphan-sidecar", "off", add_orphan_sidecar, "sidecar set mismatch"),
                (
                    "planner-mismatch",
                    "on",
                    lambda path: mutate_event(
                        path,
                        "planner_arrange",
                        lambda row: row["payload"].update({"num_updates": 1}),
                    ),
                    "planner_arrange contract mismatch",
                ),
                (
                    "checkpoint-step",
                    "off",
                    lambda path: mutate_event(
                        path,
                        "checkpoint_load",
                        lambda row: row["payload"].update({"backend_optimizer_step": 1}),
                    ),
                    "frozen zero-step checkpoint",
                ),
                (
                    "overlap-missing",
                    "on",
                    lambda path: mutate_event(
                        path,
                        "conditioner_encode_call",
                        lambda row: row["payload"].update(
                            {
                                "background_prefetch": True,
                                "thread_name": "minimax-h3-prompt-prefetch",
                                "started_monotonic_ns": 9_000_000_000,
                                "finished_monotonic_ns": 9_050_000_000,
                            }
                        ),
                    ),
                    "did not overlap denoising",
                ),
                (
                    "disk-cache",
                    "off",
                    lambda path: mutate_event(
                        path,
                        "p2_overlay_installed",
                        lambda row: row.update({"event": "cache_lookup", "payload": {"contract_violation": True}}),
                    ),
                    "persistent cache_lookup activity is forbidden",
                ),
                (
                    "prompt-digest",
                    "off",
                    lambda path: mutate_event(
                        path,
                        "sample_generation",
                        lambda row: row["payload"].update({"prompt_sha256": "f" * 64}),
                    ),
                    "prompt digest differs from manifest",
                ),
                (
                    "cleanup-missing",
                    "on",
                    lambda path: mutate_event(
                        path,
                        "p2_driver_cleanup",
                        lambda row: row["payload"].update({"pool_shutdown": False}),
                    ),
                    "driver cleanup",
                ),
            ]
        )
        for name, treatment, mutate, message in cases:
            event_root = root / f"mutation-{name}"
            output = root / f"mutation-{name}-out"
            run = f"mutation-{name}"
            _emit_events(
                event_root,
                treatment=treatment,
                campaign="p2-prefetch-abba-protocol",
                run=run,
            )
            mutate(event_root)
            expect_raises(
                GateError,
                lambda event_root=event_root, treatment=treatment, run=run, output=output: _aggregate(
                    event_root,
                    treatment=treatment,
                    run=run,
                    output=output,
                ),
                contains=message,
            )
            summary_path = output / f"runtime_summary_{run}.json"
            if summary_path.exists():
                failed = json.loads(summary_path.read_text())
                require(failed["completed"] is False, "failed aggregation wrote completed=true")


def test_protocol_gates_and_partial_summary() -> None:
    prompt_ids = [f"p{index}" for index in range(8)]
    updates = {
        "schema": UPDATE_MEMBERSHIP_SCHEMA,
        "planner_contract_sha256": "1" * 64,
        "two_update_contract_sha256": "2" * 64,
        "frozen_checkpoint_audit_sha256": "3" * 64,
        "planner_target": "unirl.train.stack.GroupInterleavedCountPlanner",
        "runtime_planner_class": EXPECTED_PLANNER_CLASS,
        "num_updates": 2,
        "optimizer_updates": 2,
        "generated_samples": GENERATED_SAMPLES,
        "group_key": "prompt_id",
        "group_preserving": True,
        "exhaustive": True,
        "disjoint": True,
        "updates": [
            {"update_index": index, "sample_ids": sample_ids}
            for index, sample_ids in enumerate(expected_update_membership(prompt_ids))
        ],
        "optimizer_transitions": [
            {"before": 0, "after": 1, "grad_norm": 1.0},
            {"before": 1, "after": 2, "grad_norm": 2.0},
        ],
        "runtime_source_ranks": [0, 2, 4, 6],
    }
    ok, issues = update_membership_gate(
        updates,
        prompt_ids=prompt_ids,
        planner_contract_sha256="1" * 64,
        two_update_contract_sha256="2" * 64,
        frozen_checkpoint_audit_sha256="3" * 64,
    )
    require(ok, f"golden two-update membership failed: {issues}")
    bad_updates = copy.deepcopy(updates)
    bad_updates["updates"] = bad_updates["updates"][:1]
    require(
        not update_membership_gate(
            bad_updates,
            prompt_ids=prompt_ids,
            planner_contract_sha256="1" * 64,
            two_update_contract_sha256="2" * 64,
            frozen_checkpoint_audit_sha256="3" * 64,
        )[0],
        "one-update mutation passed",
    )

    metrics = {
        "perf/step_time_s": 10.0,
        "perf/generate_time_s": 6.0,
        "perf/reward_time_s": 2.0,
        "perf/train_time_s": 2.0,
        "train/grad_norm": [1.0, 2.0],
        "train/optimizer_updates": 2,
        "rollout/reward_mean": 0.5,
    }
    require(metrics_gate(metrics)[0], "golden metrics failed")
    bad_metrics = dict(metrics)
    bad_metrics["rollout/reward_mean"] = float("nan")
    require(not metrics_gate(bad_metrics)[0], "non-finite reward metric passed")

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        (root / "partial-remote").mkdir()
        (root / "partial-remote" / "runtime_summary_failed.json").write_text(
            json.dumps({"completed": False}) + "\n",
            encoding="utf-8",
        )
        partial = write_partial_summary(
            root,
            campaign_id="p2-prefetch-abba-protocol",
            exit_code=17,
            failed_line=222,
            failed_command="fake-cb failure",
            current_run="protocol-on",
        )
        require(partial["completed"] is False, "partial summary became a completed result")
        require(partial["performance_usable"] is False, "partial summary became performance-usable")
        require(
            "partial-remote/runtime_summary_failed.json" in partial["available_files"],
            "partial runtime summary was not inventoried",
        )

        artifact = root / "runtime"
        artifact.mkdir()
        run = "protocol-on"
        for name, payload in (
            (
                f"sample_ledger_{run}.jsonl",
                json.dumps({"schema": SAMPLE_LEDGER_SCHEMA}) + "\n",
            ),
            (
                f"update_membership_{run}.json",
                json.dumps({"schema": UPDATE_MEMBERSHIP_SCHEMA}),
            ),
            (
                f"frozen_checkpoint_loaded_{run}.json",
                json.dumps({"schema": FROZEN_CHECKPOINT_LOADED_SCHEMA}),
            ),
        ):
            (artifact / name).write_text(payload, encoding="utf-8")
        event_root = artifact / f"runtime_receipts_{run}"
        event_dir = event_root / "events"
        event_dir.mkdir(parents=True)
        event = {
            "schema": P0_RUNTIME_EVENT_SCHEMA,
            "event": "p2_driver_cleanup",
            "sequence": 0,
            "monotonic_ns": 1,
            "context": _context(-1, "p2-prefetch-abba-protocol", run),
            "payload": {"pool_shutdown": True},
        }
        event_path = event_dir / "000000-p2_driver_cleanup.json"
        event_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
        event_path.with_name(event_path.name + ".sha256").write_text(
            f"{sha256_file(event_path)}  {event_path.name}\n",
            encoding="utf-8",
        )
        loaded = load_p0_events(event_root)
        summary = {
            "schema": P2_RUNTIME_SUMMARY_SCHEMA,
            "run_name": run,
            "treatment": "on",
            "completed": True,
            "event_schema": P0_RUNTIME_EVENT_SCHEMA,
            "event_count": 1,
            "event_tree_sha256": runtime_event_tree_digest(loaded),
            "planner_contract_sha256": "1" * 64,
            "two_update_contract_sha256": "2" * 64,
            "topology_ranks": list(range(8)),
            "sample_ledger_sha256": sha256_file(artifact / f"sample_ledger_{run}.jsonl"),
            "update_membership_sha256": sha256_file(artifact / f"update_membership_{run}.json"),
            "frozen_checkpoint_loaded_sha256": sha256_file(artifact / f"frozen_checkpoint_loaded_{run}.json"),
            "cleanup": {"pool_shutdown": True},
            "conditioner": {
                "encode_calls": 8,
                "source_ranks": [0, 2, 4, 6],
                "background_encode_calls": 4,
                "prefetch_consumed_hits": 4,
                "shutdown_ranks": list(range(8)),
                "overlap": [{"overlapped": True}] * 4,
                "all_background_calls_overlapped": True,
            },
        }
        summary_path = artifact / f"runtime_summary_{run}.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        p0 = {
            "planner_contract_sha256": "1" * 64,
            "two_update_contract_sha256": "2" * 64,
        }
        ok, issues, _ = runtime_summary_gate(
            summary_path,
            run=run,
            treatment="on",
            p0=p0,
            event_root=event_root,
        )
        require(ok, f"golden runtime summary failed: {issues}")
        event_path.write_text(json.dumps({**event, "event": "tampered"}) + "\n", encoding="utf-8")
        ok, issues, _ = runtime_summary_gate(
            summary_path,
            run=run,
            treatment="on",
            p0=p0,
            event_root=event_root,
        )
        require(not ok and any("digest mismatch" in issue for issue in issues), "event tamper passed")


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tests: list[tuple[str, Test]] = [
        ("prefetch_capacity_dedup_exception", test_prefetch_capacity_dedup_and_exception),
        ("prefetch_shutdown_cancel_join", test_prefetch_shutdown_cancels_and_joins),
        ("prompt_manifest_json_jsonl", test_prompt_manifest_json_and_jsonl),
        ("corrected_source_chain", test_corrected_source_chain_gate),
        ("abba_order_fail_closed_effects", test_abba_order_and_fail_closed_effects),
        ("runtime_golden_and_mutations", test_runtime_golden_off_on_and_fail_closed_mutations),
        ("protocol_gates_partial_summary", test_protocol_gates_and_partial_summary),
    ]
    results: list[dict[str, Any]] = []
    failed = False
    for name, test in tests:
        started = time.perf_counter()
        try:
            test()
        except BaseException as exc:
            failed = True
            results.append(
                {
                    "name": name,
                    "passed": False,
                    "duration_s": time.perf_counter() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        else:
            results.append(
                {
                    "name": name,
                    "passed": True,
                    "duration_s": time.perf_counter() - started,
                }
            )
    payload = {
        "schema": "unirl-minimax-h3-p2-prefetch-protocol-tests-v1",
        "completed": not failed,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "test_count": len(results),
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
