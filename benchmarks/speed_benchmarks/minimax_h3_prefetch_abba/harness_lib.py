#!/usr/bin/env python3
"""Shared fail-closed helpers for the MiniMax-H3 P2 prefetch AB/BA harness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_PREFIX = "unirl-minimax-h3-p2-prefetch"
P0_CONTRACT_SCHEMA = "unirl-minimax-h3-p2-integration-contract-v1"
P0_DESIGN_SCHEMA = "unirl-minimax-h3-corrected-screening-design-v1"
P0_RUNTIME_EVENT_SCHEMA = "unirl-minimax-h3-p0-runtime-event-v2"
P2_RUNTIME_SUMMARY_SCHEMA = "unirl-minimax-h3-p2-runtime-summary-v1"
FROZEN_INITIAL_LORA_AUDIT_SCHEMA = "unirl-minimax-h3-frozen-initial-lora-audit-v1"
FROZEN_CHECKPOINT_LOADED_SCHEMA = "unirl-minimax-h3-p2-frozen-checkpoint-loaded-v1"
UPDATE_MEMBERSHIP_SCHEMA = "unirl-minimax-h3-p2-two-update-membership-v1"
SAMPLE_LEDGER_SCHEMA = "unirl-minimax-h3-p2-sample-ledger-row-v1"
LEGACY_PR38_COMMIT = "fc2fb26f50028082c09fbb1b789aaf82596d7c43"
LEGACY_P2_COMMIT = "3c6522e004880ea1f60a144831caa8edbf296396"
EXPECTED_REBASED_PR38_COMMIT = "9feecf82038f827358ae4686fb8d4045fae7cc72"
EXPECTED_REBASED_PR38_TREE = "a299ce1d64cf2bde2874f3428e43d5ac4dc3040e"
EXPECTED_P0_PLANNER_COMMIT = "b0e7b241f3631bf78f80d9a20ab15413b01b6a79"
EXPECTED_P0_PLANNER_TREE = "24615687c03b1b9522a3c0243c08835597d7f92a"
EXPECTED_P1_SOURCE_PREFLIGHT_SCHEMA = "unirl-minimax-h3-pr38-p1-source-preflight-v1"
EXPECTED_P1_ARCHIVE_SHA256 = "32170d2f992f940803f58218cc3021fbee5e07f5740c99b1f1fe0d7be2e580a2"
EXPECTED_P1_FULL_TREE_SHA256 = "9b2d46261a33e69153fe38ea790ff0d1dbe1d4926fbf5768956346c2bf8988fc"
EXPECTED_P1_FILES_MANIFEST_SHA256 = "c98ce7afd9943e71dab5cd3c01b81844b24486aee4b7efc28e84f7212821e6e5"
EXPECTED_P1_SOURCE_PREFLIGHT_SHA256 = "c19f1b6522d34e4ea57aef03926119b0152d2a7243f6b118a50e9664cadab495"
EXPECTED_P0_RUNTIME_OVERLAY_SHA256 = "557c4ece42f2b3200577c6ee6aeaa387eaadcf36a2be8d381dfd1e4746714388"
EXPECTED_PLANNER_IMPLEMENTATION_SHA256 = "88d0551e8273ebc0c98bc82e30dbb0f4d2ff829f887e8700c3516542c346230b"
EXPECTED_REMOTE = "https://github.com/leviking98z-rgb/UniRL.git"
EXPECTED_PLANNER_TARGET = "unirl.train.stack.GroupInterleavedCountPlanner"
EXPECTED_PLANNER_CLASS = "unirl.train.stack.planner.count.GroupInterleavedCountPlanner"
RECIPE = "diffusion/minimax_h3/minimax_h3_t2va_trainside"
RECIPE_PATH = "examples/diffusion/minimax_h3/minimax_h3_t2va_trainside.yaml"
WORLD_SIZE = 8
SP_SIZE = 2
DP_SIZE = 4
PREFETCH_CAPACITY = 8
SAMPLES_PER_PROMPT = 4
NUM_PROMPTS = 8
GENERATED_SAMPLES = 32
NUM_UPDATES = 2
ARM_ORDER: tuple[tuple[int, str], ...] = ((1, "off"), (2, "on"), (3, "on"), (4, "off"))
FUNCTIONAL_OVERRIDES = {
    "num_devices": WORLD_SIZE,
    "batch_size": NUM_PROMPTS,
    "num_rollouts": 1,
    "data_source.args.run.seed": 42,
    "data_source.args.run.shuffle": False,
    "sampling.seed": 42,
    "sampling.height": 768,
    "sampling.width": 768,
    "sampling.num_frames": 124,
    "sampling.num_inference_steps": 10,
    "sampling.sde_indices": [0, 3, 6],
    "sampling.eta": 0.7,
    "sampling.samples_per_prompt": SAMPLES_PER_PROMPT,
    "stack.num_updates_per_batch": NUM_UPDATES,
    "stack.micro_batch_size": 1,
    "stack.micro_planner._target_": EXPECTED_PLANNER_TARGET,
    "backend.fsdp_cfg.sp_size": SP_SIZE,
    "bundle.config.prompt_embedding_share_across_sp": True,
    "bundle.config.prompt_embedding_cache_dir": None,
    "bundle.config.prompt_embedding_cache_read_only": False,
    "bundle.config.text_encoder_onload_for_embed": False,
    "bundle.config.prompt_embedding_prefetch_capacity": PREFETCH_CAPACITY,
    "bundle.config.aux_components_on_cpu": True,
    "bundle.config.vae_components_on_cpu": False,
    "rollout.forward_batch_size": 1,
    "pipeline.strategy._target_": "unirl.sde.kernels.CPSSDEStrategy",
    "algorithm._target_": "unirl.algorithms.flowgrpo.FlowGRPO",
    "reward.backend._target_": "unirl.reward.local.t2av_composite.T2AVCompositeScorer",
    "reward.backend.config.weights.videopickscore": 0.5,
    "reward.backend.config.weights.clap": 0.5,
    "reward.backend.config.scorers.videopickscore._target_": (
        "unirl.reward.local.video_pickscore.VideoPickScoreScorer"
    ),
    "reward.backend.config.scorers.videopickscore.config.frame_selection": "middle",
    "reward.backend.config.scorers.clap._target_": "unirl.reward.local.clap.CLAPRewardScorer",
    "offload_train_during_reward": True,
    "logging.log_media": False,
}
SOURCE_MARKERS: Mapping[str, tuple[str, ...]] = {
    "unirl/models/minimax_h3/prefetch.py": (
        "class BoundedPrefetcher",
        "def submit",
        "def take",
        "def shutdown",
        "PrefetchCancelledError",
    ),
    "unirl/models/minimax_h3/text_embed.py": (
        "def prefetch",
        "def _take_prefetched",
        "def _encode_prefetch",
        "MiniMax-H3 prompt prefetch digest mismatch",
        "prompt prefetch disabled",
    ),
    "unirl/models/minimax_h3/pipeline.py": (
        "def generate_with_prompt_prefetch",
        "def prefetch_prompt",
        "def shutdown_prompt_prefetch",
    ),
    "unirl/rollout/engine/trainside/engine.py": (
        "generate_with_prompt_prefetch",
        "prompt_prefetch_enabled",
        "shutdown_prompt_prefetch",
    ),
}
EMBED_RE = re.compile(
    r"MiniMaxH3 text embeds:\s*prompts=(?P<prompts>\d+)\s+"
    r"cache_hits=(?P<cache_hits>\d+)\s+misses=(?P<misses>\d+)\s+"
    r"onload=(?P<onload>True|False)\s+elapsed_s=(?P<elapsed>[0-9.eE+-]+)\s+"
    r"memory_hits=(?P<memory_hits>\d+)\s+disk_hits=(?P<disk_hits>\d+)\s+"
    r"shared_hits=(?P<shared_hits>\d+)\s+shared_source=(?P<shared_source>True|False)"
)
FAILURE_PATTERNS: Mapping[str, re.Pattern[str]] = {
    "traceback": re.compile(r"Traceback \(most recent call last\)", re.I),
    "oom": re.compile(r"CUDA out of memory|OutOfMemoryError", re.I),
    "nccl": re.compile(r"NCCL.*(?:error|failed|timeout|watchdog)", re.I),
    "nonfinite_metric": re.compile(
        r"(?:loss|reward|grad(?:ient)?(?:_norm)?|metric)[^\n=]{0,40}[=:]\s*(?:nan|[+-]?inf)(?:\s|$)",
        re.I,
    ),
    "prefetch_disabled": re.compile(r"MiniMax-H3 prompt prefetch disabled", re.I),
    "prefetch_digest": re.compile(r"prompt prefetch (?:digest|token IDs) mismatch", re.I),
    "prefetch_cancelled": re.compile(r"PrefetchCancelledError|cancelled during shutdown", re.I),
    "readonly_cache_miss": re.compile(r"read-only prompt embedding cache miss", re.I),
}


class GateError(RuntimeError):
    """A contract or artifact failed a mandatory gate."""


def canonical_json(value: Any) -> bytes:
    """Serialize JSON data deterministically for an identity digest."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise GateError(f"missing file for SHA-256: {path}") from exc
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GateError(f"missing JSON artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GateError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"expected JSON object in {path}")
    return value


def load_prompt_manifest(path: Path) -> dict[str, Any]:
    """Load and strictly validate a JSON or JSONL prompt manifest."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise GateError(f"missing prompt manifest: {path}") from exc

    stripped = text.strip()
    if not stripped:
        raise GateError(f"empty prompt manifest: {path}")
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GateError(f"invalid prompt JSONL row {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise GateError(f"prompt JSONL row {path}:{line_number} is not an object")
            rows.append(row)
        return {"prompts": rows}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GateError(f"invalid prompt JSON artifact {path}: {exc}") from exc
    if isinstance(payload, list):
        return {"prompts": payload}
    if isinstance(payload, dict):
        return payload
    raise GateError(f"prompt manifest must be a JSON object, list, or JSONL: {path}")


def read_kv(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise GateError(f"missing key/value artifact: {path}") from exc
    values: dict[str, str] = {}
    for line in lines:
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise GateError(f"malformed key/value line in {path}: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise GateError(f"duplicate/empty key in {path}: {key!r}")
        values[key] = value
    return values


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def require_sha(value: Any, *, name: str) -> str:
    if not is_sha256(value):
        raise GateError(f"{name} must be a lowercase SHA-256 digest")
    return str(value)


def require_git_oid(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value) is None:
        raise GateError(f"{name} must be a lowercase Git object ID")
    return value


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def nested(root: Mapping[str, Any], dotted: str) -> Any:
    value: Any = root
    for component in dotted.split("."):
        if not isinstance(value, Mapping) or component not in value:
            return None
        value = value[component]
    return value


def normalize_path(path: str | os.PathLike[str], *, base: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def verify_file_binding(binding: Mapping[str, Any], *, name: str, base: Path) -> dict[str, Any]:
    raw_path = binding.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise GateError(f"{name}.path is required")
    path = normalize_path(raw_path, base=base)
    if not path.is_file():
        raise GateError(f"{name} does not exist or is not a file: {path}")
    expected = require_sha(binding.get("sha256"), name=f"{name}.sha256")
    observed = sha256_file(path)
    if observed != expected:
        raise GateError(f"{name} digest mismatch: expected {expected}, observed {observed}")
    return {"path": os.fspath(path), "sha256": observed, "bytes": path.stat().st_size}


def _validate_prompt_manifest(payload: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    rows = payload.get("prompts")
    if not isinstance(rows, list) or len(rows) != NUM_PROMPTS:
        raise GateError(f"prompt manifest must contain exactly {NUM_PROMPTS} prompts")
    prompt_ids: list[str] = []
    prompt_digests: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise GateError(f"prompt manifest row {index} is not an object")
        prompt_id = row.get("prompt_id")
        prompt = row.get("prompt")
        text_sha = row.get("prompt_sha256")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise GateError(f"prompt manifest row {index} lacks prompt_id")
        if not isinstance(prompt, str) or not prompt:
            raise GateError(f"prompt manifest row {index} lacks prompt text")
        require_sha(text_sha, name=f"prompt manifest row {index}.prompt_sha256")
        observed_text_sha = sha256_bytes(prompt.encode("utf-8"))
        if text_sha != observed_text_sha:
            raise GateError(
                f"prompt manifest row {index} text digest mismatch: expected {text_sha}, observed {observed_text_sha}"
            )
        prompt_ids.append(prompt_id)
        prompt_digests.append(str(text_sha))
    if len(set(prompt_ids)) != len(prompt_ids):
        raise GateError("prompt manifest prompt_id values are not unique")
    if len(set(prompt_digests)) != len(prompt_digests):
        raise GateError("P2 prefetch benchmark requires eight distinct prompt texts")
    return prompt_ids, prompt_digests


def validate_p0_contract(path: Path) -> dict[str, Any]:
    """Validate the reviewer-authored P2 integration contract."""
    contract = load_json(path)
    base = path.parent.resolve()
    schema = contract.get("schema")
    if schema != P0_CONTRACT_SCHEMA:
        raise GateError(f"unsupported integration contract schema: {schema!r}")

    source = contract.get("source")
    planner = contract.get("planner")
    training = contract.get("training")
    two_update = contract.get("two_update")
    frozen = contract.get("frozen_checkpoint")
    runtime_overlay = contract.get("runtime_overlay")
    if not all(
        isinstance(value, Mapping) for value in (source, planner, training, two_update, frozen, runtime_overlay)
    ):
        raise GateError(
            "integration contract requires source, planner, training, two_update, "
            "frozen_checkpoint, and runtime_overlay objects"
        )

    if source.get("remote") != EXPECTED_REMOTE or source.get("official_push_disabled") is not True:
        raise GateError("integration source must be the user fork with official push disabled")
    for key in ("integration_commit", "integration_tree"):
        require_git_oid(source.get(key), name=f"source.{key}")
    require_sha(source.get("full_tree_sha256"), name="source.full_tree_sha256")
    pr38 = source.get("pr38")
    p2 = source.get("p2")
    if not isinstance(pr38, Mapping) or not isinstance(p2, Mapping):
        raise GateError("source.pr38 and source.p2 are required")
    for label, payload in (("source.pr38", pr38), ("source.p2", p2)):
        require_git_oid(payload.get("commit"), name=f"{label}.commit")
        require_git_oid(payload.get("tree"), name=f"{label}.tree")
    if (
        pr38.get("commit") != EXPECTED_REBASED_PR38_COMMIT
        or pr38.get("tree") != EXPECTED_REBASED_PR38_TREE
        or pr38.get("parent_commit") != EXPECTED_P0_PLANNER_COMMIT
        or pr38.get("rebased") is not True
        or pr38.get("supersedes_commit") != LEGACY_PR38_COMMIT
    ):
        raise GateError("source.pr38 must identify the P1-validated rebased PR38 head 9feecf8")
    if (
        p2.get("supersedes_commit") != LEGACY_P2_COMMIT
        or p2.get("commit") == LEGACY_P2_COMMIT
        or p2.get("parent_commit") != pr38.get("commit")
    ):
        raise GateError("source.p2 must be the P2 patch rebased directly onto source.pr38.commit")
    require_sha(p2.get("patch_sha256"), name="source.p2.patch_sha256")
    contains = source.get("contains_commits")
    if not isinstance(contains, list) or not {pr38["commit"], p2["commit"]}.issubset(set(contains)):
        raise GateError("integration source must declare both rebased PR38 and rebased P2 commits")
    if source.get("clean") is not True:
        raise GateError("integration source must be clean")

    source_archive = verify_file_binding(source.get("archive") or {}, name="source.archive", base=base)
    source_manifest = verify_file_binding(source.get("tree_manifest") or {}, name="source.tree_manifest", base=base)
    p1_preflight_binding = verify_file_binding(
        source.get("p1_source_preflight") or {},
        name="source.p1_source_preflight",
        base=base,
    )
    if p1_preflight_binding["sha256"] != EXPECTED_P1_SOURCE_PREFLIGHT_SHA256:
        raise GateError("source.p1_source_preflight is not the reviewed 9feecf8 receipt")
    p1_preflight = load_json(Path(p1_preflight_binding["path"]))
    p1_source = p1_preflight.get("source")
    p1_sealed = p1_source.get("sealed_source") if isinstance(p1_source, Mapping) else None
    p1_live = p1_source.get("live_source") if isinstance(p1_source, Mapping) else None
    p1_identity_keys = (
        "head",
        "tree",
        "archive_sha256",
        "files_manifest_sha256",
        "symlinks_manifest_sha256",
        "full_tree_sha256",
        "file_count",
        "symlink_count",
        "critical_sha256",
        "full_tree_verified",
    )
    if (
        p1_preflight.get("schema") != EXPECTED_P1_SOURCE_PREFLIGHT_SCHEMA
        or p1_preflight.get("passed") is not True
        or p1_preflight.get("selected_sp_size") != SP_SIZE
        or not isinstance(p1_source, Mapping)
        or p1_source.get("origin") != EXPECTED_REMOTE
        or p1_source.get("head") != EXPECTED_REBASED_PR38_COMMIT
        or p1_source.get("tree") != EXPECTED_REBASED_PR38_TREE
        or p1_source.get("archive_sha256") != EXPECTED_P1_ARCHIVE_SHA256
        or p1_source.get("derived_dp_size") != DP_SIZE
        or nested(p1_source, "canonical_contract.micro_planner") != EXPECTED_PLANNER_TARGET
        or nested(p1_source, "canonical_contract.updates") != NUM_UPDATES
        or not isinstance(p1_sealed, Mapping)
        or not isinstance(p1_live, Mapping)
        or p1_sealed.get("full_tree_sha256") != EXPECTED_P1_FULL_TREE_SHA256
        or p1_sealed.get("files_manifest_sha256") != EXPECTED_P1_FILES_MANIFEST_SHA256
        or p1_sealed.get("full_tree_verified") is not True
        or any(p1_sealed.get(key) != p1_live.get(key) for key in p1_identity_keys)
    ):
        raise GateError("source.p1_source_preflight is not the canonical P1 SP2 source receipt")

    tree_manifest = load_json(Path(source_manifest["path"]))
    if tree_manifest.get("schema") != "unirl-sealed-source-tree-v1":
        raise GateError("source tree manifest schema mismatch")
    if (
        tree_manifest.get("head") != source.get("integration_commit")
        or tree_manifest.get("tree") != source.get("integration_tree")
        or tree_manifest.get("full_tree_sha256") != source.get("full_tree_sha256")
    ):
        raise GateError("source tree manifest identity differs from integration contract")
    files = tree_manifest.get("files")
    symlinks = tree_manifest.get("symlinks")
    gitlinks = tree_manifest.get("gitlinks")
    if not isinstance(files, list) or not files or not isinstance(symlinks, list) or not isinstance(gitlinks, list):
        raise GateError("source tree manifest requires files, symlinks, and gitlinks lists")
    file_paths: set[str] = set()
    for index, row in enumerate(files):
        if not isinstance(row, Mapping):
            raise GateError(f"source tree file row {index} is not an object")
        relative = row.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in file_paths
        ):
            raise GateError(f"source tree file row {index} has an unsafe/duplicate path")
        file_paths.add(relative)
        require_sha(row.get("sha256"), name=f"source tree file row {index}.sha256")
        if not isinstance(row.get("size"), int) or row["size"] < 0:
            raise GateError(f"source tree file row {index} has invalid size")
        if not isinstance(row.get("mode"), str) or re.fullmatch(r"[0-7]{4}", row["mode"]) is None:
            raise GateError(f"source tree file row {index} has invalid mode")
    link_paths: set[str] = set()
    for index, row in enumerate(symlinks):
        if not isinstance(row, Mapping):
            raise GateError(f"source tree symlink row {index} is not an object")
        relative = row.get("path")
        target = row.get("target")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in file_paths
            or relative in link_paths
        ):
            raise GateError(f"source tree symlink row {index} has an unsafe/duplicate path")
        if not isinstance(target, str) or not target or target.startswith("/") or ".." in Path(target).parts:
            raise GateError(f"source tree symlink row {index} has an unsafe target")
        link_paths.add(relative)
    gitlink_paths: set[str] = set()
    for index, row in enumerate(gitlinks):
        if not isinstance(row, Mapping):
            raise GateError(f"source tree gitlink row {index} is not an object")
        relative = row.get("path")
        commit = row.get("commit")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in file_paths
            or relative in link_paths
            or relative in gitlink_paths
        ):
            raise GateError(f"source tree gitlink row {index} has an unsafe/duplicate path")
        require_git_oid(commit, name=f"source tree gitlink row {index}.commit")
        gitlink_paths.add(relative)

    p0_design_binding = verify_file_binding(contract.get("p0_design") or {}, name="p0_design", base=base)
    p0_design = load_json(Path(p0_design_binding["path"]))
    p0_training = p0_design.get("training_contract")
    if (
        p0_design.get("schema") != P0_DESIGN_SCHEMA
        or p0_design.get("source_commit") != EXPECTED_P0_PLANNER_COMMIT
        or not isinstance(p0_training, Mapping)
    ):
        raise GateError("p0_design is not the corrected P0 design bound to b0e7b24")

    planner_target = planner.get("target")
    if planner_target != EXPECTED_PLANNER_TARGET:
        raise GateError(f"unexpected group-aware planner target: {planner_target!r}")
    planner_commit = require_git_oid(planner.get("implementation_commit"), name="planner.implementation_commit")
    planner_tree = require_git_oid(planner.get("implementation_tree"), name="planner.implementation_tree")
    planner_sha = require_sha(planner.get("implementation_sha256"), name="planner.implementation_sha256")
    if (
        planner_commit != EXPECTED_P0_PLANNER_COMMIT
        or planner_tree != EXPECTED_P0_PLANNER_TREE
        or planner_sha != EXPECTED_PLANNER_IMPLEMENTATION_SHA256
        or pr38.get("parent_commit") != planner_commit
    ):
        raise GateError("planner implementation identity differs from corrected P0/P1 source")
    expected_planner = {
        "runtime_class": EXPECTED_PLANNER_CLASS,
        "group_key": "prompt_id",
        "sibling_count": SAMPLES_PER_PROMPT,
        "group_preserving": True,
        "deterministic": True,
    }
    if any(planner.get(key) != value for key, value in expected_planner.items()):
        raise GateError("planner contract must be deterministic and preserve four-sibling prompt groups")
    planner_contract_sha = sha256_bytes(canonical_json(dict(planner)))

    expected_training = {
        "world_size": WORLD_SIZE,
        "sp_size": SP_SIZE,
        "dp_size": DP_SIZE,
        "prompt_count": NUM_PROMPTS,
        "samples_per_prompt": SAMPLES_PER_PROMPT,
        "generated_samples": GENERATED_SAMPLES,
        "num_updates_per_batch": NUM_UPDATES,
        "micro_batch_size": 1,
        "sampling_seed": 42,
        "data_seed": 42,
        "reward": "0.5*VideoPickScore(middle)+0.5*CLAP",
    }
    observed_training = {key: training.get(key) for key in expected_training}
    if observed_training != expected_training:
        raise GateError(f"P2 training contract mismatch: {observed_training}")
    p0_expected = {
        "devices": WORLD_SIZE,
        "batch_size": NUM_PROMPTS,
        "samples_per_prompt": SAMPLES_PER_PROMPT,
        "generated_samples": GENERATED_SAMPLES,
        "num_updates_per_batch": NUM_UPDATES,
        "inference_steps": 10,
    }
    if any(p0_training.get(key) != value for key, value in p0_expected.items()):
        raise GateError("P0 design no longer proves the canonical two-update workload")
    if (
        p0_training.get("dataset", {}).get("train_prompt_slice") != [0, NUM_PROMPTS]
        or p0_training.get("dataset", {}).get("shuffle") is not False
    ):
        raise GateError("P0 design dataset slice/shuffle differs from the canonical P2 arm")
    expected_two_update = {
        "planner_target": EXPECTED_PLANNER_TARGET,
        "runtime_planner_class": EXPECTED_PLANNER_CLASS,
        "num_updates": NUM_UPDATES,
        "update_indices": [0, 1],
        "samples_per_update": [GENERATED_SAMPLES // NUM_UPDATES] * NUM_UPDATES,
        "generated_samples": GENERATED_SAMPLES,
        "group_preserving": True,
        "exhaustive": True,
        "disjoint": True,
        "optimizer_steps": NUM_UPDATES,
        "runtime_evidence_event": "planner_arrange",
    }
    if any(two_update.get(key) != value for key, value in expected_two_update.items()):
        raise GateError("two_update contract does not describe the canonical exhaustive two-update partition")
    two_update_contract_sha = sha256_bytes(canonical_json(dict(two_update)))

    topology = contract.get("topology")
    if not isinstance(topology, Mapping):
        raise GateError("integration contract requires topology")
    expected_sp_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    expected_dp_groups = [[0, 2, 4, 6], [1, 3, 5, 7]]
    if topology.get("sp_groups") != expected_sp_groups or topology.get("dp_groups") != expected_dp_groups:
        raise GateError("topology must declare the canonical contiguous SP2 mesh")

    prompt_binding = verify_file_binding(contract.get("prompt_manifest") or {}, name="prompt_manifest", base=base)
    prompt_payload = load_prompt_manifest(Path(prompt_binding["path"]))
    prompt_ids, prompt_digests = _validate_prompt_manifest(prompt_payload)

    if not isinstance(frozen.get("load_dir"), str) or not frozen["load_dir"].strip():
        raise GateError("frozen_checkpoint.load_dir is required")
    frozen_audit_binding = verify_file_binding(
        frozen.get("initial_lora_audit") or {},
        name="frozen_checkpoint.initial_lora_audit",
        base=base,
    )
    frozen_audit = load_json(Path(frozen_audit_binding["path"]))
    fingerprint = frozen_audit.get("model_fingerprint")
    frozen_tree = frozen_audit.get("tree")
    if (
        frozen_audit.get("schema") != FROZEN_INITIAL_LORA_AUDIT_SCHEMA
        or frozen_audit.get("complete") is not True
        or Path(str(frozen_audit.get("checkpoint", ""))).resolve() != normalize_path(frozen["load_dir"], base=base)
        or frozen_audit.get("optimizer_step_count") != 0
        or frozen_audit.get("rollout_step") != 0
        or frozen_audit.get("trainer_optimizer_step") != 0
        or frozen_audit.get("save_mode") != "adapter"
        or not isinstance(frozen_tree, Mapping)
        or not isinstance(fingerprint, Mapping)
    ):
        raise GateError("frozen checkpoint is not the complete P0 zero-step LoRA audit")
    frozen_tree_sha = require_sha(frozen_tree.get("tree_sha256"), name="frozen checkpoint tree.tree_sha256")
    model_summary_sha = require_sha(
        fingerprint.get("summary_sha256"),
        name="frozen checkpoint model_fingerprint.summary_sha256",
    )
    adapter_signature_sha = require_sha(
        fingerprint.get("adapter_key_signature_sha256"),
        name="frozen checkpoint model_fingerprint.adapter_key_signature_sha256",
    )
    frozen_gates = frozen_audit.get("gates")
    if (
        not isinstance(frozen_gates, Mapping)
        or not frozen_gates
        or not all(value is True for value in frozen_gates.values())
        or fingerprint.get("all_finite") is not True
        or fingerprint.get("paired_lora_modules") is not True
        or int(fingerprint.get("lora_a_nonzero_numel", 0)) <= 0
        or int(fingerprint.get("lora_b_nonzero_numel", -1)) != 0
    ):
        raise GateError("frozen checkpoint tensor/optimizer gates are incomplete")

    if runtime_overlay.get("event_schema") != P0_RUNTIME_EVENT_SCHEMA:
        raise GateError("runtime overlay must emit the canonical P0 runtime event schema")
    runtime_module = verify_file_binding(
        runtime_overlay.get("module") or {},
        name="runtime_overlay.module",
        base=base,
    )
    if runtime_module["sha256"] != EXPECTED_P0_RUNTIME_OVERLAY_SHA256:
        raise GateError("runtime overlay is not the reviewed P0 v2 module")
    module_text = Path(runtime_module["path"]).read_text(encoding="utf-8")
    if f'SCHEMA = "{P0_RUNTIME_EVENT_SCHEMA}"' not in module_text:
        raise GateError("runtime overlay module does not declare the bound P0 event schema")

    hot = contract.get("hot_cache_correctness")
    if not isinstance(hot, Mapping):
        raise GateError("integration contract requires hot_cache_correctness")
    expected_hot = {
        "mode": "in_process_lru",
        "persistent_disk_cache": False,
        "require_zero_disk_hits": True,
        "require_no_fallback": True,
        "conditioner_residency": "cpu",
    }
    if any(hot.get(key) != value for key, value in expected_hot.items()):
        raise GateError("hot-cache correctness contract is not compatible with P2 prefetch")

    correctness = contract.get("correctness")
    if not isinstance(correctness, Mapping) or correctness.get("sample_level_required") is not True:
        raise GateError("sample-level correctness must be required")
    required_fingerprints = {
        "sample_ids",
        "rewards",
        "reward_components",
        "latent",
        "video",
        "audio",
        "update_membership",
    }
    observed_fingerprints = set(correctness.get("required_fingerprints") or [])
    if not required_fingerprints.issubset(observed_fingerprints):
        raise GateError(
            f"correctness.required_fingerprints lacks {sorted(required_fingerprints - observed_fingerprints)}"
        )

    contract_sha = sha256_file(path)
    return {
        "path": os.fspath(path.resolve()),
        "sha256": contract_sha,
        "schema": schema,
        "source": dict(source),
        "source_archive": source_archive,
        "source_tree_manifest": source_manifest,
        "p1_source_preflight": p1_preflight,
        "p1_source_preflight_binding": p1_preflight_binding,
        "pr38": dict(pr38),
        "p2": dict(p2),
        "p0_design": p0_design,
        "p0_design_binding": p0_design_binding,
        "planner": dict(planner),
        "planner_contract_sha256": planner_contract_sha,
        "training": dict(training),
        "two_update": dict(two_update),
        "two_update_contract_sha256": two_update_contract_sha,
        "topology": dict(topology),
        "prompt_manifest": {**prompt_binding, "prompt_ids": prompt_ids, "prompt_sha256": prompt_digests},
        "frozen_checkpoint": {
            **dict(frozen),
            "audit_binding": frozen_audit_binding,
            "audit_payload": frozen_audit,
            "tree_sha256": frozen_tree_sha,
            "model_summary_sha256": model_summary_sha,
            "adapter_key_signature_sha256": adapter_signature_sha,
        },
        "runtime_overlay": {**dict(runtime_overlay), "module_binding": runtime_module},
        "hot_cache_correctness": dict(hot),
        "correctness": dict(correctness),
    }


def validate_source(repo: Path, *, p0: Mapping[str, Any], allow_harness_changes: bool) -> dict[str, Any]:
    """Validate the sealed rebased source chain and restrict dirt to this harness."""

    def git(*args: str, text: bool = True) -> str | bytes:
        result = subprocess.run(
            ["git", "-C", os.fspath(repo), *args],
            check=False,
            text=text,
            capture_output=True,
            timeout=120,
        )
        if result.returncode:
            stderr = result.stderr.strip()
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            raise GateError(f"git {' '.join(args)} failed: {stderr}")
        stdout = result.stdout
        return stdout.strip() if text else stdout

    source = p0["source"]
    pr38 = p0["pr38"]
    p2 = p0["p2"]
    integration_commit = str(source["integration_commit"])
    for label, commit, expected_tree in (
        ("rebased PR38", str(pr38["commit"]), str(pr38["tree"])),
        ("rebased P2", str(p2["commit"]), str(p2["tree"])),
        ("integration", integration_commit, str(source["integration_tree"])),
    ):
        observed_tree = str(git("rev-parse", f"{commit}^{{tree}}"))
        if observed_tree != expected_tree:
            raise GateError(f"{label} tree mismatch: {observed_tree} != {expected_tree}")

    p2_parents = str(git("rev-list", "--parents", "-n", "1", str(p2["commit"]))).split()
    if p2_parents != [str(p2["commit"]), str(pr38["commit"])]:
        raise GateError("rebased P2 must be a single-parent commit directly on the rebased PR38 head")
    patch = git(
        "diff",
        "--binary",
        "--full-index",
        str(pr38["commit"]),
        str(p2["commit"]),
        text=False,
    )
    if not isinstance(patch, bytes) or sha256_bytes(patch) != p2["patch_sha256"]:
        raise GateError("rebased P2 patch digest differs from source.p2.patch_sha256")
    for label, commit, descendant in (
        ("P2 vs integration", str(p2["commit"]), integration_commit),
        ("planner vs integration", str(p0["planner"]["implementation_commit"]), integration_commit),
        ("integration vs HEAD", integration_commit, "HEAD"),
    ):
        ancestor = subprocess.run(
            ["git", "-C", os.fspath(repo), "merge-base", "--is-ancestor", commit, descendant],
            check=False,
            timeout=120,
        )
        if ancestor.returncode != 0:
            raise GateError(f"source ancestry gate failed: {label}")

    planner_tree = str(git("rev-parse", f"{p0['planner']['implementation_commit']}^{{tree}}"))
    if planner_tree != p0["planner"]["implementation_tree"]:
        raise GateError("planner implementation tree differs from the P0 contract")

    head = str(git("rev-parse", "HEAD"))
    tree = str(git("rev-parse", "HEAD^{tree}"))
    remote = str(git("config", "--get", "remote.fork.url"))
    if remote != EXPECTED_REMOTE:
        raise GateError(f"fork remote mismatch: {remote!r}")
    origin_push = str(git("config", "--get", "remote.origin.pushurl"))
    if origin_push != "DISABLED_use_fork_remote":
        raise GateError("official origin push is not disabled")

    allowed_prefix = "benchmarks/speed_benchmarks/minimax_h3_prefetch_abba/"
    dirty: list[str] = []
    status = git("status", "--porcelain=v1", "--untracked-files=all")
    assert isinstance(status, str)
    for line in status.splitlines():
        if not line:
            continue
        relative = line[3:]
        if " -> " in relative:
            relative = relative.split(" -> ", 1)[1]
        if not allow_harness_changes or not relative.startswith(allowed_prefix):
            dirty.append(line)
    if dirty:
        raise GateError(f"source worktree has out-of-scope changes: {dirty[:8]}")
    if head != integration_commit:
        changed = git("diff", "--name-only", f"{integration_commit}..HEAD")
        assert isinstance(changed, str)
        committed_after_p2 = [line for line in changed.splitlines() if line and not line.startswith(allowed_prefix)]
        if committed_after_p2:
            raise GateError(
                f"commits after sealed integration source touch non-harness paths: {committed_after_p2[:8]}"
            )

    marker_checks: dict[str, bool] = {}
    for relative, markers in SOURCE_MARKERS.items():
        text = (repo / relative).read_text(encoding="utf-8")
        for marker in markers:
            marker_checks[f"{relative}:{marker}"] = marker in text
    missing = [name for name, passed in marker_checks.items() if not passed]
    if missing:
        raise GateError(f"P2 implementation marker(s) missing: {missing}")
    return {
        "repo": os.fspath(repo.resolve()),
        "head": head,
        "tree": tree,
        "integration_commit": integration_commit,
        "integration_tree": source["integration_tree"],
        "pr38_commit": pr38["commit"],
        "pr38_tree": pr38["tree"],
        "p2_commit": p2["commit"],
        "p2_tree": p2["tree"],
        "p2_patch_sha256": p2["patch_sha256"],
        "planner_commit": p0["planner"]["implementation_commit"],
        "planner_tree": p0["planner"]["implementation_tree"],
        "fork_remote": remote,
        "origin_push": origin_push,
        "allowed_harness_changes": allow_harness_changes,
        "implementation_markers": marker_checks,
    }


def load_p0_events(root: Path) -> list[dict[str, Any]]:
    """Load immutable P0 runtime events after verifying every SHA sidecar."""
    event_root = root / "events"
    if not event_root.is_dir():
        raise GateError(f"missing P0 runtime event directory: {event_root}")
    event_paths = sorted(event_root.glob("*.json"))
    expected_sidecars = {path.with_name(path.name + ".sha256") for path in event_paths}
    observed_sidecars = set(event_root.glob("*.json.sha256"))
    if observed_sidecars != expected_sidecars:
        missing = sorted(path.name for path in expected_sidecars - observed_sidecars)
        orphaned = sorted(path.name for path in observed_sidecars - expected_sidecars)
        raise GateError(f"P0 runtime event sidecar set mismatch: missing={missing} orphaned={orphaned}")
    rows: list[dict[str, Any]] = []
    for path in event_paths:
        sidecar = path.with_name(path.name + ".sha256")
        try:
            parts = sidecar.read_text(encoding="utf-8").strip().split()
        except FileNotFoundError as exc:
            raise GateError(f"missing P0 runtime event sidecar: {sidecar}") from exc
        if len(parts) != 2 or parts[1] != path.name or not is_sha256(parts[0]):
            raise GateError(f"malformed P0 runtime event sidecar: {sidecar}")
        if sha256_file(path) != parts[0]:
            raise GateError(f"P0 runtime event digest mismatch: {path}")
        row = load_json(path)
        if row.get("schema") != P0_RUNTIME_EVENT_SCHEMA:
            raise GateError(f"unexpected P0 runtime event schema in {path}")
        if not isinstance(row.get("context"), Mapping) or not isinstance(row.get("payload"), Mapping):
            raise GateError(f"malformed P0 runtime event envelope: {path}")
        row = dict(row)
        row["_path"] = os.fspath(path)
        rows.append(row)
    if not rows:
        raise GateError(f"no P0 runtime events found below {event_root}")
    return rows


def runtime_event_tree_digest(events: Sequence[Mapping[str, Any]]) -> str:
    """Hash the ordered event-file names and verified contents."""
    rows: list[dict[str, str]] = []
    for row in events:
        raw_path = row.get("_path")
        if not isinstance(raw_path, str) or not raw_path:
            raise GateError("runtime event lacks its verified source path")
        path = Path(raw_path)
        rows.append({"path": path.name, "sha256": sha256_file(path)})
    return sha256_bytes(canonical_json(rows))


def _event_fingerprint(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GateError("runtime sample fingerprint is not an object")
    required = {
        "mode",
        "shape",
        "dtype",
        "numel",
        "sample_count",
        "finite",
        "projection",
        "sample_l2",
        "sample_absmax",
        "quantized_sha256",
    }
    if not required.issubset(value):
        raise GateError(f"runtime sample fingerprint lacks {sorted(required - set(value))}")
    if value.get("mode") != "deterministic_sample_v1" or value.get("finite") is not True:
        raise GateError("runtime sample fingerprint mode/finite gate failed")
    require_sha(value.get("quantized_sha256"), name="runtime sample fingerprint quantized_sha256")
    for key in ("projection", "sample_l2", "sample_absmax"):
        if not finite_number(value.get(key)):
            raise GateError(f"runtime sample fingerprint {key} is non-finite")
    return dict(value)


def aggregate_p0_runtime_events(
    event_root: Path,
    *,
    campaign_id: str,
    run_name: str,
    treatment: str,
    prompt_ids: Sequence[str],
    prompt_sha256: Sequence[str],
    planner_contract_sha256: str,
    two_update_contract_sha256: str,
    frozen_checkpoint: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Aggregate P0 v2 events into immutable P2 correctness receipts."""
    if treatment not in {"off", "on"}:
        raise GateError(f"invalid treatment: {treatment!r}")
    events = load_p0_events(event_root)
    issues: list[str] = []
    for row in events:
        context = row["context"]
        if context.get("campaign_id") != campaign_id or context.get("run_name") != run_name:
            issues.append(f"event context campaign/run mismatch: {row['_path']}")
        if context.get("requested_sp_size") != SP_SIZE:
            issues.append(f"event requested_sp_size is not {SP_SIZE}: {row['_path']}")
        rank = context.get("rank")
        if isinstance(rank, int) and rank >= 0:
            expected_parallel = {
                "world_size": WORLD_SIZE,
                "sp_size": SP_SIZE,
                "dp_size": DP_SIZE,
                "sp_rank": rank % SP_SIZE,
                "dp_rank": rank // SP_SIZE,
                "tp_size": 1,
                "pp_size": 1,
                "ep_size": 1,
            }
            failed = [key for key, value in expected_parallel.items() if context.get(key) != value]
            if failed:
                issues.append(f"event parallel context mismatch {failed}: {row['_path']}")

    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        event = row.get("event")
        if not isinstance(event, str) or not event:
            issues.append(f"event name is invalid: {row['_path']}")
            continue
        by_name[event].append(row)

    overlay_rows = by_name.get("p2_overlay_installed", [])
    if not overlay_rows or any(
        row["payload"].get("schema") != P0_RUNTIME_EVENT_SCHEMA
        or row["payload"].get("persistent_disk_cache") is not False
        or row["payload"].get("conditioner_device") != "cpu"
        or row["payload"].get("prefetch_expected") is not (treatment == "on")
        for row in overlay_rows
    ):
        issues.append("P2 overlay installation evidence is missing or inconsistent")

    trainer_rows = [row["payload"] for row in by_name.get("trainer_contract", []) if row["context"].get("rank") == -1]
    expected_trainer = {
        "num_devices": WORLD_SIZE,
        "batch_size": NUM_PROMPTS,
        "samples_per_prompt": SAMPLES_PER_PROMPT,
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
        "prompt_embedding_prefetch_capacity": PREFETCH_CAPACITY,
        "text_encoder_onload_for_embed": False,
        "remote_handle_dereferenced": False,
    }
    if len(trainer_rows) != 1:
        issues.append("runtime events lack exactly one driver trainer_contract")
    elif any(trainer_rows[0].get(key) != value for key, value in expected_trainer.items()):
        issues.append(f"trainer runtime contract mismatch: {trainer_rows[0]}")

    data_rows = [row["payload"] for row in by_name.get("data_batch", []) if row["context"].get("rank") == -1]
    expected_root_ids = [f"prompt:{prompt_id}:sample:0" for prompt_id in prompt_ids]
    if len(data_rows) != 1 or (
        data_rows
        and (
            data_rows[0].get("batch_size") != NUM_PROMPTS
            or data_rows[0].get("shuffle") is not False
            or data_rows[0].get("ordered_root_sample_ids") != expected_root_ids
            or data_rows[0].get("ordered_prompt_sha256") != list(prompt_sha256)
        )
    ):
        issues.append("data_batch does not bind the canonical ordered prompt set")

    planner_contract_rows = by_name.get("planner_contract", [])
    planner_by_rank: dict[int, dict[str, Any]] = {}
    for row in planner_contract_rows:
        rank = row["context"].get("rank")
        if not isinstance(rank, int) or rank not in range(WORLD_SIZE):
            continue
        if rank in planner_by_rank:
            issues.append(f"duplicate planner_contract event for rank {rank}")
            continue
        planner_by_rank[rank] = dict(row["payload"])
    if set(planner_by_rank) != set(range(WORLD_SIZE)):
        issues.append(f"planner_contract rank coverage mismatch: {sorted(planner_by_rank)}")
    for rank, payload in planner_by_rank.items():
        if (
            payload.get("planner_class") != EXPECTED_PLANNER_CLASS
            or payload.get("num_updates_per_batch") != NUM_UPDATES
            or payload.get("micro_batch_size") != 1
            or payload.get("algorithm_supports_multi_update") is not True
            or payload.get("expected_num_updates_per_batch") != NUM_UPDATES
        ):
            issues.append(f"planner_contract mismatch on rank {rank}")

    arrange_by_rank: dict[int, dict[str, Any]] = {}
    for row in by_name.get("planner_arrange", []):
        rank = row["context"].get("rank")
        if not isinstance(rank, int) or rank not in range(WORLD_SIZE):
            continue
        if rank in arrange_by_rank:
            issues.append(f"duplicate planner_arrange event for rank {rank}")
            continue
        arrange_by_rank[rank] = dict(row["payload"])
    if set(arrange_by_rank) != set(range(WORLD_SIZE)):
        issues.append(f"planner_arrange rank coverage mismatch: {sorted(arrange_by_rank)}")
    observed_update_membership: list[list[str]] = [[] for _ in range(NUM_UPDATES)]
    source_ranks = {0, 2, 4, 6}
    for dp_rank in range(DP_SIZE):
        source_rank = dp_rank * SP_SIZE
        receiver_rank = source_rank + 1
        source_payload = arrange_by_rank.get(source_rank)
        receiver_payload = arrange_by_rank.get(receiver_rank)
        if source_payload is None:
            continue
        if receiver_payload is not None and canonical_json(source_payload) != canonical_json(receiver_payload):
            issues.append(f"planner replicas disagree in DP group {dp_rank}")
        if (
            source_payload.get("planner_class") != EXPECTED_PLANNER_CLASS
            or source_payload.get("num_updates") != NUM_UPDATES
            or source_payload.get("micro_batch_size") != 1
        ):
            issues.append(f"planner_arrange contract mismatch in DP group {dp_rank}")
        before = source_payload.get("before")
        after = source_payload.get("after")
        permutation = source_payload.get("permutation")
        updates = source_payload.get("updates")
        if (
            not isinstance(before, list)
            or not isinstance(after, list)
            or not isinstance(permutation, list)
            or not isinstance(updates, list)
            or len(updates) != NUM_UPDATES
            or sorted(permutation) != list(range(len(before)))
            or len(after) != len(before)
        ):
            issues.append(f"planner_arrange structure invalid in DP group {dp_rank}")
            continue
        flattened_local: list[str] = []
        for update_index, update in enumerate(updates):
            if not isinstance(update, Mapping) or update.get("update_index") != update_index:
                issues.append(f"planner update {update_index} invalid in DP group {dp_rank}")
                continue
            update_rows = update.get("rows")
            if not isinstance(update_rows, list):
                issues.append(f"planner update rows missing in DP group {dp_rank}")
                continue
            sample_ids = [str(item.get("sample_id")) for item in update_rows if isinstance(item, Mapping)]
            if len(sample_ids) != len(update_rows):
                issues.append(f"planner update row malformed in DP group {dp_rank}")
                continue
            flattened_local.extend(sample_ids)
            observed_update_membership[update_index].extend(sample_ids)
        after_ids = [str(item.get("sample_id")) for item in after if isinstance(item, Mapping)]
        if flattened_local != after_ids or len(flattened_local) != len(set(flattened_local)):
            issues.append(f"planner updates are not exhaustive/disjoint in DP group {dp_rank}")

    expected_ids = expected_sample_ids(prompt_ids)
    update_membership = expected_update_membership(prompt_ids)
    if observed_update_membership != update_membership:
        issues.append("runtime planner is not the canonical group-interleaved partition")
    update_by_sample = {
        sample_id: update_index for update_index, sample_ids in enumerate(update_membership) for sample_id in sample_ids
    }

    generation_by_id: dict[str, dict[str, Any]] = {}
    for row in by_name.get("sample_generation", []):
        payload = row["payload"]
        if payload.get("authoritative") is not True or row["context"].get("sp_rank") != 0:
            issues.append(f"non-authoritative sample_generation event: {row['_path']}")
            continue
        sample_id = payload.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            issues.append(f"sample_generation lacks sample_id: {row['_path']}")
            continue
        if sample_id in generation_by_id:
            issues.append(f"duplicate sample_generation row for {sample_id}")
            continue
        generation_by_id[sample_id] = dict(payload)

    manifests = [row for row in by_name.get("generation_manifest", []) if row["payload"].get("authoritative") is True]
    next_flags = Counter(row["payload"].get("next_prompt_supplied") for row in manifests)
    expected_next_flags = Counter({False: GENERATED_SAMPLES}) if treatment == "off" else Counter({True: 28, False: 4})
    if len(manifests) != GENERATED_SAMPLES or next_flags != expected_next_flags:
        issues.append(f"generation entrypoint/next-prompt evidence mismatch: {dict(next_flags)}")
    replica_rows = by_name.get("generation_replica_presence", [])
    if len(replica_rows) != GENERATED_SAMPLES or any(
        row["context"].get("sp_rank") != 1
        or row["payload"].get("authoritative") is not False
        or row["payload"].get("sample_count") != 1
        for row in replica_rows
    ):
        issues.append("SP receiver generation presence evidence is incomplete")

    score_by_id: dict[str, dict[str, Any]] = {}
    for row in by_name.get("sample_score", []):
        payload = row["payload"]
        sample_id = payload.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            issues.append(f"sample_score lacks sample_id: {row['_path']}")
            continue
        if sample_id in score_by_id:
            issues.append(f"duplicate sample_score row for {sample_id}")
            continue
        score_by_id[sample_id] = dict(payload)
    if set(generation_by_id) != set(expected_ids):
        issues.append("sample_generation ID set differs from the canonical 32 samples")
    if set(score_by_id) != set(expected_ids):
        issues.append("sample_score ID set differs from the canonical 32 samples")

    prompt_digest_by_id = dict(zip(prompt_ids, prompt_sha256))
    ledger_rows: list[dict[str, Any]] = []
    for sample_id in expected_ids:
        generation = generation_by_id.get(sample_id)
        score = score_by_id.get(sample_id)
        if generation is None or score is None:
            continue
        try:
            prompt_id, ordinal = parse_sample_id(sample_id)
            prompt_sha = require_sha(
                generation.get("prompt_sha256"),
                name=f"{sample_id}.generation.prompt_sha256",
            )
            if prompt_digest_by_id.get(prompt_id) != prompt_sha:
                raise GateError(f"{sample_id} prompt digest differs from manifest")
            if score.get("prompt_sha256") != prompt_sha:
                raise GateError(f"{sample_id} generation/score prompt digest mismatch")
            if (
                generation.get("root_id") != score.get("root_id")
                or generation.get("group_id") != score.get("group_id")
                or generation.get("sibling_ordinal") != ordinal
                or score.get("sibling_ordinal") != ordinal
                or score.get("rollout_id") != 0
            ):
                raise GateError(f"{sample_id} generation/score lineage mismatch")
            reward = score.get("reward")
            video_score = score.get("videopickscore")
            clap_score = score.get("clap")
            advantage = score.get("advantage")
            for key, value in (
                ("reward", reward),
                ("videopickscore", video_score),
                ("clap", clap_score),
                ("advantage", advantage),
            ):
                if not finite_number(value):
                    raise GateError(f"{sample_id} has non-finite {key}")
            expected_reward = 0.5 * float(video_score) + 0.5 * float(clap_score)
            if not math.isclose(float(reward), expected_reward, rel_tol=2e-5, abs_tol=2e-6):
                raise GateError(f"{sample_id} weighted reward parity failed")
            segment = generation.get("segment")
            if not isinstance(segment, Mapping):
                raise GateError(f"{sample_id} generation segment is missing")
            if segment.get("sde_indices") != [0, 3, 6]:
                raise GateError(f"{sample_id} SDE index contract mismatch")
            fingerprint_fields = {
                "latent_fingerprint": _event_fingerprint(segment.get("video_trajectory")),
                "audio_latent_fingerprint": _event_fingerprint(segment.get("audio_trajectory")),
                "initial_video_fingerprint": _event_fingerprint(segment.get("initial_video")),
                "initial_audio_fingerprint": _event_fingerprint(segment.get("initial_audio")),
                "final_video_latent_fingerprint": _event_fingerprint(segment.get("final_video")),
                "final_audio_latent_fingerprint": _event_fingerprint(segment.get("final_audio")),
                "video_fingerprint": _event_fingerprint(generation.get("decoded_video")),
                "audio_fingerprint": _event_fingerprint(generation.get("decoded_audio")),
            }
            generation_sha = sha256_bytes(
                canonical_json(
                    {
                        "segment": segment,
                        "decoded_video": generation.get("decoded_video"),
                        "decoded_audio": generation.get("decoded_audio"),
                    }
                )
            )
            output_sha = sha256_bytes(canonical_json(fingerprint_fields))
            ledger_rows.append(
                {
                    "schema": SAMPLE_LEDGER_SCHEMA,
                    "campaign_id": campaign_id,
                    "run_name": run_name,
                    "treatment": treatment,
                    "sample_id": sample_id,
                    "prompt_id": prompt_id,
                    "prompt_sha256": prompt_sha,
                    "root_id": generation.get("root_id"),
                    "group_id": generation.get("group_id"),
                    "sibling_ordinal": ordinal,
                    "sampling_seed": 42,
                    "update_index": update_by_sample[sample_id],
                    "reward": float(reward),
                    "reward_components": {
                        "videopickscore": float(video_score),
                        "clap": float(clap_score),
                    },
                    "advantage": float(advantage),
                    **fingerprint_fields,
                    "generation_sha256": generation_sha,
                    "output_sha256": output_sha,
                }
            )
        except GateError as exc:
            issues.append(str(exc))

    optimizer_by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    optimizer_events_by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in by_name.get("optimizer_step", []):
        rank = row["context"].get("rank")
        if isinstance(rank, int) and rank in range(WORLD_SIZE):
            optimizer_by_rank[rank].append(dict(row["payload"]))
            optimizer_events_by_rank[rank].append(row)
    canonical_optimizer: list[dict[str, Any]] = []
    if set(optimizer_by_rank) != set(range(WORLD_SIZE)):
        issues.append(f"optimizer rank coverage mismatch: {sorted(optimizer_by_rank)}")
    for rank in range(WORLD_SIZE):
        rows = sorted(optimizer_by_rank.get(rank, []), key=lambda value: int(value.get("before", -1)))
        transitions = [(row.get("before"), row.get("after")) for row in rows]
        if len(rows) != NUM_UPDATES or transitions != [(0, 1), (1, 2)]:
            issues.append(f"rank {rank} optimizer transitions mismatch: {transitions}")
        if any(
            row.get("success") is not True
            or row.get("committed") is not True
            or row.get("finite_nonzero_grad") is not True
            or row.get("update_index") not in (0, 1)
            or not finite_number(row.get("grad_norm"))
            or float(row["grad_norm"]) <= 0.0
            for row in rows
        ):
            issues.append(f"rank {rank} optimizer evidence is invalid")
        if rank == 0:
            canonical_optimizer = rows

    train_update_by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in by_name.get("train_update", []):
        rank = row["context"].get("rank")
        if isinstance(rank, int) and rank in range(WORLD_SIZE):
            train_update_by_rank[rank].append(dict(row["payload"]))
    for rank in range(WORLD_SIZE):
        updates = train_update_by_rank.get(rank, [])
        if len(updates) != NUM_UPDATES or any(
            item.get("success") is not True
            or item.get("optimizer_updates") != 1
            or item.get("after_optimizer_step") != item.get("before_optimizer_step", -2) + 1
            for item in updates
        ):
            issues.append(f"rank {rank} train_update evidence is not exactly two committed updates")

    run_updates_by_rank: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in by_name.get("run_updates", []):
        rank = row["context"].get("rank")
        if isinstance(rank, int) and rank in range(WORLD_SIZE):
            run_updates_by_rank[rank].append(dict(row["payload"]))
    for rank in range(WORLD_SIZE):
        rows = run_updates_by_rank.get(rank, [])
        if (
            len(rows) != 1
            or rows[0].get("optimizer_updates") != NUM_UPDATES
            or rows[0].get("per_update_count") != NUM_UPDATES
        ):
            issues.append(f"rank {rank} run_updates aggregate mismatch")

    train_rows = [row["payload"] for row in by_name.get("train_step", []) if row["context"].get("rank") == -1]
    if len(train_rows) != 1 or (
        train_rows
        and (
            train_rows[0].get("success") is not True
            or train_rows[0].get("optimizer_updates") != NUM_UPDATES
            or train_rows[0].get("per_update_count") != NUM_UPDATES
            or train_rows[0].get("has_backward") is not True
        )
    ):
        issues.append("driver train_step does not prove exactly two updates")

    expected_load_dir = os.fspath(Path(str(frozen_checkpoint["load_dir"])).resolve())
    checkpoint_by_rank: dict[int, dict[str, Any]] = {}
    checkpoint_events_by_rank: dict[int, dict[str, Any]] = {}
    for event in by_name.get("checkpoint_load", []):
        rank = event["context"].get("rank")
        if not isinstance(rank, int) or rank not in range(WORLD_SIZE):
            continue
        payload = event["payload"]
        if rank in checkpoint_by_rank:
            issues.append(f"duplicate checkpoint_load event for rank {rank}")
            continue
        checkpoint_by_rank[rank] = dict(payload)
        checkpoint_events_by_rank[rank] = event
    if set(checkpoint_by_rank) != set(range(WORLD_SIZE)):
        issues.append(f"checkpoint load rank coverage mismatch: {sorted(checkpoint_by_rank)}")
    for rank, payload in checkpoint_by_rank.items():
        if (
            payload.get("success") is not True
            or os.fspath(Path(str(payload.get("path", ""))).resolve()) != expected_load_dir
            or payload.get("loaded_rollout_step") != 0
            or payload.get("backend_optimizer_step") != 0
        ):
            issues.append(f"rank {rank} did not load the frozen zero-step checkpoint")
        checkpoint_time = checkpoint_events_by_rank[rank].get("monotonic_ns")
        optimizer_times = [row.get("monotonic_ns") for row in optimizer_events_by_rank.get(rank, [])]
        if not isinstance(checkpoint_time, int) or any(
            not isinstance(value, int) or value <= checkpoint_time for value in optimizer_times
        ):
            issues.append(f"rank {rank} checkpoint load was not proven before optimizer steps")
    rank0_audit = checkpoint_by_rank.get(0, {}).get("audit")
    if not isinstance(rank0_audit, Mapping) or any(
        rank0_audit.get(key) != value
        for key, value in {
            "rollout_step": 0,
            "optimizer_step_count": 0,
            "trainer_optimizer_step": 0,
            "save_mode": "adapter",
        }.items()
    ):
        issues.append("rank-0 checkpoint runtime audit is missing or nonzero")

    topology_by_rank: dict[int, dict[str, Any]] = {}
    for row in by_name.get("sp_topology", []):
        rank = row["context"].get("rank")
        if isinstance(rank, int) and rank in range(WORLD_SIZE):
            if rank in topology_by_rank:
                issues.append(f"duplicate sp_topology event for rank {rank}")
            topology_by_rank[rank] = dict(row["payload"])
    for rank in range(WORLD_SIZE):
        payload = topology_by_rank.get(rank)
        expected_group = list(range(rank // SP_SIZE * SP_SIZE, rank // SP_SIZE * SP_SIZE + SP_SIZE))
        if not isinstance(payload, Mapping) or (
            payload.get("world_size") != WORLD_SIZE
            or payload.get("observed_sp_size") != SP_SIZE
            or payload.get("observed_dp_size") != DP_SIZE
            or payload.get("sp_group_ranks") != expected_group
            or payload.get("sp_installed") is not True
            or payload.get("sp_processor_count") != 50
            or payload.get("attention_block_count") != 50
        ):
            issues.append(f"structured SP2 topology mismatch on rank {rank}")

    if by_name.get("cache_lookup"):
        issues.append("persistent cache_lookup activity is forbidden")

    encode_rows = by_name.get("conditioner_encode_call", [])
    successful_digests = [
        row["payload"].get("prompt_sha256") for row in encode_rows if row["payload"].get("success") is True
    ]
    if len(encode_rows) != NUM_PROMPTS or Counter(successful_digests) != Counter(prompt_sha256):
        issues.append(f"conditioner calls do not cover eight prompts exactly once: {len(encode_rows)}")
    encode_by_rank = Counter(row["context"].get("rank") for row in encode_rows)
    if encode_by_rank != Counter({rank: 2 for rank in source_ranks}) or any(
        row["payload"].get("success") is not True or row["payload"].get("contract_violation") is not False
        for row in encode_rows
    ):
        issues.append(f"conditioner source-rank distribution mismatch: {dict(encode_by_rank)}")

    background_rows = [row for row in encode_rows if row["payload"].get("background_prefetch") is True]
    expected_background = DP_SIZE if treatment == "on" else 0
    if len(background_rows) != expected_background:
        issues.append(f"expected {expected_background} background conditioner calls, observed {len(background_rows)}")

    denoise_intervals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in by_name.get("phase_timer", []):
        payload = row["payload"]
        rank = row["context"].get("rank")
        duration = payload.get("duration_s")
        finished = row.get("monotonic_ns")
        if (
            payload.get("timer") == "denoise_s"
            and payload.get("success") is True
            and isinstance(rank, int)
            and finite_number(duration)
            and float(duration) > 0.0
            and isinstance(finished, int)
        ):
            denoise_intervals[rank].append((finished - int(float(duration) * 1e9), finished))
    overlap_rows: list[dict[str, Any]] = []
    for row in background_rows:
        payload = row["payload"]
        rank = row["context"].get("rank")
        started = payload.get("started_monotonic_ns")
        finished = payload.get("finished_monotonic_ns")
        if (
            not isinstance(rank, int)
            or not isinstance(started, int)
            or not isinstance(finished, int)
            or finished <= started
        ):
            issues.append("background conditioner call has an invalid monotonic interval")
            continue
        overlap_ns = sum(
            max(0, min(finished, denoise_end) - max(started, denoise_start))
            for denoise_start, denoise_end in denoise_intervals.get(rank, [])
        )
        overlap_rows.append(
            {
                "rank": rank,
                "prompt_sha256": payload.get("prompt_sha256"),
                "duration_s": (finished - started) / 1e9,
                "overlap_s": overlap_ns / 1e9,
                "overlapped": overlap_ns > 0,
            }
        )
        if overlap_ns <= 0:
            issues.append(f"background conditioner call on rank {rank} did not overlap denoising")

    submit_rows = by_name.get("prefetch_submit", [])
    expected_submits = 56 if treatment == "on" else 0
    if len(submit_rows) != expected_submits or any(row["payload"].get("success") is not True for row in submit_rows):
        issues.append(f"prefetch submit protocol mismatch: observed {len(submit_rows)}")
    consume_rows = by_name.get("prefetch_consume", [])
    source_consumes = [row for row in consume_rows if row["context"].get("rank") in source_ranks]
    consumed_hits = sum(int(row["payload"].get("hits", 0)) for row in source_consumes)
    expected_consumed_hits = DP_SIZE if treatment == "on" else 0
    if (
        len(source_consumes) != 32
        or consumed_hits != expected_consumed_hits
        or any(row["payload"].get("success") is not True for row in source_consumes)
    ):
        issues.append(f"prefetch consume protocol mismatch: rows={len(source_consumes)} hits={consumed_hits}")

    shutdown_by_rank: dict[int, dict[str, Any]] = {}
    for row in by_name.get("prefetch_shutdown", []):
        rank = row["context"].get("rank")
        if isinstance(rank, int) and rank in range(WORLD_SIZE):
            shutdown_by_rank[rank] = dict(row["payload"])
    if set(shutdown_by_rank) != set(range(WORLD_SIZE)):
        issues.append(f"prefetch shutdown rank coverage mismatch: {sorted(shutdown_by_rank)}")
    for rank, payload in shutdown_by_rank.items():
        expected_shutdown = {
            "configured": treatment == "on",
            "closed": treatment == "on",
            "retained_after": 0,
            "success": True,
        }
        if any(payload.get(key) != value for key, value in expected_shutdown.items()):
            issues.append(f"prefetch shutdown mismatch on rank {rank}: {payload}")
    cleanup_rows = [row["payload"] for row in by_name.get("p2_driver_cleanup", []) if row["context"].get("rank") == -1]
    if len(cleanup_rows) != 1 or cleanup_rows[0].get("pool_shutdown") is not True:
        issues.append("driver cleanup does not prove worker/prefetch shutdown")

    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / f"sample_ledger_{run_name}.jsonl"
    updates_path = output_dir / f"update_membership_{run_name}.json"
    checkpoint_path = output_dir / f"frozen_checkpoint_loaded_{run_name}.json"
    summary_path = output_dir / f"runtime_summary_{run_name}.json"
    if any(path.exists() for path in (ledger_path, updates_path, checkpoint_path, summary_path)):
        raise GateError(f"refusing to overwrite P2 runtime receipts for {run_name}")

    event_tree_sha = runtime_event_tree_digest(events)
    if issues:
        payload = {
            "schema": P2_RUNTIME_SUMMARY_SCHEMA,
            "campaign_id": campaign_id,
            "run_name": run_name,
            "treatment": treatment,
            "completed": False,
            "event_schema": P0_RUNTIME_EVENT_SCHEMA,
            "event_count": len(events),
            "event_tree_sha256": event_tree_sha,
            "issues": issues,
        }
        write_json_atomic(summary_path, payload)
        raise GateError("P0 v2 runtime event aggregation failed: " + "; ".join(issues))

    ledger_path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in ledger_rows),
        encoding="utf-8",
    )
    updates_payload = {
        "schema": UPDATE_MEMBERSHIP_SCHEMA,
        "planner_contract_sha256": planner_contract_sha256,
        "two_update_contract_sha256": two_update_contract_sha256,
        "frozen_checkpoint_audit_sha256": frozen_checkpoint["audit_binding"]["sha256"],
        "planner_target": EXPECTED_PLANNER_TARGET,
        "runtime_planner_class": EXPECTED_PLANNER_CLASS,
        "runtime_event": "planner_arrange",
        "num_updates": NUM_UPDATES,
        "optimizer_updates": NUM_UPDATES,
        "generated_samples": GENERATED_SAMPLES,
        "group_key": "prompt_id",
        "group_preserving": True,
        "exhaustive": True,
        "disjoint": True,
        "updates": [
            {"update_index": update_index, "sample_ids": sample_ids}
            for update_index, sample_ids in enumerate(update_membership)
        ],
        "optimizer_transitions": canonical_optimizer,
        "runtime_source_ranks": sorted(source_ranks),
    }
    write_json_atomic(updates_path, updates_payload)
    frozen_payload = {
        "schema": FROZEN_CHECKPOINT_LOADED_SCHEMA,
        "audit_sha256": frozen_checkpoint["audit_binding"]["sha256"],
        "tree_sha256": frozen_checkpoint["tree_sha256"],
        "model_summary_sha256": frozen_checkpoint["model_summary_sha256"],
        "adapter_key_signature_sha256": frozen_checkpoint["adapter_key_signature_sha256"],
        "loaded_rollout_step": 0,
        "optimizer_step": 0,
        "loaded_before_optimizer_step": True,
        "load_event_count": len(checkpoint_by_rank),
        "loaded_ranks": sorted(checkpoint_by_rank),
        "runtime_rank0_audit": dict(rank0_audit),
    }
    write_json_atomic(checkpoint_path, frozen_payload)
    payload = {
        "schema": P2_RUNTIME_SUMMARY_SCHEMA,
        "campaign_id": campaign_id,
        "run_name": run_name,
        "treatment": treatment,
        "completed": True,
        "event_schema": P0_RUNTIME_EVENT_SCHEMA,
        "event_count": len(events),
        "event_tree_sha256": event_tree_sha,
        "planner_contract_sha256": planner_contract_sha256,
        "two_update_contract_sha256": two_update_contract_sha256,
        "sample_ledger_sha256": sha256_file(ledger_path),
        "update_membership_sha256": sha256_file(updates_path),
        "frozen_checkpoint_loaded_sha256": sha256_file(checkpoint_path),
        "optimizer_transitions": canonical_optimizer,
        "topology_ranks": sorted(topology_by_rank),
        "cleanup": cleanup_rows[0],
        "conditioner": {
            "encode_calls": len(encode_rows),
            "source_ranks": sorted(source_ranks),
            "background_encode_calls": len(background_rows),
            "prefetch_submit_calls": len(submit_rows),
            "prefetch_source_accepted_calls": sum(
                row["context"].get("rank") in source_ranks and row["payload"].get("accepted") is True
                for row in submit_rows
            ),
            "prefetch_consume_calls": len(source_consumes),
            "prefetch_consumed_hits": consumed_hits,
            "shutdown_ranks": sorted(shutdown_by_rank),
            "overlap": overlap_rows,
            "all_background_calls_overlapped": all(row["overlapped"] for row in overlap_rows),
        },
    }
    write_json_atomic(summary_path, payload)
    return payload


def functional_overrides(prefetch: bool, *, prompt_manifest_path: str, load_dir: str) -> dict[str, Any]:
    values = dict(FUNCTIONAL_OVERRIDES)
    values["bundle.config.prompt_embedding_prefetch"] = bool(prefetch)
    values["data_source.args.run.data_path"] = prompt_manifest_path
    values["load_dir"] = load_dir
    return values


def arm_specs(campaign_id: str) -> list[dict[str, Any]]:
    arms = []
    for period, treatment in ARM_ORDER:
        arm_id = f"p{period}-{treatment}"
        arms.append(
            {
                "period": period,
                "treatment": treatment,
                "prefetch": treatment == "on",
                "arm_id": arm_id,
                "run_name": f"{campaign_id}-{arm_id}",
            }
        )
    return arms


def expected_sample_ids(prompt_ids: Sequence[str]) -> list[str]:
    return [
        f"r0:prompt:{prompt_id}:sample:0/{sibling}" for prompt_id in prompt_ids for sibling in range(SAMPLES_PER_PROMPT)
    ]


def expected_update_membership(prompt_ids: Sequence[str]) -> list[list[str]]:
    return [
        [f"r0:prompt:{prompt_id}:sample:0/{sibling}" for prompt_id in prompt_ids for sibling in siblings]
        for siblings in ((0, 1), (2, 3))
    ]


def parse_sample_id(sample_id: str) -> tuple[str, int]:
    """Return the manifest prompt ID and sibling ordinal from a rollout sample ID."""
    prefix = "r0:prompt:"
    marker = ":sample:0/"
    if not sample_id.startswith(prefix) or marker not in sample_id:
        raise ValueError(f"unexpected sample ID grammar: {sample_id!r}")
    prompt_id, ordinal = sample_id[len(prefix) :].rsplit(marker, 1)
    if not prompt_id or not ordinal.isdigit():
        raise ValueError(f"unexpected sample ID grammar: {sample_id!r}")
    return prompt_id, int(ordinal)


def scan_failures(log: str, *, treatment: str) -> list[str]:
    failures = [name for name, pattern in FAILURE_PATTERNS.items() if pattern.search(log)]
    if treatment == "off":
        failures = [name for name in failures if name != "prefetch_disabled"]
    return failures


def parse_embedding_telemetry(log: str) -> dict[str, Any]:
    """Parse per-call debug telemetry and aggregate source/non-source behavior."""
    rows: list[dict[str, Any]] = []
    for line in log.splitlines():
        match = EMBED_RE.search(line)
        if match is None:
            continue
        rank_match = re.search(r"(?:^|[\s\[,])(?:rank|global_rank)[=: ](?P<rank>\d+)(?:[\s\],]|$)", line, re.I)
        row: dict[str, Any] = {
            key: int(value)
            for key, value in match.groupdict().items()
            if key not in {"onload", "elapsed", "shared_source"}
        }
        row["onload"] = match.group("onload") == "True"
        row["shared_source"] = match.group("shared_source") == "True"
        row["elapsed_s"] = float(match.group("elapsed"))
        row["rank"] = int(rank_match.group("rank")) if rank_match is not None else None
        rows.append(row)
    source = [row for row in rows if row["shared_source"]]
    receivers = [row for row in rows if not row["shared_source"]]

    source_histogram = Counter((row["memory_hits"], row["misses"], row["disk_hits"]) for row in source)
    receiver_histogram = Counter((row["shared_hits"], row["misses"], row["disk_hits"]) for row in receivers)
    return {
        "rows": rows,
        "row_count": len(rows),
        "source_row_count": len(source),
        "receiver_row_count": len(receivers),
        "source_memory_hits": sum(row["memory_hits"] for row in source),
        "source_misses": sum(row["misses"] for row in source),
        "source_disk_hits": sum(row["disk_hits"] for row in source),
        "receiver_shared_hits": sum(row["shared_hits"] for row in receivers),
        "receiver_misses": sum(row["misses"] for row in receivers),
        "receiver_disk_hits": sum(row["disk_hits"] for row in receivers),
        "onload_true_rows": sum(row["onload"] for row in rows),
        "source_histogram": {f"memory={m},miss={x},disk={d}": count for (m, x, d), count in source_histogram.items()},
        "receiver_histogram": {
            f"shared={s},miss={x},disk={d}": count for (s, x, d), count in receiver_histogram.items()
        },
    }


def embedding_gate(
    telemetry: Mapping[str, Any], *, treatment: str, source_ranks: Sequence[int] | None = None
) -> tuple[bool, list[str]]:
    """Enforce exact SP2 + sharing + in-process-LRU/prefetch call counts."""
    issues: list[str] = []
    rows = telemetry.get("rows")
    if not isinstance(rows, list) or len(rows) != 64:
        issues.append(
            f"expected 64 embedding telemetry rows, observed {len(rows) if isinstance(rows, list) else 'invalid'}"
        )
        return False, issues
    source = [row for row in rows if row.get("shared_source") is True]
    receivers = [row for row in rows if row.get("shared_source") is False]
    if len(source) != 32 or len(receivers) != 32:
        issues.append(f"expected source/receiver rows 32/32, observed {len(source)}/{len(receivers)}")
    parsed_ranks = [row.get("rank") for row in rows if row.get("rank") is not None]
    if parsed_ranks and len(parsed_ranks) != len(rows):
        issues.append("embedding telemetry has only partially parseable rank labels")
    elif parsed_ranks:
        ranks = Counter(int(row["rank"]) for row in rows)
        expected_ranks = Counter({rank: 8 for rank in range(WORLD_SIZE)})
        if ranks != expected_ranks:
            issues.append(f"expected eight embedding rows per rank, observed={dict(ranks)}")
        observed_source_ranks = {int(row["rank"]) for row in source}
        receiver_ranks = {int(row["rank"]) for row in receivers}
        expected_sources = set(source_ranks or (0, 2, 4, 6))
        expected_receivers = set(range(WORLD_SIZE)) - expected_sources
        if observed_source_ranks != expected_sources or receiver_ranks != expected_receivers:
            issues.append(
                f"SP2 sharing source/receiver ranks mismatch: sources={sorted(observed_source_ranks)} "
                f"receivers={sorted(receiver_ranks)}"
            )
    if any(row.get("prompts") != 1 for row in rows):
        issues.append("every embedding call must carry one prompt under forward_batch_size=1")
    if any(row.get("onload") is not False for row in rows):
        issues.append("text_encoder_onload_for_embed must remain false")
    if any(row.get("disk_hits") != 0 for row in rows):
        issues.append("persistent/disk cache activity is forbidden")
    if any(row.get("misses") != 0 or row.get("memory_hits") != 0 or row.get("shared_hits") != 1 for row in receivers):
        issues.append("non-source SP ranks must report exactly one shared hit and no local conditioner/cache work")

    by_source_pattern = Counter((row.get("memory_hits"), row.get("misses"), row.get("disk_hits")) for row in source)
    expected = Counter({(0, 1, 0): 8, (1, 0, 0): 24}) if treatment == "off" else Counter({(0, 1, 0): 4, (1, 0, 0): 28})
    if by_source_pattern != expected:
        issues.append(
            f"unexpected source hot-cache pattern: observed={dict(by_source_pattern)} expected={dict(expected)}"
        )
    return not issues, issues


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise GateError(f"missing JSONL artifact: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GateError(f"invalid JSONL row {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise GateError(f"JSONL row {path}:{line_number} is not an object")
        rows.append(row)
    return rows


def canonical_sample_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return correctness-bearing fields while excluding treatment-local metadata."""
    excluded = {
        "campaign_id",
        "arm_id",
        "run_name",
        "period",
        "treatment",
        "prefetch",
        "pid",
        "rank",
        "local_rank",
        "hostname",
        "timestamps",
        "timing",
        "duration_s",
    }
    return {key: value for key, value in row.items() if key not in excluded}


def sample_ledger_gate(
    rows: Sequence[Mapping[str, Any]], *, expected_ids: Sequence[str]
) -> tuple[bool, list[str], dict[str, str]]:
    issues: list[str] = []
    ids = [row.get("sample_id") for row in rows]
    if len(rows) != GENERATED_SAMPLES:
        issues.append(f"expected {GENERATED_SAMPLES} sample rows, observed {len(rows)}")
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in ids):
        issues.append("sample ledger has missing sample_id")
    elif len(set(ids)) != len(ids):
        issues.append("sample ledger has duplicate sample_id")
    elif set(ids) != set(expected_ids):
        issues.append("sample ledger sample_id set differs from prompt contract")

    fingerprint_fields = (
        "latent_fingerprint",
        "audio_latent_fingerprint",
        "initial_video_fingerprint",
        "initial_audio_fingerprint",
        "final_video_latent_fingerprint",
        "final_audio_latent_fingerprint",
        "video_fingerprint",
        "audio_fingerprint",
    )
    required = {
        "schema",
        "prompt_id",
        "prompt_sha256",
        "root_id",
        "group_id",
        "sibling_ordinal",
        "sampling_seed",
        "update_index",
        "reward",
        "reward_components",
        "advantage",
        "generation_sha256",
        "output_sha256",
        *fingerprint_fields,
    }
    fingerprints: dict[str, str] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        missing = sorted(field for field in required if field not in row)
        if missing:
            issues.append(f"sample {sample_id!r} missing fields {missing}")
            continue
        if row.get("schema") != SAMPLE_LEDGER_SCHEMA:
            issues.append(f"sample {sample_id!r} has an unexpected ledger schema")
        if not is_sha256(row.get("prompt_sha256")):
            issues.append(f"sample {sample_id!r} has invalid prompt_sha256")
        if not isinstance(row.get("root_id"), str) or not row.get("root_id"):
            issues.append(f"sample {sample_id!r} has invalid root_id")
        if not isinstance(row.get("group_id"), str) or not row.get("group_id"):
            issues.append(f"sample {sample_id!r} has invalid group_id")
        if row.get("sampling_seed") != 42:
            issues.append(f"sample {sample_id!r} sampling_seed is not 42")
        if row.get("sibling_ordinal") not in (0, 1, 2, 3):
            issues.append(f"sample {sample_id!r} has invalid sibling_ordinal")
        if row.get("update_index") not in (0, 1):
            issues.append(f"sample {sample_id!r} has invalid update_index")
        if not finite_number(row.get("reward")):
            issues.append(f"sample {sample_id!r} has non-finite reward")
        components = row.get("reward_components")
        if not isinstance(components, Mapping) or set(components) != {"videopickscore", "clap"}:
            issues.append(f"sample {sample_id!r} has incomplete reward components")
        elif not all(finite_number(value) for value in components.values()):
            issues.append(f"sample {sample_id!r} has non-finite reward component")
        if not finite_number(row.get("advantage")):
            issues.append(f"sample {sample_id!r} has non-finite advantage")
        normalized_fingerprints: dict[str, Any] = {}
        for field in fingerprint_fields:
            try:
                normalized_fingerprints[field] = _event_fingerprint(row.get(field))
            except GateError as exc:
                issues.append(f"sample {sample_id!r} {exc}")
        if not is_sha256(row.get("generation_sha256")):
            issues.append(f"sample {sample_id!r} has invalid generation_sha256")
        if not is_sha256(row.get("output_sha256")):
            issues.append(f"sample {sample_id!r} has invalid output_sha256")
        elif len(normalized_fingerprints) == len(fingerprint_fields):
            observed_output = sha256_bytes(canonical_json(normalized_fingerprints))
            if row.get("output_sha256") != observed_output:
                issues.append(f"sample {sample_id!r} output_sha256 does not bind all fingerprints")
        if isinstance(sample_id, str):
            try:
                prompt_id, ordinal = parse_sample_id(sample_id)
                if row.get("prompt_id") != prompt_id:
                    issues.append(f"sample {sample_id!r} prompt_id does not match sample ID")
                if row.get("sibling_ordinal") != ordinal:
                    issues.append(f"sample {sample_id!r} sibling_ordinal does not match sample ID")
                expected_update = 0 if ordinal < 2 else 1
                if row.get("update_index") != expected_update:
                    issues.append(f"sample {sample_id!r} update_index is not group-interleaved")
            except ValueError as exc:
                issues.append(str(exc))
            fingerprints[sample_id] = sha256_bytes(canonical_json(canonical_sample_view(row)))
    return not issues, issues, fingerprints


def update_membership_gate(
    payload: Mapping[str, Any],
    *,
    prompt_ids: Sequence[str],
    planner_contract_sha256: str,
    two_update_contract_sha256: str,
    frozen_checkpoint_audit_sha256: str,
) -> tuple[bool, list[str]]:
    """Validate the P2 receipt derived from runtime planner/optimizer events."""
    issues: list[str] = []
    expected_header = {
        "schema": UPDATE_MEMBERSHIP_SCHEMA,
        "planner_contract_sha256": planner_contract_sha256,
        "two_update_contract_sha256": two_update_contract_sha256,
        "frozen_checkpoint_audit_sha256": frozen_checkpoint_audit_sha256,
        "runtime_planner_class": EXPECTED_PLANNER_CLASS,
        "num_updates": NUM_UPDATES,
        "optimizer_updates": NUM_UPDATES,
        "generated_samples": GENERATED_SAMPLES,
        "group_key": "prompt_id",
        "group_preserving": True,
        "exhaustive": True,
        "disjoint": True,
        "runtime_source_ranks": [0, 2, 4, 6],
    }
    for key, value in expected_header.items():
        if payload.get(key) != value:
            issues.append(f"update membership {key} mismatch")
    target = payload.get("planner_target")
    if target != EXPECTED_PLANNER_TARGET:
        issues.append(f"planner target mismatch: {target!r}")
    updates = payload.get("updates")
    expected = expected_update_membership(prompt_ids)
    observed: list[list[str]] = []
    if not isinstance(updates, list) or len(updates) != NUM_UPDATES:
        issues.append("update membership must contain exactly two updates")
    else:
        for index, update in enumerate(updates):
            if not isinstance(update, Mapping) or update.get("update_index") != index:
                issues.append(f"invalid update membership row {index}")
                continue
            sample_ids = update.get("sample_ids")
            if not isinstance(sample_ids, list):
                issues.append(f"update {index} sample_ids missing")
                continue
            observed.append([str(value) for value in sample_ids])
    if observed != expected:
        issues.append("global update membership is not the canonical group-interleaved partition")
    flattened = [sample_id for update in observed for sample_id in update]
    if len(flattened) != GENERATED_SAMPLES or len(set(flattened)) != GENERATED_SAMPLES:
        issues.append("update membership omits or duplicates samples")

    transitions = payload.get("optimizer_transitions")
    if not isinstance(transitions, list) or len(transitions) != NUM_UPDATES:
        issues.append("update membership must contain exactly two optimizer transitions")
    else:
        observed_transitions = []
        for row in transitions:
            if not isinstance(row, Mapping):
                issues.append("optimizer transition is not an object")
                continue
            observed_transitions.append((row.get("before"), row.get("after")))
            grad_norm = row.get("grad_norm")
            if not finite_number(grad_norm) or float(grad_norm) <= 0.0:
                issues.append("optimizer transition grad_norm must be finite and nonzero")
        if observed_transitions != [(0, 1), (1, 2)]:
            issues.append(f"optimizer transitions are not exactly 0->1,1->2: {observed_transitions}")
    return not issues, issues


def metrics_gate(payload: Mapping[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    required_positive = ("perf/step_time_s", "perf/generate_time_s", "perf/reward_time_s", "perf/train_time_s")
    for key in required_positive:
        value = payload.get(key)
        if not finite_number(value) or float(value) <= 0:
            issues.append(f"missing/non-positive metric {key}")
    gradients = payload.get("train/grad_norm")
    if not isinstance(gradients, list) or len(gradients) != NUM_UPDATES:
        issues.append("train/grad_norm must contain exactly two values")
    elif not all(finite_number(value) and float(value) > 0 for value in gradients):
        issues.append("train/grad_norm must be finite and nonzero")
    if payload.get("train/optimizer_updates") != NUM_UPDATES:
        issues.append("optimizer update count is not two")
    if not finite_number(payload.get("rollout/reward_mean")):
        issues.append("rollout/reward_mean must be finite")
    return not issues, issues


def process_identity_gate(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    identities = []
    for row in rows:
        raw_pid = row.get("pid")
        pid = int(raw_pid) if isinstance(raw_pid, (int, str)) and str(raw_pid).isdigit() else 0
        starttime = str(row.get("starttime") or "")
        token = row.get("token")
        if pid <= 0 or not starttime:
            issues.append("invalid process identity")
        if not isinstance(token, str) or not token:
            issues.append("missing process token")
        identities.append((pid, starttime, token))
    if len(set(identities)) != len(identities):
        issues.append("AB/BA arms did not use four distinct fresh process identities")
    return not issues, issues


def parse_nvml(path: Path) -> dict[str, Any]:
    """Validate per-GPU coverage without requiring identical timestamps."""
    per_gpu: dict[int, list[tuple[str, float, float]]] = defaultdict(list)
    malformed = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {"complete": False, "reason": "missing", "path": os.fspath(path)}
    for line in lines:
        columns = [value.strip() for value in line.split(",")]
        if len(columns) < 7:
            malformed += 1
            continue
        try:
            index = int(columns[1])
            memory_used = float(columns[2])
            utilization = float(columns[4])
        except ValueError:
            malformed += 1
            continue
        per_gpu[index].append((columns[0], memory_used, utilization))
    gpus = sorted(per_gpu)
    complete = malformed == 0 and gpus == list(range(WORLD_SIZE)) and all(len(per_gpu[index]) >= 2 for index in gpus)
    return {
        "complete": complete,
        "path": os.fspath(path),
        "malformed_rows": malformed,
        "gpu_indices": gpus,
        "samples_per_gpu": {str(index): len(per_gpu[index]) for index in gpus},
        "peak_memory_mib": max((row[1] for values in per_gpu.values() for row in values), default=None),
        "peak_utilization_pct": max((row[2] for values in per_gpu.values() for row in values), default=None),
    }


def paired_effects(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_period = {int(record["period"]): record for record in records}
    pairs = ((1, 2), (4, 3))
    metrics = ("perf/step_time_s", "perf/generate_time_s", "perf/reward_time_s", "perf/train_time_s")
    result: dict[str, Any] = {"pairs": []}
    for off_period, on_period in pairs:
        off = by_period.get(off_period)
        on = by_period.get(on_period)
        pair: dict[str, Any] = {"off_period": off_period, "on_period": on_period, "usable": False, "metrics": {}}
        if off and on and off.get("passed") is True and on.get("passed") is True:
            pair["usable"] = True
            for metric in metrics:
                off_value = float(off["metrics"][metric])
                on_value = float(on["metrics"][metric])
                pair["metrics"][metric] = {
                    "off_s": off_value,
                    "on_s": on_value,
                    "speedup": off_value / on_value,
                    "saved_s": off_value - on_value,
                    "reduction_pct": (off_value - on_value) / off_value * 100.0,
                }
        result["pairs"].append(pair)
    for metric in metrics:
        effects = [pair["metrics"][metric]["speedup"] for pair in result["pairs"] if pair["usable"]]
        result[metric] = {
            "pair_count": len(effects),
            "geometric_mean_speedup": math.exp(sum(math.log(value) for value in effects) / len(effects))
            if effects
            else None,
            "min_speedup": min(effects) if effects else None,
            "max_speedup": max(effects) if effects else None,
        }
    result["complete"] = all(pair["usable"] for pair in result["pairs"])
    return result
