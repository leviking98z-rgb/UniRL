"""Build and validate a fail-closed MiniMax-H3 fixed-geometry trace matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from hydra import compose, initialize_config_dir
    from omegaconf import DictConfig, ListConfig, OmegaConf
except ImportError as exc:
    raise SystemExit(
        "matrix_driver.py requires hydra-core and omegaconf (install UniRL's normal dependencies)"
    ) from exc

from unirl.train.configs import FSDPConfig

MATRIX_SCHEMA = "unirl:minimax-h3:fixed-geometry-matrix:v1"
RUN_SCHEMA = "unirl:minimax-h3:fixed-geometry-run:v2"
MEASURED_EVIDENCE = {
    "kind": "measured_gpu_trace",
    "timing_source": "device-synchronized wall clock",
}
PROXY_EVIDENCE = {
    "kind": "analytical_cpu_proxy",
    "cost_domain": "geometry-only packed rows",
    "text_tokens": "fixed at zero",
    "timings": "unavailable and fixed at zero",
}
DEFAULT_SOURCE_CONFIG = "examples/diffusion/minimax_h3/minimax_h3_t2va_trainside.yaml"
DEFAULT_CONFIG_NAME = "diffusion/minimax_h3/minimax_h3_t2va_trainside"
DEFAULT_GEOMETRIES = (
    ("g0", 768, 768, 124),
    ("g1", 768, 1024, 124),
    ("g2", 768, 768, 175),
    ("g3", 768, 1024, 175),
)
_REQUIRED_SOURCE_PATHS = {
    "pipeline._target_": "unirl.models.minimax_h3.pipeline.MiniMaxH3Pipeline.from_bundle",
    "rollout._target_": "unirl.rollout.engine.trainside.engine.TrainsideRolloutEngine",
    "rollout.forward_batch_size": 1,
    "reward._target_": "unirl.reward.service.RewardService",
    "data_source._target_": "unirl.data.data_source.MultimodalRLDataSource",
    "sampling._target_": "unirl.types.sampling.DiffusionSamplingParams",
}
_FROZEN_PATHS = (
    "bundle._target_",
    "bundle.config",
    "pipeline._target_",
    "pipeline.strategy",
    "backend._target_",
    "backend.block_class_names",
    "backend.trainable_attr",
    "backend.fsdp_cfg.param_dtype",
    "backend.fsdp_cfg.cpu_offload",
    "backend.fsdp_cfg.master_dtype",
    "backend.fsdp_cfg.mixed_precision",
    "backend.fsdp_cfg.cast_forward_inputs",
    "backend.fsdp_cfg.fsdp_mode",
    "backend.fsdp_cfg.hsdp_shard_size",
    "backend.fsdp_cfg.reshard_after_forward",
    "backend.fsdp_cfg.activation_checkpointing",
    "backend.fsdp_cfg.ac_wrap_order",
    "backend.fsdp_cfg.use_torch_compile",
    "backend.fsdp_cfg.defer_grad_sync",
    "backend.fsdp_cfg.forward_prefetch",
    "backend.fsdp_cfg.root_wrap",
    "backend.fsdp_cfg.checkpoint_format",
    "backend.fsdp_cfg.checkpoint_async",
    "backend.fsdp_cfg.sp_size",
    "backend.fsdp_cfg.ep_size",
    "backend.lora_cfg",
    "rollout._target_",
    "rollout.stage_attrs",
    "rollout.forward_batch_size",
    "data_source._target_",
    "data_source.args.run.seed",
    "sampling._target_",
    "sampling.num_inference_steps",
    "sampling.guidance_scale",
    "sampling.eta",
    "sampling.sde_indices",
    "sampling.samples_per_prompt",
    "sampling.seed",
    "sampling.init_same_noise",
    "sampling.autocast_precision",
    "sampling.trajectory_precision",
    "sampling.logprob_precision",
)
_ALLOWED_ENV = (
    "CONDA_ENV",
    "CONDA_SH",
    "CUDA_RUNTIME_LIB_DIR",
    "CUDA_RUNTIME_LINK_DIR",
    "RAY_ADDRESS",
    "VENV_DIR",
)
_FSDP_DEFAULTS = {field.name: field.default for field in fields(FSDPConfig)}
_FROZEN_EXCEPTIONS = ("backend.fsdp_cfg.fsdp_mode", "backend.fsdp_cfg.hsdp_shard_size")


@dataclass(frozen=True)
class SourceBinding:
    """Immutable source/config/prompt/checkpoint identity for every matrix row."""

    source_commit: str
    source_tree: str
    source_config: str
    source_config_sha256: str
    resolved_config_sha256: str
    frozen_config_sha256: str
    pretrained_model: str
    prompts: str
    prompts_sha256: str
    prompt_count: int
    lora_checkpoint: str
    lora_checkpoint_sha256: str
    lora_metadata_sha256: str
    lora_step: int

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class MatrixError(ValueError):
    """Fail-closed matrix validation error."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise MatrixError(f"checkpoint directory has no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _repo_root(start: Path) -> Path:
    try:
        return Path(_run_git(start, "rev-parse", "--show-toplevel")).resolve()
    except subprocess.CalledProcessError as exc:
        raise MatrixError(f"not inside a Git checkout: {start}") from exc


def _assert_clean_worktree(repo: Path) -> None:
    status = _run_git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise MatrixError(f"worktree is modified or untracked; commit or clean it before prepare/run:\n{status}")


def _script_repo() -> Path:
    return _repo_root(Path(__file__).resolve().parent)


def _resolve_repo_file(repo: Path, value: str, *, label: str) -> Path:
    path = Path(value).expanduser()
    path = path if path.is_absolute() else repo / path
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise MatrixError(f"{label} must be a regular file: {resolved}")
    return resolved


def _resolve_config_path(repo: Path, config_name: str) -> Path:
    relative = str(config_name).removesuffix(".yaml") + ".yaml"
    return _resolve_repo_file(repo, f"examples/{relative}", label="Hydra config")


def _resolve_checkpoint(value: str) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise MatrixError(f"LoRA checkpoint must be a directory: {path}")
    return path


def _resolve_executable(value: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        found = shutil.which(value)
        if found is None:
            raise MatrixError(f"cannot resolve Python executable: {value}")
        candidate = Path(found)
    absolute = Path(os.path.abspath(candidate))
    if not absolute.is_file() or not os.access(absolute, os.X_OK):
        raise MatrixError(f"Python executable must be an executable file: {absolute}")
    return absolute


def _git_binding(repo: Path, path: Path) -> tuple[str, str, str, str]:
    relative = path.relative_to(repo).as_posix()
    source_commit = _run_git(repo, "rev-parse", "HEAD")
    source_tree = _run_git(repo, "rev-parse", "HEAD^{tree}")
    head_blob = _run_git(repo, "rev-parse", f"HEAD:{relative}")
    worktree_blob = _run_git(repo, "hash-object", relative)
    if head_blob != worktree_blob:
        raise MatrixError(f"source config bytes differ from HEAD: {relative}")
    return source_commit, source_tree, relative, _sha256_file(path)


def _get(cfg: Any, dotted: str) -> Any:
    value = OmegaConf.select(cfg, dotted, default=None)
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _compose_config(repo: Path, config_name: str) -> DictConfig:
    config_path = _resolve_config_path(repo, config_name)
    config_dir = repo / "examples"
    relative_name = config_path.relative_to(config_dir).as_posix().removesuffix(".yaml")
    try:
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name=relative_name)
        OmegaConf.resolve(cfg)
    except Exception as exc:
        raise MatrixError(
            f"cannot compose/resolve Hydra config {relative_name!r}; export every required environment variable: {exc}"
        ) from exc
    return cfg


def _validate_source_config(cfg: DictConfig) -> None:
    for path, expected in _REQUIRED_SOURCE_PATHS.items():
        actual = _get(cfg, path)
        if actual != expected:
            raise MatrixError(f"canonical config {path}={actual!r}; expected {expected!r}")
    if _get(cfg, "backend.lora_cfg") is None:
        raise MatrixError("canonical config must define backend.lora_cfg")
    if int(_get(cfg, "sampling.samples_per_prompt") or 0) < 1:
        raise MatrixError("canonical config sampling.samples_per_prompt must be positive")
    if int(_get(cfg, "sampling.seed") or -1) < 0:
        raise MatrixError("canonical config sampling.seed must be a fixed non-negative integer")


def _frozen_bundle_config(cfg: DictConfig) -> dict[str, Any]:
    value = _get(cfg, "bundle.config")
    if not isinstance(value, dict):
        raise MatrixError("canonical config bundle.config must be a mapping")
    return {key: item for key, item in value.items() if key != "workload_telemetry_path"}


def _frozen_value(cfg: DictConfig, path: str) -> Any:
    if path == "bundle.config":
        return _frozen_bundle_config(cfg)
    value = _get(cfg, path)
    if value is None and path.startswith("backend.fsdp_cfg."):
        field_name = path.rsplit(".", 1)[1]
        if field_name in _FSDP_DEFAULTS:
            return _FSDP_DEFAULTS[field_name]
    if value is None:
        raise MatrixError(f"canonical config is missing frozen field {path}")
    return value


def _frozen_config(cfg: DictConfig) -> dict[str, Any]:
    frozen: dict[str, Any] = {}
    for path in _FROZEN_PATHS:
        frozen[path] = _frozen_value(cfg, path)
    return frozen


def _expected_lora_metadata(cfg: DictConfig) -> dict[str, Any]:
    lora = _get(cfg, "backend.lora_cfg")
    if not isinstance(lora, dict):
        raise MatrixError("canonical config backend.lora_cfg must be a mapping")
    target_modules = lora.get("target_modules")
    module_prefix = str(lora.get("module_prefix") or "").strip(".")
    if module_prefix:
        if not isinstance(target_modules, list) or not target_modules:
            raise MatrixError("backend.lora_cfg.module_prefix requires a non-empty target_modules list")
        prefix_re = re.escape(module_prefix)
        leaves_re = "|".join(re.escape(str(target)) for target in target_modules)
        target_modules = rf"^{prefix_re}\.(?:.*\.)?(?:{leaves_re})$"
    return {
        "rank": int(lora["rank"]),
        "alpha": int(lora["alpha"]),
        "target_modules": target_modules,
        "exclude_modules": lora.get("exclude_modules"),
        "dropout": float(lora.get("dropout", 0.0)),
        "bias": str(lora.get("bias", "none")),
        "task_type": str(lora.get("task_type", "FEATURE_EXTRACTION")),
    }


def _prompt_ids(path: Path) -> list[str]:
    if path.suffix == ".txt":
        rows: list[Any] = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MatrixError(f"invalid JSONL prompt at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict) or not str(value.get("prompt", value.get("caption", ""))).strip():
                raise MatrixError(f"prompt row lacks prompt/caption at {path}:{line_number}")
            rows.append(value)
    elif path.suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MatrixError(f"invalid prompt JSON {path}: {exc}") from exc
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict) and isinstance(value.get("prompts"), list):
            rows = value["prompts"]
        else:
            raise MatrixError("JSON prompt file must be a list or an object with a prompts list")
    else:
        raise MatrixError(f"unsupported prompt suffix {path.suffix!r}; use .txt, .jsonl, or .json")
    prompt_ids = []
    for index, row in enumerate(rows):
        if isinstance(row, str):
            if not row.strip():
                raise MatrixError(f"empty prompt at {path} row {index}")
            prompt_id = f"{path.name}:{index}"
        elif isinstance(row, dict):
            if not str(row.get("prompt", row.get("caption", ""))).strip():
                raise MatrixError(f"prompt row lacks prompt/caption at {path} row {index}")
            prompt_id = str(row.get("prompt_id", f"{path.name}:{index}"))
        else:
            raise MatrixError(f"prompt row must be a string or object at {path} row {index}")
        if "/" in prompt_id:
            raise MatrixError(f"prompt_id contains the reserved lineage delimiter '/': {prompt_id!r}")
        prompt_ids.append(prompt_id)
    seen = set()
    duplicates = set()
    for prompt_id in prompt_ids:
        if prompt_id in seen:
            duplicates.add(prompt_id)
        seen.add(prompt_id)
    if duplicates:
        raise MatrixError(f"prompt source has duplicate prompt_id values: {sorted(duplicates)[:3]}")
    if not prompt_ids:
        raise MatrixError(f"prompt source has no usable prompts: {path}")
    return prompt_ids


def _count_prompts(path: Path) -> int:
    return len(_prompt_ids(path))


def _metadata_from_torch_file(path: Path) -> dict[str, Any]:
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise MatrixError(f"checkpoint metadata must be a mapping: {path}")
    return value


def _checkpoint_metadata(path: Path, expected_lora: dict[str, Any]) -> tuple[int, str, str]:
    torch_file = path / "checkpoint.pt"
    dcp_metadata = path / ".metadata"
    app_metadata = path / "metadata.pt"
    if torch_file.is_file() and dcp_metadata.exists():
        raise MatrixError(f"checkpoint mixes torch and DCP completion markers: {path}")
    if torch_file.is_file():
        metadata = _metadata_from_torch_file(torch_file)
        metadata_sha = _sha256_file(torch_file)
    elif dcp_metadata.is_file() and app_metadata.is_file():
        metadata = _metadata_from_torch_file(app_metadata)
        metadata_sha = _sha256_file(app_metadata)
    else:
        raise MatrixError(f"checkpoint lacks checkpoint.pt or complete DCP metadata: {path}")
    if str(metadata.get("save_mode", "full")) != "adapter":
        raise MatrixError("trace matrix requires a save_mode='adapter' checkpoint")
    actual_lora = metadata.get("lora_config")
    if actual_lora is None:
        raise MatrixError("adapter checkpoint has no lora_config metadata")
    if _canonical_json(actual_lora) != _canonical_json(expected_lora):
        raise MatrixError("adapter checkpoint lora_config differs from the composed source config")
    step = int(metadata.get("step") or 0)
    if step < 0:
        raise MatrixError(f"adapter checkpoint step must be >=0, got {step}")
    trainer_state = path / "trainer_state.json"
    if not trainer_state.is_file():
        raise MatrixError(f"checkpoint lacks trainer_state.json: {path}")
    try:
        json.loads(trainer_state.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MatrixError(f"invalid trainer_state.json: {exc}") from exc
    metadata_sha = _sha256_json(
        {
            "step": step,
            "save_mode": "adapter",
            "lora_config": actual_lora,
        }
    )
    return step, _sha256_tree(path), metadata_sha


def _geometry_rows() -> list[dict[str, Any]]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from unirl.utils.minimax_h3_workload import MiniMaxH3WorkloadGeometry

    rows = []
    for name, height, width, num_frames in DEFAULT_GEOMETRIES:
        geometry = MiniMaxH3WorkloadGeometry.resolve(height=height, width=width, num_frames=num_frames)
        rows.append(
            {
                "name": name,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "base_packed_rows": geometry.base_packed_rows,
            }
        )
    return rows


def _binding(
    repo: Path,
    source_config: str,
    config_name: str,
    prompts: str,
    lora_checkpoint: str,
) -> tuple[SourceBinding, DictConfig]:
    config_path = _resolve_repo_file(repo, source_config, label="source config")
    composed_path = _resolve_config_path(repo, config_name)
    if composed_path != config_path:
        raise MatrixError(
            f"source config {config_path.relative_to(repo)} does not match --config-name "
            f"{composed_path.relative_to(repo)}"
        )
    prompts_path = _resolve_repo_file(repo, prompts, label="prompt source")
    checkpoint_path = _resolve_checkpoint(lora_checkpoint)
    source_commit, source_tree, relative_config, config_sha = _git_binding(repo, config_path)
    cfg = _compose_config(repo, config_name)
    _validate_source_config(cfg)
    frozen_sha = _sha256_json(_frozen_config(cfg))
    pretrained_model = str(_get(cfg, "bundle.config.pretrained_model_ckpt_path") or "").strip()
    if not pretrained_model:
        raise MatrixError("resolved config must define a non-empty bundle.config.pretrained_model_ckpt_path")
    lora_step, checkpoint_sha, metadata_sha = _checkpoint_metadata(checkpoint_path, _expected_lora_metadata(cfg))
    return (
        SourceBinding(
            source_commit=source_commit,
            source_tree=source_tree,
            source_config=relative_config,
            source_config_sha256=config_sha,
            resolved_config_sha256=_sha256_json(cfg),
            frozen_config_sha256=frozen_sha,
            pretrained_model=pretrained_model,
            prompts=str(prompts_path),
            prompts_sha256=_sha256_file(prompts_path),
            prompt_count=_count_prompts(prompts_path),
            lora_checkpoint=str(checkpoint_path),
            lora_checkpoint_sha256=checkpoint_sha,
            lora_metadata_sha256=metadata_sha,
            lora_step=lora_step,
        ),
        cfg,
    )


def _manifest_id(
    binding: SourceBinding,
    rows: Sequence[dict[str, Any]],
    settings: dict[str, Any],
    runs: Sequence[dict[str, Any]],
) -> str:
    return _sha256_json(
        {
            "binding": binding.to_dict(),
            "geometries": list(rows),
            "settings": settings,
            "runs": list(runs),
        }
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    repo = _script_repo()
    binding, cfg = _binding(
        repo,
        args.source_config,
        args.config_name,
        args.prompts,
        args.lora_checkpoint,
    )
    rows = _geometry_rows()
    samples_per_prompt = int(_get(cfg, "sampling.samples_per_prompt"))
    num_prompts = int(args.num_prompts)
    if num_prompts < 1 or num_prompts > binding.prompt_count:
        raise MatrixError(f"num_prompts must be in [1, {binding.prompt_count}], got {num_prompts}")
    if int(args.num_devices) < 1 or int(args.sp_size) < 1 or int(args.num_devices) % int(args.sp_size):
        raise MatrixError("num_devices must be positive and divisible by sp_size")
    dp_groups = int(args.num_devices) // int(args.sp_size)
    if num_prompts % dp_groups:
        raise MatrixError(f"num_prompts={num_prompts} must be divisible by DP groups={dp_groups}")
    reward_fraction = float(_get(cfg, "reward_fraction") or 0.0)
    if reward_fraction > 0.0:
        reward_devices_f = reward_fraction * int(args.num_devices)
        reward_dp_size = int(round(reward_devices_f))
        if abs(reward_devices_f - reward_dp_size) > 1e-9:
            raise MatrixError(
                f"reward_fraction={reward_fraction} of num_devices={args.num_devices} is not an integer device count"
            )
    else:
        reward_dp_size = int(args.num_devices)
    if num_prompts % reward_dp_size:
        raise MatrixError(f"num_prompts={num_prompts} must be divisible by colocated reward DP size={reward_dp_size}")
    if int(args.group_size) < 1 or dp_groups % int(args.group_size):
        raise MatrixError(f"group_size={args.group_size} must divide DP groups={dp_groups}")
    if int(args.group_size) == 1:
        raise MatrixError("group_size=1 is a no-op; use at least 2 DP groups for a reorder decision")
    eval_eta = float(_get(cfg, "sampling.eta"))
    settings = {
        "config_name": args.config_name,
        "python_executable": str(_resolve_executable(args.python_executable)),
        "num_devices": int(args.num_devices),
        "batch_size": num_prompts,
        "sp_size": int(args.sp_size),
        "dp_groups": dp_groups,
        "reward_dp_size": reward_dp_size,
        "group_size": int(args.group_size),
        "num_prompts": num_prompts,
        "expected_root_ids": [
            f"r{binding.lora_step}:prompt:{prompt_id}:sample:0"
            for prompt_id in _prompt_ids(Path(binding.prompts))[:num_prompts]
        ],
        "samples_per_prompt": samples_per_prompt,
        "expected_records_per_geometry": num_prompts * samples_per_prompt,
        "eval_eta": eval_eta,
        "cost": args.cost,
        "tail_ratio_threshold": float(args.tail_ratio_threshold),
        "minimum_predicted_speedup": float(args.minimum_predicted_speedup),
        "runtime_overrides": (
            {"backend.fsdp_cfg.fsdp_mode": "hybrid", "backend.fsdp_cfg.hsdp_shard_size": 8}
            if int(args.num_devices) > 8 and str(_get(cfg, "backend.fsdp_cfg.fsdp_mode")) == "full"
            else {}
        ),
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.is_relative_to(repo):
        raise MatrixError(f"output_dir must be outside the source checkout: {output_dir}")
    commands = []
    for row in rows:
        trace = output_dir / "traces" / f"{row['name']}.jsonl"
        receipt = output_dir / "receipts" / f"{row['name']}.json"
        command = [
            settings["python_executable"],
            str(Path(__file__).resolve()),
            "run-one",
            "--manifest",
            str(output_dir / "matrix.json"),
            "--geometry",
            row["name"],
        ]
        commands.append({"geometry": row["name"], "trace": str(trace), "receipt": str(receipt), "command": command})
    manifest_id = _manifest_id(binding, rows, settings, commands)
    return {
        "schema": MATRIX_SCHEMA,
        "manifest_id": manifest_id,
        "binding": binding.to_dict(),
        "frozen_config": _frozen_config(cfg),
        "settings": settings,
        "geometries": rows,
        "runs": commands,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatrixError(f"expected JSON object at {path}")
    return value


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    required_fields = {"binding", "frozen_config", "geometries", "manifest_id", "runs", "schema", "settings"}
    if set(manifest) != required_fields:
        raise MatrixError(
            f"matrix fields mismatch: missing={sorted(required_fields - set(manifest))} "
            f"extra={sorted(set(manifest) - required_fields)}"
        )
    if manifest.get("schema") != MATRIX_SCHEMA:
        raise MatrixError(f"unsupported matrix schema {manifest.get('schema')!r}")
    try:
        binding = SourceBinding(**manifest["binding"])
        geometries = manifest["geometries"]
        settings = manifest["settings"]
        runs = manifest["runs"]
        expected = _manifest_id(binding, geometries, settings, runs)
    except (KeyError, TypeError) as exc:
        raise MatrixError(f"malformed matrix manifest: {exc}") from exc
    if manifest.get("manifest_id") != expected:
        raise MatrixError("matrix manifest_id does not match its binding/geometries/settings/runs")
    if geometries != _geometry_rows():
        raise MatrixError("matrix geometries differ from the fixed four-profile definition")
    required_settings = {
        "batch_size",
        "config_name",
        "cost",
        "dp_groups",
        "eval_eta",
        "expected_root_ids",
        "expected_records_per_geometry",
        "group_size",
        "minimum_predicted_speedup",
        "num_devices",
        "num_prompts",
        "python_executable",
        "reward_dp_size",
        "runtime_overrides",
        "samples_per_prompt",
        "sp_size",
        "tail_ratio_threshold",
    }
    if set(settings) != required_settings:
        raise MatrixError(
            f"matrix settings fields mismatch: missing={sorted(required_settings - set(settings))} "
            f"extra={sorted(set(settings) - required_settings)}"
        )
    if int(settings["batch_size"]) != int(settings["num_prompts"]):
        raise MatrixError("matrix batch_size must equal num_prompts for the frozen evaluation")
    positive_fields = (
        "batch_size",
        "dp_groups",
        "group_size",
        "num_devices",
        "num_prompts",
        "reward_dp_size",
        "samples_per_prompt",
        "sp_size",
    )
    if any(int(settings[field]) < 1 for field in positive_fields):
        raise MatrixError("matrix device, prompt, sample, and group counts must be positive")
    if int(settings["num_devices"]) % int(settings["sp_size"]):
        raise MatrixError("matrix num_devices must be divisible by sp_size")
    if (
        not isinstance(settings["expected_root_ids"], list)
        or len(settings["expected_root_ids"]) != int(settings["num_prompts"])
        or len(set(settings["expected_root_ids"])) != len(settings["expected_root_ids"])
        or any(
            not isinstance(root_id, str) or not root_id or "/" in root_id for root_id in settings["expected_root_ids"]
        )
    ):
        raise MatrixError("matrix expected_root_ids must contain one unique root per prompt")
    if int(settings["num_devices"]) // int(settings["sp_size"]) != int(settings["dp_groups"]):
        raise MatrixError("matrix DP geometry is inconsistent")
    if int(settings["num_prompts"]) % int(settings["dp_groups"]):
        raise MatrixError("matrix num_prompts must be divisible by dp_groups")
    if int(settings["num_prompts"]) % int(settings["reward_dp_size"]):
        raise MatrixError("matrix num_prompts must be divisible by reward_dp_size")
    if int(settings["dp_groups"]) % int(settings["group_size"]):
        raise MatrixError("matrix group_size must divide dp_groups")
    if int(settings["group_size"]) < 2:
        raise MatrixError("matrix group_size must be at least 2")
    if int(settings["expected_records_per_geometry"]) != int(settings["num_prompts"]) * int(
        settings["samples_per_prompt"]
    ):
        raise MatrixError("matrix expected record count is inconsistent")
    if settings["cost"] not in {"packed_rows", "padded_rows", "attention_rows2", "total_s", "denoise_s"}:
        raise MatrixError(f"matrix has unsupported cost {settings['cost']!r}")
    for field in ("eval_eta", "tail_ratio_threshold", "minimum_predicted_speedup"):
        if not math.isfinite(float(settings[field])):
            raise MatrixError(f"matrix {field} must be finite")
    if float(settings["tail_ratio_threshold"]) <= 1.0:
        raise MatrixError("matrix tail_ratio_threshold must be greater than 1.0")
    if float(settings["minimum_predicted_speedup"]) <= 1.0:
        raise MatrixError("matrix minimum_predicted_speedup must be greater than 1.0")
    expected_runtime_overrides = (
        {"backend.fsdp_cfg.fsdp_mode": "hybrid", "backend.fsdp_cfg.hsdp_shard_size": 8}
        if int(settings["num_devices"]) > 8 and manifest["frozen_config"]["backend.fsdp_cfg.fsdp_mode"] == "full"
        else {}
    )
    if settings["runtime_overrides"] != expected_runtime_overrides:
        raise MatrixError("matrix runtime_overrides differ from the fixed topology policy")
    _resolve_executable(str(settings["python_executable"]))
    expected_names = [row[0] for row in DEFAULT_GEOMETRIES]
    if not isinstance(runs, list) or [run.get("geometry") for run in runs if isinstance(run, dict)] != expected_names:
        raise MatrixError(f"matrix runs must contain exactly {expected_names} in order")
    output_paths = []
    repo = _script_repo()
    for run in runs:
        required_run_fields = {"geometry", "trace", "receipt", "command"}
        if set(run) != required_run_fields:
            raise MatrixError(f"matrix run fields mismatch for {run.get('geometry')!r}")
        trace, receipt = Path(run["trace"]), Path(run["receipt"])
        if not trace.is_absolute() or not receipt.is_absolute() or trace == receipt:
            raise MatrixError(f"matrix run paths must be distinct absolute paths for {run['geometry']}")
        if trace.is_relative_to(repo) or receipt.is_relative_to(repo):
            raise MatrixError(f"matrix outputs must stay outside the source checkout for {run['geometry']}")
        output_paths.extend((trace, receipt))
        expected_driver = [
            str(settings["python_executable"]),
            str(Path(__file__).resolve()),
            "run-one",
            "--manifest",
            str(path),
            "--geometry",
            run["geometry"],
        ]
        if run["command"] != expected_driver:
            raise MatrixError(f"matrix driver command differs from the expected contract for {run['geometry']}")
    if len(set(output_paths)) != len(output_paths):
        raise MatrixError("matrix trace and receipt paths must be globally unique")
    return manifest


def _geometry(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [row for row in manifest["geometries"] if row["name"] == name]
    if len(matches) != 1:
        raise MatrixError(f"geometry {name!r} is not exactly once in the manifest")
    return matches[0]


def _revalidate_binding(repo: Path, manifest: dict[str, Any]) -> None:
    _assert_clean_worktree(repo)
    bound = SourceBinding(**manifest["binding"])
    current, cfg = _binding(
        repo,
        bound.source_config,
        manifest["settings"]["config_name"],
        bound.prompts,
        bound.lora_checkpoint,
    )
    if current != bound:
        changed = [key for key in bound.__dict__ if getattr(bound, key) != getattr(current, key)]
        raise MatrixError(f"source binding changed since prepare: {changed}")
    if _frozen_config(cfg) != manifest["frozen_config"]:
        raise MatrixError("frozen canonical config differs from manifest")
    settings = manifest["settings"]
    expected_root_ids = [
        f"r{bound.lora_step}:prompt:{prompt_id}:sample:0"
        for prompt_id in _prompt_ids(Path(bound.prompts))[: int(settings["num_prompts"])]
    ]
    if settings["expected_root_ids"] != expected_root_ids:
        raise MatrixError("matrix root IDs differ from the bound prompt order and checkpoint step")


def _effective_frozen_config(manifest: dict[str, Any]) -> dict[str, Any]:
    frozen = dict(manifest["frozen_config"])
    for path, value in manifest["settings"].get("runtime_overrides", {}).items():
        if path not in _FROZEN_EXCEPTIONS:
            raise MatrixError(f"unsupported runtime override in matrix: {path}")
        frozen[path] = value
    return frozen


def _run_entry(manifest: dict[str, Any], geometry_name: str) -> dict[str, Any]:
    run = next((item for item in manifest["runs"] if item["geometry"] == geometry_name), None)
    if run is None:
        raise MatrixError(f"manifest has no run entry for {geometry_name}")
    return run


def _output_paths(manifest: dict[str, Any], geometry_name: str) -> tuple[Path, Path]:
    run = _run_entry(manifest, geometry_name)
    trace = Path(run["trace"])
    receipt = Path(run["receipt"])
    return trace, receipt


def _assert_output_slots(manifest: dict[str, Any], geometry_name: str) -> tuple[Path, Path]:
    trace, receipt = _output_paths(manifest, geometry_name)
    if trace.exists() or receipt.exists():
        raise MatrixError(f"refusing to overwrite existing trace/receipt for {geometry_name}: {trace}, {receipt}")
    return trace, receipt


def _train_command(manifest: dict[str, Any], geometry_name: str) -> list[str]:
    row = _geometry(manifest, geometry_name)
    settings = manifest["settings"]
    _effective_frozen_config(manifest)
    trace, _ = _output_paths(manifest, geometry_name)
    return [
        settings["python_executable"],
        "-m",
        "unirl.train_diffusion",
        f"--config-name={settings['config_name']}",
        f"++num_devices={settings['num_devices']}",
        f"++batch_size={settings['batch_size']}",
        f"++num_rollouts={manifest['binding']['lora_step']}",
        f"++load_dir={manifest['binding']['lora_checkpoint']}",
        "++save_interval=0",
        "++logging.report_to_wandb=false",
        "++logging.log_media=false",
        f"++bundle.config.pretrained_model_ckpt_path={manifest['binding']['pretrained_model']}",
        f"data_source.args.run.data_path={manifest['binding']['prompts']}",
        f"data_source.args.run.eval_data_path={manifest['binding']['prompts']}",
        "++data_source.args.run.shuffle=false",
        f"++eval_interval={manifest['binding']['lora_step'] + 1}",
        f"++eval_num_prompts={settings['num_prompts']}",
        f"++eval_chunk_prompts={settings['num_prompts']}",
        f"++eval_samples_per_prompt={settings['samples_per_prompt']}",
        f"++eval_eta={settings['eval_eta']}",
        f"sampling.height={row['height']}",
        f"sampling.width={row['width']}",
        f"sampling.num_frames={row['num_frames']}",
        f"++eval_sampling.height={row['height']}",
        f"++eval_sampling.width={row['width']}",
        f"++eval_sampling.num_frames={row['num_frames']}",
        f"++backend.fsdp_cfg.sp_size={settings['sp_size']}",
        f"++bundle.config.workload_telemetry_path={trace}",
        *(f"++{path}={value}" for path, value in settings.get("runtime_overrides", {}).items()),
    ]


def _command(
    manifest: dict[str, Any],
    geometry_name: str,
    *,
    require_empty_outputs: bool = True,
) -> tuple[list[str], dict[str, str], Path, Path]:
    _revalidate_binding(_script_repo(), manifest)
    trace, receipt = (
        _assert_output_slots(manifest, geometry_name)
        if require_empty_outputs
        else _output_paths(manifest, geometry_name)
    )
    if os.environ.get("RAY_ADDRESS") != "auto":
        raise MatrixError("P3 requires an existing P0 Ray allocation; export RAY_ADDRESS=auto")
    command = _train_command(manifest, geometry_name)
    env = {key: os.environ[key] for key in _ALLOWED_ENV if key in os.environ}
    env.update(
        {
            "REPORT_TO_WANDB": "false",
            "WANDB_MODE": "disabled",
            "MATRIX_MANIFEST_ID": str(manifest["manifest_id"]),
            "MATRIX_GEOMETRY": geometry_name,
        }
    )
    return command, env, trace, receipt


def expected_command(manifest: dict[str, Any], geometry_name: str) -> list[str]:
    """Return the immutable train command without requiring empty output slots."""
    return _train_command(manifest, geometry_name)


def validate_trace_records(
    manifest: dict[str, Any],
    geometry: dict[str, Any],
    records: Sequence[Any],
) -> dict[str, Any]:
    """Validate one fixed-geometry trace against its immutable matrix row."""
    expected_count = int(manifest["settings"]["expected_records_per_geometry"])
    if len(records) != expected_count:
        raise MatrixError(f"trace has {len(records)} records, expected {expected_count}")
    expected_geometry = (geometry["height"], geometry["width"], geometry["num_frames"])
    actual_geometries = {(row.height, row.width, row.num_frames) for row in records}
    if actual_geometries != {expected_geometry}:
        raise MatrixError(f"trace geometry mismatch: expected {expected_geometry}, got {sorted(actual_geometries)}")
    settings = manifest["settings"]
    if {row.sp_size for row in records} != {int(settings["sp_size"])}:
        raise MatrixError("trace SP size does not match the matrix")
    if {row.sp_rank for row in records} != {0}:
        raise MatrixError("trace must contain only SP-group head records")
    dp_groups = int(settings["dp_groups"])
    if not records or any(record.dp_rank < 0 or record.dp_rank >= dp_groups for record in records):
        raise MatrixError(f"trace DP ranks must be in [0, {dp_groups})")

    expected_roots = list(settings["expected_root_ids"])
    expected_siblings = int(settings["samples_per_prompt"])
    expected_samples = {
        root_id: [f"{root_id}/{sibling}" for sibling in range(expected_siblings)] for root_id in expected_roots
    }
    by_root: dict[str, list[Any]] = {}
    seen_samples = set()
    for record in records:
        if record.sample_id in seen_samples:
            raise MatrixError(f"trace has duplicate sample_id {record.sample_id!r}")
        seen_samples.add(record.sample_id)
        by_root.setdefault(record.root_id, []).append(record)
    if set(by_root) != set(expected_roots):
        raise MatrixError("trace root IDs differ from the prompt-bound matrix")

    roots_per_dp = len(expected_roots) // dp_groups
    observed_order_by_rank: dict[int, list[str]] = {rank: [] for rank in range(dp_groups)}
    for root_index, root_id in enumerate(expected_roots):
        siblings = by_root[root_id]
        if [record.sample_id for record in siblings] != expected_samples[root_id]:
            raise MatrixError(f"trace root {root_id!r} does not contain its ordered sibling set")
        ranks = {record.dp_rank for record in siblings}
        expected_rank = root_index // roots_per_dp
        if ranks != {expected_rank}:
            raise MatrixError(f"trace root {root_id!r} is on DP ranks {sorted(ranks)}, expected {expected_rank}")
        if len({record.text_tokens for record in siblings}) != 1:
            raise MatrixError(f"trace root {root_id!r} has inconsistent text token counts")

    seen_roots = set()
    for record in records:
        if record.root_id not in seen_roots:
            observed_order_by_rank[record.dp_rank].append(record.root_id)
            seen_roots.add(record.root_id)
    for rank in range(dp_groups):
        expected_rank_roots = expected_roots[rank * roots_per_dp : (rank + 1) * roots_per_dp]
        if observed_order_by_rank[rank] != expected_rank_roots:
            raise MatrixError(f"trace root order differs from deterministic DP-shard order on rank {rank}")
    return {"records": len(records), "roots": len(by_root)}


def _write_receipt(
    path: Path,
    *,
    manifest: dict[str, Any],
    geometry: dict[str, Any],
    trace: Path,
    command: Sequence[str],
    evidence: dict[str, str],
) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from unirl.utils.minimax_h3_workload import read_workload_records

    records = read_workload_records([trace])
    counts = validate_trace_records(manifest, geometry, records)
    _write_json(
        path,
        {
            "schema": RUN_SCHEMA,
            "manifest_id": manifest["manifest_id"],
            "binding": manifest["binding"],
            "geometry": geometry,
            "evidence": evidence,
            "trace": str(trace),
            "trace_sha256": _sha256_file(trace),
            "records": counts["records"],
            "roots": counts["roots"],
            "command": list(command),
        },
    )


def prepare(args: argparse.Namespace) -> None:
    _assert_clean_worktree(_script_repo())
    manifest = build_manifest(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "matrix.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MatrixError(f"output directory must not exist or must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)
    print(f"wrote {manifest_path}")
    print(f"manifest_id={manifest['manifest_id']}")
    for run in manifest["runs"]:
        print(shlex.join(run["command"]))


def show(args: argparse.Namespace) -> None:
    manifest = _load_manifest(Path(args.manifest).expanduser().resolve(strict=True))
    command, env, _, _ = _command(manifest, args.geometry)
    for key, value in sorted(env.items()):
        print(f"export {key}={shlex.quote(value)}")
    print(shlex.join(command))


def run_one(args: argparse.Namespace) -> None:
    if os.environ.get("P3_ALLOW_GPU_RUN") != "1":
        raise MatrixError("GPU launch is locked; set P3_ALLOW_GPU_RUN=1 explicitly after reviewing `show`")
    manifest = _load_manifest(Path(args.manifest).expanduser().resolve(strict=True))
    command, env, trace, receipt = _command(manifest, args.geometry)
    trace.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    run_env.update(env)
    subprocess.run(command, check=True, cwd=_script_repo(), env=run_env)
    _write_receipt(
        receipt,
        manifest=manifest,
        geometry=_geometry(manifest, args.geometry),
        trace=trace,
        command=command,
        evidence=MEASURED_EVIDENCE,
    )
    print(f"validated {trace}")
    print(f"wrote {receipt}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _speedup(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a speedup ratio greater than 1.0")
    return parsed


def _tail_ratio(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a tail ratio greater than 1.0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="bind inputs and write the immutable four-run matrix")
    prepare_parser.add_argument("--source-config", default=DEFAULT_SOURCE_CONFIG)
    prepare_parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    prepare_parser.add_argument("--prompts", required=True)
    prepare_parser.add_argument("--lora-checkpoint", required=True)
    prepare_parser.add_argument("--output-dir", required=True)
    prepare_parser.add_argument("--python-executable", default=sys.executable)
    prepare_parser.add_argument("--num-devices", type=_positive_int, required=True)
    prepare_parser.add_argument("--sp-size", type=_positive_int, default=8)
    prepare_parser.add_argument("--group-size", type=_positive_int, default=2)
    prepare_parser.add_argument("--num-prompts", type=_positive_int, required=True)
    prepare_parser.add_argument(
        "--cost",
        choices=("packed_rows", "padded_rows", "attention_rows2", "total_s", "denoise_s"),
        default="total_s",
    )
    prepare_parser.add_argument("--tail-ratio-threshold", type=_tail_ratio, default=1.05)
    prepare_parser.add_argument("--minimum-predicted-speedup", type=_speedup, default=1.05)

    for name in ("show", "run-one"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--geometry", choices=tuple(row[0] for row in DEFAULT_GEOMETRIES), required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "show":
            show(args)
        else:
            run_one(args)
    except (MatrixError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
