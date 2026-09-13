"""Adapt the digest-bound P0 v2 runtime receipt overlay for P2 prefetch."""

from __future__ import annotations

import functools
import hashlib
import inspect
import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import p0_runtime_receipts_base as p0

SCHEMA = "unirl-minimax-h3-p0-runtime-event-v2"
EXPECTED_PLANNER_CLASS = "unirl.train.stack.planner.count.GroupInterleavedCountPlanner"
_INSTALLED = False


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"P2 runtime overlay requires {name}")
    return value


def _install_conditioner_hooks() -> None:
    from unirl.models.minimax_h3.text_embed import MiniMaxH3TextEmbedStage

    original_resolve = MiniMaxH3TextEmbedStage._resolve_with_disk_cache

    @functools.wraps(original_resolve)
    def resolve(
        self: Any,
        prompts: Iterable[str],
        resolved: dict[str, Any],
    ) -> tuple[int, int]:
        prompt_list = list(prompts)
        p0._guarded_emit(
            "cache_lookup",
            {
                "prompt_count": len(prompt_list),
                "prompt_sha256": [hashlib.sha256(str(prompt).encode()).hexdigest() for prompt in prompt_list],
                "disk_hits": None,
                "misses": None,
                "duration_s": 0.0,
                "success": False,
                "read_only": bool(getattr(self, "_disk_cache_read_only", False)),
                "cache_root": getattr(self, "_disk_cache_dir", None),
                "contract_violation": True,
            },
            owner=self,
        )
        raise RuntimeError("P2 canonical prefetch run forbids persistent prompt-cache access")

    MiniMaxH3TextEmbedStage._resolve_with_disk_cache = p0._mark_wrapper(resolve)
    original_encode = MiniMaxH3TextEmbedStage._encode_prompt

    @functools.wraps(original_encode)
    def encode(self: Any, prompt: str, *args: Any, **kwargs: Any) -> Any:
        started_ns = time.monotonic_ns()
        success = False
        error_type: str | None = None
        try:
            result = original_encode(self, prompt, *args, **kwargs)
            success = True
            return result
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            finished_ns = time.monotonic_ns()
            thread_name = threading.current_thread().name
            p0._guarded_emit(
                "conditioner_encode_call",
                {
                    "prompt_sha256": hashlib.sha256(str(prompt).encode()).hexdigest(),
                    "started_monotonic_ns": started_ns,
                    "finished_monotonic_ns": finished_ns,
                    "duration_s": (finished_ns - started_ns) / 1e9,
                    "thread_name": thread_name,
                    "background_prefetch": thread_name == "minimax-h3-prompt-prefetch",
                    "success": success,
                    "error_type": error_type,
                    "contract_violation": False,
                },
                owner=self,
            )

    MiniMaxH3TextEmbedStage._encode_prompt = p0._mark_wrapper(encode)
    p0._install_wrapper(MiniMaxH3TextEmbedStage, "embed", p0._timer_wrapper("conditioner_s"))

    def wrap_prefetch(original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def prefetch(self: Any, texts: Any) -> bool:
            prompt_sha = [hashlib.sha256(str(prompt).encode()).hexdigest() for prompt in texts.texts]
            accepted = False
            success = False
            error_type: str | None = None
            try:
                accepted = bool(original(self, texts))
                success = True
                return accepted
            except BaseException as exc:
                error_type = type(exc).__name__
                raise
            finally:
                p0._guarded_emit(
                    "prefetch_submit",
                    {
                        "prompt_count": len(prompt_sha),
                        "prompt_sha256": prompt_sha,
                        "accepted": accepted,
                        "enabled": bool(getattr(self, "prefetch_enabled", False)),
                        "success": success,
                        "error_type": error_type,
                    },
                    owner=self,
                )

        return prefetch

    def wrap_take(original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def take(self: Any, prompts: Any, resolved: Any) -> int:
            hits = 0
            success = False
            error_type: str | None = None
            try:
                hits = int(original(self, prompts, resolved))
                success = True
                return hits
            except BaseException as exc:
                error_type = type(exc).__name__
                raise
            finally:
                p0._guarded_emit(
                    "prefetch_consume",
                    {
                        "requested": len(prompts),
                        "hits": hits,
                        "enabled": bool(getattr(self, "prefetch_enabled", False)),
                        "success": success,
                        "error_type": error_type,
                    },
                    owner=self,
                )

        return take

    def wrap_shutdown(original: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(original)
        def shutdown(self: Any) -> None:
            prefetcher = getattr(self, "_prefetch", None)
            retained_before = int(prefetcher.retained) if prefetcher is not None else 0
            success = False
            error_type: str | None = None
            try:
                original(self)
                success = True
            except BaseException as exc:
                error_type = type(exc).__name__
                raise
            finally:
                p0._guarded_emit(
                    "prefetch_shutdown",
                    {
                        "configured": prefetcher is not None,
                        "closed": bool(prefetcher is not None and prefetcher.closed),
                        "retained_before": retained_before,
                        "retained_after": (int(prefetcher.retained) if prefetcher is not None else 0),
                        "success": success,
                        "error_type": error_type,
                    },
                    owner=self,
                )

        return shutdown

    p0._install_wrapper(MiniMaxH3TextEmbedStage, "prefetch", wrap_prefetch)
    p0._install_wrapper(MiniMaxH3TextEmbedStage, "_take_prefetched", wrap_take)
    p0._install_wrapper(MiniMaxH3TextEmbedStage, "shutdown", wrap_shutdown)


def _install_generation_hooks() -> None:
    from unirl.models.minimax_h3.diffusion import MiniMaxH3DiffusionStage
    from unirl.models.minimax_h3.pipeline import MiniMaxH3Pipeline
    from unirl.models.minimax_h3.vae import (
        MiniMaxH3AudioDecodeStage,
        MiniMaxH3VideoDecodeStage,
    )

    p0._install_wrapper(MiniMaxH3DiffusionStage, "generate", p0._timer_wrapper("denoise_s"))
    p0._install_wrapper(MiniMaxH3VideoDecodeStage, "decode", p0._timer_wrapper("video_decode_s"))
    p0._install_wrapper(MiniMaxH3AudioDecodeStage, "decode", p0._timer_wrapper("audio_decode_s"))

    original_generate = MiniMaxH3Pipeline._generate

    @functools.wraps(original_generate)
    def generate(self: Any, sample: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_generate(self, sample, *args, **kwargs)
        parallel = p0._parallel_context(owner=self)
        authoritative = parallel["sp_rank"] == 0
        if not authoritative:
            p0._guarded_emit(
                "generation_replica_presence",
                {
                    "authoritative": False,
                    "sample_count": len(result.parts[-1].sample_ids),
                    "entrypoint": "MiniMaxH3Pipeline._generate",
                },
                owner=self,
            )
            return result

        started = time.perf_counter()
        rows = p0._generation_rows(result)
        for row in rows:
            p0._guarded_emit(
                "sample_generation",
                {"authoritative": True, **row},
                owner=self,
            )
        p0._guarded_emit(
            "generation_manifest",
            {
                "authoritative": True,
                "sample_count": len(rows),
                "ordered_sample_ids": [row["sample_id"] for row in rows],
                "entrypoint": "MiniMaxH3Pipeline._generate",
                "next_prompt_supplied": kwargs.get("next_sample") is not None,
            },
            owner=self,
        )
        p0._guarded_emit(
            "phase_timer",
            {
                "timer": "audit_receipt_s",
                "duration_s": time.perf_counter() - started,
                "success": True,
                "method": "sample_generation_fingerprints",
            },
            owner=self,
        )
        return result

    MiniMaxH3Pipeline._generate = p0._mark_wrapper(generate)


def _install_contract_hooks() -> None:
    from unirl.train.stack.base import TrainStack
    from unirl.trainer.diffusion import DiffusionTrainer

    original_stack_init = TrainStack.__init__

    @functools.wraps(original_stack_init)
    def stack_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_stack_init(self, *args, **kwargs)
        expected = int(_require_env("P0_EXPECTED_NUM_UPDATES"))
        planner_class = f"{type(self.micro_planner).__module__}.{type(self.micro_planner).__qualname__}"
        payload = {
            "planner_class": planner_class,
            "num_updates_per_batch": int(self.num_updates_per_batch),
            "micro_batch_size": int(self.micro_batch_size),
            "algorithm_class": f"{type(self.algorithm).__module__}.{type(self.algorithm).__qualname__}",
            "algorithm_supports_multi_update": bool(getattr(self.algorithm, "supports_multi_update", False)),
            "expected_num_updates_per_batch": expected,
        }
        p0._guarded_emit("planner_contract", payload, owner=self)
        if p0._strict() and (
            expected != 2
            or payload["num_updates_per_batch"] != expected
            or payload["micro_batch_size"] != 1
            or payload["algorithm_supports_multi_update"] is not True
            or planner_class != EXPECTED_PLANNER_CLASS
        ):
            raise RuntimeError(f"P2 planner contract violated: {payload}")

    TrainStack.__init__ = p0._mark_wrapper(stack_init)
    original_trainer_init = DiffusionTrainer.__init__
    signature = inspect.signature(original_trainer_init)

    @functools.wraps(original_trainer_init)
    def trainer_init(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind_partial(self, *args, **kwargs)
        bundle_cfg = bound.arguments.get("bundle_cfg")
        data_source_cfg = bound.arguments.get("data_source_cfg")
        captured = {
            "data_path": p0._cfg_get(data_source_cfg, "args", "run", "data_path"),
            "data_shuffle": bool(p0._cfg_get(data_source_cfg, "args", "run", "shuffle", default=True)),
            "aux_components_on_cpu": bool(p0._cfg_get(bundle_cfg, "config", "aux_components_on_cpu", default=False)),
            "cache_dir": p0._cfg_get(bundle_cfg, "config", "prompt_embedding_cache_dir"),
            "cache_read_only": bool(
                p0._cfg_get(
                    bundle_cfg,
                    "config",
                    "prompt_embedding_cache_read_only",
                    default=False,
                )
            ),
            "share": bool(
                p0._cfg_get(
                    bundle_cfg,
                    "config",
                    "prompt_embedding_share_across_sp",
                    default=False,
                )
            ),
            "prefetch": bool(p0._cfg_get(bundle_cfg, "config", "prompt_embedding_prefetch", default=False)),
            "prefetch_capacity": int(
                p0._cfg_get(
                    bundle_cfg,
                    "config",
                    "prompt_embedding_prefetch_capacity",
                    default=0,
                )
            ),
            "onload": bool(
                p0._cfg_get(
                    bundle_cfg,
                    "config",
                    "text_encoder_onload_for_embed",
                    default=False,
                )
            ),
        }
        original_trainer_init(self, *args, **kwargs)
        diffusion = self.sampling_params["diffusion"]
        data_path = Path(str(captured["data_path"] or self.data_source.data_path)).resolve()
        payload = {
            "num_devices": int(self.num_devices),
            "batch_size": int(self.batch_size),
            "samples_per_prompt": int(diffusion.samples_per_prompt),
            "num_inference_steps": int(diffusion.num_inference_steps),
            "height": int(diffusion.height),
            "width": int(diffusion.width),
            "num_frames": int(diffusion.num_frames),
            "eta": float(diffusion.eta),
            "sde_indices": [int(value) for value in diffusion.sde_indices],
            "data_path": str(data_path),
            "data_sha256": p0._sha256_file(data_path),
            "data_shuffle": captured["data_shuffle"],
            "data_source_runtime_shuffle": bool(self.data_source.shuffle),
            "offload_train_during_reward": bool(self._offload_train_during_reward),
            "aux_components_on_cpu": captured["aux_components_on_cpu"],
            "prompt_embedding_cache_dir": captured["cache_dir"],
            "prompt_embedding_cache_read_only": captured["cache_read_only"],
            "prompt_embedding_share_across_sp": captured["share"],
            "prompt_embedding_prefetch": captured["prefetch"],
            "prompt_embedding_prefetch_capacity": captured["prefetch_capacity"],
            "text_encoder_onload_for_embed": captured["onload"],
            "source_commit": _require_env("P0_SOURCE_COMMIT"),
            "remote_handle_dereferenced": False,
        }
        p0._guarded_emit("trainer_contract", payload, rank_override=-1)
        expected = {
            "num_devices": 8,
            "batch_size": 8,
            "samples_per_prompt": 4,
            "num_inference_steps": 10,
            "height": 768,
            "width": 768,
            "num_frames": 124,
            "eta": 0.7,
            "sde_indices": [0, 3, 6],
            "data_sha256": _require_env("P0_DATASET_SHA256"),
            "data_shuffle": False,
            "data_source_runtime_shuffle": False,
            "offload_train_during_reward": True,
            "aux_components_on_cpu": True,
            "prompt_embedding_cache_dir": None,
            "prompt_embedding_cache_read_only": False,
            "prompt_embedding_share_across_sp": True,
            "prompt_embedding_prefetch": _require_env("P2_EXPECTED_PREFETCH") == "1",
            "prompt_embedding_prefetch_capacity": 8,
            "text_encoder_onload_for_embed": False,
            "source_commit": _require_env("P0_EXPECTED_SOURCE_COMMIT"),
            "remote_handle_dereferenced": False,
        }
        failed = [key for key, value in expected.items() if payload.get(key) != value]
        if p0._strict() and failed:
            raise RuntimeError(f"P2 trainer contract mismatch in {failed}: {payload}")

    DiffusionTrainer.__init__ = p0._mark_wrapper(trainer_init)

    original_train = DiffusionTrainer.train

    @functools.wraps(original_train)
    def train(self: Any, *args: Any, **kwargs: Any) -> Any:
        succeeded = False
        cleanup_error: BaseException | None = None
        try:
            result = original_train(self, *args, **kwargs)
            succeeded = True
            return result
        finally:
            started = time.perf_counter()
            try:
                self.pool.shutdown()
            except BaseException as exc:
                cleanup_error = exc
                if succeeded:
                    raise
            finally:
                p0._guarded_emit(
                    "p2_driver_cleanup",
                    {
                        "pool_shutdown": cleanup_error is None,
                        "training_succeeded": succeeded,
                        "error_type": (None if cleanup_error is None else type(cleanup_error).__name__),
                        "duration_s": time.perf_counter() - started,
                    },
                    rank_override=-1,
                )

    DiffusionTrainer.train = p0._mark_wrapper(train)


def _verify_base_overlay() -> None:
    if p0.SCHEMA != SCHEMA:
        raise RuntimeError(f"P0 runtime event schema mismatch: {p0.SCHEMA!r}")
    expected = _require_env("P2_BASE_OVERLAY_SHA256")
    observed = p0._sha256_file(Path(p0.__file__).resolve())
    if observed != expected:
        raise RuntimeError(f"P0 runtime overlay digest mismatch: expected {expected}, observed {observed}")
    required = (
        "_guarded_emit",
        "_install_topology_and_rank_hooks",
        "_install_data_source_hook",
        "_install_planner_hook",
        "_install_replay_hooks",
        "_install_reward_hooks",
        "_install_train_checkpoint_hooks",
        "_parallel_context",
        "_generation_rows",
    )
    missing = [name for name in required if not callable(getattr(p0, name, None))]
    if missing:
        raise RuntimeError(f"P0 v2 runtime overlay API mismatch: missing {missing}")


def install() -> None:
    """Install P0 v2 correctness hooks plus narrow P2 compatibility hooks."""
    global _INSTALLED
    if _INSTALLED:
        return
    _verify_base_overlay()
    for name in (
        "P0_RECEIPT_DIR",
        "P0_CAMPAIGN_ID",
        "P0_ARM_ID",
        "P0_PHASE",
        "P0_EVIDENCE_CLASS",
        "P0_LAUNCH_UUID",
        "P0_RUN_NAME",
        "P0_NODE",
        "P0_SP_SIZE",
        "P0_EXPECTED_NUM_UPDATES",
        "P0_SOURCE_COMMIT",
        "P0_EXPECTED_SOURCE_COMMIT",
        "P0_DATASET_SHA256",
        "P2_EXPECTED_PREFETCH",
    ):
        _require_env(name)

    installed: list[str] = []
    try:
        p0._install_topology_and_rank_hooks()
        installed.append("topology_rank")
        _install_conditioner_hooks()
        installed.append("p2_conditioner")
        _install_generation_hooks()
        installed.append("p2_generation")
        p0._install_data_source_hook()
        installed.append("data_source")
        p0._install_planner_hook()
        installed.append("planner")
        p0._install_replay_hooks()
        installed.append("replay")
        p0._install_reward_hooks()
        installed.append("reward")
        p0._install_train_checkpoint_hooks()
        installed.append("train_checkpoint")
        _install_contract_hooks()
        installed.append("p2_contracts")
    except Exception as exc:
        p0._guarded_emit(
            "p2_overlay_install_failure",
            {
                "installed_groups": installed,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    _INSTALLED = True
    p0._INSTALLED = True
    p0._guarded_emit(
        "overlay_installed",
        {
            "installed_groups": installed,
            "strict": p0._strict(),
            "adapter": "p2-next-batch-prefetch",
            "base_event_schema": p0.SCHEMA,
        },
    )
    p0._guarded_emit(
        "p2_overlay_installed",
        {
            "schema": SCHEMA,
            "base_overlay_sha256": _require_env("P2_BASE_OVERLAY_SHA256"),
            "planner_class": EXPECTED_PLANNER_CLASS,
            "persistent_disk_cache": False,
            "conditioner_device": "cpu",
            "prefetch_expected": _require_env("P2_EXPECTED_PREFETCH") == "1",
            "finite": math.isfinite(time.monotonic()),
        },
    )


__all__ = ["SCHEMA", "install"]
