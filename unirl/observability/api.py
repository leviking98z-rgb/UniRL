"""Backend-neutral observability contract consumed by trainer programs."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Protocol, runtime_checkable


def emit_progress(
    rollout_id: int,
    num_rollouts: int,
    results: Any,
    mean_reward: float,
    *,
    extra: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Emit the provider-independent one-line rollout progress summary."""
    log = logger if logger is not None else logging.getLogger(__name__)

    def metric(metrics: Any, key: str) -> Optional[float]:
        value = (metrics or {}).get(key) if metrics is not None else None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def format_result(result: Any) -> str:
        parts = f"loss={result.loss:.4f} gn={result.grad_norm:.4f} lr={result.lr:.2e}"
        metrics = getattr(result, "metrics", None)
        ratio_mean = metric(metrics, "ratio_mean")
        ratio_std = metric(metrics, "ratio_std")
        clip_fraction = metric(metrics, "clip_fraction")
        if ratio_mean is not None:
            parts += f" ratio={ratio_mean:.4f}"
            if ratio_std is not None:
                parts += f"±{ratio_std:.4f}"
        if clip_fraction is not None:
            parts += f" clip={clip_fraction:.2f}"
        k3_mean = metric(metrics, "k3_mean")
        absdiff_mean = metric(metrics, "rollout_replay_logp_absdiff_mean")
        if k3_mean is not None:
            parts += f" k3={k3_mean:.2e}"
        if absdiff_mean is not None:
            parts += f" |Δlogp|={absdiff_mean:.2e}"
        return parts

    if isinstance(results, dict):
        body = "  ".join(f"{name}[{format_result(result)}]" for name, result in results.items())
    else:
        body = format_result(results)
    suffix = ("  " + " ".join(f"{key}={value}" for key, value in extra.items())) if extra else ""
    log.info(
        "rollout %d/%d  reward=%.4f  %s%s",
        rollout_id + 1,
        num_rollouts,
        mean_reward,
        body,
        suffix,
    )


@runtime_checkable
class Observer(Protocol):
    """Structural interface for metrics, media, progress, and run state."""

    run_id: Optional[str]
    media_max_items: int

    @property
    def initialized(self) -> bool: ...

    @property
    def optimizer_step(self) -> int: ...

    def bind_memory_monitor(self, monitor: Any) -> None: ...

    def should_log_media(self, rollout_id: int) -> bool: ...

    def log_generated_media(self, step: int, media_preview: Any, **kwargs: Any) -> None: ...

    def log_rollout_step(
        self,
        rollout_id: int,
        results: Any,
        sample: Any,
        **kwargs: Any,
    ) -> None: ...

    def log_rollout(self, rollout_id: int, metrics: Dict[str, Any]) -> None: ...

    def log_step(self, step: int, metrics: Dict[str, Any], prefix: str = "train/") -> None: ...

    def log_eval(self, step: int, metrics: Dict[str, Any]) -> None: ...

    def log_progress(
        self,
        rollout_id: int,
        num_rollouts: int,
        results: Any,
        mean_reward: float,
        *,
        extra: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None: ...

    def finish(self) -> None: ...


class NullObserver:
    """Disabled observer that preserves console and memory-monitor behavior."""

    def __init__(
        self,
        *,
        run_id: Optional[str] = None,
        optimizer_step: int = 0,
        media_max_items: int = 8,
    ) -> None:
        self.run_id = run_id
        self.media_max_items = max(1, int(media_max_items))
        self._optimizer_step = int(optimizer_step)
        self._memory_monitor: Any = None

    @property
    def initialized(self) -> bool:
        return False

    @property
    def optimizer_step(self) -> int:
        return self._optimizer_step

    def bind_memory_monitor(self, monitor: Any) -> None:
        self._memory_monitor = monitor

    def should_log_media(self, rollout_id: int) -> bool:
        return False

    def log_generated_media(self, step: int, media_preview: Any, **kwargs: Any) -> None:
        return None

    def log_rollout_step(
        self,
        rollout_id: int,
        results: Any,
        sample: Any,
        **kwargs: Any,
    ) -> None:
        if self._memory_monitor is not None:
            self._memory_monitor.step_summary(step=rollout_id + 1)

    def log_rollout(self, rollout_id: int, metrics: Dict[str, Any]) -> None:
        return None

    def log_step(self, step: int, metrics: Dict[str, Any], prefix: str = "train/") -> None:
        return None

    def log_eval(self, step: int, metrics: Dict[str, Any]) -> None:
        return None

    def log_progress(
        self,
        rollout_id: int,
        num_rollouts: int,
        results: Any,
        mean_reward: float,
        *,
        extra: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        emit_progress(
            rollout_id,
            num_rollouts,
            results,
            mean_reward,
            extra=extra,
            logger=logger,
        )

    def finish(self) -> None:
        return None


def observer_state_dict(observer: Observer) -> Dict[str, Any]:
    """Checkpoint provider-neutral run identity and metric step state.

    ``wandb_run_id`` remains as a compatibility alias so an older UniRL checkout
    can resume a checkpoint produced after the observer API migration.
    """

    return {
        "observer_run_id": observer.run_id,
        "wandb_run_id": observer.run_id,
        "optimizer_step": observer.optimizer_step,
    }


__all__ = ["NullObserver", "Observer", "emit_progress", "observer_state_dict"]
