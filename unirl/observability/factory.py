"""Build an Observer from the stable ``logging`` recipe block."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

from .api import NullObserver, Observer


def _tags(value: Any) -> Optional[list[str]]:
    if isinstance(value, str):
        tags = [tag.strip() for tag in value.split(",") if tag.strip()]
        return tags or None
    if value:
        tags = [str(tag).strip() for tag in value if str(tag).strip()]
        return tags or None
    return None


def create_observer(
    logging_cfg: Optional[Mapping[str, Any]],
    *,
    run_config: Dict[str, Any],
    resume_state: Optional[Mapping[str, Any]] = None,
    rank: int = 0,
) -> Observer:
    """Create the configured sink without exposing its provider to trainers.

    Existing recipes implicitly select ``wandb`` through
    ``report_to_wandb: true``. ``provider: none|wandb`` and generic ``enabled``
    are also accepted so the framework-facing contract is no longer named after
    one sink.
    """

    cfg = logging_cfg or {}
    report_to_wandb = bool(cfg.get("report_to_wandb", False))
    provider = str(cfg.get("provider") or ("wandb" if report_to_wandb else "none")).strip().lower()
    if provider not in {"none", "wandb"}:
        raise ValueError(f"Unknown observability provider {provider!r}; expected 'none' or 'wandb'.")

    project = cfg.get("project_name")
    enabled = provider == "wandb" and bool(cfg.get("enabled", report_to_wandb)) and bool(project)
    state = resume_state or {}
    run_id = state.get("observer_run_id") or state.get("wandb_run_id")
    optimizer_step = int(state.get("optimizer_step") or 0)
    media_max_items = int(cfg.get("media_max_items", 8))
    if not enabled:
        return NullObserver(
            run_id=run_id,
            optimizer_step=optimizer_step,
            media_max_items=media_max_items,
        )

    from unirl.utils.wandb_logger import init_logger

    return init_logger(
        project=str(project) if project else None,
        run_name=cfg.get("run_name"),
        config=run_config,
        log_dir=cfg.get("logging_dir"),
        rank=rank,
        tags=_tags(cfg.get("tags")),
        entity=(cfg.get("entity") or os.environ.get("WANDB_ENTITY") or None),
        log_media=bool(cfg.get("log_media", False)),
        media_max_items=media_max_items,
        media_log_interval=int(cfg.get("media_log_interval", 1)),
        enabled=enabled,
        run_id=run_id,
        optimizer_step=optimizer_step,
    )


__all__ = ["create_observer"]
