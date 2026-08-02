"""Standard-library logging configuration for UniRL processes."""

from __future__ import annotations

import logging
import os
from typing import Optional


def configure_logger(
    level: int = logging.INFO,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
) -> None:
    """Configure console and optional file logging for a training run."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=level, format=format_str, handlers=handlers)
    logging.getLogger("ray").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


__all__ = ["configure_logger"]
