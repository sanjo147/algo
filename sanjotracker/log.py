"""Loguru setup shared by scripts, scheduler and dashboard."""

from __future__ import annotations

import sys

from loguru import logger

from sanjotracker.config import PROJECT_ROOT, get_config


def setup_logging(level: str | None = None, to_file: bool = True) -> None:
    """Console logging at `level` (default: app.log_level from config), plus a
    rotating DEBUG file in logs/. Call it first thing in every script."""
    logger.remove()  # before loading config, so its debug line isn't printed raw
    level = level or get_config().app.log_level
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )
    if to_file:
        logger.add(
            PROJECT_ROOT / "logs" / "sanjotracker.log",
            level="DEBUG",
            rotation="5 MB",
            retention=5,
            encoding="utf-8",
        )
