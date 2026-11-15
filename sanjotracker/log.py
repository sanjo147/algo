"""Loguru setup shared by scripts, scheduler and dashboard."""

from __future__ import annotations

import sys

from loguru import logger

from sanjotracker.config import PROJECT_ROOT


def setup_logging(level: str = "INFO", to_file: bool = True) -> None:
    """Console logging at `level`, plus a rotating file in logs/ at DEBUG."""
    logger.remove()
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
