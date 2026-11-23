"""Loguru setup shared by scripts, scheduler and dashboard."""

from __future__ import annotations

import sys

from loguru import logger

from sanjotracker.config import PROJECT_ROOT, get_config


def _utf8_console() -> None:
    """Windows consoles default to cp1252, which can't print emoji. (None under pythonw.)"""
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        if (stream.encoding or "").lower() != "utf-8":
            stream.reconfigure(encoding="utf-8", errors="replace")


def setup_logging(level: str | None = None, to_file: bool = True) -> None:
    """Console logging at `level` (default: app.log_level from config), plus a
    rotating DEBUG file in logs/. Call it first thing in every script."""
    _utf8_console()
    logger.remove()  # before loading config, so its debug line isn't printed raw
    level = level or get_config().app.log_level
    if sys.stderr is not None:  # pythonw.exe has no console: file log only
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
