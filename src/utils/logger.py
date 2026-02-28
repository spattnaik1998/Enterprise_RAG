"""Structured logging setup using Loguru."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_level: str = "INFO", log_file: str = "logs/pipeline.log") -> None:
    """
    Configure loguru for the pipeline.

    - Console: coloured, human-readable
    - File: structured, rotating, compressed
    """
    logger.remove()

    # Console - coloured
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File - rotating, compressed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,
    )

    logger.info(f"Logger initialised | level={log_level} | file={log_file}")
