"""Logging configuration for MOSADE experiments."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "mosade",
    level: int = logging.INFO,
    log_file: Path | str | None = None,
) -> logging.Logger:
    """Configure and return a logger.

    Parameters
    ----------
    name : str
        Logger name.
    level : int
        Logging level.
    log_file : Path or str, optional
        If given, also write log output to this file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    has_console = any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_console:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file).resolve()
        has_file = any(
            isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == log_path
            for handler in logger.handlers
        )
        if not has_file:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_path), encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
