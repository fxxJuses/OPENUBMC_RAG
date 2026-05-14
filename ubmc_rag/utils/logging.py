"""Logging utilities."""

from __future__ import annotations

import logging

import rich.logging


def setup_logging(level: str = "INFO") -> None:
    handler = rich.logging.RichHandler(show_time=False, show_path=False, markup=True)
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        handlers=[handler],
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
