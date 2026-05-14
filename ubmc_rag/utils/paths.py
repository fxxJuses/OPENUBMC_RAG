"""Path resolution helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_data_dir(config_path: str) -> Path:
    """Resolve the data directory relative to config file location."""
    return Path(config_path).parent / "data"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
