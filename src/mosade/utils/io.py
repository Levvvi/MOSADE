"""I/O helpers for loading configs and saving results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and return its contents as a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    return cfg


def make_run_dir(base: str | Path, tag: str | None = None) -> Path:
    """Create a timestamped directory under *base* for a single experiment run.

    Layout: base / YYYYMMDD_HHMMSS[_tag] /
    """
    base = Path(base)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{tag}" if tag else ts
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_objectives(path: str | Path, objectives: np.ndarray) -> None:
    """Save an (N, M) objective matrix as a space-delimited text file."""
    np.savetxt(str(path), objectives, fmt="%.12e")


def save_json(path: str | Path, data: dict) -> None:
    """Save a dict as a JSON file (numpy types are coerced)."""

    class _Enc(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_Enc)
