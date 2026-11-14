"""Artifact helpers shared by model packages: versioning, joblib files, model cards."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prahari.config import Settings


def new_version() -> str:
    """UTC timestamp version string, e.g. ``20261114T093000``."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def data_hash(*frames: pd.DataFrame | np.ndarray) -> str:
    """Short, stable hash of the training data (values only)."""
    h = hashlib.sha256()
    for f in frames:
        arr = f.to_numpy() if isinstance(f, pd.DataFrame) else np.asarray(f)
        h.update(np.ascontiguousarray(arr.astype(np.float64, copy=False)).tobytes())
    return h.hexdigest()[:16]


def save_artifact(settings: Settings, name: str, version: str, obj: dict, card: dict) -> Path:
    """Write ``artifacts/<name>/<version>/model.joblib`` and ``<version>.card.json``."""
    root = Path(settings.artifacts_dir) / name
    (root / version).mkdir(parents=True, exist_ok=True)
    path = root / version / "model.joblib"
    joblib.dump(obj, path)
    (root / f"{version}.card.json").write_text(json.dumps(card, indent=1, default=float))
    return path


def resolve_version(settings: Settings, name: str, version: str | None) -> str:
    """Return ``version`` or the newest version directory that has a model file."""
    root = Path(settings.artifacts_dir) / name
    if version:
        return version
    found = sorted(p.name for p in root.glob("*") if (p / "model.joblib").exists())
    if not found:
        raise FileNotFoundError(f"no trained {name} artifact under {root}; run its train module")
    return found[-1]


def load_artifact(settings: Settings, name: str, version: str | None) -> tuple[str, dict]:
    """Load the joblib dict for ``name`` and return ``(version, obj)``."""
    v = resolve_version(settings, name, version)
    return v, joblib.load(Path(settings.artifacts_dir) / name / v / "model.joblib")
