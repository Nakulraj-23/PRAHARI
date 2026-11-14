"""Time-based train/val/test split of incidents (by incident start time)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from prahari.config import Settings, get_settings
from prahari.db import get_engine


def splits_path(settings: Settings) -> Path:
    return Path(settings.artifacts_dir) / "splits.json"


def make_splits(settings: Settings | None = None, train: float = 0.70, val: float = 0.15) -> dict:
    """Split incidents chronologically and write ``artifacts/splits.json``.

    Boundaries are incident start times, so alarms and KPI windows can be split by time too.
    """
    settings = settings or get_settings()
    inc = pd.read_sql(
        "SELECT incident_id, start_ts FROM incidents ORDER BY start_ts, incident_id",
        get_engine(settings),
        parse_dates=["start_ts"],
    )
    n = len(inc)
    i_tr, i_va = int(n * train), int(n * (train + val))
    out = {
        "method": "chronological by incident start_ts",
        "fractions": {"train": train, "val": val, "test": round(1 - train - val, 4)},
        "boundaries": {
            "train_end": inc["start_ts"].iloc[i_tr].isoformat(),
            "val_end": inc["start_ts"].iloc[i_va].isoformat(),
        },
        "train": inc["incident_id"].iloc[:i_tr].tolist(),
        "val": inc["incident_id"].iloc[i_tr:i_va].tolist(),
        "test": inc["incident_id"].iloc[i_va:].tolist(),
    }
    path = splits_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def load_splits(settings: Settings | None = None) -> dict:
    """Read ``artifacts/splits.json`` (run ``make seed`` first)."""
    settings = settings or get_settings()
    return json.loads(splits_path(settings).read_text())
