"""Shared fixtures for the Q1 data audit tests: planned datasets and a small seeded DB."""

import shutil

import numpy as np
import pandas as pd
import pytest

from prahari.config import Settings
from prahari.pipeline.splits import make_splits
from prahari.pipeline.store import KpiStore
from prahari.sim.generate import build_dataset, write_dataset


@pytest.fixture(scope="session")
def plan30():
    """30-day plan (topology, faults, alarms; KPIs not materialised). ~0.2 s."""
    return build_dataset(days=30, seed=42)


@pytest.fixture(scope="session")
def ds7():
    """7-day dataset with an in-memory KPI store (no DB)."""
    ds = build_dataset(days=7, seed=42)
    blocks = [b for _, b in ds.kpi_chunks()]
    store = KpiStore(ds.start, np.concatenate(blocks), ds.net.node_ids)
    return ds, store


@pytest.fixture(scope="session")
def small_db(tmp_path_factory):
    """2-day seeded SQLite DB with splits; treat as read-only (copy before corrupting)."""
    d = tmp_path_factory.mktemp("q1db")
    s = Settings(db_url=f"sqlite:///{d / 'q1.db'}", artifacts_dir=d / "art")
    write_dataset(build_dataset(days=2, seed=42), s)
    make_splits(s)
    return s, d / "q1.db"


@pytest.fixture()
def db_copy(small_db, tmp_path):
    """A private copy of the small DB, safe to corrupt."""
    _, path = small_db
    dst = tmp_path / "copy.db"
    shutil.copy(path, dst)
    return Settings(db_url=f"sqlite:///{dst}", artifacts_dir=tmp_path / "art")


def time_split_mask(ts: pd.Series, frac: float = 0.7) -> np.ndarray:
    """True for rows before the ``frac`` time quantile (chronological train mask)."""
    cut = ts.sort_values().iloc[int(len(ts) * frac)]
    return (ts < cut).to_numpy()
