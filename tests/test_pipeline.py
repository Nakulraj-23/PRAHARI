"""Feature pipeline, quality checks, splits."""

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from prahari.config import Settings
from prahari.db import get_engine
from prahari.pipeline.features import (
    build_alarm_features,
    build_incident_graphs,
    build_node_window_features,
    feature_names,
)
from prahari.pipeline.quality import DataQualityError, run_quality_checks
from prahari.pipeline.splits import load_splits, make_splits
from prahari.pipeline.store import ALARM_COLUMNS, load_alarms, load_kpi
from prahari.sim.constants import FAULT_TYPES
from prahari.sim.generate import build_dataset, write_dataset
from prahari.sim.topology import NetworkModel


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("pipe")
    s = Settings(db_url=f"sqlite:///{d / 'p.db'}", artifacts_dir=d / "art")
    write_dataset(build_dataset(days=3, seed=42), s)
    return s


def test_quality_passes_and_fails_loudly(env):
    rep = run_quality_checks(get_engine(env))
    assert not rep.failures and len(rep.checks) > 30
    eng = get_engine(env)
    with eng.begin() as c:
        c.execute(text("UPDATE kpi_samples SET cpu_pct = 250 WHERE id = 1"))
    with pytest.raises(DataQualityError, match="range:kpi_samples.cpu_pct"):
        run_quality_checks(eng)
    with eng.begin() as c:
        c.execute(text("UPDATE kpi_samples SET cpu_pct = 40 WHERE id = 1"))
        c.execute(
            text(
                "UPDATE kpi_samples SET ts = (SELECT ts FROM kpi_samples WHERE id = 1 + 120) "
                "WHERE id = 1 + 240"
            )
        )
    with pytest.raises(DataQualityError, match="duplicate|monotonic"):
        run_quality_checks(eng)


def test_no_label_columns_in_features(env):
    eng = get_engine(env)
    net = NetworkModel.from_db(eng)
    store = load_kpi(eng, net)
    alarms = load_alarms(eng)
    assert not any(c.startswith("label_") for c in alarms.columns)
    assert set(alarms.columns) == set(ALARM_COLUMNS)
    # even if a caller passes label columns, they are dropped
    raw = pd.read_sql("SELECT * FROM alarms", eng, parse_dates=["ts"]).sort_values("ts")
    assert "label_is_true" in raw.columns
    f = build_alarm_features(raw, net, store)
    assert not any(c.startswith("label_") for c in f.columns)
    assert len(f) == len(raw) and np.isfinite(f.drop(columns=["alarm_id", "ts"]).to_numpy()).all()
    w = build_node_window_features("2026-10-02 10:00", "2026-10-02 10:15", env)
    assert not any(c.startswith("label_") for c in w.columns)
    assert w.shape[0] == 120 and np.isfinite(w.to_numpy()).all()
    assert not any(c.startswith("label_") for c in feature_names())


def test_incident_graphs_and_split(env):
    splits = make_splits(env)
    assert load_splits(env)["train"] == splits["train"]
    eng = get_engine(env)
    inc = pd.read_sql("SELECT incident_id, start_ts FROM incidents", eng, parse_dates=["start_ts"])
    st = inc.set_index("incident_id").start_ts
    assert st[splits["train"]].max() <= st[splits["val"]].min()
    assert st[splits["val"]].max() <= st[splits["test"]].min()
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(inc)
    graphs = build_incident_graphs(env, split="test", with_ids=True)
    assert len(graphs) == len(splits["test"])
    _, x, a, y_root, y_fault = graphs[0]
    assert x.shape == (120, len(feature_names())) and a.shape == (120, 120)
    assert (a == a.T).all() and 0 <= y_root < 120 and 0 <= y_fault < len(FAULT_TYPES)
    # the root node carries more alarm evidence than the typical node in most incidents
    names = feature_names()
    col = names.index("alarm_total")
    wins = sum(g[1][g[3], col] > np.median(g[1][:, col]) for g in graphs)
    assert wins > 0.6 * len(graphs)
