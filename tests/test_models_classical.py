"""A2: anomaly detector and alarm validator."""

import inspect
import json
import time

import numpy as np
import pandas as pd
import pytest

from prahari.config import Settings
from prahari.db import get_engine
from prahari.models.anomaly import load as load_anomaly
from prahari.models.anomaly import model as anomaly_mod
from prahari.models.anomaly import train as train_anomaly
from prahari.models.validator import build_features
from prahari.models.validator import load as load_validator
from prahari.models.validator import model as validator_mod
from prahari.models.validator import train as train_validator
from prahari.pipeline.splits import make_splits
from prahari.pipeline.store import load_alarms, load_kpi
from prahari.sim.generate import build_dataset, write_dataset
from prahari.sim.topology import NetworkModel


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    d = tmp_path_factory.mktemp("a2")
    s = Settings(db_url=f"sqlite:///{d / 'a2.db'}", artifacts_dir=d / "art")
    write_dataset(build_dataset(days=3, seed=42), s)
    make_splits(s)
    return s


@pytest.fixture(scope="module")
def trained(env):
    t0 = time.perf_counter()
    v = train_validator(env)
    a = train_anomaly(env)
    return {"v": v, "a": a, "secs": time.perf_counter() - t0}


def test_tiny_training_is_fast_and_writes_artifacts(env, trained):
    assert trained["secs"] < 20
    for name, ver in (
        ("alarm_validator", trained["v"]["version"]),
        ("anomaly_iforest", trained["a"]["version"]),
    ):
        root = env.artifacts_dir / name
        assert (root / ver / "model.joblib").exists()
        card = json.loads((root / f"{ver}.card.json").read_text())
        assert {
            "name",
            "version",
            "trained_at",
            "data_hash",
            "features",
            "metrics",
            "baseline_metrics",
            "intended_use",
            "limitations",
        } <= card.keys()


def test_validator_predict_deterministic(env, trained):
    pred = load_validator(settings=env)
    eng = get_engine(env)
    al = load_alarms(eng).head(40)
    payload = {
        "alarms": al[["alarm_id", "ts", "node_id", "alarm_type", "severity"]]
        .astype({"ts": str})
        .to_dict("records")
    }
    a, b = pred.predict(payload), pred.predict(payload)
    assert a["results"] == b["results"] and len(a["results"]) == 40
    assert all(0 <= r["valid_prob"] <= 1 for r in a["results"])


def test_anomaly_predict_deterministic(env, trained):
    pred = load_anomaly(settings=env)
    eng = get_engine(env)
    net = NetworkModel.from_db(eng)
    st = load_kpi(eng, net, None, None)
    payload = {
        "kpi": {n: st.values[300:400, i, :].tolist() for i, n in enumerate(net.node_ids[:5])}
    }
    a, b = pred.predict(payload), pred.predict(payload)
    assert a["scores"] == b["scores"] and len(a["scores"]) == 5


def test_validator_features_ignore_labels(env):
    eng = get_engine(env)
    net = NetworkModel.from_db(eng)
    al = load_alarms(eng).head(200)
    base = build_features(al, net)
    withlab = al.assign(
        label_is_true=np.random.default_rng(0).integers(0, 2, len(al)), label_incident_id="X"
    )
    pd.testing.assert_frame_equal(base, build_features(withlab, net))
    assert not [c for c in base.columns if c.startswith("label_")]


def test_features_use_only_past_alarms(env):
    """Counts for an alarm do not change when later alarms are removed."""
    eng = get_engine(env)
    net = NetworkModel.from_db(eng)
    al = load_alarms(eng).head(150).reset_index(drop=True)
    full = build_features(al, net)
    cut = build_features(al.iloc[:100], net)
    pd.testing.assert_frame_equal(full.iloc[:100].reset_index(drop=True), cut)


def test_source_never_reads_label_columns_in_features():
    for mod in (validator_mod.build_features, anomaly_mod.rolling_features):
        assert "label_" not in inspect.getsource(mod)
