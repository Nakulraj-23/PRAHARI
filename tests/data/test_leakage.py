"""Leakage audit: can alarm validity or root cause be read off trivially?

Measured on the 30-day seed-42 DB (Q1 handoff): depth-2 tree on kpi_z_max alone AUC 0.976 vs
0.998 for gradient boosting on all features; 9 of 15 alarm types are label-pure. Tests that
encode a known shortcut were xfail(strict); A1 fixed them (Q1 F1-F3) and the markers are gone.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from prahari.pipeline.features import _window_features, build_alarm_features, feature_names
from tests.data.conftest import time_split_mask


def _tree_auc(x: np.ndarray, y: np.ndarray, tr: np.ndarray, depth: int = 2) -> float:
    m = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(x[tr], y[tr])
    return float(roc_auc_score(y[~tr], m.predict_proba(x[~tr])[:, 1]))


@pytest.fixture(scope="module")
def alarm_xy(ds7):
    ds, store = ds7
    df = ds.alarms.reset_index(drop=True)
    f = build_alarm_features(df, ds.net, store).drop(columns=["alarm_id", "ts"])
    return df, f, df["label_is_true"].astype(int).to_numpy(), time_split_mask(df["ts"])


def test_alarm_id_order_carries_no_validity_signal(plan30):
    df = plan30.alarms.reset_index(drop=True)
    y = df["label_is_true"].astype(int).to_numpy()
    ordinal = df["alarm_id"].str[4:].astype(int).to_numpy()[:, None].astype(float)
    assert _tree_auc(ordinal, y, time_split_mask(df["ts"])) < 0.55


def test_raw_text_has_no_label_tokens(plan30):
    df = plan30.alarms
    inc_ids = {f.incident_id for f in plan30.faults}
    text = " ".join(df["raw_text"]).lower()
    assert "label" not in text and "incident" not in text and "noise" not in text
    assert not any(i in text.upper() for i in list(inc_ids)[:50])
    y = df["label_is_true"].astype(int).to_numpy()
    cv = CountVectorizer(token_pattern=r"[A-Za-z_]+|\d+", binary=True, min_df=5)
    x = cv.fit_transform(df["raw_text"]).toarray().astype(float)
    # tokens encode alarm type/severity, so some signal is expected; not more than type itself
    assert _tree_auc(x, y, time_split_mask(df["ts"])) < 0.80


def test_no_alarm_type_is_label_pure(plan30):
    ct = pd.crosstab(plan30.alarms["alarm_type"], plan30.alarms["label_is_true"])
    assert ((ct == 0).sum(axis=1) == 0).all()


def test_critical_severity_is_not_label_pure(plan30):
    a = plan30.alarms
    assert (~a.loc[a["severity"] == "critical", "label_is_true"].astype(bool)).any()


def test_single_kpi_feature_does_not_solve_validation(alarm_xy):
    _, f, y, tr = alarm_xy
    auc = _tree_auc(f[["kpi_z_max"]].to_numpy(float), y, tr)
    assert auc < 0.90, f"depth-2 tree on kpi_z_max alone: AUC {auc:.3f}"


def _graphs(ds, store, shift_min: float = 0.0):
    alarms = ds.alarms.drop(columns=["label_is_true", "label_incident_id"])
    names = feature_names()
    xs, ys = [], []
    for f in ds.faults:
        st = f.start - pd.Timedelta(minutes=shift_min)
        if st - pd.Timedelta(minutes=120) < ds.start:
            continue
        x = _window_features(ds.net, store, alarms, st, st + pd.Timedelta(minutes=15))
        xs.append(x[names].to_numpy(np.float32))
        ys.append(f.root_idx)
    return np.stack(xs), np.array(ys)


def test_root_not_trivially_max_latency_delta(ds7):
    ds, store = ds7
    x, y = _graphs(ds, store)
    k = feature_names().index("d_latency_ms")
    top1 = float((x[:, :, k].argmax(axis=1) == y).mean())
    assert top1 < 0.6, f"argmax d_latency_ms finds the root in {top1:.1%} of incidents"


def test_window_features_ignore_labels(ds7):
    """Shuffling the label columns must not change any feature."""
    ds, store = ds7
    df = ds.alarms.reset_index(drop=True)
    shuffled = df.copy()
    rng = np.random.default_rng(0)
    shuffled["label_is_true"] = rng.permutation(df["label_is_true"].to_numpy())
    shuffled["label_incident_id"] = rng.permutation(df["label_incident_id"].to_numpy())
    a = build_alarm_features(df, ds.net, store)
    b = build_alarm_features(shuffled, ds.net, store)
    pd.testing.assert_frame_equal(a, b)
    st = ds.faults[10].start
    w1 = _window_features(ds.net, store, df, st, st + pd.Timedelta(minutes=15))
    w2 = _window_features(ds.net, store, shuffled, st, st + pd.Timedelta(minutes=15))
    pd.testing.assert_frame_equal(w1, w2)
    assert not any(c.startswith("label_") for c in a.columns.tolist() + w1.columns.tolist())
