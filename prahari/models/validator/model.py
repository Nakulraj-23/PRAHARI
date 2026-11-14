"""Alarm validator: real fault symptom (1) or noise (0).

Every feature uses only information available at alarm time: alarm type, severity, node type and
vendor, KPI z-scores of the last 5 min vs the previous hour (up to the alarm minute), alarms on
the same node in the last 10 min (flap count), same node+type in the last 10 min, and alarms on
graph neighbours in the last 5 min. Counts use strictly earlier alarms. ``label_*`` columns are
read only in ``train`` to build the target.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from prahari.config import Settings, get_settings
from prahari.db import get_engine
from prahari.models._io import data_hash, load_artifact, new_version, save_artifact
from prahari.pipeline.features import _kpi_z_at
from prahari.pipeline.splits import load_splits
from prahari.pipeline.store import load_alarms, load_kpi
from prahari.sim.constants import ALARM_TYPES, KPI_COLUMNS, NODE_TYPES, VENDORS
from prahari.sim.topology import NetworkModel

NAME = "alarm_validator"
_SEV = {"minor": 1, "major": 2, "critical": 3}
INPUT_COLUMNS = ["alarm_id", "ts", "node_id", "alarm_type", "severity"]
TARGET_RECALL = 0.98


def feature_names() -> list[str]:
    """Model input columns, in order."""
    return (
        ["sev_num"]
        + [f"type_{t}" for t in ALARM_TYPES]
        + [f"ntype_{t}" for t in NODE_TYPES]
        + [f"vendor_{v}" for v in VENDORS]
        + ["flap_node_10m", "flap_type_10m", "neigh_5m"]
        + [f"kpi_z_{k}" for k in KPI_COLUMNS]
        + ["kpi_z_max"]
    )


def _prior_counts(key: np.ndarray, t: np.ndarray, window: float) -> np.ndarray:
    """Per row: rows with the same key and time in [t - window, t), ties broken by input order."""
    n = len(key)
    order = np.lexsort((np.arange(n), t, key))
    k, tt = key[order], t[order]
    comp = k.astype(np.float64) * 1e12 + tt
    hi = np.arange(n)  # rows before me in sorted order (same key & time sorted by input order)
    lo = np.searchsorted(comp, comp - window, side="left")
    out = np.empty(n, dtype=np.int64)
    out[order] = np.maximum(hi - lo, 0)
    return out


def build_features(
    alarms: pd.DataFrame,
    net: NetworkModel,
    kpi_z: np.ndarray | None = None,
    context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Feature matrix for ``alarms`` (columns ``INPUT_COLUMNS``; any ``label_*`` is ignored).

    ``context`` holds earlier alarms (e.g. the last 10 min before the batch) that count toward
    flap and neighbour features but are not scored. ``kpi_z`` is an (n, 6) array of signed KPI
    z-scores at each alarm's time (zeros if omitted).
    """
    a = alarms[INPUT_COLUMNS].reset_index(drop=True)
    n_a = len(a)
    allx = (
        a
        if context is None or context.empty
        else pd.concat([context[INPUT_COLUMNS], a], ignore_index=True)
    )
    allx = allx.reset_index(drop=True)
    ts = pd.to_datetime(allx["ts"])
    sec = ((ts - pd.Timestamp("2000-01-01")) / pd.Timedelta(seconds=1)).to_numpy(np.float64)
    node = allx["node_id"].map(net.index).to_numpy()
    atype = allx["alarm_type"].map({t: i for i, t in enumerate(ALARM_TYPES)}).to_numpy()
    off = len(allx) - n_a
    out = pd.DataFrame(index=range(n_a))
    sl = slice(off, None)
    out["sev_num"] = allx["severity"].map(_SEV).to_numpy()[sl]
    for i, t in enumerate(ALARM_TYPES):
        out[f"type_{t}"] = (atype[sl] == i).astype(np.int8)
    for i, t in enumerate(NODE_TYPES):
        out[f"ntype_{t}"] = (net.type_idx[node[sl]] == i).astype(np.int8)
    for i, v in enumerate(VENDORS):
        out[f"vendor_{v}"] = (net.vendor_idx[node[sl]] == i).astype(np.int8)
    out["flap_node_10m"] = _prior_counts(node, sec, 600)[sl]
    out["flap_type_10m"] = _prior_counts(node * len(ALARM_TYPES) + atype, sec, 600)[sl]
    # neighbours: alarms on 1-hop neighbours in [t - 5 min, t)
    neigh = np.zeros(len(allx), dtype=np.int64)
    order = np.argsort(sec, kind="stable")
    by_node = [np.sort(sec[node == i]) for i in range(net.n)]
    nbrs = [np.flatnonzero(net.adj[i]) for i in range(net.n)]
    for r in range(len(allx)):
        t = sec[r]
        neigh[r] = sum(
            np.searchsorted(by_node[j], t, "left") - np.searchsorted(by_node[j], t - 300, "left")
            for j in nbrs[node[r]]
        )
    del order
    out["neigh_5m"] = neigh[sl]
    z = np.zeros((n_a, len(KPI_COLUMNS))) if kpi_z is None else np.asarray(kpi_z, dtype=float)
    for k, name in enumerate(KPI_COLUMNS):
        out[f"kpi_z_{name}"] = z[:, k]
    out["kpi_z_max"] = np.abs(z).max(axis=1) if n_a else np.zeros(0)
    return out[feature_names()].astype(np.float32)


def alarm_kpi_z(alarms: pd.DataFrame, net: NetworkModel, store) -> np.ndarray:
    """Signed KPI z-scores (n, 6) at each alarm's minute from a KpiStore."""
    node = alarms["node_id"].map(net.index).to_numpy()
    t_idx = np.array([store.minute_index(t) for t in alarms["ts"]])
    return _kpi_z_at(store, t_idx, node)


@dataclass
class ValidatorPredictor:
    """Calibrated alarm validator. ``predict`` is a pure function of its payload."""

    model: object  # fitted classifier with predict_proba
    calibrator: IsotonicRegression
    threshold: float
    net: NetworkModel
    version: str

    def proba(self, feats: pd.DataFrame) -> np.ndarray:
        """Calibrated P(alarm is real) for a feature frame."""
        raw = self.model.predict_proba(feats[feature_names()])[:, 1]
        return self.calibrator.predict(raw)

    def predict(self, payload: dict) -> dict:
        """Score alarms.

        payload: ``{"alarms": [{alarm_id, ts, node_id, alarm_type, severity}, ...],
        "context_alarms": [... earlier alarms, optional],
        "kpi_z": {alarm_id: [6 z-scores]} optional}``.
        Output: ``{"results": [{alarm_id, valid_prob, is_valid_pred}], "model", "latency_ms"}``.
        """
        t0 = time.perf_counter()
        alarms = pd.DataFrame(payload["alarms"])
        if alarms.empty:
            res: list[dict] = []
        else:
            alarms["ts"] = pd.to_datetime(alarms["ts"])
            ctx = pd.DataFrame(payload.get("context_alarms") or [], columns=INPUT_COLUMNS)
            if len(ctx):
                ctx["ts"] = pd.to_datetime(ctx["ts"])
            kz = payload.get("kpi_z")
            z = np.array([kz.get(i, [0.0] * 6) for i in alarms["alarm_id"]]) if kz else None
            p = self.proba(build_features(alarms, self.net, z, ctx))
            res = [
                {
                    "alarm_id": i,
                    "valid_prob": round(float(q), 4),
                    "is_valid_pred": bool(q >= self.threshold),
                }
                for i, q in zip(alarms["alarm_id"], p, strict=True)
            ]
        return {
            "results": res,
            "model": {"name": NAME, "version": self.version},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def predict_window(self, start, end, settings: Settings | None = None) -> pd.DataFrame:
        """DB convenience for serving: score stored alarms in [start, end) (labels never read)."""
        settings = settings or get_settings()
        engine = get_engine(settings)
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        ctx_start = start - pd.Timedelta(minutes=10)
        al = load_alarms(engine, ctx_start, end)
        store = load_kpi(engine, self.net, start - pd.Timedelta(minutes=70), end)
        inwin = (al["ts"] >= start).to_numpy()
        z = alarm_kpi_z(al[inwin], self.net, store)
        feats = build_features(al[inwin], self.net, z, al[~inwin])
        out = al[inwin][["alarm_id", "ts", "node_id", "alarm_type", "severity"]].copy()
        out["valid_prob"] = self.proba(feats)
        out["is_valid_pred"] = out["valid_prob"] >= self.threshold
        return out.reset_index(drop=True)


def load(version: str | None = None, settings: Settings | None = None) -> ValidatorPredictor:
    """Load the newest (or given) validator artifact."""
    settings = settings or get_settings()
    v, o = load_artifact(settings, NAME, version)
    net = NetworkModel(o["nodes"], o["links"])
    return ValidatorPredictor(o["model"], o["calibrator"], o["threshold"], net, v)


def _metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    return {
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
    }


def _load_labelled(
    settings: Settings, net: NetworkModel, engine
) -> tuple[pd.DataFrame, np.ndarray]:
    """Alarms (label-free frame) and the TARGET label, read separately (training only)."""
    al = load_alarms(engine)
    lab = pd.read_sql("SELECT alarm_id, label_is_true FROM alarms", engine)
    y = (
        al[["alarm_id"]]
        .merge(lab, on="alarm_id", how="left")["label_is_true"]
        .to_numpy()
        .astype(int)
    )
    return al, y


def train(settings: Settings | None = None, max_alarms: int | None = None) -> dict:
    """Train LR baseline and HistGradientBoosting, calibrate, choose a recall-favouring threshold.

    Time split from ``artifacts/splits.json`` boundaries. Isotonic calibration and the threshold
    are fitted on validation; test is touched once for reporting. ``max_alarms`` (testing) keeps
    only the first N alarms.
    """
    settings = settings or get_settings()
    engine = get_engine(settings)
    net = NetworkModel.from_db(engine)
    al, y = _load_labelled(settings, net, engine)
    if max_alarms:
        al, y = al.iloc[:max_alarms], y[:max_alarms]
    sp = load_splits(settings)
    t_tr, t_va = (
        pd.Timestamp(sp["boundaries"]["train_end"]),
        pd.Timestamp(sp["boundaries"]["val_end"]),
    )
    if max_alarms:
        q = al["ts"].quantile([0.6, 0.8])
        t_tr, t_va = q.iloc[0], q.iloc[1]
    store = load_kpi(
        engine,
        net,
        al["ts"].min() - pd.Timedelta(minutes=70),
        al["ts"].max() + pd.Timedelta(minutes=1),
    )
    x = build_features(al, net, alarm_kpi_z(al, net, store))
    tr, va = (al["ts"] < t_tr).to_numpy(), ((al["ts"] >= t_tr) & (al["ts"] < t_va)).to_numpy()
    te = (al["ts"] >= t_va).to_numpy()

    lr = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=settings.seed)
    ).fit(x[tr], y[tr])
    with threadpool_limits(1):  # ~5k rows: OpenMP spin-waits dominate (and fight other jobs)
        hgb = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=settings.seed,
        ).fit(x[tr], y[tr])

    out: dict = {}
    cals = {}
    for name, m in (("logreg", lr), ("hgb", hgb)):
        raw_va = m.predict_proba(x[va])[:, 1]
        cals[name] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw_va, y[va])
    p_va = {
        k: cals[k].predict(m.predict_proba(x[va])[:, 1]) for k, m in (("logreg", lr), ("hgb", hgb))
    }
    p_te = {
        k: cals[k].predict(m.predict_proba(x[te])[:, 1]) for k, m in (("logreg", lr), ("hgb", hgb))
    }
    # threshold: largest value that still keeps recall of true alarms >= TARGET_RECALL on val
    thr = {}
    for k in p_va:
        cand = np.unique(p_va[k][y[va] == 1])
        ok = [c for c in cand if (p_va[k][y[va] == 1] >= c).mean() >= TARGET_RECALL]
        thr[k] = float(max(ok)) if ok else 0.5
    for k in p_va:
        out[k] = {
            "test": _metrics(y[te], p_te[k], thr[k]),
            "val": _metrics(y[va], p_va[k], thr[k]),
            "threshold": round(thr[k], 4),
            "test_at_0.5": _metrics(y[te], p_te[k], 0.5),
        }
    # uncalibrated brier for reference
    out["hgb"]["test_uncalibrated_brier"] = round(
        float(brier_score_loss(y[te], hgb.predict_proba(x[te])[:, 1])), 4
    )
    out["n"] = {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}
    out["test_true_rate"] = round(float(y[te].mean()), 4)

    version = new_version()
    card = {
        "name": NAME,
        "version": version,
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "data_hash": data_hash(x.to_numpy(), y),
        "features": feature_names(),
        "metrics": {
            **out["hgb"]["test"],
            "threshold": out["hgb"]["threshold"],
            "threshold_rule": f"max threshold with val recall >= {TARGET_RECALL}",
        },
        "baseline_metrics": {**out["logreg"]["test"], "model": "LogisticRegression"},
        "extra": {
            "n": out["n"],
            "test_true_rate": out["test_true_rate"],
            "seed": settings.seed,
            "hgb_at_0.5": out["hgb"]["test_at_0.5"],
        },
        "intended_use": "Down-rank likely-noise alarms before root-cause analysis; the threshold "
        "favours keeping true alarms (recall) over removing every noise alarm.",
        "limitations": "Trained on synthetic simulator data; separation here is easier than on a "
        "real network. Calibrated and thresholded on one validation period; needs the KPI z-scores "
        "at alarm time (zeros if absent degrades it) and ~10 min of preceding alarms as context.",
    }
    nodes = pd.read_sql("SELECT node_id, node_type, region, vendor FROM nodes", engine)
    links = pd.read_sql("SELECT link_id, src, dst, capacity_mbps FROM links", engine)
    save_artifact(
        settings,
        NAME,
        version,
        {
            "model": hgb,
            "calibrator": cals["hgb"],
            "threshold": thr["hgb"],
            "nodes": nodes,
            "links": links,
        },
        card,
    )
    return {"version": version, **out}
