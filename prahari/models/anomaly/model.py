"""Per-node KPI anomaly detection (IsolationForest) with a 3-sigma baseline.

Features at minute t for node n and KPI k (causal, only data up to t):
``z5``  = (mean of last 5 min - mean of minutes [t-65, t-5)) / std of that baseline,
``z15`` = same with a 15 min recent mean. Std is floored at 2% of the baseline mean.
Ground truth (incidents, alarm labels) is used ONLY to build evaluation targets in ``train``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve

from prahari.config import Settings, get_settings
from prahari.db import get_engine
from prahari.models._io import (
    data_hash,
    load_artifact,
    new_version,
    save_artifact,
)
from prahari.pipeline.splits import load_splits
from prahari.pipeline.store import load_kpi
from prahari.sim.constants import KPI_COLUMNS
from prahari.sim.topology import NetworkModel

NAME = "anomaly_iforest"
BASE = 60  # baseline minutes
GAP = 75  # baseline ends GAP minutes before t (> longest fault, 70 min)
STEP = 5  # evaluation grid (minutes)
NEED = BASE + GAP  # minutes of history needed before t
RECENT = (5, 15)


def feature_names() -> list[str]:
    """Order of the 14 features: z5 per KPI, z15 per KPI, then max|z5| and max|z15|."""
    return [f"z{w}_{k}" for w in RECENT for k in KPI_COLUMNS] + [f"zmax{w}" for w in RECENT]


def _window_sum(c: np.ndarray, hi: np.ndarray, w: int) -> np.ndarray:
    """Sum over minutes [hi-w+1, hi] from cumulative sums (c[i] = sum of first i)."""
    return c[hi + 1] - c[hi + 1 - w]


def rolling_features(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Features at grid indices ``idx`` of a (T, N, 6) tensor -> (len(idx), N, 14).

    Requires ``idx >= NEED - 1``. NaN samples are ignored in means and stds.
    """
    nk = values.shape[2]
    out = np.zeros((len(idx), values.shape[1], 2 * nk + len(RECENT)), dtype=np.float32)
    for k in range(values.shape[2]):
        v = values[:, :, k].astype(np.float64)
        ok = ~np.isnan(v)
        v = np.where(ok, v, 0.0)
        z = np.zeros((1, v.shape[1]))
        cs = np.concatenate([z, np.cumsum(v, axis=0)])
        cq = np.concatenate([z, np.cumsum(v * v, axis=0)])
        cn = np.concatenate([z, np.cumsum(ok.astype(np.float64), axis=0)])
        b_hi = idx - GAP
        n_b = np.maximum(_window_sum(cn, b_hi, BASE), 1)
        mu = _window_sum(cs, b_hi, BASE) / n_b
        sd = np.sqrt(np.maximum(_window_sum(cq, b_hi, BASE) / n_b - mu**2, 0))
        sd = np.maximum(sd, np.maximum(0.02 * np.abs(mu), 1e-3))
        for j, w in enumerate(RECENT):
            n_c = _window_sum(cn, idx, w)
            cur = _window_sum(cs, idx, w) / np.maximum(n_c, 1)
            z5 = np.where(n_c > 0, (cur - mu) / sd, 0.0)
            out[:, :, j * nk + k] = np.clip(z5, -50, 50)
    for j in range(len(RECENT)):
        out[:, :, 2 * nk + j] = np.abs(out[:, :, j * nk : (j + 1) * nk]).max(axis=2)
    return out


@dataclass
class AnomalyPredictor:
    """Scores node KPI histories. ``predict`` is a pure function of its payload."""

    model: IsolationForest
    threshold: float
    node_ids: list[str]
    version: str
    sigma_threshold: float = 3.0

    def score_matrix(self, values: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Anomaly scores (higher = more anomalous) and 3-sigma max|z5| for (T,N,6) values."""
        f = rolling_features(values, idx)
        n = f.shape[1]
        s = -self.model.score_samples(f.reshape(-1, f.shape[2])).reshape(len(idx), n)
        zmax = np.abs(f[:, :, : len(KPI_COLUMNS)]).max(axis=2)
        return s, zmax

    def predict(self, payload: dict) -> dict:
        """Score nodes at the END of the supplied history.

        payload: ``{"kpi": {node_id: [[latency, loss, thr, cpu, mem, temp], ...]}}`` with one row
        per minute, oldest first, at least 65 rows (more are ignored from the left), NaN allowed.
        Returns per-node ``score``, ``is_anomaly`` (score >= validation-chosen threshold),
        ``z_max`` (3-sigma baseline) and the model block.
        """
        t0 = time.perf_counter()
        nodes = list(payload["kpi"])
        arrs = [np.asarray(payload["kpi"][n], dtype=np.float64) for n in nodes]
        t = min(a.shape[0] for a in arrs)
        if t < NEED:
            raise ValueError(f"need at least {NEED} minutes of KPI history, got {t}")
        vals = np.stack([a[-t:] for a in arrs], axis=1)
        s, zmax = self.score_matrix(vals, np.array([t - 1]))
        scores = {
            n: {
                "score": float(s[0, i]),
                "is_anomaly": bool(s[0, i] >= self.threshold),
                "z_max": float(zmax[0, i]),
            }
            for i, n in enumerate(nodes)
        }
        return {
            "scores": scores,
            "model": {"name": NAME, "version": self.version},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    def predict_window(self, start, end, settings: Settings | None = None, store=None) -> dict:
        """DB convenience for serving: per-node max score over grid points in [start, end]."""
        settings = settings or get_settings()
        engine = get_engine(settings)
        net = NetworkModel.from_db(engine)
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        if store is None:
            store = load_kpi(
                engine, net, start - pd.Timedelta(minutes=NEED), end + pd.Timedelta(minutes=1)
            )
        lo = max(store.minute_index(start), NEED - 1)
        hi = min(store.minute_index(end), store.n_minutes - 1)
        idx = np.arange(lo, hi + 1, STEP) if hi >= lo else np.array([store.n_minutes - 1])
        s, zmax = self.score_matrix(store.values, idx)
        return {
            "window": {"start": start.isoformat(), "end": end.isoformat()},
            "scores": {
                n: {
                    "score": float(s[:, i].max()),
                    "is_anomaly": bool(s[:, i].max() >= self.threshold),
                    "z_max": float(zmax[:, i].max()),
                }
                for i, n in enumerate(store.node_ids)
            },
            "model": {"name": NAME, "version": self.version},
        }


def load(version: str | None = None, settings: Settings | None = None) -> AnomalyPredictor:
    """Load the newest (or given) anomaly artifact."""
    settings = settings or get_settings()
    v, obj = load_artifact(settings, NAME, version)
    return AnomalyPredictor(obj["model"], obj["threshold"], obj["node_ids"], v)


VISIBLE = 0.15  # simulated KPI effect (impact x gain) at or above this counts as KPI-visible


def _targets(engine, net, store, idx, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Evaluation targets from ground truth (never a feature).

    y is True where minute t lies in an incident and the node's SIMULATED KPI effect
    (propagation impact x per-node gain) is >= ``VISIBLE``; KPI-silent nodes are negatives
    because nothing in the KPIs could reveal them. ``since`` holds minutes since the incident
    start for positives (-1 elsewhere).
    """
    from prahari.sim.faults import fault_gain, propagate

    inc = pd.read_sql(
        "SELECT incident_id, start_ts, end_ts, root_node_id, fault_type FROM incidents",
        engine,
        parse_dates=["start_ts", "end_ts"],
    )
    y = np.zeros((len(idx), net.n), dtype=bool)
    since = np.full(y.shape, -1, dtype=np.int32)
    for r in inc.itertuples():
        ri = net.index[r.root_node_id]
        eff = propagate(net, ri, r.fault_type) * fault_gain(net, r.incident_id, ri, seed)
        cols = np.flatnonzero(eff >= VISIBLE)
        a_min = store.minute_index(r.start_ts)
        a = int(np.searchsorted(idx, a_min, side="left"))
        b = int(np.searchsorted(idx, store.minute_index(r.end_ts), side="right"))
        if b > a and len(cols):
            y[a:b, cols] = True
            since[a:b, cols] = (idx[a:b] - a_min)[:, None]
    return y, since


def _prf(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = s >= thr
    tp = int((pred & y).sum())
    p = tp / max(int(pred.sum()), 1)
    r = tp / max(int(y.sum()), 1)
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / max(p + r, 1e-12), 4),
    }


def _best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f))])


def train(settings: Settings | None = None, days: float | None = None) -> dict:
    """Train IsolationForest, choose the threshold on validation, evaluate on test.

    ``days`` (testing) limits the KPI range loaded from the start of the DB. Returns metrics and
    writes ``artifacts/anomaly_iforest/<version>/model.joblib`` plus the model card.
    """
    settings = settings or get_settings()
    engine = get_engine(settings)
    net = NetworkModel.from_db(engine)
    sp = load_splits(settings)
    t_tr, t_va = (
        pd.Timestamp(sp["boundaries"]["train_end"]),
        pd.Timestamp(sp["boundaries"]["val_end"]),
    )
    end = None
    if days is not None:
        first = pd.read_sql("SELECT MIN(ts) AS m FROM kpi_samples", engine, parse_dates=["m"])["m"][
            0
        ]
        end = first + pd.Timedelta(days=days)
    store = load_kpi(engine, net, None, end)
    idx = np.arange(NEED - 1, store.n_minutes, STEP)
    ts = store.t0 + pd.to_timedelta(idx, unit="min")
    tr, va, te = ts < t_tr, (ts >= t_tr) & (ts < t_va), ts >= t_va
    if days is not None:  # tiny runs: split the loaded range by thirds instead
        n = len(ts)
        tr, va, te = (
            np.arange(n) < n * 0.6,
            (np.arange(n) >= n * 0.6) & (np.arange(n) < n * 0.8),
            np.arange(n) >= n * 0.8,
        )
    feats = rolling_features(store.values, idx)
    y, since = _targets(engine, net, store, idx, settings.seed)
    flat = lambda m: feats[m].reshape(-1, feats.shape[2])  # noqa: E731
    rng = np.random.default_rng(settings.seed)
    x_tr = flat(tr)
    sub = rng.choice(len(x_tr), size=min(len(x_tr), 300_000), replace=False)
    model = IsolationForest(
        n_estimators=200,
        max_samples=512,
        contamination="auto",
        random_state=settings.seed,
        n_jobs=-1,
    ).fit(x_tr[sub])

    def score(m):
        return -model.score_samples(flat(m))

    s_va, s_te = score(va), score(te)
    y_va, y_te = y[va].reshape(-1), y[te].reshape(-1)
    thr = _best_f1_threshold(y_va, s_va)
    zmax_va = np.abs(feats[va][:, :, : len(KPI_COLUMNS)]).max(axis=2).reshape(-1)
    zmax_te = np.abs(feats[te][:, :, : len(KPI_COLUMNS)]).max(axis=2).reshape(-1)
    base_thr_tuned = _best_f1_threshold(y_va, zmax_va)
    metrics = {
        "model": {
            "pr_auc": round(float(average_precision_score(y_te, s_te)), 4),
            **_prf(y_te, s_te, thr),
            "threshold_val_f1": round(thr, 4),
        },
        "baseline_3sigma": {
            "pr_auc": round(float(average_precision_score(y_te, zmax_te)), 4),
            **_prf(y_te, zmax_te, 3.0),
        },
        "baseline_sigma_tuned_on_val": {
            **_prf(y_te, zmax_te, base_thr_tuned),
            "sigma_threshold": round(base_thr_tuned, 3),
        },
        "test_positive_rate": round(float(y_te.mean()), 4),
        "n_test_rows": int(len(y_te)),
        "val_f1_at_threshold": _prf(y_va, s_va, thr)["f1"],
    }
    since_te = since[te].reshape(-1)
    bins = [(0, 5), (5, 15), (15, 30), (30, 60), (60, 999)]

    def by_onset(sc, thr_):
        out = {}
        for lo, hi in bins:
            m = (since_te >= lo) & (since_te < hi)
            out[f"{lo}-{hi}min"] = {
                "recall": round(float((sc[m] >= thr_).mean()), 4) if m.any() else None,
                "n": int(m.sum()),
            }
        return out

    metrics["recall_by_minutes_since_onset"] = {
        "iforest": by_onset(s_te, thr),
        "3sigma": by_onset(zmax_te, 3.0),
    }
    version = new_version()
    card = {
        "name": NAME,
        "version": version,
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "data_hash": data_hash(x_tr[sub]),
        "features": feature_names(),
        "metrics": {**metrics["model"], "test_positive_rate": metrics["test_positive_rate"]},
        "baseline_metrics": metrics["baseline_3sigma"],
        "extra": {
            "baseline_sigma_tuned_on_val": metrics["baseline_sigma_tuned_on_val"],
            "grid_minutes": STEP,
            "baseline_minutes": BASE,
            "baseline_lag_minutes": GAP,
            "recall_by_minutes_since_onset": metrics["recall_by_minutes_since_onset"],
            "seed": settings.seed,
        },
        "intended_use": "Flag nodes whose recent KPIs depart from their own lagged baseline "
        f"(an hour that ends {GAP} min earlier, so a lasting fault is not absorbed); "
        "an input signal to triage, not a root-cause decision.",
        "limitations": "Trained on synthetic simulator data. Target = minute lies inside an "
        f"incident and the node's simulated KPI effect is >= {VISIBLE}; KPI-silent nodes are "
        "negatives. A node hit again within the lag window has a contaminated baseline. "
        "Threshold maximises F1 on validation, so it is tuned to this fault density.",
    }
    save_artifact(
        settings, NAME, version, {"model": model, "threshold": thr, "node_ids": net.node_ids}, card
    )
    return {"version": version, **metrics}
