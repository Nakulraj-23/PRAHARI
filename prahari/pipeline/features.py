"""Vectorised feature builders for alarms, node windows and incident graphs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prahari.config import Settings, get_settings
from prahari.db import get_engine
from prahari.pipeline.splits import load_splits
from prahari.pipeline.store import (
    KpiStore,
    load_alarms,
    load_ground_truth_incidents,
    load_kpi,
)
from prahari.sim.constants import ALARM_TYPES, FAULT_TYPES, KPI_COLUMNS, NODE_TYPES, VENDORS
from prahari.sim.topology import NetworkModel

_SEV = {"minor": 1, "major": 2, "critical": 3}
BASELINE_MINUTES = 120
_BIG = 1e12


def _drop_labels(df: pd.DataFrame) -> pd.DataFrame:
    return df[[c for c in df.columns if not c.startswith("label_")]]


def _count_before(group: np.ndarray, t: np.ndarray, window: float) -> np.ndarray:
    """For each row: number of OTHER rows in the same group with time in [t - window, t]."""
    key = group.astype(np.float64) * _BIG + t
    order = np.argsort(key, kind="stable")
    ks = key[order]
    hi = np.searchsorted(ks, ks, side="right")
    lo = np.searchsorted(ks, ks - window, side="left")
    out = np.empty(len(key), dtype=np.int64)
    out[order] = hi - lo - 1
    return out


def _kpi_z_at(
    store: KpiStore, t_idx: np.ndarray, n_idx: np.ndarray, recent: int = 5, base: int = 60
) -> np.ndarray:
    """|z| of the last ``recent`` minutes vs the ``base`` minutes before them, per alarm and KPI."""
    offs = np.arange(-(recent + base) + 1, 1)
    ti = np.clip(t_idx[:, None] + offs[None, :], 0, store.n_minutes - 1)
    seg = store.values[ti, n_idx[:, None], :]  # (A, recent+base, 6)
    ref, cur = seg[:, :base, :], seg[:, base:, :]
    with np.errstate(all="ignore"):
        mu, sd = np.nanmean(ref, axis=1), np.nanstd(ref, axis=1)
        sd = np.maximum(sd, np.maximum(0.02 * np.abs(mu), 1e-3))
        z = (np.nanmean(cur, axis=1) - mu) / sd
    return np.clip(np.nan_to_num(z), -50, 50)


def build_alarm_features(
    df: pd.DataFrame,
    net: NetworkModel | None = None,
    kpi: KpiStore | None = None,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Per-alarm features for validity classification. Label columns are always dropped.

    Features: severity, type/node-type/vendor one-hots, hour-of-day, flapping counts (same node
    and type in the last 5 / 30 min), alarms on the node in the last 5 min, alarms on 1-hop
    neighbours within +-5 min, network-wide alarm rate within +-5 min and, when ``kpi`` is given,
    KPI z-scores around the alarm time.
    """
    df = _drop_labels(df).reset_index(drop=True)
    if net is None:
        net = NetworkModel.from_db(get_engine(settings or get_settings()))
    out = df[["alarm_id", "ts"]].copy()
    ts = df["ts"].astype("datetime64[ns]")
    sec = (
        ((ts - ts.min()) // pd.Timedelta(seconds=1)).to_numpy(dtype=np.float64)
        if len(df)
        else np.zeros(0)
    )
    node = df["node_id"].map(net.index).to_numpy()
    atype = df["alarm_type"].map({t: i for i, t in enumerate(ALARM_TYPES)}).to_numpy()
    out["sev_num"] = df["severity"].map(_SEV).to_numpy()
    for i, t in enumerate(ALARM_TYPES):
        out[f"type_{t}"] = (atype == i).astype(np.int8)
    for i, t in enumerate(NODE_TYPES):
        out[f"ntype_{t}"] = (net.type_idx[node] == i).astype(np.int8)
    for i, v in enumerate(VENDORS):
        out[f"vendor_{v}"] = (net.vendor_idx[node] == i).astype(np.int8)
    hod = (ts.dt.hour + ts.dt.minute / 60.0).to_numpy()
    out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
    out["n_same_5m"] = _count_before(node * len(ALARM_TYPES) + atype, sec, 300)
    out["n_same_30m"] = _count_before(node * len(ALARM_TYPES) + atype, sec, 1800)
    out["n_node_5m"] = _count_before(node, sec, 300)
    out["n_net_5m"] = _count_before(np.zeros(len(df)), sec, 600) if len(df) else np.zeros(0)
    # neighbours within +-1 five-minute bucket via an (N, B) count matrix times adjacency
    bucket = (sec // 300).astype(int)
    counts = np.zeros((net.n, (bucket.max() + 3) if len(df) else 3))
    np.add.at(counts, (node, bucket + 1), 1)
    nb = net.adj.astype(float) @ counts
    out["n_neigh_5m"] = nb[node, bucket] + nb[node, bucket + 1] + nb[node, bucket + 2]
    if kpi is not None and kpi.n_minutes:
        t_idx = np.array([kpi.minute_index(t) for t in ts])
        z = _kpi_z_at(kpi, t_idx, node)
        for k, name in enumerate(KPI_COLUMNS):
            out[f"kpi_z_{name}"] = z[:, k]
        out["kpi_z_max"] = np.abs(z).max(axis=1)
    return out


def _window_features(
    net: NetworkModel,
    store: KpiStore,
    alarms: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    baseline_minutes: int = BASELINE_MINUTES,
) -> pd.DataFrame:
    """Per-node features for [start, end): alarm counts, KPI deltas and z vs a rolling baseline."""
    a = alarms[(alarms["ts"] >= start) & (alarms["ts"] < end)]
    cols: dict[str, np.ndarray] = {}
    nidx = a["node_id"].map(net.index).to_numpy()
    for t in ALARM_TYPES:
        cols[f"alarm_{t}"] = np.bincount(nidx[(a["alarm_type"] == t).to_numpy()], minlength=net.n)
    cols["alarm_total"] = np.bincount(nidx, minlength=net.n)
    cols["alarm_critical"] = np.bincount(
        nidx[(a["severity"] == "critical").to_numpy()], minlength=net.n
    )
    first = np.full(net.n, (end - start) / pd.Timedelta(minutes=1))
    if len(a):
        mins = ((a["ts"] - start) / pd.Timedelta(minutes=1)).to_numpy()
        np.minimum.at(first, nidx, mins)
    cols["first_alarm_min"] = first
    cur = store.window(start, end)
    ref = store.window(start - pd.Timedelta(minutes=baseline_minutes), start)
    with np.errstate(all="ignore"):
        cur_mean = np.nanmean(cur, axis=0) if len(cur) else np.full((net.n, 6), np.nan)
        if len(ref):
            mu, sd = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0)
        else:
            mu, sd = cur_mean, np.zeros_like(cur_mean)
        sd = np.maximum(sd, np.maximum(0.02 * np.abs(mu), 1e-3))
        delta = np.nan_to_num(cur_mean - mu)
        z = np.nan_to_num(delta / sd)
    for k, name in enumerate(KPI_COLUMNS):
        cols[f"d_{name}"] = delta[:, k]
        cols[f"z_{name}"] = np.clip(z[:, k], -50, 50)
    for i, t in enumerate(NODE_TYPES):
        cols[f"ntype_{t}"] = (net.type_idx == i).astype(float)
    return pd.DataFrame(cols, index=pd.Index(net.node_ids, name="node_id")).astype(np.float32)


def build_node_window_features(
    start,
    end,
    settings: Settings | None = None,
    baseline_minutes: int = BASELINE_MINUTES,
    *,
    net: NetworkModel | None = None,
    store: KpiStore | None = None,
    alarms: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Node-level feature frame (index node_id) for the window [start, end).

    Reads KPIs (including ``baseline_minutes`` before ``start``) and alarms from the DB unless
    ``store``/``alarms`` are supplied. Uses no ``label_*`` column.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    engine = get_engine(settings or get_settings())
    net = net or NetworkModel.from_db(engine)
    if store is None:
        store = load_kpi(engine, net, start - pd.Timedelta(minutes=baseline_minutes), end)
    if alarms is None:
        alarms = load_alarms(engine, start, end)
    return _window_features(net, store, _drop_labels(alarms), start, end, baseline_minutes)


def feature_names() -> list[str]:
    """Column order of the node feature matrix X in :func:`build_incident_graphs`."""
    return (
        [f"alarm_{t}" for t in ALARM_TYPES]
        + ["alarm_total", "alarm_critical", "first_alarm_min"]
        + [f"d_{k}" for k in KPI_COLUMNS]
        + [f"z_{k}" for k in KPI_COLUMNS]
        + [f"ntype_{t}" for t in NODE_TYPES]
    )


def build_incident_graphs(
    settings: Settings | None = None,
    split: str | None = None,
    window_minutes: int = 15,
    with_ids: bool = False,
) -> list:
    """Training graphs, one per ground-truth incident: ``(X, A, y_root, y_fault)``.

    X is (N, F) node features for [start, start + window_minutes) (columns: ``feature_names()``),
    A the symmetric (N, N) 0/1 adjacency, y_root the root node index (node order = sorted node_id),
    y_fault the index into ``FAULT_TYPES``. ``split`` in train/val/test restricts incidents using
    ``artifacts/splits.json``. With ``with_ids`` each item is prefixed by the incident_id.
    Ground truth is read only for the targets; X never contains labels.
    """
    settings = settings or get_settings()
    engine = get_engine(settings)
    net = NetworkModel.from_db(engine)
    inc = load_ground_truth_incidents(engine)
    if split is not None:
        inc = inc[inc["incident_id"].isin(set(load_splits(settings)[split]))]
    if inc.empty:
        return []
    lo = inc["start_ts"].min() - pd.Timedelta(minutes=BASELINE_MINUTES)
    hi = inc["start_ts"].max() + pd.Timedelta(minutes=window_minutes)
    store = load_kpi(engine, net, lo, hi)
    alarms = load_alarms(engine, lo, hi)
    adj = net.adj.astype(np.float32)
    names = feature_names()
    out = []
    for r in inc.itertuples():
        start = r.start_ts
        x = _window_features(
            net, store, alarms, start, start + pd.Timedelta(minutes=window_minutes)
        )[names]
        item = (
            x.to_numpy(np.float32),
            adj,
            net.index[r.root_node_id],
            FAULT_TYPES.index(r.fault_type),
        )
        out.append((r.incident_id, *item) if with_ids else item)
    return out
