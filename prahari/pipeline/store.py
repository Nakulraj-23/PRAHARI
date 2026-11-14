"""Loaders: KPI tensor, label-free alarm frames and (training-only) ground truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from prahari.sim.constants import KPI_COLUMNS
from prahari.sim.topology import NetworkModel

# Explicit allow-list: label_* columns are never selected by the feature pipeline.
ALARM_COLUMNS = ["alarm_id", "ts", "node_id", "alarm_type", "severity", "raw_text"]
_MIN = pd.Timedelta(minutes=1)


@dataclass
class KpiStore:
    """KPIs as a (T, N, 6) float32 tensor on a 1-minute grid starting at ``t0`` (NaN = missing)."""

    t0: pd.Timestamp
    values: np.ndarray
    node_ids: list[str]

    @property
    def n_minutes(self) -> int:
        return self.values.shape[0]

    def minute_index(self, ts) -> int:
        """Grid index of a timestamp (floored to the minute)."""
        return int((pd.Timestamp(ts) - self.t0) // _MIN)

    def window(self, start, end) -> np.ndarray:
        """KPIs in [start, end), clipped to the available grid; shape (t, N, 6)."""
        lo = max(self.minute_index(start), 0)
        hi = min(self.minute_index(end), self.n_minutes)
        return self.values[lo : max(hi, lo)]


def load_kpi(engine: Engine, net: NetworkModel, start=None, end=None) -> KpiStore:
    """Load KPI samples in [start, end) into a tensor in ``net`` node order."""
    where, params = "", {}
    if start is not None:
        where += " AND k.ts >= :s"
        params["s"] = pd.Timestamp(start).to_pydatetime()
    if end is not None:
        where += " AND k.ts < :e"
        params["e"] = pd.Timestamp(end).to_pydatetime()
    cols = ", ".join(f"k.{c}" for c in KPI_COLUMNS)
    # join from nodes so the (node_id, ts) index serves the time range
    sql = text(
        f"SELECT k.ts, k.node_id, {cols} FROM nodes n JOIN kpi_samples k "  # noqa: S608
        f"ON k.node_id = n.node_id WHERE 1=1{where}"
    )
    df = pd.read_sql(sql, engine, params=params, parse_dates=["ts"])
    if df.empty:
        t0 = pd.Timestamp(start) if start is not None else pd.Timestamp("1970-01-01")
        return KpiStore(t0, np.zeros((0, net.n, 6), dtype=np.float32), net.node_ids)
    off = df["ts"] != df["ts"].dt.floor("min")
    if off.any():  # never floor silently: that would overwrite a neighbouring minute
        from prahari.pipeline.quality import DataQualityError

        raise DataQualityError(
            f"grid: {int(off.sum())} KPI timestamps are not on the 1-minute grid"
        )
    dup = df.duplicated(["ts", "node_id"])
    if dup.any():
        from prahari.pipeline.quality import DataQualityError

        raise DataQualityError(f"duplicate: {int(dup.sum())} duplicate (ts, node_id) KPI rows")
    t0 = df["ts"].min()
    t_idx = ((df["ts"] - t0) // _MIN).to_numpy()
    n_idx = df["node_id"].map(net.index).to_numpy()
    values = np.full((int(t_idx.max()) + 1, net.n, 6), np.nan, dtype=np.float32)
    values[t_idx, n_idx] = df[KPI_COLUMNS].to_numpy(dtype=np.float32)
    return KpiStore(t0, values, net.node_ids)


def load_alarms(engine: Engine, start=None, end=None) -> pd.DataFrame:
    """Alarms in [start, end) with feature-safe columns only (no ``label_*``), time-sorted."""
    where, params = [], {}
    if start is not None:
        where.append("ts >= :s")
        params["s"] = pd.Timestamp(start).to_pydatetime()
    if end is not None:
        where.append("ts < :e")
        params["e"] = pd.Timestamp(end).to_pydatetime()
    sql = f"SELECT {', '.join(ALARM_COLUMNS)} FROM alarms"  # noqa: S608
    if where:
        sql += " WHERE " + " AND ".join(where)
    return pd.read_sql(
        text(sql + " ORDER BY ts, alarm_id"), engine, params=params, parse_dates=["ts"]
    )


def load_ground_truth_incidents(engine: Engine) -> pd.DataFrame:
    """TRAINING/EVALUATION ONLY: ground-truth incidents. Never call this from the serving path."""
    return pd.read_sql(
        "SELECT incident_id, start_ts, end_ts, root_node_id, fault_type FROM incidents "
        "ORDER BY start_ts, incident_id",
        engine,
        parse_dates=["start_ts", "end_ts"],
    )
