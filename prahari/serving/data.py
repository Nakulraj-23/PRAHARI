"""Serving-path DB access: explicit column lists, never `label_*` or the `incidents` table."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from prahari.db import AuditLog, Prediction
from prahari.sim.constants import KPI_COLUMNS

ALARM_COLS = "alarm_id, ts, node_id, alarm_type, severity, raw_text"  # no label_* columns
KPI_SQL = ", ".join(KPI_COLUMNS)
HISTORY = 240  # minutes of KPI history sent to the anomaly model (it uses what it needs)
Z_RECENT, Z_BASE = 5, 60  # validator KPI z-score: last 5 min vs the 60 min before them


def naive(dt: datetime) -> datetime:
    """UTC-naive datetime (the DB stores naive timestamps)."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def utcnow() -> datetime:
    """Naive UTC now."""
    return datetime.now(UTC).replace(tzinfo=None)


class Db:
    """Thin read helpers over one engine."""

    def __init__(self, engine: Engine):
        self.engine = engine

    # ---- topology ---------------------------------------------------------------------
    def topology(self) -> dict[str, list[dict]]:
        with self.engine.connect() as c:
            nodes = c.execute(
                text("SELECT node_id, node_type, region, vendor FROM nodes ORDER BY node_id")
            ).mappings()
            nodes = [dict(r) for r in nodes]
            links = c.execute(
                text("SELECT link_id, src, dst, capacity_mbps FROM links ORDER BY link_id")
            ).mappings()
            links = [dict(r) for r in links]
        return {"nodes": nodes, "links": links}

    def neighbourhood(self, hops: int = 2) -> dict[str, set[str]]:
        """node -> set of nodes within ``hops`` links (itself included)."""
        adj: dict[str, set[str]] = {}
        for lk in self.topology()["links"]:
            adj.setdefault(lk["src"], set()).add(lk["dst"])
            adj.setdefault(lk["dst"], set()).add(lk["src"])
        out = {}
        for n in adj:
            seen, frontier = {n}, {n}
            for _ in range(hops):
                frontier = {m for f in frontier for m in adj.get(f, ())} - seen
                seen |= frontier
            out[n] = seen
        return out

    def descendants(self, node_id: str) -> set[str]:
        """Nodes strictly downstream of ``node_id`` (reachable through deeper tiers only)."""
        depth = {"core": 0, "agg": 1, "access": 2, "cell": 3}
        topo = self.topology()
        d = {n["node_id"]: depth[n["node_type"]] for n in topo["nodes"]}
        adj: dict[str, set[str]] = {}
        for lk in topo["links"]:
            adj.setdefault(lk["src"], set()).add(lk["dst"])
            adj.setdefault(lk["dst"], set()).add(lk["src"])
        seen, stack = {node_id}, [node_id]
        while stack:
            cur = stack.pop()
            for n in adj.get(cur, ()):
                if n not in seen and d[n] > d[cur]:
                    seen.add(n)
                    stack.append(n)
        return seen - {node_id}

    def has_kpi(self, start: datetime, end: datetime) -> bool:
        """Any KPI sample in [start, end)? Probes one node so the (node_id, ts) index is used."""
        with self.engine.connect() as c:
            return (
                c.execute(
                    text(
                        "SELECT 1 FROM kpi_samples WHERE node_id = "
                        "(SELECT MIN(node_id) FROM nodes) AND ts >= :s AND ts < :e LIMIT 1"
                    ),
                    {"s": start, "e": end},
                ).first()
                is not None
            )

    # ---- alarms -----------------------------------------------------------------------
    def alarms(self, where: str = "", params: dict | None = None, tail: str = "") -> pd.DataFrame:
        sql = f"SELECT {ALARM_COLS} FROM alarms {where} {tail}"  # noqa: S608
        return pd.read_sql(text(sql), self.engine, params=params or {}, parse_dates=["ts"])

    def alarms_by_ids(self, ids: list[str]) -> pd.DataFrame:
        if not ids:
            return pd.DataFrame(columns=ALARM_COLS.split(", "))
        stmt = text(f"SELECT {ALARM_COLS} FROM alarms WHERE alarm_id IN :ids").bindparams(  # noqa: S608
            bindparam("ids", expanding=True)
        )
        return pd.read_sql(stmt, self.engine, params={"ids": ids}, parse_dates=["ts"])

    def max_alarm_id(self) -> str | None:
        with self.engine.connect() as c:
            return c.execute(text("SELECT MAX(alarm_id) FROM alarms")).scalar()

    # ---- KPI --------------------------------------------------------------------------
    def kpi_series(self, node_id: str, start: datetime, end: datetime, limit: int) -> pd.DataFrame:
        sql = (
            f"SELECT ts, {KPI_SQL} FROM kpi_samples WHERE node_id = :n AND ts >= :s AND ts <= :e "  # noqa: S608
            "ORDER BY ts LIMIT :l"
        )
        return pd.read_sql(
            text(sql),
            self.engine,
            params={"n": node_id, "s": start, "e": end, "l": limit},
            parse_dates=["ts"],
        )

    def last_kpi_ts(self) -> datetime | None:
        with self.engine.connect() as c:
            v = c.execute(text("SELECT MAX(ts) FROM kpi_samples")).scalar()
        return pd.Timestamp(v).to_pydatetime() if v is not None else None

    def kpi_matrix(self, node_ids: list[str], end: datetime, minutes: int) -> np.ndarray:
        """(minutes, N, 6) KPI grid ending at the minute of ``end`` (inclusive); NaN = missing."""
        last = pd.Timestamp(end).floor("min")
        first = last - pd.Timedelta(minutes=minutes - 1)
        sql = (
            f"SELECT ts, node_id, {KPI_SQL} FROM kpi_samples "  # noqa: S608
            "WHERE ts >= :s AND ts < :e"
        )
        df = pd.read_sql(
            text(sql),
            self.engine,
            params={
                "s": first.to_pydatetime(),
                "e": (last + pd.Timedelta(minutes=1)).to_pydatetime(),
            },
            parse_dates=["ts"],
        )
        out = np.full((minutes, len(node_ids), 6), np.nan)
        if df.empty:
            return out
        idx = {n: i for i, n in enumerate(node_ids)}
        df = df[df["node_id"].isin(idx)]
        t = ((df["ts"].dt.floor("min") - first) // pd.Timedelta(minutes=1)).to_numpy()
        ok = (t >= 0) & (t < minutes)
        n = df["node_id"].map(idx).to_numpy()
        out[t[ok], n[ok]] = df[KPI_COLUMNS].to_numpy(float)[ok]
        return out

    def alarm_kpi_z(self, alarms: pd.DataFrame) -> dict[str, list[float]]:
        """Signed KPI z-scores per alarm: mean of the last 5 min vs mean/std of the 60 min before.

        Same definition the validator was trained with (documented in its module docstring); one
        indexed query per alarm node-minute. Missing history gives zeros.
        """
        out: dict[str, list[float]] = {}
        span = Z_RECENT + Z_BASE
        with self.engine.connect() as c:
            for r in alarms.itertuples():
                end = pd.Timestamp(r.ts).floor("min")
                start = end - pd.Timedelta(minutes=span - 1)
                rows = c.execute(
                    text(
                        f"SELECT ts, {KPI_SQL} FROM kpi_samples "  # noqa: S608
                        "WHERE node_id = :n AND ts >= :s AND ts < :e"
                    ),
                    {
                        "n": r.node_id,
                        "s": start.to_pydatetime(),
                        "e": (end + pd.Timedelta(minutes=1)).to_pydatetime(),
                    },
                ).all()
                grid = np.full((span, 6), np.nan)
                for row in rows:
                    k = int((pd.Timestamp(row[0]).floor("min") - start) // pd.Timedelta(minutes=1))
                    if 0 <= k < span:
                        grid[k] = row[1:]
                out[r.alarm_id] = _z(grid)
        return out

    # ---- logs -------------------------------------------------------------------------
    def latency_samples(self, model_name: str, n: int = 500) -> list[float]:
        with self.engine.connect() as c:
            rows = c.execute(
                text(
                    "SELECT latency_ms FROM predictions WHERE model_name = :m "
                    "ORDER BY ts DESC LIMIT :n"
                ),
                {"m": model_name, "n": n},
            ).all()
        return [float(r[0]) for r in rows]


def _z(grid: np.ndarray) -> list[float]:
    ref, cur = grid[:Z_BASE], grid[Z_BASE:]
    with np.errstate(all="ignore"):
        if np.isnan(ref).all() or np.isnan(cur).all():
            return [0.0] * 6
        mu, sd = np.nanmean(ref, axis=0), np.nanstd(ref, axis=0)
        sd = np.maximum(sd, np.maximum(0.02 * np.abs(mu), 1e-3))
        z = (np.nanmean(cur, axis=0) - mu) / sd
    return [float(v) for v in np.clip(np.nan_to_num(z), -50, 50)]


def log_prediction(
    session, kind: str, model: dict, input_ref: str, output: Any, latency_ms: float
) -> None:
    """Add a ``predictions`` row to ``session`` (caller commits)."""
    session.add(
        Prediction(
            pred_id=uuid.uuid4().hex,
            ts=utcnow(),
            kind=kind,
            model_name=model["name"],
            model_version=str(model["version"]),
            input_ref=input_ref[:500],
            output_json=output,
            latency_ms=float(latency_ms),
        )
    )


def audit(session, actor: str, event: str, payload: dict) -> None:
    """Add an ``audit_log`` row to ``session`` (caller commits)."""
    session.add(AuditLog(ts=utcnow(), actor=actor, event=event, payload_json=payload))
