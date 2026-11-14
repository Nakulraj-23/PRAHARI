"""Model registry and scoring wrappers (models used only via ``load`` and ``predict``)."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prahari.config import Settings
from prahari.serving.data import HISTORY, Db, log_prediction
from prahari.serving.obs import ApiError

LOG = logging.getLogger("prahari.api")
MODEL_NAMES = {"validator": "alarm_validator", "anomaly": "anomaly_iforest", "rca": "rca_gnn"}
CONTEXT_MIN = 10  # validator flap/neighbour features look back this far
ANOMALY_STEP = 5  # minutes between anomaly scoring points inside a window
MAX_POINTS = 12


def _loaders() -> dict[str, Any]:
    from prahari.models import anomaly, rca, validator

    return {"validator": validator.load, "anomaly": anomaly.load, "rca": rca.load}


class Registry:
    """Loads each model once; a model that fails to load makes its endpoints return 503."""

    def __init__(self, settings: Settings, load: bool = True):
        self.settings = settings
        self.models: dict[str, Any] = {}
        self.errors: dict[str, str] = {}
        if load:
            for key, fn in _loaders().items():
                try:
                    try:
                        self.models[key] = fn(settings=settings)  # artifacts dir + DB of this app
                    except TypeError:  # a loader that only takes ``version``
                        self.models[key] = fn()
                except Exception as e:  # noqa: BLE001 - missing artifact, corrupt file, ...
                    self.errors[key] = f"{type(e).__name__}: {e}"
                    LOG.warning("model %s not loaded: %s", key, self.errors[key])

    def get(self, key: str) -> Any:
        m = self.models.get(key)
        if m is None:
            why = self.errors.get(key, "not loaded")
            raise ApiError(503, f"model {MODEL_NAMES[key]} unavailable: {why}", "model_unavailable")
        return m

    def ref(self, key: str) -> dict:
        return {"name": MODEL_NAMES[key], "version": str(self.get(key).version)}

    def cards(self) -> list[dict]:
        """Model cards (flattened numeric metrics) for every loaded model."""
        out = []
        for key, m in self.models.items():
            p = Path(self.settings.artifacts_dir) / MODEL_NAMES[key] / f"{m.version}.card.json"
            if not p.exists():
                continue
            c = json.loads(p.read_text())
            c["metrics"] = _flat(c.get("metrics", {}))
            c["baseline_metrics"] = _flat(c.get("baseline_metrics", {}))
            c["version"] = str(c.get("version", m.version))
            c.setdefault("intended_use", "")
            c.setdefault("limitations", "")
            out.append(c)
        return out


def _flat(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, dict):
            out |= _flat(v, f"{prefix}{k}.")
        elif isinstance(v, int | float) and np.isfinite(v):
            out[f"{prefix}{k}"] = float(v)
    return out


def latency_stats(samples: list[float]) -> dict | None:
    """Summary of logged model latencies (ms); None when there are none."""
    if not samples:
        return None
    a = np.asarray(samples)
    return {
        "count": len(a),
        "mean_ms": round(float(a.mean()), 2),
        "p50_ms": round(float(np.percentile(a, 50)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2),
        "max_ms": round(float(a.max()), 2),
    }


@dataclass
class Scored:
    """Validator output for one alarm."""

    alarm_id: str
    ts: datetime
    node_id: str
    valid_prob: float
    is_valid: bool


class Scoring:
    """Alarm validation, anomaly scoring and RCA on top of the registry and the DB."""

    def __init__(self, registry: Registry, db: Db, sessions, cache_size: int = 50_000):
        self.reg, self.db, self.sessions = registry, db, sessions
        self.cache: OrderedDict[str, Scored] = OrderedDict()
        self._lock = threading.Lock()
        self.cache_size = cache_size

    # ---- alarm validation -------------------------------------------------------------
    def _remember(self, scored: list[Scored]) -> None:
        with self._lock:
            for s in scored:
                self.cache[s.alarm_id] = s
                self.cache.move_to_end(s.alarm_id)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

    def cached_in(self, start: datetime, end: datetime) -> list[Scored]:
        """Cached validator scores with ``start <= ts < end``."""
        with self._lock:
            return [s for s in self.cache.values() if start <= s.ts < end]

    def validate(self, alarms: pd.DataFrame, log: bool = True) -> tuple[list[Scored], float]:
        """Score alarms (alarm_id, ts, node_id, alarm_type, severity); returns latency ms."""
        model = self.reg.get("validator")
        if alarms.empty:
            return [], 0.0
        alarms = alarms.sort_values(["ts", "alarm_id"]).reset_index(drop=True)
        lo, hi = alarms["ts"].min(), alarms["ts"].max()
        ctx = self.db.alarms(
            "WHERE ts >= :s AND ts <= :e",
            {
                "s": (lo - timedelta(minutes=CONTEXT_MIN)).to_pydatetime(),
                "e": hi.to_pydatetime(),
            },
        )
        ctx = ctx[~ctx["alarm_id"].isin(alarms["alarm_id"])]
        cols = ["alarm_id", "ts", "node_id", "alarm_type", "severity"]
        payload = {
            "alarms": alarms[cols].to_dict("records"),
            "context_alarms": ctx[cols].to_dict("records"),
            "kpi_z": self.db.alarm_kpi_z(alarms),
        }
        res = model.predict(payload)
        by_id = {r["alarm_id"]: r for r in res["results"]}
        scored = [
            Scored(
                r.alarm_id,
                r.ts.to_pydatetime(),
                r.node_id,
                float(by_id[r.alarm_id]["valid_prob"]),
                bool(by_id[r.alarm_id]["is_valid_pred"]),
            )
            for r in alarms.itertuples()
        ]
        self._remember(scored)
        if log:
            with self.sessions() as s:
                log_prediction(
                    s,
                    "alarm_validation",
                    self.reg.ref("validator"),
                    f"{len(scored)} alarms {lo.isoformat()}..{hi.isoformat()}",
                    {"n": len(scored), "n_valid": sum(x.is_valid for x in scored)},
                    res["latency_ms"],
                )
                s.commit()
        return scored, float(res["latency_ms"])

    def scores_for(self, alarms: pd.DataFrame) -> dict[str, Scored]:
        """Scores for stored alarms: cached ones reused, the rest validated now."""
        have = {a: self.cache[a] for a in alarms["alarm_id"] if a in self.cache}
        missing = alarms[~alarms["alarm_id"].isin(have)]
        if len(missing):
            scored, _ = self.validate(missing)
            have |= {s.alarm_id: s for s in scored}
        return have

    # ---- anomaly ----------------------------------------------------------------------
    def anomaly(self, start: datetime, end: datetime, node_ids: list[str] | None) -> dict:
        """Per-node max anomaly score at 5-minute points inside [start, end].

        The model reads the last N minutes of history it needs and ignores older rows, so we send
        ``HISTORY`` rows; if a retrained model asks for more it says so and we retry once.
        """
        try:
            return self._anomaly(start, end, node_ids, HISTORY)
        except ValueError as e:
            m = re.search(r"need at least (\d+)", str(e))
            if not m or int(m.group(1)) <= HISTORY:
                raise
            return self._anomaly(start, end, node_ids, int(m.group(1)) + 15)

    def _anomaly(
        self, start: datetime, end: datetime, node_ids: list[str] | None, history: int
    ) -> dict:
        model = self.reg.get("anomaly")
        nodes = [n["node_id"] for n in self.db.topology()["nodes"]]
        if node_ids:
            bad = sorted(set(node_ids) - set(nodes))
            if bad:
                raise ApiError(422, f"unknown node_id(s): {bad}", "validation_error")
        end_m = pd.Timestamp(end).floor("min")
        points = [end_m - timedelta(minutes=ANOMALY_STEP * k) for k in range(MAX_POINTS)]
        points = sorted(p for p in points if p >= pd.Timestamp(start).floor("min")) or [end_m]
        span = int((points[-1] - points[0]) / timedelta(minutes=1))
        grid = self.db.kpi_matrix(nodes, end_m.to_pydatetime(), history + span)
        if np.isnan(grid[-1]).all():
            raise ApiError(422, "no KPI telemetry at the end of the requested window", "no_data")
        keep = [i for i, n in enumerate(nodes) if not node_ids or n in node_ids]
        best = {nodes[i]: 0.0 for i in keep}
        flag = {nodes[i]: False for i in keep}
        latency, version = 0.0, None
        for p in points:
            upto = history + span - int((points[-1] - p) / timedelta(minutes=1))
            hist = grid[upto - history : upto]
            res = model.predict({"kpi": {nodes[i]: hist[:, i, :].tolist() for i in keep}})
            latency += res["latency_ms"]
            version = res["model"]["version"]
            for n, v in res["scores"].items():
                best[n] = max(best[n], float(v["score"]))
                flag[n] = flag[n] or bool(v["is_anomaly"])
        with self.sessions() as s:
            log_prediction(
                s,
                "anomaly",
                {"name": MODEL_NAMES["anomaly"], "version": version},
                f"{start.isoformat()}..{end.isoformat()}",
                {"n_anomalous": sum(flag.values()), "n_nodes": len(flag)},
                latency,
            )
            s.commit()
        scores = [
            {"node_id": n, "score": best[n], "is_anomalous": flag[n]}
            for n in sorted(best, key=lambda k: -best[k])
        ]
        return {
            "window": {"start": start, "end": end},
            "scores": scores,
            "model": {"name": MODEL_NAMES["anomaly"], "version": str(version)},
            "latency_ms": round(latency, 2),
        }

    # ---- RCA --------------------------------------------------------------------------
    def rca(
        self, start: datetime, end: datetime, valid: int | None = None, noise: int | None = None
    ):
        """RCA contract dict for a window; valid/noise counts come from the validator."""
        model = self.reg.get("rca")
        if valid is None:
            al = self.db.alarms("WHERE ts >= :s AND ts < :e", {"s": start, "e": end})
            sc = self.scores_for(al) if len(al) else {}
            valid = sum(s.is_valid for s in sc.values())
            noise = len(sc) - valid
        out = model.predict(
            {
                "start": start,
                "end": end,
                "valid_alarm_count": int(valid),
                "noise_alarm_count": int(noise or 0),
            }
        )
        with self.sessions() as s:
            log_prediction(
                s,
                "rca",
                out["model"],
                f"{start.isoformat()}..{end.isoformat()}",
                {
                    "top": out["root_candidates"][:3],
                    "fault_type": out["fault_type"],
                    "valid": valid,
                    "noise": noise,
                },
                out["latency_ms"],
            )
            s.commit()
        return out
