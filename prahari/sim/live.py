"""Live mode for the API: inject faults, apply remediation actions, run scenarios.

The simulator keeps a clock at the end of the data in the DB. Every call appends new KPI
samples and alarms after that clock, so the serving path sees them like real telemetry.
Ground truth (incidents, ``label_*``) is written alongside, as in the offline generator.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import text

from prahari.config import Settings, get_settings
from prahari.db import Base, get_engine
from prahari.sim.alarms import apply_blips, assemble_alarms, noise_alarms, noise_blips
from prahari.sim.constants import FAULT_TYPES, SIM_START, VENDORS
from prahari.sim.faults import (
    RAMP_DOWN,
    ROOT_TIER_P,
    Fault,
    apply_fault,
    fault_alarms,
    new_fault,
)
from prahari.sim.kpi import block_to_frame, clip_kpis, drop_missing, kpi_block
from prahari.sim.topology import NetworkModel

OBSERVE_MIN = 10  # minutes of telemetry written when a fault is injected
RECOVERY_MIN = RAMP_DOWN + 2
DRIFT_MULT = 1.25
ROOT_MISSING_FRAC = 0.15  # same as the offline generator
_MIN = pd.Timedelta(minutes=1)


@dataclass
class LiveSimulator:
    """Appends fault/scenario telemetry to the DB. One instance per API process."""

    settings: Settings = field(default_factory=get_settings)
    seed: int | None = None

    def __post_init__(self) -> None:
        self.engine = get_engine(self.settings)
        Base.metadata.create_all(self.engine)
        self.seed = self.settings.seed if self.seed is None else self.seed
        self.net = NetworkModel.from_db(self.engine)
        self._lock = threading.RLock()
        self._state_path = self.settings.artifacts_dir / "sim_state.json"
        self.state: dict[str, Any] = {"events": 0, "vendor_latency_mult": {}}
        if self._state_path.exists():
            self.state.update(json.loads(self._state_path.read_text()))
        self.active: dict[str, Fault] = {}
        self._restore_active()

    # ---- state -------------------------------------------------------------------------
    @property
    def clock(self) -> pd.Timestamp:
        """Next minute to be written."""
        with self.engine.connect() as c:
            last = c.execute(text("SELECT MAX(ts) FROM kpi_samples")).scalar()
        return pd.Timestamp(last).floor("min") + _MIN if last else pd.Timestamp(SIM_START)

    def _restore_active(self) -> None:
        """Re-create faults still open (end_ts after the clock) after a restart."""
        clock = self.clock
        df = pd.read_sql(
            text(
                "SELECT incident_id, start_ts, end_ts, root_node_id, fault_type FROM incidents "
                "WHERE end_ts > :c"
            ),
            self.engine,
            params={"c": clock.to_pydatetime()},
            parse_dates=["start_ts", "end_ts"],
        )
        for r in df.itertuples():
            j = self.net.index[r.root_node_id]
            self.active[r.incident_id] = new_fault(
                self.net,
                r.incident_id,
                r.fault_type,
                j,
                r.start_ts,
                int((r.end_ts - r.start_ts) / _MIN),
                int(self.seed),
            )

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self.state))

    def _rng(self) -> np.random.Generator:
        self.state["events"] += 1
        return np.random.default_rng([int(self.seed), 5005, self.state["events"]])

    def _next_id(self, table: str, col: str, prefix: str, width: int) -> int:
        with self.engine.connect() as c:
            last = c.execute(text(f"SELECT MAX({col}) FROM {table}")).scalar()  # noqa: S608
        return int(str(last)[len(prefix) :]) + 1 if last else 1

    # ---- writing -----------------------------------------------------------------------
    def _write_kpis(
        self,
        start: pd.Timestamp,
        minutes: int,
        rng: np.random.Generator,
        faults: list[Fault],
        early_end: dict[str, pd.Timestamp] | None = None,
        blips: list[dict] | None = None,
    ) -> int:
        idx = pd.date_range(start, periods=minutes, freq="min")
        block = kpi_block(self.net, idx, rng, self.state["vendor_latency_mult"])
        for f in faults:
            apply_fault(block, start, f, self.net, (early_end or {}).get(f.incident_id))
        apply_blips(block, start, blips or [])
        clip_kpis(block)
        drop_missing(block, rng)
        df = block_to_frame(self.net, idx, block)
        df.to_sql("kpi_samples", self.engine, if_exists="append", index=False, chunksize=50_000)
        return len(df)

    def _write_alarms(self, rows: list[dict], rng: np.random.Generator, until: pd.Timestamp) -> int:
        rows = [r for r in rows if r["ts"] < until]
        if not rows:
            return 0
        first = self._next_id("alarms", "alarm_id", "ALM-", 7)
        df = assemble_alarms(self.net, rows, rng, first)
        df.to_sql("alarms", self.engine, if_exists="append", index=False)
        return len(df)

    # ---- public API --------------------------------------------------------------------
    def inject(self, fault_type: str, node_id: str | None = None, duration_min: int = 30) -> dict:
        """Start a fault at the clock and write ``OBSERVE_MIN`` minutes of telemetry and alarms."""
        if fault_type not in FAULT_TYPES:
            raise ValueError(f"unknown fault_type {fault_type!r}; expected one of {FAULT_TYPES}")
        with self._lock:
            rng = self._rng()
            if node_id is None:
                tier = int(rng.choice(4, p=ROOT_TIER_P[fault_type]))
                cand = np.nonzero(self.net.type_idx == tier)[0]
                root = int(rng.choice(cand))
            elif node_id in self.net.index:
                root = self.net.index[node_id]
            else:
                raise ValueError(f"unknown node_id {node_id!r}")
            start = self.clock
            n = self._next_id("incidents", "incident_id", "INC-", 5)
            fault = new_fault(
                self.net, f"INC-{n:05d}", fault_type, root, start, int(duration_min), int(self.seed)
            )
            pd.DataFrame(
                [
                    dict(
                        incident_id=fault.incident_id,
                        start_ts=start,
                        end_ts=fault.end,
                        root_node_id=self.net.node_ids[root],
                        fault_type=fault_type,
                    )
                ]
            ).to_sql("incidents", self.engine, if_exists="append", index=False)
            self.active[fault.incident_id] = fault
            end = start + OBSERVE_MIN * _MIN
            rows = fault_alarms(
                self.net, fault, rng, root_missing=bool(rng.random() < ROOT_MISSING_FRAC)
            )
            n_true = len(rows)
            rows += noise_alarms(self.net, max(1, round(n_true / 3)), start, OBSERVE_MIN, rng)
            blips = noise_blips(rows, rng)
            n_alarms = self._write_alarms(rows, rng, end)
            self._write_kpis(start, OBSERVE_MIN, rng, list(self.active.values()), blips=blips)
            self._save_state()
            return dict(
                incident_id=fault.incident_id,
                fault_type=fault_type,
                root_node_id=self.net.node_ids[root],
                start_ts=start.isoformat(),
                affected_nodes=int((fault.impact > 0).sum()),
                alarms_written=n_alarms,
                clock=end.isoformat(),
            )

    def tick(self, minutes: int = 5) -> dict:
        """Advance the clock; open faults keep degrading the network."""
        with self._lock:
            rng = self._rng()
            start = self.clock
            n = self._write_kpis(start, minutes, rng, list(self.active.values()))
            self._save_state()
            return dict(kpi_samples=n, clock=(start + minutes * _MIN).isoformat())

    def apply_action(self, action: Any) -> dict:
        """Apply a remediation action (dict or object with ``target_node_id``).

        An action on a node that is the root of an open fault recovers the network: KPIs ramp
        back to baseline and the incident's ``end_ts`` is set. Any other target changes nothing.
        """
        target = action.get("target_node_id") if isinstance(action, dict) else action.target_node_id
        with self._lock:
            hit = [f for f in self.active.values() if self.net.node_ids[f.root_idx] == target]
            if not hit:
                return dict(
                    recovered=[],
                    reason=f"{target} is not the root of an open fault",
                    clock=self.clock.isoformat(),
                )
            rng = self._rng()
            start = self.clock
            early = {f.incident_id: start for f in hit}
            with self.engine.begin() as c:
                for f in hit:
                    c.execute(
                        text("UPDATE incidents SET end_ts = :e WHERE incident_id = :i"),
                        {"e": start.to_pydatetime(), "i": f.incident_id},
                    )
            self._write_kpis(start, RECOVERY_MIN, rng, list(self.active.values()), early)
            for f in hit:
                self.active.pop(f.incident_id)
            self._save_state()
            return dict(
                recovered=[f.incident_id for f in hit],
                target_node_id=target,
                clock=(start + RECOVERY_MIN * _MIN).isoformat(),
            )

    def scenario(self, name: str, vendor: str | None = None, minutes: int = 360) -> dict:
        """Run a named scenario. ``firmware_drift``: +25% latency baseline for one vendor."""
        if name != "firmware_drift":
            raise ValueError(f"unknown scenario {name!r}; expected 'firmware_drift'")
        with self._lock:
            rng = self._rng()
            vendor = vendor or VENDORS[int(rng.integers(len(VENDORS)))]
            self.state["vendor_latency_mult"][vendor] = DRIFT_MULT
            start = self.clock
            n = self._write_kpis(start, minutes, rng, list(self.active.values()))
            self._save_state()
            return dict(
                scenario=name,
                vendor=vendor,
                latency_mult=DRIFT_MULT,
                kpi_samples=n,
                start_ts=start.isoformat(),
                end_ts=(start + minutes * _MIN).isoformat(),
            )

    def reset_scenarios(self) -> None:
        """Clear vendor drift so later telemetry returns to baseline."""
        with self._lock:
            self.state["vendor_latency_mult"] = {}
            self._save_state()
