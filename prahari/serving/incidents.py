"""Incident detector (validated alarms -> cluster -> RCA -> agent -> governance -> action) and
the human decision flow. Reads only label-free alarm columns and its own tables."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import func, select

from prahari.agent import analyze
from prahari.agent.agent import explain as agent_explain
from prahari.config import Settings
from prahari.db import Action, AuditLog, DetectedIncident, DriftReport
from prahari.serving.data import Db, audit, log_prediction, utcnow
from prahari.serving.obs import ApiError, Metrics
from prahari.serving.services import Scored, Scoring

LOG = logging.getLogger("prahari.api")
MIN_ALARMS = 2  # valid alarms needed to open an incident
GAP_MIN = 5  # alarms closer than this in time ...
HOPS = 2  # ... and this many links apart belong to one cluster
ATTACH_MIN = 10  # a cluster this close (in time) to an open incident joins it
WINDOW_BEFORE_MIN, WINDOW_MIN = 5, 15  # RCA window: 5 min before the first valid alarm, 15 long
PENDING_MIN = 15  # single valid alarms wait this long for company
RECENT_ACTION_MIN = 60
CURSOR_EVENT = "detector_cursor"


class Incidents:
    """Detector and decision logic. One instance per API process."""

    def __init__(self, settings: Settings, db: Db, sessions, scoring: Scoring, metrics: Metrics):
        self.settings, self.db, self.sessions = settings, db, sessions
        self.scoring, self.metrics = scoring, metrics
        self.sim: Any = None  # LiveSimulator, set by the app
        self.lock = threading.RLock()
        self.pending: list[Scored] = []
        self.cursor = self._initial_cursor()

    # ---- helpers ----------------------------------------------------------------------
    def _initial_cursor(self) -> str:
        """Resume after the last processed alarm; on first start only alarms from now on count."""
        with self.sessions() as s:
            row = s.execute(
                select(AuditLog.payload_json)
                .where(AuditLog.event == CURSOR_EVENT)
                .order_by(AuditLog.id.desc())
                .limit(1)
            ).first()
        if row:
            return row[0]["last_alarm_id"]
        return self.db.max_alarm_id() or ""

    def drift_status(self) -> str | None:
        with self.sessions() as s:
            return s.execute(
                select(DriftReport.status).order_by(DriftReport.id.desc()).limit(1)
            ).scalar()

    def _recent_actions(self, s, incident_id: str) -> list[dict]:
        since = utcnow() - timedelta(minutes=RECENT_ACTION_MIN)
        rows = s.execute(
            select(Action).where(
                Action.incident_ref != incident_id,
                Action.created_at >= since,
                Action.status.in_(["proposed", "approved", "executed"]),
            )
        ).scalars()
        return [
            {"target_node_id": a.target_node_id, "action": a.proposed_action, "ts": a.created_at}
            for a in rows
        ]

    @staticmethod
    def _next_id(s, col, prefix: str) -> str:
        last = s.execute(select(func.max(col))).scalar()
        n = int(last[len(prefix) :]) + 1 if last else 1
        return f"{prefix}{n:05d}"

    # ---- clustering -------------------------------------------------------------------
    def _clusters(self, alarms: list[Scored]) -> list[list[Scored]]:
        near = self.db.neighbourhood(HOPS)
        parent = list(range(len(alarms)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, a in enumerate(alarms):
            for j in range(i):
                b = alarms[j]
                if abs((a.ts - b.ts).total_seconds()) <= GAP_MIN * 60 and (
                    b.node_id in near.get(a.node_id, {a.node_id})
                ):
                    parent[find(i)] = find(j)
        groups: dict[int, list[Scored]] = {}
        for i, a in enumerate(alarms):
            groups.setdefault(find(i), []).append(a)
        return sorted(groups.values(), key=lambda g: min(x.ts for x in g))

    def _open_incidents(self, s) -> list[DetectedIncident]:
        return list(
            s.execute(select(DetectedIncident).where(DetectedIncident.status == "open")).scalars()
        )

    def _matching_open(self, s, cluster: list[Scored]) -> DetectedIncident | None:
        near = self.db.neighbourhood(HOPS)
        for inc in self._open_incidents(s):
            al = self.db.alarms_by_ids(list(inc.alarm_ids))
            if al.empty:
                continue
            last = al["ts"].max().to_pydatetime()
            nodes = set(al["node_id"])
            first = min(x.ts for x in cluster)
            if first - last <= timedelta(minutes=ATTACH_MIN) and any(
                x.node_id in near.get(n, {n}) for x in cluster for n in nodes
            ):
                return inc
        return None

    # ---- main cycle -------------------------------------------------------------------
    def detect_once(self) -> list[str]:
        """Process alarms that arrived since the last cycle; return ids of incidents touched."""
        with self.lock:
            try:
                touched = self._detect()
                self.metrics.detector_runs.labels("ok").inc()
                return touched
            except ApiError as e:
                LOG.warning("detector skipped: %s", e.message)
                self.metrics.detector_runs.labels("error").inc()
                return []
            except Exception:
                LOG.exception("detector failed")
                self.metrics.detector_runs.labels("error").inc()
                return []

    def _detect(self) -> list[str]:
        df = self.db.alarms(
            "WHERE alarm_id > :a", {"a": self.cursor}, "ORDER BY alarm_id LIMIT 2000"
        )
        if df.empty:
            return []
        scored, _ = self.scoring.validate(df)
        self.cursor = str(df["alarm_id"].max())
        with self.sessions() as s:
            audit(
                s,
                "detector",
                CURSOR_EVENT,
                {
                    "last_alarm_id": self.cursor,
                    "alarms": len(df),
                    "valid": sum(x.is_valid for x in scored),
                },
            )
            s.commit()
        pool = self.pending + [x for x in scored if x.is_valid]
        newest = max((x.ts for x in pool), default=None)
        touched: list[str] = []
        self.pending = []
        for cl in self._clusters(pool):
            if len(cl) < MIN_ALARMS:
                if newest and newest - cl[0].ts <= timedelta(minutes=PENDING_MIN):
                    self.pending += cl
                continue
            with self.sessions() as s:
                inc = self._matching_open(s, cl)
                iid = self._attach(s, inc, cl) if inc else self._open(s, cl)
                s.commit()
            touched.append(iid)
        return touched

    # ---- open / update ----------------------------------------------------------------
    def _analyse(self, s, incident_id: str, first_ts: datetime, alarm_ids: list[str]):
        ws = pd.Timestamp(first_ts).floor("min").to_pydatetime() - timedelta(
            minutes=WINDOW_BEFORE_MIN
        )
        we = ws + timedelta(minutes=WINDOW_MIN)
        inwin = self.scoring.cached_in(ws, we)
        valid = sum(x.is_valid for x in inwin)
        rca = self.scoring.rca(ws, we, valid, len(inwin) - valid)
        out, gov = analyze(
            rca,
            self.db.engine,
            lang="en",
            incident_id=incident_id,
            drift_status=self.drift_status(),
            recent_actions=self._recent_actions(s, incident_id),
            settings=self.settings,
        )
        log_prediction(
            s,
            "agent",
            out.llm.model_dump(),
            incident_id,
            {
                "action": out.proposed_action.model_dump(),
                "decision": gov.decision,
                "steps": out.steps,
            },
            0.0,
        )
        return ws, we, rca, out, gov

    def _open(self, s, cl: list[Scored]) -> str:
        iid = self._next_id(s, DetectedIncident.incident_id, "DET-")
        ids = [x.alarm_id for x in cl]
        first = min(x.ts for x in cl)
        ws, we, rca, out, gov = self._analyse(s, iid, first, ids)
        aid = self._next_id(s, Action.action_id, "ACT-")
        status = {"blocked": "blocked"}.get(gov.decision, "proposed")
        s.add(
            Action(
                action_id=aid,
                incident_ref=iid,
                proposed_action=out.proposed_action.action,
                target_node_id=out.proposed_action.target_node_id,
                status=status,
                governance_json=gov.model_dump(mode="json"),
                created_at=utcnow(),
            )
        )
        s.add(
            DetectedIncident(
                incident_id=iid,
                detected_at=utcnow(),
                start_ts=first,
                end_ts=None,
                status="open",
                window_start=ws,
                window_end=we,
                alarm_ids=ids,
                rca_json=rca,
                agent_json={"en": out.model_dump(mode="json")},
                governance_json=gov.model_dump(mode="json"),
                action_id=aid,
            )
        )
        audit(
            s,
            "detector",
            "incident_detected",
            {
                "incident_id": iid,
                "alarms": len(ids),
                "root": rca["root_candidates"][0],
                "fault_type": rca["fault_type"],
            },
        )
        audit(
            s,
            "agent",
            "action_proposed",
            {
                "action_id": aid,
                "incident_id": iid,
                "action": out.proposed_action.model_dump(),
                "decision": gov.decision,
                "failed_checks": [c.id for c in gov.checks if c.status == "fail"],
            },
        )
        s.flush()
        self.metrics.incidents.inc()
        if gov.decision == "auto_approved":
            self._execute(s, s.get(Action, aid), "governance:auto_approved", note="auto-approved")
        return iid

    def _attach(self, s, inc: DetectedIncident, cl: list[Scored]) -> str:
        ids = list(dict.fromkeys(list(inc.alarm_ids) + [x.alarm_id for x in cl]))
        act = s.get(Action, inc.action_id) if inc.action_id else None
        inc.alarm_ids = ids
        audit(
            s, "detector", "incident_updated", {"incident_id": inc.incident_id, "alarms": len(ids)}
        )
        if act is not None and act.status == "proposed":
            ws, we, rca, out, gov = self._analyse(s, inc.incident_id, inc.start_ts, ids)
            inc.rca_json, inc.agent_json = rca, {"en": out.model_dump(mode="json")}
            inc.governance_json = gov.model_dump(mode="json")
            inc.window_start, inc.window_end = ws, we
            act.proposed_action = out.proposed_action.action
            act.target_node_id = out.proposed_action.target_node_id
            act.governance_json = gov.model_dump(mode="json")
            act.status = "blocked" if gov.decision == "blocked" else "proposed"
        return inc.incident_id

    # ---- decisions --------------------------------------------------------------------
    def _execute(self, s, act: Action, actor: str, note: str = "") -> dict:
        """Apply the action; mark it executed and resolve the incident if it recovered."""
        if self.sim is None:
            raise ApiError(503, "simulator unavailable; cannot execute action", "sim_unavailable")
        act.status, act.decided_by, act.decided_at = "approved", actor, utcnow()
        if act.proposed_action == "no_action":
            res = {"recovered": [], "reason": "no_action"}
        else:
            try:
                res = self.sim.apply_action({"target_node_id": act.target_node_id})
            except Exception as e:  # noqa: BLE001
                audit(
                    s, actor, "action_execute_failed", {"action_id": act.action_id, "error": str(e)}
                )
                s.commit()
                raise ApiError(
                    502, f"action approved but execution failed: {e}", "action_failed"
                ) from e
        act.status = "executed"
        inc = s.execute(
            select(DetectedIncident).where(DetectedIncident.incident_id == act.incident_ref)
        ).scalar_one_or_none()
        recovered = bool(res.get("recovered"))
        if inc is not None and recovered:
            inc.status, inc.end_ts = "resolved", pd.Timestamp(res["clock"]).to_pydatetime()
        audit(
            s,
            actor,
            "action_executed",
            {"action_id": act.action_id, "note": note, "recovered": recovered, "result": res},
        )
        return res

    def decide(self, action_id: str, approve: bool, user: str, note: str) -> Action:
        """Human approve/reject. Blocked actions cannot be approved (409)."""
        with self.lock, self.sessions() as s:
            act = s.get(Action, action_id)
            if act is None:
                raise ApiError(404, f"action {action_id} not found")
            if act.status == "blocked" and approve:
                raise ApiError(409, "action is blocked by governance and cannot be approved")
            if act.status not in ("proposed", "blocked"):
                raise ApiError(409, f"action already {act.status}")
            if not approve:
                act.status, act.decided_by, act.decided_at = "rejected", user, utcnow()
                audit(s, user, "action_rejected", {"action_id": action_id, "note": note})
            else:
                audit(s, user, "action_approved", {"action_id": action_id, "note": note})
                self._execute(s, act, user, note)
            s.commit()
            return act

    # ---- views ------------------------------------------------------------------------
    @staticmethod
    def action_out(a: Action | None) -> dict | None:
        if a is None:
            return None
        return {
            "action_id": a.action_id,
            "incident_ref": a.incident_ref,
            "proposed_action": a.proposed_action,
            "target_node_id": a.target_node_id,
            "status": a.status,
            "governance": a.governance_json,
            "decided_by": a.decided_by,
            "decided_at": a.decided_at,
        }

    def incident_out(self, s, inc: DetectedIncident) -> dict:
        act = s.get(Action, inc.action_id) if inc.action_id else None
        return {
            "incident_id": inc.incident_id,
            "start_ts": inc.start_ts,
            "end_ts": inc.end_ts,
            "status": inc.status,
            "rca": inc.rca_json,
            "action": self.action_out(act),
            "governance": act.governance_json if act else inc.governance_json,
        }

    def explain(self, incident_id: str, lang: str):
        """Narrative in ``lang`` (generated once per language, then stored)."""
        with self.lock, self.sessions() as s:
            inc = s.get(DetectedIncident, incident_id)
            if inc is None:
                raise ApiError(404, f"incident {incident_id} not found")
            stored = dict(inc.agent_json)
            if lang not in stored:
                out = agent_explain(
                    inc.rca_json,
                    lang=lang,
                    incident_id=incident_id,
                    engine=self.db.engine,
                    settings=self.settings,
                )
                stored[lang] = out.model_dump(mode="json")
                inc.agent_json = stored
                audit(
                    s,
                    "agent",
                    "explanation_generated",
                    {"incident_id": incident_id, "lang": lang, "llm": stored[lang]["llm"]},
                )
                s.commit()
            return stored[lang]
