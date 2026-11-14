"""FastAPI app: ``prahari.serving.app:app`` (``make serve``)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from prahari import schemas as sc
from prahari.config import Settings, get_settings
from prahari.db import AuditLog, DetectedIncident, DriftReport, ensure_schema, get_engine
from prahari.serving import obs
from prahari.serving.data import Db, audit, naive
from prahari.serving.incidents import Incidents
from prahari.serving.obs import ApiError
from prahari.serving.services import Registry, Scoring, latency_stats

LOG = logging.getLogger("prahari.api")
KPI_MAX_POINTS = 5000
KPI_DEFAULT_MIN = 120


def _default_drift_runner(settings: Settings) -> Any:
    """A6's ``run_drift`` if present; it stores its report in ``drift_reports``."""
    from prahari.monitoring import run_drift  # type: ignore[attr-defined]

    return run_drift()


def create_app(
    settings: Settings | None = None,
    load_models: bool = True,
    start_sim: bool = True,
    drift_runner: Any = None,
) -> FastAPI:
    """Build the app. ``load_models=False`` / ``start_sim=False`` are for tests."""
    settings = settings or get_settings()
    obs.setup_logging()
    metrics = obs.Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = get_engine(settings)
        ensure_schema(engine)
        sessions = sessionmaker(engine, expire_on_commit=False)
        db = Db(engine)
        registry = Registry(settings, load=load_models)
        scoring = Scoring(registry, db, sessions)
        incidents = Incidents(settings, db, sessions, scoring, metrics)
        if start_sim:
            try:
                from prahari.sim.live import LiveSimulator

                incidents.sim = LiveSimulator(settings)
            except Exception as e:  # noqa: BLE001
                LOG.warning("simulator unavailable: %s", e)
        scoring.warmup()
        app.state.s = dict(
            engine=engine,
            sessions=sessions,
            db=db,
            registry=registry,
            scoring=scoring,
            incidents=incidents,
        )
        task = None
        if settings.detector_enabled:

            async def loop() -> None:
                while True:
                    await asyncio.sleep(settings.detector_interval_s)
                    await asyncio.to_thread(incidents.detect_once)

            task = asyncio.create_task(loop())
        yield
        if task:
            task.cancel()
        engine.dispose()

    app = FastAPI(title="Prahari API", version="1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["x-request-id"],
    )
    obs.install(app, metrics)

    def S(request: Request) -> dict:
        return request.app.state.s

    # ---- health / metrics ---------------------------------------------------------------
    @app.get("/health", response_model=sc.HealthResponse)
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def prom():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    # ---- topology / alarms / KPI --------------------------------------------------------
    @app.get("/v1/topology", response_model=sc.TopologyResponse)
    def topology(request: Request):
        return S(request)["db"].topology()

    @app.get("/v1/alarms", response_model=sc.AlarmsResponse)
    def alarms(
        request: Request,
        since: datetime | None = None,
        limit: int = Query(100, ge=1, le=1000),
    ):
        st = S(request)
        where, params = ("WHERE ts >= :s", {"s": naive(since)}) if since else ("", {})
        df = st["db"].alarms(where, params, f"ORDER BY alarm_id DESC LIMIT {int(limit)}")
        df = df.sort_values(["ts", "alarm_id"]).reset_index(drop=True)
        try:
            scores = st["scoring"].scores_for(df) if len(df) else {}
        except ApiError:
            scores = {}  # validator missing: still serve the alarms, unscored
        rows = []
        for r in df.itertuples():
            s_ = scores.get(r.alarm_id)
            rows.append(
                {
                    "alarm_id": r.alarm_id,
                    "ts": r.ts.to_pydatetime(),
                    "node_id": r.node_id,
                    "alarm_type": r.alarm_type,
                    "severity": r.severity,
                    "raw_text": r.raw_text,
                    "is_valid_pred": s_.is_valid if s_ else None,
                    "valid_prob": s_.valid_prob if s_ else None,
                }
            )
        return {"alarms": rows}

    @app.post("/v1/alarms/validate", response_model=sc.ValidateResponse)
    def validate(body: sc.ValidateRequest, request: Request):
        st = S(request)
        nodes = {n["node_id"] for n in st["db"].topology()["nodes"]}
        bad = sorted({a.node_id for a in body.alarms} - nodes)
        if bad:
            raise ApiError(422, f"unknown node_id(s): {bad}", "validation_error")
        df = pd.DataFrame(
            [{**a.model_dump(), "ts": pd.Timestamp(naive(a.ts))} for a in body.alarms]
        )
        if df.empty:
            return {"scores": [], "model": st["registry"].ref("validator"), "latency_ms": 0.0}
        res, ms = st["scoring"].validate(df)
        by = {x.alarm_id: x for x in res}
        return {
            "scores": [
                {
                    "alarm_id": a.alarm_id,
                    "is_valid_pred": by[a.alarm_id].is_valid,
                    "valid_prob": by[a.alarm_id].valid_prob,
                }
                for a in body.alarms
            ],
            "model": st["registry"].ref("validator"),
            "latency_ms": ms,
        }

    @app.get("/v1/kpi", response_model=sc.KpiResponse)
    def kpi(
        request: Request,
        node_id: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = Query(KPI_MAX_POINTS, ge=1, le=KPI_MAX_POINTS),
    ):
        db: Db = S(request)["db"]
        if node_id not in {n["node_id"] for n in db.topology()["nodes"]}:
            raise ApiError(404, f"node {node_id} not found")
        if end is None:
            last = db.last_kpi_ts()
            if last is None:
                raise ApiError(404, "no KPI data")
            e = last
        else:
            e = naive(end)
        s_ = naive(start) if start else e - timedelta(minutes=KPI_DEFAULT_MIN)
        if s_ > e:
            raise ApiError(422, "start must be <= end", "validation_error")
        df = db.kpi_series(node_id, s_, e, limit)
        pts = [
            {
                k: (None if pd.isna(v) else (v.to_pydatetime() if k == "ts" else float(v)))
                for k, v in row.items()
            }
            for row in df.to_dict("records")
        ]
        return {"node_id": node_id, "start": s_, "end": e, "points": pts}

    # ---- models -------------------------------------------------------------------------
    @app.post("/v1/anomaly/score", response_model=sc.AnomalyScoreResponse)
    def anomaly(body: sc.AnomalyScoreRequest, request: Request):
        if body.start > body.end:
            raise ApiError(422, "start must be <= end", "validation_error")
        return S(request)["scoring"].anomaly(naive(body.start), naive(body.end), body.node_ids)

    @app.post("/v1/rca", response_model=sc.RcaOutput)
    def rca(body: sc.WindowRequest, request: Request):
        if body.start >= body.end:
            raise ApiError(422, "start must be before end", "validation_error")
        return S(request)["scoring"].rca(naive(body.start), naive(body.end))

    @app.get("/v1/models", response_model=sc.ModelsResponse)
    def models(request: Request):
        st = S(request)
        out = []
        for c in st["registry"].cards():
            c["latency"] = latency_stats(st["db"].latency_samples(c["name"]))
            out.append(c)
        return {"models": out}

    # ---- incidents / actions ------------------------------------------------------------
    @app.get("/v1/incidents", response_model=sc.IncidentsResponse)
    def incidents(
        request: Request,
        limit: int = Query(50, ge=1, le=500),
        status: Literal["open", "resolved"] | None = None,
    ):
        st = S(request)
        q = select(DetectedIncident).order_by(DetectedIncident.start_ts.desc()).limit(limit)
        if status:
            q = q.where(DetectedIncident.status == status)
        with st["sessions"]() as s:
            return {"incidents": [st["incidents"].incident_out(s, i) for i in s.scalars(q)]}

    @app.get("/v1/incidents/{incident_id}", response_model=sc.IncidentOut)
    def incident(incident_id: str, request: Request):
        st = S(request)
        with st["sessions"]() as s:
            inc = s.get(DetectedIncident, incident_id)
            if inc is None:
                raise ApiError(404, f"incident {incident_id} not found")
            return st["incidents"].incident_out(s, inc)

    @app.post("/v1/incidents/{incident_id}/explain", response_model=sc.ExplainResponse)
    def explain(incident_id: str, request: Request, lang: Literal["en", "hi"] = "en"):
        out = S(request)["incidents"].explain(incident_id, lang)
        return {
            "incident_id": incident_id,
            "lang": lang,
            "narrative": out["narrative"],
            "runbook": out["runbook"],
            "llm": out["llm"],
        }

    def _decide(action_id: str, approve: bool, body: sc.DecisionRequest, request: Request):
        inc = S(request)["incidents"]
        act = inc.decide(action_id, approve, body.user, body.note)
        return {"action": inc.action_out(act)}

    @app.post("/v1/actions/{action_id}/approve", response_model=sc.DecisionResponse)
    def approve(action_id: str, body: sc.DecisionRequest, request: Request):
        return _decide(action_id, True, body, request)

    @app.post("/v1/actions/{action_id}/reject", response_model=sc.DecisionResponse)
    def reject(action_id: str, body: sc.DecisionRequest, request: Request):
        return _decide(action_id, False, body, request)

    # ---- drift --------------------------------------------------------------------------
    def _drift_out(row: DriftReport) -> dict:
        rep = dict(row.report_json or {})
        feats = rep.get("features", [])
        if isinstance(feats, dict):
            feats = [{"feature": k, **v} for k, v in feats.items()]
        return {
            "id": row.id,
            "ts": row.ts,
            "status": row.status,
            "features": feats,
            "summary": rep.get("summary", ""),
            "extra": {k: v for k, v in rep.items() if k not in ("features", "summary")},
        }

    @app.get("/v1/drift/latest", response_model=sc.DriftReportOut)
    def drift_latest(request: Request):
        with S(request)["sessions"]() as s:
            row = s.execute(select(DriftReport).order_by(DriftReport.id.desc()).limit(1)).scalar()
            if row is None:
                raise ApiError(404, "no drift report yet; POST /v1/drift/run")
            return _drift_out(row)

    @app.post("/v1/drift/run", response_model=sc.DriftReportOut)
    def drift_run(request: Request):
        st = S(request)
        runner = drift_runner or (lambda: _default_drift_runner(settings))
        try:
            runner()
        except ImportError as e:
            raise ApiError(503, f"drift monitor unavailable: {e}", "model_unavailable") from e
        with st["sessions"]() as s:
            row = s.execute(select(DriftReport).order_by(DriftReport.id.desc()).limit(1)).scalar()
            if row is None:
                raise ApiError(503, "drift monitor produced no report", "model_unavailable")
            audit(s, "api", "drift_run", {"report_id": row.id, "status": row.status})
            s.commit()
            return _drift_out(row)

    # ---- simulator (demo only) ----------------------------------------------------------
    def _sim(request: Request):
        sim = S(request)["incidents"].sim
        if sim is None:
            raise ApiError(503, "simulator unavailable", "sim_unavailable")
        return sim

    @app.post("/v1/sim/inject", response_model=sc.InjectResponse)
    def inject(body: sc.InjectRequest, request: Request):
        sim = _sim(request)
        try:
            r = sim.inject(body.fault_type, body.node_id)
        except ValueError as e:
            raise ApiError(422, str(e), "validation_error") from e
        with S(request)["sessions"]() as s:
            audit(
                s,
                "api",
                "fault_injected",
                {k: r[k] for k in ("incident_id", "fault_type", "root_node_id")},
            )
            s.commit()
        return {
            "incident_id": r["incident_id"],
            "fault_type": r["fault_type"],
            "node_id": r["root_node_id"],
            "injected_at": pd.Timestamp(r["start_ts"]).to_pydatetime(),
        }

    @app.post("/v1/sim/scenario", response_model=sc.ScenarioResponse)
    def scenario(body: sc.ScenarioRequest, request: Request):
        sim = _sim(request)
        try:
            r = sim.scenario(body.name)
        except ValueError as e:
            raise ApiError(422, str(e), "validation_error") from e
        with S(request)["sessions"]() as s:
            audit(s, "api", "scenario_started", {"name": body.name, "vendor": r.get("vendor")})
            s.commit()
        return {
            "name": body.name,
            "started": True,
            "detail": f"vendor {r['vendor']} latency x{r['latency_mult']}",
        }

    # ---- audit --------------------------------------------------------------------------
    @app.get("/v1/audit", response_model=sc.AuditResponse)
    def audit_log(request: Request, limit: int = Query(100, ge=1, le=1000)):
        with S(request)["sessions"]() as s:
            rows = s.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(limit))
            return {
                "entries": [
                    {
                        "id": r.id,
                        "ts": r.ts,
                        "actor": r.actor,
                        "event": r.event,
                        "payload": r.payload_json,
                    }
                    for r in rows
                ]
            }

    return app


app = create_app()
