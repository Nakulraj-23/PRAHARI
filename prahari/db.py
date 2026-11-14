"""SQLAlchemy models for every table in CONTRACTS.md.

Columns prefixed ``label_`` and the ``incidents`` table are ground truth and must
never be used as model features or read by the serving path.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from prahari.config import Settings, get_settings


class Base(DeclarativeBase):
    """Declarative base."""


class Node(Base):
    __tablename__ = "nodes"
    node_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    node_type: Mapped[str] = mapped_column(String(16))  # core/agg/access/cell
    region: Mapped[str] = mapped_column(String(64))
    vendor: Mapped[str] = mapped_column(String(64))


class Link(Base):
    __tablename__ = "links"
    link_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    src: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"))
    dst: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"))
    capacity_mbps: Mapped[float] = mapped_column(Float)


class KpiSample(Base):
    __tablename__ = "kpi_samples"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"))
    latency_ms: Mapped[float] = mapped_column(Float)
    packet_loss_pct: Mapped[float] = mapped_column(Float)
    throughput_mbps: Mapped[float] = mapped_column(Float)
    cpu_pct: Mapped[float] = mapped_column(Float)
    mem_pct: Mapped[float] = mapped_column(Float)
    temp_c: Mapped[float] = mapped_column(Float)
    __table_args__ = (Index("ix_kpi_node_ts", "node_id", "ts"),)


class Alarm(Base):
    __tablename__ = "alarms"
    alarm_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime)
    node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"))
    alarm_type: Mapped[str] = mapped_column(String(64))
    severity: Mapped[str] = mapped_column(String(16))  # minor/major/critical
    raw_text: Mapped[str] = mapped_column(String)
    # ground truth: never a model feature, never read by serving
    label_is_true: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    label_incident_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    __table_args__ = (Index("ix_alarms_ts", "ts"),)


class Incident(Base):
    """Ground-truth incidents injected by the simulator."""

    __tablename__ = "incidents"
    incident_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    start_ts: Mapped[datetime] = mapped_column(DateTime)
    end_ts: Mapped[datetime] = mapped_column(DateTime)
    root_node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"))
    fault_type: Mapped[str] = mapped_column(String(32))


class Prediction(Base):
    __tablename__ = "predictions"
    pred_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime)
    kind: Mapped[str] = mapped_column(String(32))
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64))
    input_ref: Mapped[str] = mapped_column(String)
    output_json: Mapped[dict] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float)


class Action(Base):
    __tablename__ = "actions"
    action_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    incident_ref: Mapped[str] = mapped_column(String(64))
    proposed_action: Mapped[str] = mapped_column(String)
    target_node_id: Mapped[str] = mapped_column(ForeignKey("nodes.node_id"))
    # proposed/approved/rejected/executed/blocked
    status: Mapped[str] = mapped_column(String(16))
    governance_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # wall-clock UTC time the action was proposed; feeds governance G9 (duplicate-action check)
    created_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class DetectedIncident(Base):
    """Incidents found by the serving-path detector (not ground truth; see ``Incident``)."""

    __tablename__ = "detected_incidents"
    incident_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime)
    start_ts: Mapped[datetime] = mapped_column(DateTime)  # first valid alarm
    end_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(16))  # open/resolved
    window_start: Mapped[datetime] = mapped_column(DateTime)
    window_end: Mapped[datetime] = mapped_column(DateTime)
    alarm_ids: Mapped[list] = mapped_column(JSON)  # valid alarms in the cluster
    rca_json: Mapped[dict] = mapped_column(JSON)
    agent_json: Mapped[dict] = mapped_column(JSON)  # {lang: AgentOutput}
    governance_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    action_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    __table_args__ = (Index("ix_detected_start", "start_ts"),)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime)
    actor: Mapped[str] = mapped_column(String(64))
    event: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[dict] = mapped_column(JSON)


class DriftReport(Base):
    __tablename__ = "drift_reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(8))  # green/amber/red
    report_json: Mapped[dict] = mapped_column(JSON)


def get_engine(settings: Settings | None = None) -> Engine:
    """Create an engine from settings."""
    settings = settings or get_settings()
    if not settings.db_url.startswith("sqlite"):
        return create_engine(settings.db_url, future=True)
    engine = create_engine(settings.db_url, future=True, connect_args={"timeout": 30})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _):
        """WAL lets readers run during a write; busy_timeout waits instead of failing at once."""
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    return engine


def get_sessionmaker(settings: Settings | None = None) -> sessionmaker:
    """Return a session factory bound to the configured engine."""
    return sessionmaker(get_engine(settings), expire_on_commit=False)


def ensure_schema(engine: Engine) -> None:
    """Create missing tables and add columns introduced after a DB was first seeded."""
    Base.metadata.create_all(engine)
    cols = {c["name"] for c in inspect(engine).get_columns("actions")}
    if "created_at" not in cols:
        with engine.begin() as c:
            c.execute(text("ALTER TABLE actions ADD COLUMN created_at TIMESTAMP"))


def init_db(settings: Settings | None = None) -> Engine:
    """Create all tables (idempotent) and return the engine."""
    engine = get_engine(settings)
    ensure_schema(engine)
    return engine
