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
    return create_engine(settings.db_url, future=True)


def get_sessionmaker(settings: Settings | None = None) -> sessionmaker:
    """Return a session factory bound to the configured engine."""
    return sessionmaker(get_engine(settings), expire_on_commit=False)


def init_db(settings: Settings | None = None) -> Engine:
    """Create all tables (idempotent) and return the engine."""
    engine = get_engine(settings)
    Base.metadata.create_all(engine)
    return engine
