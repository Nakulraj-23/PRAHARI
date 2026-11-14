"""Pydantic models for every API request/response and the shared RCA/governance contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

NodeType = Literal["core", "agg", "access", "cell"]
Severity = Literal["minor", "major", "critical"]
FaultType = Literal[
    "fibre_cut", "power_failure", "congestion", "hardware_degradation", "config_error"
]
ActionStatus = Literal["proposed", "approved", "rejected", "executed", "blocked"]
DriftStatus = Literal["green", "amber", "red"]
GovDecision = Literal["auto_approved", "needs_approval", "blocked"]
CheckStatus = Literal["pass", "warn", "fail"]


# ---- health / topology -------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"


class NodeOut(BaseModel):
    node_id: str
    node_type: NodeType
    region: str
    vendor: str


class LinkOut(BaseModel):
    link_id: str
    src: str
    dst: str
    capacity_mbps: float


class TopologyResponse(BaseModel):
    nodes: list[NodeOut]
    links: list[LinkOut]


# ---- alarms ------------------------------------------------------------------
class AlarmIn(BaseModel):
    """An alarm as submitted for validation (no ground-truth labels)."""

    alarm_id: str
    ts: datetime
    node_id: str
    alarm_type: str
    severity: Severity
    raw_text: str


class AlarmOut(AlarmIn):
    is_valid_pred: bool | None = None
    valid_prob: float | None = Field(default=None, ge=0, le=1)


class AlarmsResponse(BaseModel):
    alarms: list[AlarmOut]


class AlarmScore(BaseModel):
    alarm_id: str
    is_valid_pred: bool
    valid_prob: float = Field(ge=0, le=1)


class ValidateRequest(BaseModel):
    alarms: list[AlarmIn]


class ValidateResponse(BaseModel):
    scores: list[AlarmScore]
    model: ModelRef
    latency_ms: float


# ---- anomaly -----------------------------------------------------------------
class WindowRequest(BaseModel):
    start: datetime
    end: datetime


class AnomalyScoreRequest(BaseModel):
    start: datetime
    end: datetime
    node_ids: list[str] | None = None


class NodeAnomaly(BaseModel):
    node_id: str
    score: float
    is_anomalous: bool


class AnomalyScoreResponse(BaseModel):
    window: Window
    scores: list[NodeAnomaly]
    model: ModelRef
    latency_ms: float


# ---- RCA (contract shared by A3, A4, A5, A7) ---------------------------------
class Window(BaseModel):
    start: datetime
    end: datetime


class ModelRef(BaseModel):
    name: str
    version: str


class RootCandidate(BaseModel):
    node_id: str
    prob: float = Field(ge=0, le=1)


class FaultTypePred(BaseModel):
    label: FaultType
    prob: float = Field(ge=0, le=1)


class RcaEvidence(BaseModel):
    alarmed_nodes: list[str]
    valid_alarm_count: int
    noise_alarm_count: int
    kpi_deltas: dict[str, dict[str, float]]


class RcaOutput(BaseModel):
    window: Window
    root_candidates: list[RootCandidate]
    fault_type: FaultTypePred
    evidence: RcaEvidence
    model: ModelRef
    latency_ms: float


# ---- governance (A5 produces, A4 stores, A7 renders) -------------------------
class GovCheck(BaseModel):
    id: str
    name: str
    status: CheckStatus
    detail: str


class GovernanceReport(BaseModel):
    decision: GovDecision
    checks: list[GovCheck]


# ---- incidents / actions -----------------------------------------------------
class ActionOut(BaseModel):
    action_id: str
    incident_ref: str
    proposed_action: str
    target_node_id: str
    status: ActionStatus
    governance: GovernanceReport | None = None
    decided_by: str | None = None
    decided_at: datetime | None = None


class IncidentOut(BaseModel):
    incident_id: str
    start_ts: datetime
    end_ts: datetime | None = None
    status: Literal["open", "resolved"] = "open"
    rca: RcaOutput
    action: ActionOut | None = None
    governance: GovernanceReport | None = None


class IncidentsResponse(BaseModel):
    incidents: list[IncidentOut]


class ExplainResponse(BaseModel):
    incident_id: str
    lang: Literal["en", "hi"]
    narrative: str
    runbook: RunbookCitation
    llm: ModelRef


class RunbookCitation(BaseModel):
    runbook_id: str
    title: str
    section: str
    excerpt: str


class DecisionRequest(BaseModel):
    user: str
    note: str = ""


class DecisionResponse(BaseModel):
    action: ActionOut


# ---- models ------------------------------------------------------------------
class ModelCard(BaseModel):
    name: str
    version: str
    trained_at: datetime
    data_hash: str
    features: list[str]
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    intended_use: str
    limitations: str


class ModelsResponse(BaseModel):
    models: list[ModelCard]


# ---- drift -------------------------------------------------------------------
class DriftFeature(BaseModel):
    feature: str
    psi: float
    ks_pvalue: float | None = None
    status: DriftStatus


class DriftReportOut(BaseModel):
    id: int
    ts: datetime
    status: DriftStatus
    features: list[DriftFeature]
    summary: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


# ---- simulator (demo only) ---------------------------------------------------
class InjectRequest(BaseModel):
    fault_type: FaultType
    node_id: str | None = None


class InjectResponse(BaseModel):
    incident_id: str
    fault_type: FaultType
    node_id: str
    injected_at: datetime


class ScenarioRequest(BaseModel):
    name: str  # e.g. "firmware_drift"


class ScenarioResponse(BaseModel):
    name: str
    started: bool
    detail: str = ""


# ---- audit -------------------------------------------------------------------
class AuditEntry(BaseModel):
    id: int
    ts: datetime
    actor: str
    event: str
    payload: dict[str, Any]


class AuditResponse(BaseModel):
    entries: list[AuditEntry]


for _m in (ValidateResponse, AnomalyScoreResponse, ExplainResponse):
    _m.model_rebuild()
