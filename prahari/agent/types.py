"""Shared types for the RCA agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from prahari.schemas import ModelRef, RunbookCitation

ACTIONS = (
    "reroute_traffic",
    "restart_node",
    "rollback_config",
    "dispatch_field_team",
    "throttle_traffic",
    "no_action",
)
ActionName = Literal[
    "reroute_traffic",
    "restart_node",
    "rollback_config",
    "dispatch_field_team",
    "throttle_traffic",
    "no_action",
]


class ProposedAction(BaseModel):
    """An action proposal; never executed by the agent itself."""

    action: ActionName
    target_node_id: str
    rationale: str = ""
    risk: str = "unknown"


class AgentOutput(BaseModel):
    """What the agent returns for one incident."""

    incident_id: str | None = None
    lang: Literal["en", "hi"] = "en"
    narrative: str
    runbook: RunbookCitation
    proposed_action: ProposedAction
    evidence_refs: list[str] = Field(default_factory=list)
    llm: ModelRef
    steps: int = 0
    redactions: int = 0
