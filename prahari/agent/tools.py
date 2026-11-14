"""The four agent tools and their JSON schemas."""

from __future__ import annotations

from typing import Any

from sqlalchemy.engine import Engine

from prahari.agent.retrieval import RunbookIndex
from prahari.agent.sqlguard import SqlRejected, run_sql
from prahari.agent.types import ACTIONS

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "sql_query",
        "description": "Run one read-only SELECT on nodes, links, kpi_samples, "
        "alarms (name columns), predictions, actions, audit_log, drift_reports. Max 200 rows.",
        "parameters": {
            "type": "object",
            "properties": {"sql": {"type": "string"}},
            "required": ["sql"],
        },
    },
    {
        "name": "get_rca",
        "description": "Return the RCA result for this incident.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "search_runbooks",
        "description": "TF-IDF search over runbooks; returns top-k sections.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["query"],
        },
    },
    {
        "name": "propose_action",
        "description": "Propose one remediation action (not executed).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "target_node_id": {"type": "string"},
                "rationale": {"type": "string"},
            },
            "required": ["action", "target_node_id"],
        },
    },
]


class Toolbox:
    """Holds tool state for one incident and records which evidence was used."""

    def __init__(self, rca: dict, index: RunbookIndex, engine: Engine | None = None):
        self.rca, self.index, self.engine = rca, index, engine
        self.evidence_refs: list[str] = []
        self.runbook_hit: dict | None = None
        self.proposal: dict | None = None

    def call(self, name: str, args: dict) -> Any:
        """Dispatch a tool call; errors are returned as {"error": ...} for the model."""
        try:
            fn = {
                "sql_query": self.sql_query,
                "get_rca": self.get_rca,
                "search_runbooks": self.search_runbooks,
                "propose_action": self.propose_action,
            }[name]
        except KeyError:
            return {"error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except (SqlRejected, ValueError, TypeError) as e:
            return {"error": str(e)}

    def sql_query(self, sql: str) -> dict:
        """SELECT-only query; see `sqlguard`."""
        if self.engine is None:
            raise ValueError("no database attached")
        res = run_sql(self.engine, sql)
        self.evidence_refs.append(f"sql:{' '.join(sql.split())[:80]}")
        return res

    def get_rca(self) -> dict:
        """The RCA output for this incident."""
        self.evidence_refs.append("rca")
        return self.rca

    def search_runbooks(self, query: str, k: int = 3) -> list[dict]:
        """Top-k runbook sections with scores."""
        hits = self.index.search(query, max(1, min(int(k), 10)))
        if hits and self.runbook_hit is None:
            self.runbook_hit = hits[0]
        for h in hits[:1]:
            self.evidence_refs.append(f"runbook:{h['runbook_id']}#{h['section']}")
        return hits

    def propose_action(self, action: str, target_node_id: str, rationale: str = "") -> dict:
        """Validate the action against the fixed enum and record the proposal."""
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {list(ACTIONS)}")
        known = {c["node_id"] for c in self.rca.get("root_candidates", [])}
        if known and target_node_id not in known:
            raise ValueError("target_node_id must be a root candidate from the RCA")
        risk = (self.runbook_hit or {}).get("risk") or "unknown"
        self.proposal = {
            "action": action,
            "target_node_id": target_node_id,
            "rationale": rationale,
            "risk": risk,
        }
        self.evidence_refs.append(f"action:{action}")
        return self.proposal
