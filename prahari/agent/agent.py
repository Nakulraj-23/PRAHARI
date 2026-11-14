"""The RCA agent: tool loop (LLM providers) or deterministic plan (`none`)."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.engine import Engine

from prahari.agent import templates
from prahari.agent.llm import LLMProvider, NoneProvider, make_provider
from prahari.agent.retrieval import RunbookIndex, load_runbooks
from prahari.agent.tools import TOOL_SCHEMAS, Toolbox
from prahari.agent.types import AgentOutput, ProposedAction
from prahari.config import Settings, get_settings
from prahari.governance.redact import redact
from prahari.schemas import ModelRef, RunbookCitation

MAX_STEPS = 6
TIER_WORDS = {
    "core": "core",
    "agg": "aggregation",
    "access": "access cell edge",
    "cell": "access cell edge",
}
PREFIX_TYPE = {"CORE": "core", "AGG": "agg", "ACC": "access", "CELL": "cell"}
SYSTEM = (
    "You are a telecom NOC assistant. Use get_rca, search_runbooks and optionally sql_query, "
    "then call propose_action once, then write a short narrative. Mention only node ids and "
    "numbers that appear in the RCA evidence. Never reveal secrets or personal data."
)

_index_cache: dict[str, RunbookIndex] = {}


def get_index(settings: Settings) -> RunbookIndex:
    """Build (and cache) the runbook index."""
    key = str(settings.runbooks_dir)
    if key not in _index_cache:
        _index_cache[key] = RunbookIndex(load_runbooks(Path(settings.runbooks_dir)))
    return _index_cache[key]


def _node_type(tb: Toolbox, node: str) -> str:
    if tb.engine is not None:
        res = tb.call("sql_query", {"sql": f"SELECT node_type FROM nodes WHERE node_id = '{node}'"})
        if isinstance(res, dict) and res.get("rows"):
            return res["rows"][0][0]
    return PREFIX_TYPE.get(node.split("-")[0], "access")


def _top_kpi(rca: dict, node: str) -> tuple[str, float] | None:
    deltas = rca["evidence"].get("kpi_deltas", {}).get(node) or {}
    if not deltas:
        return None
    name = max(deltas, key=lambda k: abs(deltas[k]))
    return name, float(deltas[name])


def _deterministic(tb: Toolbox, lang: str) -> tuple[str, int]:
    rca = tb.call("get_rca", {})
    node = rca["root_candidates"][0]["node_id"]
    fault = rca["fault_type"]["label"]
    ntype = _node_type(tb, node)
    query = f"{fault.replace('_', ' ')} {TIER_WORDS.get(ntype, ntype)}"
    hits = tb.call("search_runbooks", {"query": query, "k": 3})
    top = hits[0]
    rem = next(
        (
            h
            for h in tb.index.search(f"{top['runbook_id']} remediation", 20)
            if h["runbook_id"] == top["runbook_id"] and h["section"] == "Remediation"
        ),
        top,
    )
    tb.runbook_hit = {**rem, "action": top["action"], "risk": top["risk"]}
    action = top["action"] or "no_action"
    tb.call(
        "propose_action",
        {"action": action, "target_node_id": node, "rationale": f"runbook {top['runbook_id']}"},
    )
    facts = {
        "node": node,
        "pct": round(rca["root_candidates"][0]["prob"] * 100),
        "fault": fault,
        "valid": rca["evidence"]["valid_alarm_count"],
        "noise": rca["evidence"]["noise_alarm_count"],
        "n_alarmed": len(rca["evidence"]["alarmed_nodes"]),
        "kpi": _top_kpi(rca, node),
        "runbook": top["runbook_id"],
        "action": action,
    }
    if facts["kpi"]:
        tb.evidence_refs.append(f"rca.evidence.kpi_deltas.{node}.{facts['kpi'][0]}")
    tb.evidence_refs += [
        "rca.root_candidates[0]",
        "rca.fault_type",
        "rca.evidence.valid_alarm_count",
    ]
    return templates.render(lang, facts), 4


def _llm_loop(tb: Toolbox, provider: LLMProvider, lang: str, max_steps: int) -> tuple[str, int]:
    messages: list[dict] = [
        {
            "role": "user",
            "content": "Analyse this incident; answer in "
            + ("Hindi." if lang == "hi" else "English."),
        }
    ]
    text, steps = "", 0
    for step in range(1, max_steps + 1):
        steps = step
        resp = provider.chat(SYSTEM, messages, TOOL_SCHEMAS)
        if not resp.tool_calls:
            text = resp.text
            break
        messages.append(
            {
                "role": "assistant",
                "text": resp.text,
                "tool_calls": [
                    {"id": c.id, "name": c.name, "args": c.args} for c in resp.tool_calls
                ],
            }
        )
        for c in resp.tool_calls:
            out = tb.call(c.name, c.args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": c.id,
                    "name": c.name,
                    "content": json.dumps(out, default=str),
                }
            )
    return text, steps


def explain(
    rca: dict,
    lang: str = "en",
    incident_id: str | None = None,
    engine: Engine | None = None,
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    max_steps: int = MAX_STEPS,
) -> AgentOutput:
    """Produce narrative, cited runbook, proposed action and evidence refs for one RCA result.

    With the `none` provider (default) everything is deterministic. With an LLM provider the
    tool loop runs for at most `max_steps` turns; if the model does not finish (no narrative,
    no runbook search or no proposal) the deterministic result fills the gap.
    """
    settings = settings or get_settings()
    provider = provider or make_provider(settings.llm_provider, settings.llm_model)
    tb = Toolbox(rca, get_index(settings), engine)
    llm_used = not isinstance(provider, NoneProvider)
    steps, text = 0, ""
    if llm_used:
        text, steps = _llm_loop(tb, provider, lang, max_steps)
    served_by = provider
    if not (llm_used and text and tb.runbook_hit and tb.proposal):
        det_tb = Toolbox(rca, tb.index, engine)
        det_text, n = _deterministic(det_tb, lang)
        if not text:
            text, served_by = det_text, NoneProvider()
        tb.runbook_hit = tb.runbook_hit or det_tb.runbook_hit
        tb.proposal = tb.proposal or det_tb.proposal
        tb.evidence_refs += det_tb.evidence_refs
        steps += n
    text, n_red = redact(text)
    hit = tb.runbook_hit or {}
    return AgentOutput(
        incident_id=incident_id,
        lang=lang,
        narrative=text,
        runbook=RunbookCitation(
            runbook_id=hit["runbook_id"],
            title=hit["title"],
            section=hit["section"],
            excerpt=hit["excerpt"],
        ),
        proposed_action=ProposedAction(**(tb.proposal or {})),
        evidence_refs=list(dict.fromkeys(tb.evidence_refs)),
        llm=ModelRef(name=served_by.name, version=served_by.model or "0"),
        steps=steps,
        redactions=n_red,
    )
