"""A5 agent tests: SQL guard, retrieval, tools, LLM adapters, deterministic output."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import text

from prahari.agent import analyze, explain
from prahari.agent.agent import get_index
from prahari.agent.flow import dependent_count, window_severity
from prahari.agent.llm import (
    AnthropicProvider,
    LLMProvider,
    LLMResponse,
    OpenAICompatibleProvider,
    ToolCall,
    make_provider,
)
from prahari.agent.sqlguard import SqlRejected, run_sql, validate_sql
from prahari.agent.tools import Toolbox
from prahari.config import Settings
from prahari.db import init_db

ROOT = Path(__file__).resolve().parent.parent
RCA = json.loads((ROOT / "fixtures" / "rca.json").read_text())
S = Settings(llm_provider="none")


@pytest.fixture()
def engine(tmp_path):
    eng = init_db(Settings(db_url=f"sqlite:///{tmp_path / 'a.db'}"))
    with eng.begin() as c:
        for nid, t in [
            ("CORE-02", "core"),
            ("AGG-07", "agg"),
            ("ACC-21", "access"),
            ("CELL-113", "cell"),
            ("CELL-114", "cell"),
        ]:
            c.execute(text("INSERT INTO nodes VALUES (:i,:t,'north','Nexa')"), {"i": nid, "t": t})
        for i, (a, b) in enumerate(
            [
                ("CORE-02", "AGG-07"),
                ("AGG-07", "ACC-21"),
                ("ACC-21", "CELL-113"),
                ("ACC-21", "CELL-114"),
            ]
        ):
            c.execute(
                text("INSERT INTO links VALUES (:i,:a,:b,1000)"), {"i": f"L{i}", "a": a, "b": b}
            )
        c.execute(
            text(
                "INSERT INTO alarms VALUES "
                "('A1','2026-03-02 10:11:00','AGG-07','los','critical','x',1,'INC-9')"
            )
        )
        c.execute(
            text(
                "INSERT INTO incidents VALUES ('INC-9','2026-03-02 10:10:00',"
                "'2026-03-02 11:00:00','AGG-07','fibre_cut')"
            )
        )
    return eng


# ---- SQL tool ----------------------------------------------------------------
@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE nodes",
        "UPDATE nodes SET region='x'",
        "DELETE FROM nodes",
        "SELECT 1 FROM nodes; DROP TABLE nodes",
        "SELECT * FROM nodes; SELECT * FROM links",
        "SELECT * FROM incidents",
        "SELECT root_node_id FROM incidents",
        "SELECT label_is_true FROM alarms",
        "SELECT alarm_id, LABEL_INCIDENT_ID FROM alarms",
        "SELECT * FROM alarms",
        "SELECT a.* FROM alarms a",
        "SELECT * FROM nodes, incidents",
        "SELECT * FROM sqlite_master",
        "WITH x AS (SELECT * FROM incidents) SELECT * FROM x",
        "PRAGMA table_info(nodes)",
        "",
    ],
)
def test_sql_rejects(engine, sql):
    with pytest.raises(SqlRejected):
        run_sql(engine, sql)


def test_sql_comment_cannot_hide_second_statement():
    with pytest.raises(SqlRejected):
        validate_sql("SELECT 1 FROM nodes /* ; */ ; DROP TABLE nodes")


def test_sql_allows_select_and_limits_rows(engine):
    res = run_sql(engine, "SELECT node_id FROM nodes ORDER BY node_id", limit=2)
    assert res["columns"] == ["node_id"] and len(res["rows"]) == 2 and res["truncated"]
    ok = run_sql(engine, "SELECT alarm_id, severity FROM alarms;")
    assert ok["rows"] == [["A1", "critical"]]
    assert run_sql(engine, "SELECT count(*) FROM alarms")["rows"] == [[1]]


def test_authorizer_blocks_even_if_parser_is_bypassed(engine):
    """Defence in depth: the sqlite authorizer denies label_ reads on its own."""
    raw = engine.raw_connection()
    try:
        from prahari.agent.sqlguard import _authorizer

        raw.driver_connection.set_authorizer(_authorizer)
        with pytest.raises(Exception, match="prohibited"):
            raw.cursor().execute("SELECT label_is_true FROM alarms").fetchall()
    finally:
        raw.close()


def test_tool_errors_are_returned_not_raised(engine):
    tb = Toolbox(RCA, get_index(S), engine)
    assert "error" in tb.call("sql_query", {"sql": "DROP TABLE nodes"})
    assert "error" in tb.call(
        "propose_action", {"action": "format_disk", "target_node_id": "AGG-07"}
    )
    assert "error" in tb.call(
        "propose_action", {"action": "restart_node", "target_node_id": "CELL-1"}
    )
    assert "error" in tb.call("nope", {})
    assert (
        tb.call("propose_action", {"action": "no_action", "target_node_id": "AGG-07"})["action"]
        == "no_action"
    )


# ---- retrieval ---------------------------------------------------------------
def test_runbook_count_and_retrieval_accuracy():
    idx = get_index(S)
    ids = {c.runbook_id for c in idx.chunks}
    assert len(ids) == 15
    tiers = {"core": "core", "agg": "aggregation", "edge": "access cell edge"}
    ok = total = 0
    for fault in [
        "fibre_cut",
        "power_failure",
        "congestion",
        "hardware_degradation",
        "config_error",
    ]:
        for tier, words in tiers.items():
            total += 1
            top = idx.search(f"{fault.replace('_', ' ')} {words}", 1)[0]
            ok += top["runbook_id"].endswith(tier.upper()) and fault in {
                c.meta["fault_type"] for c in idx.chunks if c.runbook_id == top["runbook_id"]
            }
    assert ok == total == 15


def test_search_returns_scores_sorted():
    hits = get_index(S).search("fibre cut loss of signal", 3)
    assert len(hits) == 3 and hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"] > 0


# ---- deterministic agent -----------------------------------------------------
def test_none_provider_is_complete_english_and_hindi():
    en = explain(RCA, "en", "INC-1", settings=S)
    hi = explain(RCA, "hi", "INC-1", settings=S)
    assert en.llm.name == "none" and en.proposed_action.action == "reroute_traffic"
    assert en.runbook.runbook_id == "RB-FIBRE-AGG" and "AGG-07" in en.narrative
    assert any("devanagari" in __import__("unicodedata").name(ch).lower() for ch in hi.narrative)
    assert "AGG-07" in hi.narrative and "83%" in hi.narrative
    assert "rca" in en.evidence_refs
    assert (
        explain(RCA, "en", settings=S).model_dump() == explain(RCA, "en", settings=S).model_dump()
    )


def test_agent_uses_db_for_node_type(engine):
    out = explain(RCA, "en", engine=engine, settings=S)
    assert any(r.startswith("sql:") for r in out.evidence_refs)


# ---- LLM adapters ------------------------------------------------------------
class Scripted(LLMProvider):
    name, model = "scripted", "s1"

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def chat(self, system, messages, tools):
        self.calls += 1
        return (
            self.script.pop(0)
            if self.script
            else LLMResponse(tool_calls=[ToolCall("x", "get_rca", {})])
        )


def test_llm_tool_loop_completes():
    p = Scripted(
        [
            LLMResponse(tool_calls=[ToolCall("1", "get_rca", {})]),
            LLMResponse(
                tool_calls=[ToolCall("2", "search_runbooks", {"query": "fibre cut aggregation"})]
            ),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "3",
                        "propose_action",
                        {"action": "dispatch_field_team", "target_node_id": "AGG-07"},
                    )
                ]
            ),
            LLMResponse(text="Fibre cut on AGG-07 (83%)."),
        ]
    )
    out = explain(RCA, settings=S, provider=p)
    assert out.llm.name == "scripted" and out.steps == 4
    assert out.proposed_action.action == "dispatch_field_team"
    assert out.narrative == "Fibre cut on AGG-07 (83%)."


def test_llm_loop_max_steps_falls_back_to_template():
    p = Scripted([])  # always asks for get_rca, never finishes
    out = explain(RCA, settings=S, provider=p, max_steps=3)
    assert p.calls == 3
    assert out.llm.name == "none" and out.proposed_action.action == "reroute_traffic"


def test_narrative_is_redacted():
    p = Scripted(
        [LLMResponse(text="Contact ops@example.com or +91 98765 43210, host 10.1.2.3 on AGG-07")]
    )
    out = explain(RCA, settings=S, provider=p)
    assert "@" not in out.narrative and "10.1.2.3" not in out.narrative and out.redactions >= 3


def test_anthropic_wire_format_and_parsing():
    seen = {}

    def transport(url, headers, body):
        seen.update(url=url, headers=headers, body=body)
        return {
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "tool_use", "id": "t1", "name": "get_rca", "input": {}},
            ]
        }

    p = AnthropicProvider("m", api_key="k", transport=transport)
    msgs = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "text": "",
            "tool_calls": [{"id": "a", "name": "get_rca", "args": {}}],
        },
        {"role": "tool", "tool_call_id": "a", "name": "get_rca", "content": "{}"},
    ]
    r = p.chat(
        "sys", msgs, [{"name": "get_rca", "description": "d", "parameters": {"type": "object"}}]
    )
    assert seen["url"].endswith("/v1/messages") and seen["headers"]["x-api-key"] == "k"
    assert seen["body"]["messages"][-1]["content"][0]["type"] == "tool_result"
    assert r.text == "hi" and r.tool_calls[0].name == "get_rca"


def test_openai_compatible_wire_format_and_parsing():
    seen = {}

    def transport(url, headers, body):
        seen.update(url=url, headers=headers, body=body)
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "get_rca", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }

    p = OpenAICompatibleProvider("m", base_url="http://x/v1/", api_key="k", transport=transport)
    r = p.chat("sys", [{"role": "user", "content": "go"}], [])
    assert (
        seen["url"] == "http://x/v1/chat/completions"
        and seen["headers"]["authorization"] == "Bearer k"
    )
    assert r.tool_calls[0].id == "c1"


def test_make_provider():
    assert make_provider("none").name == "none" and make_provider("stub").name == "none"
    assert make_provider("anthropic", api_key="k").name == "anthropic"
    assert make_provider("openai_compatible", base_url="http://x").name == "openai_compatible"
    with pytest.raises(ValueError):
        make_provider("bogus")


# ---- flow --------------------------------------------------------------------
def test_dependents_and_severity(engine):
    assert dependent_count(engine, "AGG-07") == 3 and dependent_count(engine, "CELL-113") == 0
    assert dependent_count(engine, "NOPE") is None
    assert window_severity(engine, RCA) == "critical"


def test_analyze_end_to_end(engine, tmp_path):
    out, gov = analyze(
        RCA,
        engine=engine,
        settings=Settings(artifacts_dir=tmp_path),
        drift_status="green",
        now=datetime(2026, 3, 2, 10, 20),
    )
    by = {c.id: c for c in gov.checks}
    assert out.proposed_action.target_node_id == "AGG-07"
    assert by["G3"].status == "pass" and by["G4"].status == "pass" and by["G10"].status == "warn"
    assert by["G7"].status == "fail" and gov.decision == "blocked"  # no card in tmp artifacts
