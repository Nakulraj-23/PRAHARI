"""One passing and one failing case per governance check, plus the decision rule."""

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from prahari.agent import explain
from prahari.config import Settings
from prahari.governance import REGISTRY, GovContext, decide, evaluate, load_policy
from prahari.governance.redact import find_sensitive, redact
from prahari.schemas import GovCheck

ROOT = Path(__file__).resolve().parent.parent
RCA = json.loads((ROOT / "fixtures" / "rca.json").read_text())
RCA["model"] = {"name": "rca_gnn", "version": "v1"}  # tests use a tmp artifacts dir only
POLICY = load_policy(ROOT / "prahari" / "governance" / "policy.yaml")
NOW = datetime(2026, 3, 2, 10, 30)


@pytest.fixture()
def arts(tmp_path):
    d = tmp_path / "rca_gnn"
    d.mkdir()
    (d / "v1.card.json").write_text("{}")
    return tmp_path


def make_ctx(arts, **over):
    out = explain(RCA, "en", settings=Settings(llm_provider="none")).model_dump(mode="json")
    base = dict(
        rca=copy.deepcopy(RCA),
        output=out,
        artifacts_dir=arts,
        drift_status="green",
        severity="minor",
        dependents=3,
        recent_actions=[],
        now=NOW,
    )
    base.update(over)
    return GovContext(**base)


def run(cid, ctx):
    return REGISTRY[cid](POLICY).evaluate(ctx)


def with_rca(arts, **mut):
    rca = copy.deepcopy(RCA)
    for k, v in mut.items():
        rca[k] = v
    return make_ctx(arts, rca=rca)


def test_registry_has_ten_checks():
    assert sorted(REGISTRY, key=lambda x: int(x[1:])) == [f"G{i}" for i in range(1, 11)]


def test_g1(arts):
    assert run("G1", make_ctx(arts)).status == "pass"
    low = with_rca(
        arts,
        root_candidates=[{"node_id": "AGG-07", "prob": 0.4}, {"node_id": "CORE-02", "prob": 0.1}],
    )
    assert run("G1", low).status == "fail"
    mid = with_rca(
        arts,
        root_candidates=[{"node_id": "AGG-07", "prob": 0.6}, {"node_id": "CORE-02", "prob": 0.1}],
    )
    assert run("G1", mid).status == "warn"


def test_g2(arts):
    assert run("G2", make_ctx(arts)).status == "pass"
    tie = with_rca(
        arts,
        root_candidates=[{"node_id": "AGG-07", "prob": 0.5}, {"node_id": "CORE-02", "prob": 0.48}],
    )
    assert run("G2", tie).status == "fail"


def test_g3(arts):
    assert run("G3", make_ctx(arts)).status == "pass"
    ctx = make_ctx(arts)
    ctx.output["narrative"] += " Also CORE-99 shows 77 errors."
    c = run("G3", ctx)
    assert c.status == "fail" and "CORE-99" in c.detail and "77" in c.detail


def test_g3_hindi_passes(arts):
    out = explain(RCA, "hi", settings=Settings(llm_provider="none")).model_dump(mode="json")
    assert run("G3", make_ctx(arts, output=out)).status == "pass"


def test_g4(arts):
    assert run("G4", make_ctx(arts)).status == "pass"
    assert run("G4", make_ctx(arts, dependents=10)).status == "warn"
    assert run("G4", make_ctx(arts, dependents=100)).status == "fail"
    ctx = make_ctx(arts)
    ctx.output["proposed_action"]["action"] = "format_disk"
    assert run("G4", ctx).status == "fail"


def test_g5(arts):
    assert run("G5", make_ctx(arts)).status == "pass"
    ctx = make_ctx(arts)
    ctx.output["narrative"] += " mail ops@example.com"
    assert run("G5", ctx).status == "fail"


def test_g6(arts):
    assert run("G6", make_ctx(arts)).status == "pass"
    assert run("G6", make_ctx(arts, drift_status="amber")).status == "warn"
    assert run("G6", make_ctx(arts, drift_status="red")).status == "fail"


def test_g7(arts, tmp_path_factory):
    assert run("G7", make_ctx(arts)).status == "pass"
    assert run("G7", make_ctx(tmp_path_factory.mktemp("empty"))).status == "fail"


def test_g8(arts):
    assert run("G8", make_ctx(arts)).status == "pass"
    ctx = make_ctx(arts)
    del ctx.rca["fault_type"]
    assert run("G8", ctx).status == "fail"


def test_g9(arts):
    assert run("G9", make_ctx(arts)).status == "pass"
    recent = [
        {"target_node_id": "AGG-07", "action": "restart_node", "ts": NOW - timedelta(minutes=5)}
    ]
    assert run("G9", make_ctx(arts, recent_actions=recent)).status == "fail"
    old = [
        {
            "target_node_id": "AGG-07",
            "action": "restart_node",
            "ts": (NOW - timedelta(minutes=45)).isoformat(),
        }
    ]
    assert run("G9", make_ctx(arts, recent_actions=old)).status == "pass"


def test_g10(arts):
    assert run("G10", make_ctx(arts)).status == "pass"
    assert run("G10", make_ctx(arts, severity="major")).status == "warn"
    assert run("G10", make_ctx(arts, severity="critical")).status == "warn"
    assert run("G10", make_ctx(arts, severity=None)).status == "warn"


def test_check_error_fails_closed(arts):
    ctx = make_ctx(arts)
    ctx.rca["root_candidates"] = []
    assert run("G1", ctx).status == "fail"


def test_decision_rule(arts):
    assert evaluate(make_ctx(arts), POLICY).decision == "auto_approved"
    assert evaluate(make_ctx(arts, severity="major"), POLICY).decision == "needs_approval"
    assert evaluate(make_ctx(arts, drift_status="amber"), POLICY).decision == "needs_approval"
    assert (
        evaluate(make_ctx(arts, drift_status="red", severity="major"), POLICY).decision == "blocked"
    )
    gc = lambda i, s: GovCheck(id=i, name="n", status=s, detail="")  # noqa: E731
    assert decide([gc("G1", "pass"), gc("G10", "warn")]) == "needs_approval"
    assert decide([gc("G1", "warn"), gc("G2", "fail")]) == "blocked"


def test_redaction():
    t, n = redact("a@b.io 10.0.0.1 +91 98765 43210 sk-abcdefghijklmnop1234")
    assert n == 4 and "@" not in t and "10.0.0.1" not in t and "sk-" not in t
    assert find_sensitive("AGG-07 38.2 41 valid alarms 83% 0.83") == {}


def test_g1_fault_type_prob(arts):
    def ctx(p):
        rca = copy.deepcopy(RCA)
        rca["fault_type"]["prob"] = p
        return make_ctx(arts, rca=rca)

    assert run("G1", ctx(0.91)).status == "pass"
    mid = run("G1", ctx(0.5))
    assert mid.status == "warn" and "fault_type 0.50" in mid.detail
    low = run("G1", ctx(0.36))
    assert low.status == "fail" and "fault_type 0.36" in low.detail
    assert evaluate(ctx(0.36), POLICY).decision == "blocked"
    assert evaluate(ctx(0.5), POLICY).decision == "needs_approval"


def test_reevaluate_at_approve_time(arts):
    first = evaluate(make_ctx(arts), POLICY)
    assert first.decision == "auto_approved"
    later = evaluate(make_ctx(arts, drift_status="red"), POLICY)  # drift turned red after proposal
    assert later.decision == "blocked" and len(later.checks) == 10
