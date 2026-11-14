"""Governance checks G1-G10: abstract `Check`, one subclass each, and a registry."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from prahari.governance.redact import find_sensitive
from prahari.schemas import GovCheck, RcaOutput

NODE_RE = re.compile(r"\b(?:CORE|AGG|ACC|CELL)-\d+\b")
RB_RE = re.compile(r"\bRB-[A-Z]+-[A-Z]+\b")
NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")


@dataclass
class GovContext:
    """Everything a check may look at. Callers fill what they know; unknowns are handled."""

    rca: dict
    output: dict  # AgentOutput as dict
    artifacts_dir: Path = Path("artifacts")
    drift_status: str | None = None
    severity: str | None = None  # max alarm severity in the window
    dependents: int | None = None  # downstream node count of the target
    recent_actions: list[dict] = field(default_factory=list)  # {target_node_id, action, ts}
    now: datetime = field(default_factory=datetime.utcnow)


class Check(ABC):
    """One governance check. Subclasses set `id`, `name` and implement `run`."""

    id: str = ""
    name: str = ""

    def __init__(self, policy: dict | None = None):
        self.p = (policy or {}).get(self.id, {}) or {}

    @abstractmethod
    def run(self, ctx: GovContext) -> tuple[str, str]:
        """Return (status, detail); status in pass/warn/fail."""

    def evaluate(self, ctx: GovContext) -> GovCheck:
        """Run the check; an exception becomes a fail (fail closed)."""
        try:
            status, detail = self.run(ctx)
        except Exception as e:  # noqa: BLE001
            status, detail = "fail", f"check error: {type(e).__name__}: {e}"
        return GovCheck(id=self.id, name=self.name, status=status, detail=detail)


REGISTRY: dict[str, type[Check]] = {}


def register(cls: type[Check]) -> type[Check]:
    """Class decorator adding a check to the registry."""
    REGISTRY[cls.id] = cls
    return cls


def _probs(ctx: GovContext) -> list[float]:
    return sorted((c["prob"] for c in ctx.rca["root_candidates"]), reverse=True)


@register
class ConfidenceThreshold(Check):
    id, name = "G1", "confidence_threshold"

    def run(self, ctx):
        top = _probs(ctx)[0]
        ok, bad = self.p.get("pass_at", 0.70), self.p.get("fail_below", 0.50)
        fp = ctx.rca["fault_type"]["prob"]
        fok, fbad = self.p.get("fault_pass_at", 0.60), self.p.get("fault_fail_below", 0.40)
        rank = {"pass": 0, "warn": 1, "fail": 2}
        root = "pass" if top >= ok else "warn" if top >= bad else "fail"
        fault = "pass" if fp >= fok else "warn" if fp >= fbad else "fail"
        status = max(root, fault, key=rank.get)
        detail = (
            f"root {top:.2f} {'>=' if root == 'pass' else '<'} "
            f"{ok if root != 'fail' else bad:.2f}; "
            f"fault_type {fp:.2f} {'>=' if fault == 'pass' else '<'} "
            f"{fok if fault != 'fail' else fbad:.2f}"
        )
        return status, detail


@register
class Margin(Check):
    id, name = "G2", "top1_top2_margin"

    def run(self, ctx):
        p = _probs(ctx)
        margin = p[0] - (p[1] if len(p) > 1 else 0.0)
        ok, bad = self.p.get("pass_at", 0.15), self.p.get("fail_below", 0.05)
        if margin >= ok:
            return "pass", f"margin {margin:.2f} >= {ok:.2f}"
        if margin >= bad:
            return "warn", f"margin {margin:.2f} < {ok:.2f}"
        return "fail", f"margin {margin:.2f} < {bad:.2f}"


def allowed_numbers(rca: dict) -> set[str]:
    """Number strings the narrative may use, all derived from the RCA output."""
    ev = rca["evidence"]
    vals: list[float] = [c["prob"] for c in rca["root_candidates"]] + [rca["fault_type"]["prob"]]
    out = {str(round(v * 100)) for v in vals} | {f"{v:.2f}" for v in vals}
    ints = [ev["valid_alarm_count"], ev["noise_alarm_count"], len(ev["alarmed_nodes"])]
    out |= {str(i) for i in ints}
    for d in ev.get("kpi_deltas", {}).values():
        for v in d.values():
            out |= {f"{v:.1f}", f"{v:.0f}", f"{abs(v):.1f}", f"{abs(v):.0f}"}
    return out


@register
class Groundedness(Check):
    id, name = "G3", "groundedness"

    def run(self, ctx):
        text = ctx.output["narrative"]
        ev = ctx.rca["evidence"]
        known = (
            set(ev["alarmed_nodes"])
            | set(ev.get("kpi_deltas", {}))
            | {c["node_id"] for c in ctx.rca["root_candidates"]}
        )
        ids = set(NODE_RE.findall(text))
        stripped = RB_RE.sub(" ", NODE_RE.sub(" ", text))
        nums = set(NUM_RE.findall(stripped))
        bad_ids, bad_nums = sorted(ids - known), sorted(nums - allowed_numbers(ctx.rca))
        if bad_ids or bad_nums:
            return "fail", f"not in evidence: ids {bad_ids} numbers {bad_nums}"
        return "pass", f"{len(ids)} node ids and {len(nums)} numbers all in evidence"


@register
class ActionBlastRadius(Check):
    id, name = "G4", "action_allowlist_blast_radius"

    def run(self, ctx):
        act = ctx.output["proposed_action"]["action"]
        allowed = self.p.get("allowed_actions", [])
        if act not in allowed:
            return "fail", f"action {act} not in allow-list"
        if ctx.dependents is None:
            return "warn", f"{act} allowed; blast radius unknown"
        if act == "no_action":
            return "pass", "no_action has no blast radius"
        n, warn, fail = ctx.dependents, self.p.get("warn_above", 5), self.p.get("fail_above", 40)
        if n > fail:
            return "fail", f"affects {n} downstream nodes > {fail}"
        if n > warn:
            return "warn", f"affects {n} downstream nodes > {warn}"
        return "pass", f"affects {n} downstream nodes"


@register
class Redaction(Check):
    id, name = "G5", "pii_secret_redaction"

    def run(self, ctx):
        found = find_sensitive(
            ctx.output["narrative"] + " " + ctx.output["proposed_action"].get("rationale", "")
        )
        if found:
            return "fail", f"unredacted: {found}"
        n = ctx.output.get("redactions", 0)
        return "pass", f"clean ({n} redacted upstream)" if n else "clean"


@register
class DriftNotRed(Check):
    id, name = "G6", "drift_status"

    def run(self, ctx):
        s = ctx.drift_status
        if s == "red":
            return "fail", "drift status red"
        if s == "amber":
            return "warn", "drift status amber"
        if s == "green":
            return "pass", "drift status green"
        return "warn", "drift status unknown"


@register
class ModelCard(Check):
    id, name = "G7", "model_card_registered"

    def run(self, ctx):
        m = ctx.rca["model"]
        path = Path(ctx.artifacts_dir) / m["name"] / f"{m['version']}.card.json"
        if path.exists():
            return "pass", f"card found for {m['name']}/{m['version']}"
        return "fail", f"no card at {m['name']}/{m['version']}.card.json"


@register
class SchemaValid(Check):
    id, name = "G8", "output_schema_valid"

    def run(self, ctx):
        from prahari.agent.types import AgentOutput

        try:
            RcaOutput.model_validate(ctx.rca)
            AgentOutput.model_validate(ctx.output)
        except ValidationError as e:
            return "fail", f"{e.error_count()} schema errors: {e.errors()[0]['loc']}"
        return "pass", "RCA and agent output valid"


@register
class NoDuplicateAction(Check):
    id, name = "G9", "no_duplicate_action"

    def run(self, ctx):
        act = ctx.output["proposed_action"]
        if act["action"] == "no_action":
            return "pass", "no_action"
        win = timedelta(minutes=self.p.get("window_minutes", 30))
        for r in ctx.recent_actions:
            if r["target_node_id"] == act["target_node_id"] and r.get("action") != "no_action":
                ts = r["ts"] if isinstance(r["ts"], datetime) else datetime.fromisoformat(r["ts"])
                if timedelta(0) <= ctx.now - ts < win:
                    return "fail", (
                        f"{r.get('action')} already proposed on "
                        f"{act['target_node_id']} {int((ctx.now - ts).total_seconds() // 60)} min "
                        "ago"
                    )
        return (
            "pass",
            f"no action on {act['target_node_id']} in last {int(win.total_seconds() // 60)} min",
        )


@register
class SeverityHuman(Check):
    id, name = "G10", "severity_requires_human"

    def run(self, ctx):
        needs = self.p.get("human_required_severities", ["major", "critical"])
        s = ctx.severity
        if s is None:
            if self.p.get("unknown_severity_requires_human", True):
                return "warn", "severity unknown; human approval required"
            return "pass", "severity unknown"
        if s in needs:
            return "warn", f"severity {s} requires human approval"
        return "pass", f"severity {s}"
