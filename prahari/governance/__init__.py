"""Governance gate: run checks G1-G10 and combine them into a decision."""

from __future__ import annotations

from pathlib import Path

import yaml

from prahari.config import Settings, get_settings
from prahari.governance.checks import REGISTRY, Check, GovContext
from prahari.schemas import GovernanceReport

__all__ = ["Check", "GovContext", "REGISTRY", "decide", "evaluate", "load_policy"]


def load_policy(path: Path | None = None) -> dict:
    """Read policy.yaml."""
    path = path or get_settings().policy_path
    return yaml.safe_load(Path(path).read_text()) or {}


def decide(checks: list) -> str:
    """Any fail -> blocked; G10 or any warn -> needs_approval; else auto_approved."""
    if any(c.status == "fail" for c in checks):
        return "blocked"
    if any(c.status == "warn" or (c.id == "G10" and c.status != "pass") for c in checks):
        return "needs_approval"
    return "auto_approved"


def evaluate(
    ctx: GovContext, policy: dict | None = None, settings: Settings | None = None
) -> GovernanceReport:
    """Run every registered check in id order and return the governance report."""
    settings = settings or get_settings()
    policy = policy if policy is not None else load_policy(settings.policy_path)
    ordered = sorted(REGISTRY.values(), key=lambda c: int(c.id[1:]))
    checks = [cls(policy).evaluate(ctx) for cls in ordered]
    return GovernanceReport(decision=decide(checks), checks=checks)
