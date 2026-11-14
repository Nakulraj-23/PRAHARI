"""Glue for the serving layer: agent output + governance report for one RCA result."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import text
from sqlalchemy.engine import Engine

from prahari.agent.agent import explain
from prahari.agent.types import AgentOutput
from prahari.config import Settings, get_settings
from prahari.governance import GovContext, evaluate
from prahari.schemas import GovernanceReport

DEPTH = {"core": 0, "agg": 1, "access": 2, "cell": 3}
SEV_RANK = {"minor": 1, "major": 2, "critical": 3}


def dependent_count(engine: Engine, node_id: str) -> int | None:
    """Number of nodes downstream of `node_id` (reachable through strictly deeper tiers)."""
    with engine.connect() as c:
        types = dict(c.execute(text("SELECT node_id, node_type FROM nodes")).all())
        links = c.execute(text("SELECT src, dst FROM links")).all()
    if node_id not in types:
        return None
    nbrs: dict[str, set[str]] = {}
    for a, b in links:
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)
    seen, stack = {node_id}, [node_id]
    while stack:
        cur = stack.pop()
        for n in nbrs.get(cur, ()):
            if n not in seen and DEPTH[types[n]] > DEPTH[types[cur]]:
                seen.add(n)
                stack.append(n)
    return len(seen) - 1


def window_severity(engine: Engine, rca: dict) -> str | None:
    """Highest alarm severity on alarmed nodes in the RCA window (label columns never read)."""
    nodes = rca["evidence"]["alarmed_nodes"]
    if not nodes:
        return None
    marks = ",".join(f":n{i}" for i in range(len(nodes)))
    params = {f"n{i}": n for i, n in enumerate(nodes)}
    params |= {
        "s": datetime.fromisoformat(str(rca["window"]["start"])),
        "e": datetime.fromisoformat(str(rca["window"]["end"])),
    }
    with engine.connect() as c:
        rows = c.execute(
            text(
                f"SELECT DISTINCT severity FROM alarms WHERE node_id IN ({marks}) "
                "AND ts >= :s AND ts <= :e"
            ),
            params,
        ).all()
    sev = [r[0] for r in rows if r[0] in SEV_RANK]
    return max(sev, key=SEV_RANK.get) if sev else None


def analyze(
    rca: dict,
    engine: Engine | None = None,
    lang: str = "en",
    incident_id: str | None = None,
    drift_status: str | None = None,
    recent_actions: list[dict] | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
    **agent_kw,
) -> tuple[AgentOutput, GovernanceReport]:
    """Run the agent, then the governance gate. Unknown inputs make checks warn, not pass."""
    settings = settings or get_settings()
    out = explain(
        rca, lang=lang, incident_id=incident_id, engine=engine, settings=settings, **agent_kw
    )
    target = out.proposed_action.target_node_id
    ctx = GovContext(
        rca=rca,
        output=out.model_dump(mode="json"),
        artifacts_dir=settings.artifacts_dir,
        drift_status=drift_status,
        recent_actions=recent_actions or [],
        severity=window_severity(engine, rca) if engine is not None else None,
        dependents=dependent_count(engine, target) if engine is not None else None,
        **({"now": now} if now else {}),
    )
    return out, evaluate(ctx, settings=settings)
