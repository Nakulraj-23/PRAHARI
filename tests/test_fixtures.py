"""Every fixture must validate against its schema in prahari.schemas."""

import json
from pathlib import Path

import pytest

from prahari import schemas as s

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
MAP = {
    "health": s.HealthResponse,
    "topology": s.TopologyResponse,
    "alarms": s.AlarmsResponse,
    "alarms_validate": s.ValidateResponse,
    "anomaly_score": s.AnomalyScoreResponse,
    "rca": s.RcaOutput,
    "incidents": s.IncidentsResponse,
    "incident_detail": s.IncidentOut,
    "explain": s.ExplainResponse,
    "action_decision": s.DecisionResponse,
    "models": s.ModelsResponse,
    "drift_latest": s.DriftReportOut,
    "drift_run": s.DriftReportOut,
    "sim_inject": s.InjectResponse,
    "sim_scenario": s.ScenarioResponse,
    "audit": s.AuditResponse,
}


@pytest.mark.parametrize("name", sorted(MAP))
def test_fixture_validates(name):
    data = json.loads((FIXTURES / f"{name}.json").read_text())
    MAP[name].model_validate(data)


def test_every_json_fixture_is_mapped():
    on_disk = {p.stem for p in FIXTURES.glob("*.json")}
    assert on_disk == set(MAP)
