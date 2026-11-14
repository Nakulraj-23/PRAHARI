"""DB init creates every contract table and the required indexes."""

from sqlalchemy import inspect

from prahari.config import Settings
from prahari.db import init_db


def test_init_db(tmp_path):
    engine = init_db(Settings(db_url=f"sqlite:///{tmp_path / 't.db'}"))
    insp = inspect(engine)
    assert set(insp.get_table_names()) == {
        "nodes",
        "links",
        "kpi_samples",
        "alarms",
        "incidents",
        "predictions",
        "actions",
        "audit_log",
        "drift_reports",
    }
    assert any("node_id" in i["column_names"] for i in insp.get_indexes("kpi_samples"))
    assert any(i["column_names"] == ["ts"] for i in insp.get_indexes("alarms"))
    cols = {c["name"] for c in insp.get_columns("alarms")}
    assert {"label_is_true", "label_incident_id"} <= cols
