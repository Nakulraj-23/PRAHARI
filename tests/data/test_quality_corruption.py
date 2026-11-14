"""Corrupt a copy of a seeded DB and check that the quality checks fail loudly.

Every corruption, including the ones Q1 found undetected (F6), must now fail loudly.
NULLs never reach the checks: the schema's NOT NULL constraints reject them first.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from prahari.db import get_engine
from prahari.pipeline.quality import DataQualityError, run_quality_checks

CAUGHT = {
    "neg_latency": ("UPDATE kpi_samples SET latency_ms = -1 WHERE id = 5", "range:kpi_samples"),
    "loss_over_100": (
        "UPDATE kpi_samples SET packet_loss_pct = 140 WHERE id = 9",
        "range:kpi_samples",
    ),
    "dup_kpi": (
        "INSERT INTO kpi_samples (ts, node_id, latency_ms, packet_loss_pct, throughput_mbps, "
        "cpu_pct, mem_pct, temp_c) SELECT ts, node_id, latency_ms, packet_loss_pct, "
        "throughput_mbps, cpu_pct, mem_pct, temp_c FROM kpi_samples WHERE id = 1",
        "duplicate:kpi_samples|monotonic:kpi_samples",
    ),
    "orphan_alarm_node": (
        "UPDATE alarms SET node_id = 'NOPE-1' WHERE alarm_id = 'ALM-0000003'",
        "referential:alarms",
    ),
    "bad_severity": (
        "UPDATE alarms SET severity = 'CRITICAL' WHERE alarm_id = 'ALM-0000003'",
        "range:alarms.severity",
    ),
    "alarm_out_of_order": (
        "UPDATE alarms SET ts = '2026-09-01 00:00:00' WHERE alarm_id = 'ALM-0000050'",
        "monotonic:alarms",
    ),
    "incident_end_before_start": (
        "UPDATE incidents SET end_ts = start_ts WHERE incident_id = 'INC-00002'",
        "range:incidents.end_after_start",
    ),
    "orphan_label_incident": (
        "UPDATE alarms SET label_incident_id = 'INC-99999' WHERE label_is_true = 1 "
        "AND alarm_id = (SELECT MIN(alarm_id) FROM alarms WHERE label_is_true = 1)",
        "referential:alarms.label_incident_id",
    ),
}

MISSED = {
    "kpi_gap_30min": (
        "DELETE FROM kpi_samples WHERE node_id = 'AGG-03' "
        "AND ts >= '2026-10-01 10:00:00' AND ts < '2026-10-01 10:30:00'",
        "completeness:kpi_samples gap",
    ),
    "kpi_off_grid_ts": (
        "UPDATE kpi_samples SET ts = '2026-10-01 10:00:30.000000' "
        "WHERE node_id = 'AGG-03' AND ts LIKE '2026-10-01 10:01:00%'",
        "grid:kpi_samples",
    ),
    "stuck_sensor": (
        "UPDATE kpi_samples SET cpu_pct = 41.0 WHERE node_id = 'AGG-03' "
        "AND ts >= '2026-10-01 06:00:00' AND ts < '2026-10-01 12:00:00'",
        "flatline:kpi_samples",
    ),
    "alarm_in_future": (
        "UPDATE alarms SET ts = '2099-01-01 00:00:00' "
        "WHERE alarm_id = (SELECT MAX(alarm_id) FROM alarms)",
        "time-bounds:alarms",
    ),
    "true_alarm_without_incident": (
        "UPDATE alarms SET label_incident_id = NULL "
        "WHERE alarm_id = (SELECT MIN(alarm_id) FROM alarms WHERE label_is_true = 1)",
        "label-consistency",
    ),
    "noise_alarm_with_incident": (
        "UPDATE alarms SET label_incident_id = 'INC-00001' "
        "WHERE alarm_id = (SELECT MIN(alarm_id) FROM alarms WHERE label_is_true = 0)",
        "label-consistency",
    ),
    "self_loop_link": (
        "UPDATE links SET dst = src WHERE link_id = (SELECT MIN(link_id) FROM links)",
        "topology:self-loop",
    ),
    "throughput_over_capacity": (
        "UPDATE kpi_samples SET throughput_mbps = 900000 "
        "WHERE node_id LIKE 'CELL-%' AND id = (SELECT MIN(id) FROM kpi_samples "
        "WHERE node_id LIKE 'CELL-%')",
        "range:kpi_samples.throughput_vs_tier",
    ),
    "link_capacity_3000x": (
        "UPDATE links SET capacity_mbps = capacity_mbps * 3000 "
        "WHERE link_id = (SELECT MIN(link_id) FROM links WHERE capacity_mbps = 1000)",
        "range:links.capacity",
    ),
}


def test_clean_copy_passes(db_copy):
    assert not run_quality_checks(get_engine(db_copy)).failures


@pytest.mark.parametrize("case", sorted(CAUGHT))
def test_corruption_is_caught(db_copy, case):
    sql, pattern = CAUGHT[case]
    eng = get_engine(db_copy)
    with eng.begin() as c:
        assert c.execute(text(sql)).rowcount >= 1
    with pytest.raises(DataQualityError, match=pattern):
        run_quality_checks(eng)


@pytest.mark.parametrize("case", sorted(MISSED))
def test_formerly_missed_corruption_is_caught(db_copy, case):
    sql, pattern = MISSED[case]
    eng = get_engine(db_copy)
    with eng.begin() as c:
        assert c.execute(text(sql)).rowcount >= 1
    with pytest.raises(DataQualityError, match=pattern):
        run_quality_checks(eng)


def test_off_grid_ts_is_not_silently_floored(db_copy):
    from prahari.pipeline.store import load_kpi
    from prahari.sim.topology import NetworkModel

    eng = get_engine(db_copy)
    with eng.begin() as c:
        c.execute(
            text(
                "UPDATE kpi_samples SET ts = '2026-10-01 10:00:30.000000' "
                "WHERE node_id = 'AGG-03' AND ts LIKE '2026-10-01 10:01:00%'"
            )
        )
    with pytest.raises(DataQualityError, match="grid"):
        load_kpi(eng, NetworkModel.from_db(eng))


def test_nulls_rejected_by_schema(db_copy):
    with pytest.raises(IntegrityError), get_engine(db_copy).begin() as c:
        c.execute(text("UPDATE kpi_samples SET latency_ms = NULL WHERE id = 5"))
