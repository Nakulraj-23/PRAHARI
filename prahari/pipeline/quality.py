"""Data-quality checks: null, range, duplicate, monotonic-timestamp, referential. Fails loudly."""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.engine import Engine

from prahari.sim.constants import ALARM_TYPES, FAULT_TYPES, KPI_COLUMNS, NODE_TYPES, SEVERITIES

RANGES = {
    "latency_ms": (0, 5000),
    "packet_loss_pct": (0, 100),
    "throughput_mbps": (0, 1e6),
    "cpu_pct": (0, 100),
    "mem_pct": (0, 100),
    "temp_c": (-10, 120),
}
NOT_NULL = {
    "nodes": ["node_id", "node_type", "region", "vendor"],
    "links": ["link_id", "src", "dst", "capacity_mbps"],
    "kpi_samples": ["ts", "node_id", *KPI_COLUMNS],
    "alarms": ["alarm_id", "ts", "node_id", "alarm_type", "severity", "raw_text"],
    "incidents": ["incident_id", "start_ts", "end_ts", "root_node_id", "fault_type"],
}


class DataQualityError(RuntimeError):
    """Raised when one or more data-quality checks fail."""


@dataclass
class QualityReport:
    checks: list[dict] = field(default_factory=list)

    @property
    def failures(self) -> list[dict]:
        return [c for c in self.checks if not c["ok"]]

    def add(self, name: str, bad: int, detail: str = "") -> None:
        self.checks.append({"name": name, "ok": bad == 0, "bad_rows": int(bad), "detail": detail})


def _count(conn, sql: str, **params) -> int:
    return int(conn.execute(text(sql), params).scalar() or 0)


def _in(values) -> str:
    return ",".join(f"'{v}'" for v in values)


def run_quality_checks(engine: Engine, raise_on_fail: bool = True) -> QualityReport:
    """Run all checks against the DB. Raises ``DataQualityError`` listing every failure."""
    rep = QualityReport()
    with engine.connect() as c:
        for table, cols in NOT_NULL.items():  # nulls
            for col in cols:
                rep.add(
                    f"null:{table}.{col}",
                    _count(c, f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL"),
                )
        for col, (lo, hi) in RANGES.items():  # ranges
            rep.add(
                f"range:kpi_samples.{col}",
                _count(
                    c,
                    f"SELECT COUNT(*) FROM kpi_samples WHERE {col} < :lo OR {col} > :hi",
                    lo=lo,
                    hi=hi,
                ),
                f"[{lo}, {hi}]",
            )
        rep.add(
            "range:alarms.severity",
            _count(c, f"SELECT COUNT(*) FROM alarms WHERE severity NOT IN ({_in(SEVERITIES)})"),
        )
        rep.add(
            "range:alarms.alarm_type",
            _count(c, f"SELECT COUNT(*) FROM alarms WHERE alarm_type NOT IN ({_in(ALARM_TYPES)})"),
        )
        rep.add(
            "range:nodes.node_type",
            _count(c, f"SELECT COUNT(*) FROM nodes WHERE node_type NOT IN ({_in(NODE_TYPES)})"),
        )
        rep.add(
            "range:incidents.fault_type",
            _count(
                c, f"SELECT COUNT(*) FROM incidents WHERE fault_type NOT IN ({_in(FAULT_TYPES)})"
            ),
        )
        rep.add(
            "range:incidents.end_after_start",
            _count(c, "SELECT COUNT(*) FROM incidents WHERE end_ts <= start_ts"),
        )
        rep.add(
            "range:links.capacity", _count(c, "SELECT COUNT(*) FROM links WHERE capacity_mbps <= 0")
        )
        rep.add(
            "duplicate:kpi_samples(node_id,ts)",
            _count(
                c,
                "SELECT COUNT(*) FROM (SELECT 1 FROM kpi_samples "
                "GROUP BY node_id, ts HAVING COUNT(*) > 1) d",
            ),
        )
        rep.add(
            "duplicate:alarms.alarm_id",
            _count(
                c,
                "SELECT COUNT(*) FROM (SELECT 1 FROM alarms "
                "GROUP BY alarm_id HAVING COUNT(*) > 1) d",
            ),
        )
        rep.add(
            "monotonic:kpi_samples.ts per node",
            _count(
                c,
                "SELECT COUNT(*) FROM (SELECT ts, "
                "LAG(ts) OVER (PARTITION BY node_id ORDER BY id) p "
                "FROM kpi_samples) x WHERE p IS NOT NULL AND ts <= p",
            ),
        )
        rep.add(
            "monotonic:alarms.ts by alarm_id",
            _count(
                c,
                "SELECT COUNT(*) FROM (SELECT ts, LAG(ts) OVER (ORDER BY alarm_id) p "
                "FROM alarms) x "
                "WHERE p IS NOT NULL AND ts < p",
            ),
        )
        for tbl, col, ref in [
            ("kpi_samples", "node_id", "nodes"),
            ("alarms", "node_id", "nodes"),
            ("incidents", "root_node_id", "nodes"),
            ("links", "src", "nodes"),
            ("links", "dst", "nodes"),
        ]:
            rep.add(
                f"referential:{tbl}.{col}->{ref}",
                _count(
                    c,
                    f"SELECT COUNT(*) FROM {tbl} t LEFT JOIN {ref} r ON t.{col} = r.node_id "
                    f"WHERE r.node_id IS NULL",
                ),
            )
        rep.add(
            "referential:alarms.label_incident_id->incidents",
            _count(
                c,
                "SELECT COUNT(*) FROM alarms a LEFT JOIN incidents i "
                "ON a.label_incident_id = i.incident_id "
                "WHERE a.label_incident_id IS NOT NULL AND i.incident_id IS NULL",
            ),
        )
    if raise_on_fail and rep.failures:
        lines = [f"  {f['name']}: {f['bad_rows']} bad rows {f['detail']}" for f in rep.failures]
        raise DataQualityError("data quality checks failed:\n" + "\n".join(lines))
    return rep
