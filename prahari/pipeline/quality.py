"""Data-quality checks: null, range, duplicate, monotonic, referential, completeness, grid,
flatline, time-bounds, label-consistency, topology sanity. Fails loudly."""

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
        _structural_checks(c, rep)
    if raise_on_fail and rep.failures:
        lines = [f"  {f['name']}: {f['bad_rows']} bad rows {f['detail']}" for f in rep.failures]
        raise DataQualityError("data quality checks failed:\n" + "\n".join(lines))
    return rep


# KPIs whose real values are never exactly constant for an hour (throughput and loss can be 0)
FLATLINE_KPIS = ["latency_ms", "cpu_pct", "mem_pct", "temp_c"]
MAX_GAP_MIN = 15  # longest tolerated hole in a node's 1-minute KPI series
MIN_COVERAGE = 0.90  # share of grid minutes a node must have samples for
TIER_CAPACITY = {"core": 40000.0, "agg": 10000.0, "access": 2000.0, "cell": 300.0}
THROUGHPUT_OVER_TIER = 2.0  # throughput above 2x the tier's nominal capacity is physically wrong


def _minutes(c, later: str, earlier: str) -> str:
    """SQL expression for ``later - earlier`` in minutes, per dialect."""
    if c.dialect.name == "sqlite":
        return f"((julianday({later}) - julianday({earlier})) * 1440.0)"
    return f"(EXTRACT(EPOCH FROM ({later} - {earlier})) / 60.0)"


def _structural_checks(c, rep: QualityReport) -> None:
    """Completeness, grid, flatline, time-bounds, label-consistency and topology sanity."""
    sqlite = c.dialect.name == "sqlite"
    off_grid = "strftime('%f', ts) <> '00.000'" if sqlite else "EXTRACT(SECOND FROM ts) <> 0"
    rep.add(
        "grid:kpi_samples.ts not on the 1-minute grid",
        _count(c, f"SELECT COUNT(*) FROM kpi_samples WHERE {off_grid}"),
    )
    span = _minutes(c, "s.b", "s.a")
    rep.add(
        "completeness:kpi_samples node coverage",
        _count(
            c,
            "SELECT COUNT(*) FROM "
            "(SELECT node_id, COUNT(*) n FROM kpi_samples GROUP BY node_id) x, "
            "(SELECT MIN(ts) a, MAX(ts) b FROM kpi_samples) s "
            f"WHERE x.n < {MIN_COVERAGE} * ({span} + 1)",
        ),
        f"each node needs >= {MIN_COVERAGE:.0%} of grid minutes",
    )
    gap = _minutes(c, "ts", "p")
    rep.add(
        "completeness:kpi_samples gap",
        _count(
            c,
            "SELECT COUNT(*) FROM (SELECT ts, LAG(ts) OVER (PARTITION BY node_id ORDER BY ts) p "
            f"FROM kpi_samples) x WHERE p IS NOT NULL AND {gap} > {MAX_GAP_MIN}",
        ),
        f"no hole longer than {MAX_GAP_MIN} min",
    )
    hour = "substr(CAST(ts AS VARCHAR), 1, 13)"
    flat = " OR ".join(f"MIN({k}) = MAX({k})" for k in FLATLINE_KPIS)
    rep.add(
        "flatline:kpi_samples stuck sensor",
        _count(
            c,
            f"SELECT COUNT(*) FROM (SELECT 1 FROM kpi_samples GROUP BY node_id, {hour} "
            f"HAVING COUNT(*) >= 30 AND ({flat})) x",
        ),
        "an hour of identical readings on " + "/".join(FLATLINE_KPIS),
    )
    rep.add(
        "time-bounds:alarms outside the KPI time range",
        _count(
            c,
            "SELECT COUNT(*) FROM alarms a, (SELECT MIN(ts) lo, MAX(ts) hi FROM kpi_samples) k "
            f"WHERE a.ts < k.lo OR {_minutes(c, 'a.ts', 'k.hi')} > 60",
        ),
    )
    rep.add(
        "time-bounds:incidents outside the KPI time range",
        _count(
            c,
            "SELECT COUNT(*) FROM incidents i, (SELECT MIN(ts) lo, MAX(ts) hi FROM kpi_samples) k "
            "WHERE i.start_ts < k.lo OR i.start_ts > k.hi OR "
            f"{_minutes(c, 'i.end_ts', 'k.hi')} > 240",
        ),
    )
    rep.add(
        "label-consistency:true alarm without incident / noise alarm with incident",
        _count(
            c,
            "SELECT COUNT(*) FROM alarms WHERE label_is_true IS NULL "
            "OR (label_is_true AND label_incident_id IS NULL) "
            "OR (NOT label_is_true AND label_incident_id IS NOT NULL)",
        ),
    )
    rep.add(
        "label-consistency:true alarm before its incident starts",
        _count(
            c,
            "SELECT COUNT(*) FROM alarms a JOIN incidents i ON a.label_incident_id = i.incident_id "
            "WHERE a.ts < i.start_ts",
        ),
    )
    rep.add("topology:self-loop links", _count(c, "SELECT COUNT(*) FROM links WHERE src = dst"))
    rep.add(
        "range:links.capacity_mbps",
        _count(
            c, "SELECT COUNT(*) FROM links WHERE capacity_mbps < 100 OR capacity_mbps > 1000000"
        ),
        "[100, 1e6]",
    )
    cases = " ".join(f"WHEN '{k}' THEN {v}" for k, v in TIER_CAPACITY.items())
    rep.add(
        "range:kpi_samples.throughput_vs_tier_capacity",
        _count(
            c,
            "SELECT COUNT(*) FROM kpi_samples k JOIN nodes n ON n.node_id = k.node_id "
            f"WHERE k.throughput_mbps > {THROUGHPUT_OVER_TIER} * CASE n.node_type {cases} END",
        ),
        f"throughput <= {THROUGHPUT_OVER_TIER}x tier capacity",
    )
