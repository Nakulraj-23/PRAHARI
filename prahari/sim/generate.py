"""Generate a synthetic dataset into the DB: ``python -m prahari.sim.generate --days 30``."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prahari.config import ROOT, Settings, get_settings
from prahari.db import Base, get_engine
from prahari.sim.alarms import build_alarm_table
from prahari.sim.constants import SIM_START
from prahari.sim.faults import Fault, apply_fault, schedule_faults
from prahari.sim.kpi import block_to_frame, clip_kpis, kpi_block
from prahari.sim.topology import (
    ALT_SEED,
    ALT_SPEC,
    MAIN_SPEC,
    NetworkModel,
    TopologySpec,
    generate_topology,
    topology_frames,
)

CHUNK_MINUTES = 1440


@dataclass
class Dataset:
    """In-memory simulation plan: topology, ground-truth faults and alarms (KPIs are streamed)."""

    net: NetworkModel
    nodes: pd.DataFrame
    links: pd.DataFrame
    faults: list[Fault]
    alarms: pd.DataFrame
    start: pd.Timestamp
    n_minutes: int
    seed: int

    def incidents_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                dict(
                    incident_id=f.incident_id,
                    start_ts=f.start,
                    end_ts=f.end,
                    root_node_id=self.net.node_ids[f.root_idx],
                    fault_type=f.fault_type,
                )
                for f in self.faults
            ]
        )

    def kpi_chunks(self):
        """Yield (minutes, block) per day; deterministic in (seed, chunk index)."""
        for c, off in enumerate(range(0, self.n_minutes, CHUNK_MINUTES)):
            n = min(CHUNK_MINUTES, self.n_minutes - off)
            t0 = self.start + pd.Timedelta(minutes=off)
            minutes = pd.date_range(t0, periods=n, freq="min")
            block = kpi_block(self.net, minutes, np.random.default_rng([self.seed, 4004, c]))
            for f in self.faults:
                apply_fault(block, t0, f, self.net)
            yield minutes, clip_kpis(block)


def build_dataset(
    days: int = 30,
    seed: int = 42,
    spec: TopologySpec = MAIN_SPEC,
    topo_seed: int | None = None,
    start: pd.Timestamp | None = None,
    per_day: float = 23.0,
) -> Dataset:
    """Plan a dataset (topology, faults, alarms). Pure function of its arguments."""
    start = pd.Timestamp(start or SIM_START)
    g = generate_topology(seed if topo_seed is None else topo_seed, spec)
    net = NetworkModel.from_graph(g)
    nodes, links = topology_frames(g)
    n_minutes = days * 1440
    faults = schedule_faults(net, start, n_minutes, seed, per_day=per_day)
    alarms = build_alarm_table(net, faults, start, n_minutes, seed)
    return Dataset(net, nodes, links, faults, alarms, start, n_minutes, seed)


def write_dataset(ds: Dataset, settings: Settings, reset: bool = True) -> dict:
    """Write a dataset to the DB at ``settings.db_url`` and return row counts."""
    engine = get_engine(settings)
    if reset:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    ds.nodes.to_sql("nodes", engine, if_exists="append", index=False)
    ds.links.to_sql("links", engine, if_exists="append", index=False)
    ds.incidents_frame().to_sql("incidents", engine, if_exists="append", index=False)
    ds.alarms.to_sql("alarms", engine, if_exists="append", index=False)
    n_kpi = 0
    for minutes, block in ds.kpi_chunks():
        df = block_to_frame(ds.net, minutes, block)
        df.to_sql("kpi_samples", engine, if_exists="append", index=False, chunksize=100_000)
        n_kpi += len(df)
    return dict(
        nodes=len(ds.nodes),
        links=len(ds.links),
        incidents=len(ds.faults),
        alarms=len(ds.alarms),
        kpi_samples=n_kpi,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--alt",
        action="store_true",
        help="second topology (seed 7, ~80 nodes) for generalisation tests only",
    )
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--per-day", type=float, default=23.0, help="target incidents per day")
    ap.add_argument("--no-checks", action="store_true", help="skip quality checks and splits")
    args = ap.parse_args(argv)

    settings = get_settings()
    seed = settings.seed if args.seed is None else args.seed
    if args.alt:
        db_url = args.db_url or f"sqlite:///{ROOT / 'prahari_alt.db'}"
        spec, topo_seed = ALT_SPEC, ALT_SEED
    else:
        db_url, spec, topo_seed = args.db_url or settings.db_url, MAIN_SPEC, None
    settings = settings.model_copy(update={"db_url": db_url, "seed": seed})

    t0 = time.time()
    ds = build_dataset(args.days, seed, spec, topo_seed, per_day=args.per_day)
    counts = write_dataset(ds, settings)
    print(f"wrote {counts} to {db_url} in {time.time() - t0:.1f}s")
    if not args.no_checks:
        from prahari.pipeline.quality import run_quality_checks
        from prahari.pipeline.splits import make_splits

        report = run_quality_checks(get_engine(settings))
        print(f"quality checks passed: {len(report.checks)} checks")
        if not args.alt:
            splits = make_splits(settings)
            print(
                "splits:", {k: len(v) for k, v in splits.items() if k in ("train", "val", "test")}
            )
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
