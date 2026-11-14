"""Simulator: topology, propagation, determinism, live mode."""

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from prahari.config import Settings
from prahari.db import get_engine
from prahari.sim.constants import FAULT_TYPES
from prahari.sim.faults import ATTEN, propagate
from prahari.sim.generate import build_dataset, write_dataset
from prahari.sim.live import LiveSimulator
from prahari.sim.topology import (
    ALT_SEED,
    ALT_SPEC,
    MAIN_SPEC,
    generate_topology,
)


@pytest.fixture(scope="module")
def ds():
    return build_dataset(days=2, seed=42)


@pytest.fixture(scope="module")
def net(ds):
    return ds.net


def test_topology_shapes():
    g = generate_topology(42, MAIN_SPEC)
    assert g.number_of_nodes() == 120
    assert len({d["region"] for _, d in g.nodes(data=True)}) == 3
    assert len({d["vendor"] for _, d in g.nodes(data=True)}) == 3
    alt = generate_topology(ALT_SEED, ALT_SPEC)
    assert alt.number_of_nodes() == 80
    assert set(alt.nodes) != set(g.nodes)
    # redundant links exist: more links than a tree
    assert g.number_of_edges() > g.number_of_nodes() - 1


@pytest.mark.parametrize("ftype", [f for f in FAULT_TYPES if f != "config_error"])
def test_propagation_follows_topology(net, ftype):
    """Impact reaches only downstream dependents, never upstream, and decays with distance."""
    for root in (
        int(np.nonzero(net.type_idx == 1)[0][0]),
        int(np.nonzero(net.type_idx == 2)[0][0]),
    ):
        impact = propagate(net, root, ftype)
        hit = np.nonzero(impact)[0]
        assert impact[root] == 1.0
        for v in hit:
            assert net.depth[v] >= net.depth[root]
            assert impact[v] <= ATTEN ** net.dist[root, v] + 1e-9
        # every affected non-root node is reachable via a chain of affected upstream parents
        for v in hit:
            if v != root:
                assert (net.parents[v] * impact).sum() > 0
        # upstream and unrelated nodes are untouched
        assert not impact[net.depth < net.depth[root]].any()


def test_redundant_uplink_reduces_impact(net):
    """A dependent with a second, healthy parent is hit less than a single-homed one."""
    root = int(np.nonzero(net.type_idx == 1)[0][0])
    impact = propagate(net, root, "fibre_cut")
    children = [v for v in np.nonzero(net.parents[:, root])[0] if impact[v] > 0]
    n_par = net.parents.sum(axis=1)
    single = [impact[v] for v in children if n_par[v] == 1]
    multi = [impact[v] for v in children if n_par[v] > 1]
    if single and multi:
        assert max(multi) < min(single)


def test_alarm_delay_grows_with_hops(ds):
    f = next(f for f in ds.faults if f.fault_type == "fibre_cut" and (f.impact > 0).sum() > 5)
    a = ds.alarms[ds.alarms.label_incident_id == f.incident_id]
    hops = a.node_id.map(lambda n: ds.net.dist[f.root_idx, ds.net.index[n]])
    delay = (a.ts - f.start).dt.total_seconds()
    assert np.corrcoef(hops, delay)[0, 1] > 0


def test_determinism_under_seed():
    a, b = build_dataset(days=1, seed=3), build_dataset(days=1, seed=3)
    c = build_dataset(days=1, seed=4)
    pd.testing.assert_frame_equal(a.alarms, b.alarms)
    pd.testing.assert_frame_equal(a.incidents_frame(), b.incidents_frame())
    ka = next(a.kpi_chunks())[1]
    kb = next(b.kpi_chunks())[1]
    assert np.array_equal(ka, kb)
    assert not a.incidents_frame().equals(c.incidents_frame())


def test_realism_knobs(ds):
    al = ds.alarms
    assert 0.2 <= (~al.label_is_true.astype(bool)).mean() <= 0.3
    assert set(al.raw_text.str[:2]) >= {"%%", '{"'} or al.raw_text.str.contains("ALM").any()
    assert not al.raw_text.str.contains("label|true|noise", case=False).any()


@pytest.fixture()
def live(tmp_path):
    s = Settings(db_url=f"sqlite:///{tmp_path / 'l.db'}", artifacts_dir=tmp_path / "art")
    write_dataset(build_dataset(days=1, seed=42), s)
    return LiveSimulator(s)


def test_live_inject_and_recover(live):
    clock0 = live.clock
    root = "AGG-03"
    res = live.inject("fibre_cut", root)
    assert res["root_node_id"] == root and res["alarms_written"] > 0
    eng = get_engine(live.settings)
    with eng.connect() as c:
        row = c.execute(
            text("SELECT root_node_id, fault_type FROM incidents WHERE incident_id = :i"),
            {"i": res["incident_id"]},
        ).one()
        loss = c.execute(
            text("SELECT MAX(packet_loss_pct) FROM kpi_samples WHERE node_id = :n AND ts >= :t"),
            {"n": root, "t": clock0.to_pydatetime()},
        ).scalar()
    assert tuple(row) == (root, "fibre_cut") and loss > 30
    wrong = live.apply_action({"target_node_id": "CORE-01"})
    assert wrong["recovered"] == []
    ok = live.apply_action({"target_node_id": root})
    assert ok["recovered"] == [res["incident_id"]]
    live.tick(10)
    with eng.connect() as c:
        loss_after = c.execute(
            text(
                "SELECT packet_loss_pct FROM kpi_samples "
                "WHERE node_id = :n ORDER BY ts DESC LIMIT 1"
            ),
            {"n": root},
        ).scalar()
    assert loss_after < 2


def test_live_firmware_drift(live):
    """Latency baseline of one vendor rises ~25% after the scenario; other vendors do not move."""
    eng = get_engine(live.settings)
    start = live.clock
    res = live.scenario("firmware_drift", vendor="Orion", minutes=24 * 60)
    assert res["vendor"] == "Orion"
    df = pd.read_sql(
        "SELECT k.ts, k.latency_ms, n.vendor FROM kpi_samples k "
        "JOIN nodes n ON n.node_id = k.node_id "
        "WHERE n.node_type = 'cell'",
        eng,
        parse_dates=["ts"],
    )
    before, after = df[df.ts < start], df[df.ts >= start]
    ratio = after.groupby("vendor").latency_ms.mean() / before.groupby("vendor").latency_ms.mean()
    assert 1.2 < ratio["Orion"] < 1.3
    assert ratio.drop("Orion").between(0.95, 1.05).all()
