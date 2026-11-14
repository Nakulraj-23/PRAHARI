"""Hierarchical core -> agg -> access -> cell topology and the vectorised network model."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd

from prahari.sim.constants import DEPTH, NODE_TYPES, REGIONS, VENDORS


@dataclass(frozen=True)
class TopologySpec:
    """Node counts per tier and the share of nodes with a redundant uplink."""

    n_core: int
    n_agg: int
    n_access: int
    n_cell: int
    agg_redundancy: float
    access_redundancy: float
    cell_redundancy: float

    @property
    def n_nodes(self) -> int:
        return self.n_core + self.n_agg + self.n_access + self.n_cell


MAIN_SPEC = TopologySpec(6, 18, 30, 66, 0.5, 0.3, 0.1)  # 120 nodes
ALT_SPEC = TopologySpec(3, 9, 22, 46, 0.25, 0.15, 0.2)  # 80 nodes, generalisation tests only
ALT_SEED = 7

_VENDOR_P = {"North": (0.6, 0.25, 0.15), "West": (0.2, 0.6, 0.2), "South": (0.15, 0.25, 0.6)}
_CAP = {
    ("core", "core"): 100000,
    ("core", "agg"): 40000,
    ("agg", "access"): 10000,
    ("access", "cell"): 1000,
}
_PREFIX = {"core": "CORE-%02d", "agg": "AGG-%02d", "access": "ACC-%03d", "cell": "CELL-%03d"}


def generate_topology(seed: int = 42, spec: TopologySpec = MAIN_SPEC) -> nx.Graph:
    """Build a hierarchical topology with some redundant uplinks, 3 regions, 3 vendors."""
    rng = np.random.default_rng([seed, 1001])
    g = nx.Graph()
    tiers: dict[str, list[str]] = {t: [] for t in NODE_TYPES}

    def add(ntype: str, i: int, region: str) -> str:
        nid = _PREFIX[ntype] % (i + 1)
        vendor = str(rng.choice(VENDORS, p=_VENDOR_P[region]))
        g.add_node(nid, node_type=ntype, region=region, vendor=vendor, depth=DEPTH[ntype])
        tiers[ntype].append(nid)
        return nid

    def link(a: str, b: str) -> None:
        if a == b or g.has_edge(a, b):
            return
        if g.nodes[a]["depth"] > g.nodes[b]["depth"]:
            a, b = b, a  # upstream endpoint first
        cap = _CAP[(g.nodes[a]["node_type"], g.nodes[b]["node_type"])]
        g.add_edge(a, b, link_id=f"L-{a}-{b}", capacity_mbps=float(cap))

    for i in range(spec.n_core):
        add("core", i, REGIONS[i % 3])
    for i in range(spec.n_agg):
        add("agg", i, REGIONS[i % 3])
    cores = tiers["core"]
    for i, c in enumerate(cores):  # ring plus chords between cores
        link(c, cores[(i + 1) % len(cores)])
        if len(cores) >= 6 and i < len(cores) // 2:
            link(c, cores[i + len(cores) // 2])
    for a in tiers["agg"]:
        same = [c for c in cores if g.nodes[c]["region"] == g.nodes[a]["region"]] or cores
        link(a, same[int(rng.integers(len(same)))])
        if rng.random() < spec.agg_redundancy:
            others = [c for c in cores if not g.has_edge(a, c)]
            if others:
                link(a, others[int(rng.integers(len(others)))])

    def attach(child_t: str, parent_t: str, n: int, redundancy: float) -> None:
        parents = tiers[parent_t]
        order = np.resize(rng.permutation(len(parents)), n)  # balanced: every parent gets children
        for i in range(n):
            p = parents[int(order[i])]
            c = add(child_t, i, g.nodes[p]["region"])
            link(c, p)
            if rng.random() < redundancy:
                same = [
                    q for q in parents if q != p and g.nodes[q]["region"] == g.nodes[p]["region"]
                ]
                if same:
                    link(c, same[int(rng.integers(len(same)))])

    attach("access", "agg", spec.n_access, spec.access_redundancy)
    attach("cell", "access", spec.n_cell, spec.cell_redundancy)
    if not nx.is_connected(g):  # pragma: no cover - construction guarantees connectivity
        raise RuntimeError("generated topology is not connected")
    return g


def topology_frames(g: nx.Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (nodes, links) DataFrames matching the DB tables."""
    nodes = pd.DataFrame(
        [
            {
                "node_id": n,
                "node_type": d["node_type"],
                "region": d["region"],
                "vendor": d["vendor"],
            }
            for n, d in g.nodes(data=True)
        ]
    )
    links = pd.DataFrame(
        [
            {"link_id": d["link_id"], "src": u, "dst": v, "capacity_mbps": d["capacity_mbps"]}
            for u, v, d in g.edges(data=True)
        ]
    )
    return nodes, links


def _node_factor(node_id: str, salt: str) -> float:
    """Deterministic per-node factor in [0.9, 1.1], independent of any RNG state."""
    return 0.9 + 0.2 * (zlib.crc32(f"{salt}:{node_id}".encode()) % 10007) / 10006.0


class NetworkModel:
    """Array view of the topology used by the simulator and the pipeline."""

    def __init__(self, nodes: pd.DataFrame, links: pd.DataFrame):
        nodes = nodes.sort_values("node_id").reset_index(drop=True)
        self.node_ids: list[str] = nodes["node_id"].tolist()
        self.index = {n: i for i, n in enumerate(self.node_ids)}
        self.n = len(self.node_ids)
        self.node_type = nodes["node_type"].to_numpy()
        self.vendor = nodes["vendor"].to_numpy()
        self.region = nodes["region"].to_numpy()
        self.depth = np.array([DEPTH[t] for t in self.node_type])
        self.type_idx = np.array([NODE_TYPES.index(t) for t in self.node_type])
        self.vendor_idx = np.array([VENDORS.index(v) for v in self.vendor])
        self.adj = np.zeros((self.n, self.n), dtype=bool)
        cap = np.zeros(self.n)
        for s, d, c in zip(links["src"], links["dst"], links["capacity_mbps"], strict=True):
            i, j = self.index[s], self.index[d]
            self.adj[i, j] = self.adj[j, i] = True
            cap[i], cap[j] = max(cap[i], c), max(cap[j], c)
        self.link_capacity = cap
        # parents[v, p] = 1 if p is an upstream neighbour of v (one tier closer to the core)
        self.parents = (self.adj & (self.depth[None, :] == self.depth[:, None] - 1)).astype(float)
        self.dist = self._hop_distances()
        self.node_factor = np.array([_node_factor(n, "kpi") for n in self.node_ids])

    @classmethod
    def from_graph(cls, g: nx.Graph) -> NetworkModel:
        nodes, links = topology_frames(g)
        return cls(nodes, links)

    @classmethod
    def from_db(cls, engine) -> NetworkModel:
        nodes = pd.read_sql("SELECT node_id, node_type, region, vendor FROM nodes", engine)
        links = pd.read_sql("SELECT link_id, src, dst, capacity_mbps FROM links", engine)
        return cls(nodes, links)

    def _hop_distances(self) -> np.ndarray:
        dist = np.full((self.n, self.n), 99, dtype=np.int16)
        np.fill_diagonal(dist, 0)
        frontier = np.eye(self.n, dtype=bool)
        seen = frontier.copy()
        for h in range(1, self.n):
            frontier = (frontier.astype(np.uint8) @ self.adj.astype(np.uint8) > 0) & ~seen
            if not frontier.any():
                break
            dist[frontier] = h
            seen |= frontier
        return dist
