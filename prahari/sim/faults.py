"""Fault model: propagation over the topology, KPI effects, scheduling and symptomatic alarms."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prahari.sim.constants import ALARM_TYPES, FAULT_TYPES, KPI_COLUMNS
from prahari.sim.topology import NetworkModel

ATTEN = 0.8  # impact multiplier per hop downstream
MIN_IMPACT = 0.12  # below this a node is considered unaffected
RAMP_DOWN = 3  # minutes of recovery tail after a fault ends

K = {name: i for i, name in enumerate(KPI_COLUMNS)}

# KPI effects: ("add", delta) or ("mul", factor); scaled by node impact. ramp = minutes to full.
PROFILES: dict[str, dict] = {
    "fibre_cut": dict(
        ramp=2,
        duration=(20, 60),
        root={
            "packet_loss_pct": ("add", 65),
            "latency_ms": ("add", 90),
            "throughput_mbps": ("mul", 0.08),
        },
        down={
            "packet_loss_pct": ("add", 55),
            "latency_ms": ("add", 70),
            "throughput_mbps": ("mul", 0.15),
        },
    ),
    "power_failure": dict(
        ramp=1,
        duration=(15, 50),
        root={
            "packet_loss_pct": ("add", 92),
            "latency_ms": ("add", 120),
            "throughput_mbps": ("mul", 0.0),
            "cpu_pct": ("mul", 0.03),
            "temp_c": ("add", -12),
        },
        down={
            "packet_loss_pct": ("add", 50),
            "latency_ms": ("add", 60),
            "throughput_mbps": ("mul", 0.2),
        },
    ),
    "congestion": dict(
        ramp=4,
        duration=(10, 40),
        root={
            "latency_ms": ("add", 45),
            "packet_loss_pct": ("add", 9),
            "throughput_mbps": ("mul", 1.6),
            "cpu_pct": ("add", 22),
        },
        down={
            "latency_ms": ("add", 28),
            "packet_loss_pct": ("add", 4),
            "throughput_mbps": ("mul", 0.8),
        },
    ),
    "hardware_degradation": dict(
        ramp=9,
        duration=(25, 70),
        root={
            "temp_c": ("add", 28),
            "cpu_pct": ("add", 32),
            "mem_pct": ("add", 14),
            "packet_loss_pct": ("add", 4),
            "latency_ms": ("add", 18),
        },
        down={"latency_ms": ("add", 10), "packet_loss_pct": ("add", 1.5)},
    ),
    "config_error": dict(
        ramp=1,
        duration=(10, 45),
        root={
            "packet_loss_pct": ("add", 18),
            "latency_ms": ("add", 35),
            "throughput_mbps": ("mul", 0.55),
            "cpu_pct": ("add", 5),
        },
        down={
            "latency_ms": ("add", 25),
            "packet_loss_pct": ("add", 8),
            "throughput_mbps": ("mul", 0.7),
        },
    ),
}

# Root-node tier mix per fault type: core, agg, access, cell
ROOT_TIER_P = {
    "fibre_cut": (0.10, 0.45, 0.45, 0.0),
    "power_failure": (0.0, 0.20, 0.40, 0.40),
    "congestion": (0.10, 0.40, 0.50, 0.0),
    "hardware_degradation": (0.15, 0.30, 0.30, 0.25),
    "config_error": (0.15, 0.35, 0.35, 0.15),
}
FAULT_MIX = (0.20, 0.15, 0.25, 0.20, 0.20)  # order of FAULT_TYPES

# Symptomatic alarms: (type, base probability, severity or None = from impact)
ALARM_SPEC: dict[str, dict[str, list]] = {
    "fibre_cut": {
        "root": [("LOSS_OF_SIGNAL", 1.0, "critical"), ("LINK_DOWN", 0.8, "critical")],
        "down": [
            ("UPSTREAM_UNREACHABLE", 0.9, None),
            ("HIGH_PACKET_LOSS", 0.8, None),
            ("HIGH_LATENCY", 0.5, "minor"),
            ("THROUGHPUT_DROP", 0.4, "minor"),
        ],
    },
    "power_failure": {
        "root": [("POWER_FAILURE", 1.0, "critical"), ("LINK_DOWN", 0.5, "critical")],
        "down": [("UPSTREAM_UNREACHABLE", 0.9, None), ("HIGH_PACKET_LOSS", 0.6, None)],
    },
    "congestion": {
        "root": [
            ("HIGH_UTILISATION", 1.0, "major"),
            ("HIGH_LATENCY", 0.9, "major"),
            ("HIGH_PACKET_LOSS", 0.6, "major"),
        ],
        "down": [("HIGH_LATENCY", 0.7, "minor"), ("THROUGHPUT_DROP", 0.4, "minor")],
    },
    "hardware_degradation": {
        "root": [
            ("HIGH_TEMP", 0.95, "major"),
            ("HIGH_CPU", 0.9, "major"),
            ("HIGH_PACKET_LOSS", 0.5, "minor"),
        ],
        "down": [("HIGH_LATENCY", 0.5, "minor")],
    },
    "config_error": {
        "root": [("CONFIG_MISMATCH", 1.0, "major"), ("HIGH_PACKET_LOSS", 0.8, "major")],
        "down": [("HIGH_LATENCY", 0.6, "minor"), ("THROUGHPUT_DROP", 0.5, "minor")],
    },
}


@dataclass
class Fault:
    """A ground-truth fault; ``end`` is when the fault stops being active."""

    incident_id: str
    fault_type: str
    root_idx: int
    start: pd.Timestamp
    duration_min: int
    impact: np.ndarray  # (N,) in [0, 1], 1 at the root
    gain: np.ndarray | None = None  # (N,) per-node KPI effect scale (root: fault magnitude)

    @property
    def end(self) -> pd.Timestamp:
        return self.start + pd.Timedelta(minutes=self.duration_min)


def fault_gain(net: NetworkModel, incident_id: str, root_idx: int, seed: int) -> np.ndarray:
    """Per-incident KPI effect scale, deterministic in (seed, incident_id).

    The root magnitude varies per incident (10% are nearly KPI-silent) and each dependent gets
    its own gain, so downstream deltas can exceed the root's; ~40% of dependents show almost no
    KPI change even though their alarms still fire.
    """
    rng = np.random.default_rng([seed, 7007, zlib.crc32(incident_id.encode())])
    gain = np.where(
        rng.random(net.n) < 0.4, rng.uniform(0.02, 0.12, net.n), rng.uniform(0.5, 2.2, net.n)
    )
    mag = rng.uniform(0.3, 1.2)
    gain[root_idx] = mag * (0.15 if rng.random() < 0.10 else 1.0)
    return gain


def new_fault(
    net: NetworkModel,
    incident_id: str,
    fault_type: str,
    root_idx: int,
    start: pd.Timestamp,
    duration_min: int,
    seed: int,
) -> Fault:
    """Build a fault with its propagation impact and per-node gains."""
    return Fault(
        incident_id,
        fault_type,
        root_idx,
        start,
        duration_min,
        propagate(net, root_idx, fault_type),
        fault_gain(net, incident_id, root_idx, seed),
    )


def propagate(net: NetworkModel, root_idx: int, fault_type: str) -> np.ndarray:
    """Impact per node: 1 at the root, attenuated per hop to dependent (downstream) nodes.

    A dependent with several upstream links only loses the share routed through failed
    parents, so redundant links reduce the impact. ``config_error`` also leaks to neighbours.
    """
    impact = np.zeros(net.n)
    impact[root_idx] = 1.0
    for d in range(int(net.depth[root_idx]) + 1, int(net.depth.max()) + 1):
        level = net.depth == d
        n_par = net.parents.sum(axis=1)
        share = np.divide(net.parents @ impact, n_par, out=np.zeros(net.n), where=n_par > 0)
        impact[level] = ATTEN * share[level]
    if fault_type == "config_error":
        nb = net.adj[root_idx].copy()
        impact[nb] = np.maximum(impact[nb], 0.35)
    impact[impact < MIN_IMPACT] = 0.0
    return impact


def envelope(length: int, ramp_up: int, duration: int, ramp_down: int = RAMP_DOWN) -> np.ndarray:
    """Fault intensity over time: linear ramp up, plateau, linear ramp down."""
    t = np.arange(length, dtype=float)
    up = np.clip((t + 1) / max(ramp_up, 1), 0, 1)
    down = np.clip((duration + ramp_down - t) / max(ramp_down, 1), 0, 1)
    return np.minimum(up, down)


def apply_fault(
    block: np.ndarray,
    block_start: pd.Timestamp,
    fault: Fault,
    net: NetworkModel,
    end_override: pd.Timestamp | None = None,
) -> None:
    """Add a fault's KPI effects into ``block`` (T, N, 6) in place; may end early."""
    prof = PROFILES[fault.fault_type]
    duration = fault.duration_min
    if end_override is not None:
        duration = int((end_override - fault.start) / pd.Timedelta(minutes=1))
    i0 = int((fault.start - block_start) / pd.Timedelta(minutes=1))
    total = duration + RAMP_DOWN
    lo, hi = max(i0, 0), min(i0 + total, block.shape[0])
    if hi <= lo:
        return
    env = envelope(total, prof["ramp"], duration)
    # when ended early the ramp-up may be cut short: fine, envelope handles both
    env = env[lo - i0 : hi - i0][:, None]
    nodes = np.nonzero(fault.impact)[0]
    gain = fault.gain if fault.gain is not None else np.ones(net.n)
    imp = (fault.impact * gain)[nodes][None, :]
    for who, sel in (("root", nodes == fault.root_idx), ("down", nodes != fault.root_idx)):
        if not sel.any():
            continue
        cols = nodes[sel]
        w = env * imp[:, sel]
        for kpi, (kind, val) in prof[who].items():
            k = K[kpi]
            if kind == "mul":
                block[lo:hi, cols, k] *= (1 - np.minimum(w, 1.0) * (1 - val)).astype(np.float32)
            else:
                block[lo:hi, cols, k] += (w * val).astype(np.float32)


def schedule_faults(
    net: NetworkModel,
    start: pd.Timestamp,
    n_minutes: int,
    seed: int,
    per_day: float = 6.5,
    overlap_frac: float = 0.10,
    margin_start: int = 180,
    margin_end: int = 90,
    first_id: int = 1,
) -> list[Fault]:
    """Schedule faults: mostly sequential, ``overlap_frac`` start during the previous one."""
    rng = np.random.default_rng([seed, 2002])
    horizon = n_minutes - margin_end
    mean_dur = 32.0
    mean_gap = max(1440.0 / per_day - mean_dur - RAMP_DOWN, 3.0) / (1 - overlap_frac)
    faults: list[Fault] = []
    t = float(margin_start)
    by_tier = {
        tier: np.nonzero(net.type_idx == i)[0]
        for i, tier in enumerate(["core", "agg", "access", "cell"])
    }
    while True:
        ftype = FAULT_TYPES[int(rng.choice(len(FAULT_TYPES), p=FAULT_MIX))]
        lo, hi = PROFILES[ftype]["duration"]
        dur = int(rng.integers(lo, hi + 1))
        if faults and rng.random() < overlap_frac:
            prev = faults[-1]
            s = (prev.start - start) / pd.Timedelta(minutes=1) + rng.uniform(
                2, max(prev.duration_min - 2, 3)
            )
            far = np.nonzero(net.dist[prev.root_idx] >= 2)[0]
        else:
            s = t + rng.exponential(mean_gap)
            far = np.arange(net.n)
        if s + dur + RAMP_DOWN > horizon:
            break
        tier = int(rng.choice(4, p=ROOT_TIER_P[ftype]))
        cand = np.intersect1d(by_tier[["core", "agg", "access", "cell"][tier]], far)
        if cand.size == 0:
            continue
        root = int(rng.choice(cand))
        s_min = int(s)
        fault = new_fault(
            net,
            f"INC-{first_id + len(faults):05d}",
            ftype,
            root,
            start + pd.Timedelta(minutes=s_min),
            dur,
            seed,
        )
        faults.append(fault)
        t = max(t, s_min + dur + RAMP_DOWN)
    return faults


def fault_alarms(
    net: NetworkModel,
    fault: Fault,
    rng: np.random.Generator,
    drop_frac: float = 0.10,
    root_missing: bool = False,
) -> list[dict]:
    """Symptomatic alarms for one fault: delay grows with hop distance; some are dropped."""
    spec = ALARM_SPEC[fault.fault_type]
    out: list[dict] = []
    slow = 240.0 if fault.fault_type == "hardware_degradation" else 0.0
    for j in np.nonzero(fault.impact)[0]:
        is_root = j == fault.root_idx
        if is_root and root_missing:
            continue
        hop = int(net.dist[fault.root_idx, j])
        for atype, p, sev in spec["root" if is_root else "down"]:
            if rng.random() > p * (0.4 + 0.6 * fault.impact[j]):
                continue
            if rng.random() < drop_frac:
                continue
            delay = (
                hop * rng.uniform(15, 45)
                + rng.exponential(40)
                + rng.uniform(5, 30)
                + (rng.uniform(0, slow) if slow else 0.0)
            )
            severity = sev or ("major" if fault.impact[j] > 0.55 else "minor")
            out.append(
                dict(
                    ts=fault.start + pd.Timedelta(seconds=float(delay)),
                    node_idx=int(j),
                    alarm_type=atype,
                    severity=severity,
                    label_is_true=True,
                    label_incident_id=fault.incident_id,
                )
            )
    out += _collateral_alarms(net, fault, rng, root_missing)
    if not out:  # keep every incident observable by at least its first symptom
        atype, _, sev = spec["root"][0]
        out.append(
            dict(
                ts=fault.start + pd.Timedelta(seconds=float(rng.uniform(10, 60))),
                node_idx=fault.root_idx,
                alarm_type=atype,
                severity=sev or "major",
                label_is_true=True,
                label_incident_id=fault.incident_id,
            )
        )
    return out


def _collateral_alarms(
    net: NetworkModel, fault: Fault, rng: np.random.Generator, root_missing: bool
) -> list[dict]:
    """Extra true alarms of arbitrary type (~15% of affected nodes), so no type or severity is
    exclusive to noise and the alarm type alone does not reveal the fault."""
    out = []
    for j in np.nonzero(fault.impact)[0]:
        if (j == fault.root_idx and root_missing) or rng.random() > 0.15:
            continue
        out.append(
            dict(
                ts=fault.start + pd.Timedelta(seconds=float(rng.uniform(10, 300))),
                node_idx=int(j),
                alarm_type=ALARM_TYPES[int(rng.integers(len(ALARM_TYPES)))],
                severity=str(rng.choice(["minor", "major", "critical"], p=[0.5, 0.35, 0.15])),
                label_is_true=True,
                label_incident_id=fault.incident_id,
            )
        )
    return out
