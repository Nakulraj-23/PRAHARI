"""KPI simulator: 1-minute samples with daily seasonality, vendor baselines and Gaussian noise."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prahari.sim.constants import KPI_COLUMNS
from prahari.sim.topology import NetworkModel

# per node-type base values: latency ms, loss %, utilisation of capacity, cpu %, mem %, temp C
_BASE = {
    "core": (5.0, 0.04, 0.30, 32.0, 52.0, 38.0),
    "agg": (10.0, 0.08, 0.35, 36.0, 55.0, 41.0),
    "access": (18.0, 0.12, 0.40, 40.0, 58.0, 44.0),
    "cell": (30.0, 0.20, 0.45, 44.0, 60.0, 47.0),
}
_NODE_CAP = {"core": 40000.0, "agg": 10000.0, "access": 2000.0, "cell": 300.0}
# per vendor: latency mult, loss mult, cpu offset, temp offset, throughput mult
_VENDOR = {
    "Nexa": (1.00, 1.00, 0.0, 0.0, 1.00),
    "Orion": (1.12, 1.25, 5.0, 3.0, 0.95),
    "Zenith": (0.92, 0.85, -3.0, -2.0, 1.05),
}
_NOISE_SD = np.array([0.04, 0.0, 0.05, 1.5, 0.6, 0.5])  # latency, throughput relative; others abs


def node_baselines(
    net: NetworkModel,
    minutes: pd.DatetimeIndex,
    vendor_latency_mult: dict[str, float] | None = None,
) -> np.ndarray:
    """Noise-free expected KPIs, shape (T, N, 6), with daily and weekly seasonality."""
    hod = minutes.hour.to_numpy() + minutes.minute.to_numpy() / 60.0
    season = 0.5 * (1 + np.cos(2 * np.pi * (hod - 20.0) / 24.0))  # peak 20:00, trough 08:00
    weekend = np.where(minutes.day_of_week.to_numpy() >= 5, 0.9, 1.0)
    base = np.empty((len(minutes), net.n, 6), dtype=np.float32)
    for j in range(net.n):
        lat, loss, util, cpu, mem, temp = _BASE[net.node_type[j]]
        vl, vloss, vcpu, vtemp, vthr = _VENDOR[net.vendor[j]]
        f = net.node_factor[j]
        lat_mult = (vendor_latency_mult or {}).get(net.vendor[j], 1.0)
        cap = _NODE_CAP[net.node_type[j]]
        base[:, j, 0] = lat * vl * f * lat_mult * (1 + 0.12 * season)
        base[:, j, 1] = loss * vloss * f * (1 + 0.5 * season)
        base[:, j, 2] = cap * util * vthr * f * (0.45 + 0.9 * season * weekend)
        base[:, j, 3] = cpu + vcpu + 14.0 * season * f
        base[:, j, 4] = mem + 3.0 * season * f + 2.0 * (f - 1.0) * 10
        base[:, j, 5] = temp + vtemp + 4.0 * season
    return base


def kpi_block(
    net: NetworkModel,
    minutes: pd.DatetimeIndex,
    rng: np.random.Generator,
    vendor_latency_mult: dict[str, float] | None = None,
) -> np.ndarray:
    """Baseline plus Gaussian noise, shape (T, N, 6), before fault effects are applied."""
    base = node_baselines(net, minutes, vendor_latency_mult)
    noise = rng.standard_normal(base.shape, dtype=np.float32)
    out = base.copy()
    for k in (0, 2):  # relative noise
        out[:, :, k] = base[:, :, k] * (1 + _NOISE_SD[k] * noise[:, :, k])
    out[:, :, 1] = base[:, :, 1] + 0.02 * noise[:, :, 1] * (0.5 + base[:, :, 1])
    for k in (3, 4, 5):
        out[:, :, k] = base[:, :, k] + _NOISE_SD[k] * noise[:, :, k]
    return out


def clip_kpis(block: np.ndarray) -> np.ndarray:
    """Clip to physical ranges in place."""
    np.clip(block[:, :, 0], 0, None, out=block[:, :, 0])
    np.clip(block[:, :, 1], 0, 100, out=block[:, :, 1])
    np.clip(block[:, :, 2], 0, None, out=block[:, :, 2])
    for k in (3, 4):
        np.clip(block[:, :, k], 0, 100, out=block[:, :, k])
    np.clip(block[:, :, 5], -10, 120, out=block[:, :, 5])
    return block


def block_to_frame(net: NetworkModel, minutes: pd.DatetimeIndex, block: np.ndarray) -> pd.DataFrame:
    """Flatten a (T, N, 6) block to the kpi_samples row layout (ts-major, node-sorted)."""
    t, n, _ = block.shape
    df = pd.DataFrame(block.reshape(t * n, 6).astype(np.float64), columns=KPI_COLUMNS)
    df.insert(0, "node_id", np.tile(np.array(net.node_ids, dtype=object), t))
    df.insert(0, "ts", np.repeat(minutes.to_numpy(), n))
    return df
