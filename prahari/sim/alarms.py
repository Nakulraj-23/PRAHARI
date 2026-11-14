"""Noise alarms, vendor-specific raw_text formats and alarm table assembly."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prahari.sim.constants import ALARM_TYPES, SEVERITIES
from prahari.sim.faults import Fault, fault_alarms
from prahari.sim.topology import NetworkModel

_DESC = {
    "LOSS_OF_SIGNAL": "Loss of signal on uplink port",
    "LINK_DOWN": "Link down",
    "UPSTREAM_UNREACHABLE": "Upstream neighbour unreachable",
    "HIGH_PACKET_LOSS": "Packet loss above threshold",
    "HIGH_LATENCY": "Latency above threshold",
    "THROUGHPUT_DROP": "Throughput dropped below floor",
    "POWER_FAILURE": "Mains power failure, on battery",
    "HIGH_CPU": "CPU utilisation above threshold",
    "HIGH_TEMP": "Chassis temperature above threshold",
    "HIGH_UTILISATION": "Link utilisation above 90 pct",
    "CONFIG_MISMATCH": "Running config differs from golden config",
    "INTERFACE_FLAP": "Interface state changed",
    "FAN_WARNING": "Fan speed out of range",
    "NTP_SYNC_LOST": "NTP synchronisation lost",
    "CRC_ERRORS": "CRC error counter increasing",
}
_NEXA_CODE = {t: f"{t[:3]}_{100 + i}" for i, t in enumerate(ALARM_TYPES)}
_ORION_NAME = {t: "".join(w.capitalize() for w in t.split("_")) for t in ALARM_TYPES}
_ZENITH_SEV = {"minor": 2, "major": 3, "critical": 5}

# Noise alarms use every alarm type (so none is class-exclusive); flapping favours link types.
_SPURIOUS_TYPES = list(ALARM_TYPES)
_FLAP_TYPES = [
    "INTERFACE_FLAP",
    "LINK_DOWN",
    "HIGH_LATENCY",
    "CRC_ERRORS",
    "LOSS_OF_SIGNAL",
    "UPSTREAM_UNREACHABLE",
]
_SEV_P = [0.6, 0.28, 0.12]  # minor, major, critical
BLIP_FRAC = 0.30  # share of noise alarms that coincide with a real, short KPI blip

# KPI blip per alarm type: (kpi column index, kind, value); scaled by a random magnitude
_BLIP = {
    "HIGH_LATENCY": [(0, "add", 40.0)],
    "HIGH_PACKET_LOSS": [(1, "add", 10.0)],
    "THROUGHPUT_DROP": [(2, "mul", 0.5)],
    "HIGH_CPU": [(3, "add", 25.0)],
    "HIGH_TEMP": [(5, "add", 15.0)],
    "FAN_WARNING": [(5, "add", 8.0)],
    "HIGH_UTILISATION": [(2, "mul", 1.5), (0, "add", 15.0)],
    "INTERFACE_FLAP": [(1, "add", 6.0), (0, "add", 20.0)],
    "LINK_DOWN": [(1, "add", 15.0)],
    "CRC_ERRORS": [(1, "add", 5.0)],
    "LOSS_OF_SIGNAL": [(1, "add", 25.0), (0, "add", 30.0)],
    "UPSTREAM_UNREACHABLE": [(1, "add", 20.0), (0, "add", 25.0)],
    "POWER_FAILURE": [(5, "add", -6.0)],
    "CONFIG_MISMATCH": [(0, "add", 12.0)],
    "NTP_SYNC_LOST": [(0, "add", 8.0)],
}


def format_raw_text(
    vendor: str,
    node_id: str,
    alarm_type: str,
    severity: str,
    ts: pd.Timestamp,
    seq: int,
    rng: np.random.Generator,
) -> str:
    """Vendor-specific alarm text. Carries no label information."""
    port = f"{int(rng.integers(1, 4))}/{int(rng.integers(1, 3))}/{int(rng.integers(1, 25))}"
    if vendor == "Nexa":
        return (
            f"{ts:%Y-%m-%d %H:%M:%S} ALM {node_id} SEV={severity.upper()} "
            f'CODE={_NEXA_CODE[alarm_type]} "{_DESC[alarm_type]}" port={port}'
        )
    if vendor == "Orion":
        return (
            f"%%01ALARM/*NE={node_id}*/ ALMID={seq} LEVEL={severity.capitalize()} "
            f"NAME={_ORION_NAME[alarm_type]} LOC={port} RAISED={ts:%d-%b-%Y %H:%M:%S}"
        )
    return json.dumps(
        {
            "src": node_id,
            "evt": alarm_type,
            "sev": _ZENITH_SEV[severity],
            "epoch": int(ts.timestamp()),
            "desc": _DESC[alarm_type],
            "if": port,
        },
        separators=(",", ":"),
    )


def noise_alarms(
    net: NetworkModel, n_noise: int, start: pd.Timestamp, n_minutes: int, rng: np.random.Generator
) -> list[dict]:
    """Flapping bursts (~55% of volume) and spurious single alarms (~45%)."""
    out: list[dict] = []
    n_flap = int(n_noise * 0.55)
    span = n_minutes * 60.0
    while n_flap > 0:
        node = int(rng.integers(net.n))
        atype = _FLAP_TYPES[int(rng.integers(len(_FLAP_TYPES)))]
        reps = min(int(rng.integers(3, 9)), n_flap)
        t = float(rng.uniform(0, span))
        sev = str(rng.choice(SEVERITIES, p=_SEV_P))
        for _ in range(reps):
            out.append(
                dict(
                    ts=start + pd.Timedelta(seconds=t),
                    node_idx=node,
                    alarm_type=atype,
                    severity=sev,
                    label_is_true=False,
                    label_incident_id=None,
                )
            )
            t += float(rng.uniform(30, 240))
        n_flap -= reps
    for _ in range(n_noise - int(n_noise * 0.55)):
        out.append(
            dict(
                ts=start + pd.Timedelta(seconds=float(rng.uniform(0, span))),
                node_idx=int(rng.integers(net.n)),
                alarm_type=_SPURIOUS_TYPES[int(rng.integers(len(_SPURIOUS_TYPES)))],
                severity=str(rng.choice(SEVERITIES, p=_SEV_P)),
                label_is_true=False,
                label_incident_id=None,
            )
        )
    return out


def build_alarm_table(
    net: NetworkModel,
    faults: list[Fault],
    start: pd.Timestamp,
    n_minutes: int,
    seed: int,
    noise_frac: float = 0.25,
    root_missing_frac: float = 0.15,
    first_alarm_id: int = 1,
) -> tuple[pd.DataFrame, list[dict]]:
    """All alarms (symptomatic + noise) time-sorted with ids, plus the noise KPI blips."""
    rng = np.random.default_rng([seed, 3003])
    rows: list[dict] = []
    for f in faults:
        rows += fault_alarms(net, f, rng, root_missing=bool(rng.random() < root_missing_frac))
    n_noise = int(round(len(rows) * noise_frac / (1 - noise_frac)))
    rows += noise_alarms(net, n_noise, start, n_minutes, rng)
    blips = noise_blips(rows, rng)
    return assemble_alarms(net, rows, rng, first_alarm_id), blips


def assemble_alarms(
    net: NetworkModel, rows: list[dict], rng: np.random.Generator, first_alarm_id: int = 1
) -> pd.DataFrame:
    """Sort by time, assign ids and render raw_text for alarm rows with ``node_idx``."""
    cols = [
        "alarm_id",
        "ts",
        "node_id",
        "alarm_type",
        "severity",
        "raw_text",
        "label_is_true",
        "label_incident_id",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).sort_values("ts", kind="stable").reset_index(drop=True)
    df["ts"] = df["ts"].dt.floor("s")
    df["alarm_id"] = [f"ALM-{first_alarm_id + i:07d}" for i in range(len(df))]
    df["node_id"] = [net.node_ids[j] for j in df["node_idx"]]
    df["raw_text"] = [
        format_raw_text(net.vendor[j], net.node_ids[j], t, s, ts, first_alarm_id + i, rng)
        for i, (j, t, s, ts) in enumerate(
            zip(df["node_idx"], df["alarm_type"], df["severity"], df["ts"], strict=True)
        )
    ]
    return df[cols]


def noise_blips(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    """Short KPI blips for ~30% of noise alarms: dict(ts, node_idx, alarm_type, mag, dur)."""
    return [
        dict(
            ts=r["ts"],
            node_idx=r["node_idx"],
            alarm_type=r["alarm_type"],
            mag=float(rng.uniform(0.4, 1.4)),
            dur=int(rng.integers(3, 9)),
        )
        for r in rows
        if not r["label_is_true"] and rng.random() < BLIP_FRAC
    ]


def apply_blips(block: np.ndarray, block_start: pd.Timestamp, blips: list[dict]) -> None:
    """Add triangular KPI blips into ``block`` (T, N, 6) in place."""
    for b in blips:
        i0 = int((b["ts"] - block_start) // pd.Timedelta(minutes=1))
        lo, hi = max(i0, 0), min(i0 + b["dur"], block.shape[0])
        if hi <= lo:
            continue
        tri = 1 - np.abs(np.linspace(-1, 1, b["dur"]))[lo - i0 : hi - i0]
        for k, kind, val in _BLIP[b["alarm_type"]]:
            w = tri * b["mag"]
            if kind == "mul":
                block[lo:hi, b["node_idx"], k] *= (1 - w * (1 - val)).astype(np.float32)
            else:
                block[lo:hi, b["node_idx"], k] += (w * val).astype(np.float32)
