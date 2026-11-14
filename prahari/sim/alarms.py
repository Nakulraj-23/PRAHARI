"""Noise alarms, vendor-specific raw_text formats and alarm table assembly."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from prahari.sim.constants import ALARM_TYPES
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

# noise alarm types: spurious ones deliberately overlap with symptomatic types
_SPURIOUS_TYPES = [
    "HIGH_LATENCY",
    "HIGH_CPU",
    "HIGH_PACKET_LOSS",
    "FAN_WARNING",
    "NTP_SYNC_LOST",
    "CRC_ERRORS",
    "INTERFACE_FLAP",
    "THROUGHPUT_DROP",
    "HIGH_TEMP",
]
_FLAP_TYPES = ["INTERFACE_FLAP", "LINK_DOWN", "HIGH_LATENCY", "CRC_ERRORS"]


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
        sev = "minor" if rng.random() < 0.7 else "major"
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
                severity=str(rng.choice(["minor", "major"], p=[0.75, 0.25])),
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
) -> pd.DataFrame:
    """All alarms (symptomatic + noise), time-sorted, with ids assigned after sorting."""
    rng = np.random.default_rng([seed, 3003])
    rows: list[dict] = []
    for f in faults:
        rows += fault_alarms(net, f, rng, root_missing=bool(rng.random() < root_missing_frac))
    n_noise = int(round(len(rows) * noise_frac / (1 - noise_frac)))
    rows += noise_alarms(net, n_noise, start, n_minutes, rng)
    return assemble_alarms(net, rows, rng, first_alarm_id)


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
