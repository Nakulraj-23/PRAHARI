"""Shared constants for the simulator and the feature pipeline."""

from __future__ import annotations

from datetime import datetime

FAULT_TYPES = ["fibre_cut", "power_failure", "congestion", "hardware_degradation", "config_error"]
KPI_COLUMNS = ["latency_ms", "packet_loss_pct", "throughput_mbps", "cpu_pct", "mem_pct", "temp_c"]
NODE_TYPES = ["core", "agg", "access", "cell"]
DEPTH = {"core": 0, "agg": 1, "access": 2, "cell": 3}
VENDORS = ["Nexa", "Orion", "Zenith"]
REGIONS = ["North", "West", "South"]
SEVERITIES = ["minor", "major", "critical"]
ALARM_TYPES = [
    "LOSS_OF_SIGNAL",
    "LINK_DOWN",
    "UPSTREAM_UNREACHABLE",
    "HIGH_PACKET_LOSS",
    "HIGH_LATENCY",
    "THROUGHPUT_DROP",
    "POWER_FAILURE",
    "HIGH_CPU",
    "HIGH_TEMP",
    "HIGH_UTILISATION",
    "CONFIG_MISMATCH",
    "INTERFACE_FLAP",
    "FAN_WARNING",
    "NTP_SYNC_LOST",
    "CRC_ERRORS",
]
SIM_START = datetime(2026, 10, 1)
