"""Deterministic English and Hindi narrative templates (the `none` provider)."""

from __future__ import annotations

FAULT_EN = {
    "fibre_cut": "A fibre cut",
    "power_failure": "A power failure",
    "congestion": "Congestion",
    "hardware_degradation": "Hardware degradation",
    "config_error": "A configuration error",
}
FAULT_HI = {
    "fibre_cut": "फ़ाइबर कटने की समस्या",
    "power_failure": "बिजली गुल होने की समस्या",
    "congestion": "नेटवर्क भीड़भाड़",
    "hardware_degradation": "हार्डवेयर की गिरावट",
    "config_error": "कॉन्फ़िगरेशन की त्रुटि",
}
KPI_HI = {
    "latency_ms": "लेटेंसी",
    "packet_loss_pct": "पैकेट लॉस",
    "throughput_mbps": "थ्रूपुट",
    "cpu_pct": "सीपीयू",
    "mem_pct": "मेमोरी",
    "temp_c": "तापमान",
}
ACTION_EN = {
    "reroute_traffic": "reroute traffic to the backup path",
    "restart_node": "restart the node",
    "rollback_config": "roll back the configuration",
    "dispatch_field_team": "dispatch a field team",
    "throttle_traffic": "throttle low-priority traffic",
    "no_action": "take no action",
}
ACTION_HI = {
    "reroute_traffic": "ट्रैफ़िक को बैकअप मार्ग पर भेजें",
    "restart_node": "नोड को रीस्टार्ट करें",
    "rollback_config": "कॉन्फ़िगरेशन वापस पुराने संस्करण पर करें",
    "dispatch_field_team": "फ़ील्ड टीम भेजें",
    "throttle_traffic": "कम प्राथमिकता वाले ट्रैफ़िक को सीमित करें",
    "no_action": "कोई कार्रवाई न करें",
}


def render(lang: str, facts: dict) -> str:
    """Render the narrative. Only numbers and node ids present in the RCA evidence are used."""
    node, pct = facts["node"], facts["pct"]
    kpi = facts.get("kpi")  # (name, value) or None
    if lang == "hi":
        s = (
            f"{FAULT_HI[facts['fault']]} सबसे अधिक संभावना से {node} पर है ({pct}% विश्वास)। "
            f"{facts['valid']} वैध अलार्म और {facts['noise']} शोर वाले अलार्म "
            f"{facts['n_alarmed']} नोड पर दिखे।"
        )
        if kpi:
            s += f" {node} पर {KPI_HI.get(kpi[0], kpi[0])} में {kpi[1]:.1f} का बदलाव हुआ।"
        s += (
            f" रनबुक {facts['runbook']} के अनुसार: {ACTION_HI[facts['action']]}। "
            f"प्रस्तावित कार्रवाई: {facts['action']} ({node})।"
        )
        return s
    s = (
        f"{FAULT_EN[facts['fault']]} is most likely on {node} ({pct}% confidence). "
        f"{facts['valid']} valid alarms and {facts['noise']} noise alarms were seen across "
        f"{facts['n_alarmed']} alarmed nodes."
    )
    if kpi:
        s += f" {kpi[0]} on {node} changed by {kpi[1]:.1f}."
    s += (
        f" Per runbook {facts['runbook']}, {ACTION_EN[facts['action']]}. "
        f"Proposed action: {facts['action']} on {node}."
    )
    return s
