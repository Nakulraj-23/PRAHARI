"""PII and secret redaction (emails, phone numbers, IPv4 addresses, API keys)."""

from __future__ import annotations

import re

PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "key": re.compile(
        r"\b(?:sk|pk|key|tok|token|api)[-_][A-Za-z0-9_-]{12,}\b|\b[A-Za-z0-9]{32,}\b", re.I
    ),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "phone": re.compile(
        r"(?<![\w.])(?:\+?\d{1,3}[\s-]?)?\d{5}[\s-]?\d{5}(?![\w.])"
        r"|(?<![\w.])\(?\d{3}\)?[\s-]\d{3}[\s-]\d{4}(?![\w.])"
    ),
}


def find_sensitive(text: str) -> dict[str, int]:
    """Count matches per category."""
    out = {}
    for name, rx in PATTERNS.items():
        n = len(rx.findall(text))
        if n:
            out[name] = n
    return out


def redact(text: str) -> tuple[str, int]:
    """Replace sensitive spans with [REDACTED:<kind>]; returns (text, count)."""
    total = 0
    for name, rx in PATTERNS.items():
        text, n = rx.subn(f"[REDACTED:{name}]", text)
        total += n
    return text, total
