"""Monitoring (A6): drift (``run_drift``), retraining gate, benchmarks, load test."""


def run_drift(window=None, settings=None, store: bool = True) -> dict:
    """Drift report dict (``DriftReportOut`` shape); see ``drift.run_drift``."""
    from prahari.monitoring.drift import run_drift as _run

    return _run(window, settings, store)


def latest_report(settings=None):
    """Latest stored drift report or None."""
    from prahari.monitoring.drift import latest_report as _latest

    return _latest(settings)
