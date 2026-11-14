"""RCA graph neural network (A3). Interface: ``train(settings)``, ``load(version)``."""

from __future__ import annotations


def train(settings=None):
    """Train the RCA model (see :func:`prahari.models.rca.train.train`)."""
    from prahari.models.rca.train import train as _train

    return _train(settings)


def load(version: str | None = None, settings=None):
    """Load a trained RCA predictor (latest version by default)."""
    from prahari.models.rca.predictor import load as _load

    return _load(version, settings)
