"""Rule baselines for root-cause selection, operating on the node feature matrix X."""

from __future__ import annotations

import numpy as np

from prahari.pipeline.features import feature_names

_NAMES = feature_names()
_TOTAL = _NAMES.index("alarm_total")
_FIRST = _NAMES.index("first_alarm_min")
_NT = [_NAMES.index(f"ntype_{t}") for t in ("core", "agg", "access", "cell")]


def upstream_scores(x: np.ndarray) -> np.ndarray:
    """Most-upstream alarmed node: lowest tier wins, earliest first alarm breaks ties."""
    depth = x[:, _NT].argmax(1)
    s = -(depth * 1000.0) - x[:, _FIRST]
    return np.where(x[:, _TOTAL] > 0, s, -1e9)


def most_alarms_scores(x: np.ndarray) -> np.ndarray:
    """Node with the most alarms; earliest first alarm breaks ties."""
    return x[:, _TOTAL] * 1000.0 - x[:, _FIRST]


def topk_hit(scores: np.ndarray, y: int, k: int) -> bool:
    """True if node y is within the top-k scores (ties resolved by node index)."""
    order = np.argsort(-scores, kind="stable")[:k]
    return bool(y in order)
