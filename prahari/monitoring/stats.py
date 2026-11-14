"""Drift statistics in plain NumPy: PSI and the two-sample Kolmogorov-Smirnov test."""

from __future__ import annotations

import numpy as np

PSI_AMBER = 0.1
PSI_RED = 0.2
EPS = 1e-4


def psi_from_props(expected: np.ndarray, actual: np.ndarray, eps: float = EPS) -> float:
    """Population stability index for two bin-proportion vectors: sum (a - e) ln(a / e).

    Proportions are floored at ``eps`` and renormalised so empty bins stay finite.
    """
    e = np.maximum(np.asarray(expected, float), eps)
    a = np.maximum(np.asarray(actual, float), eps)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def quantile_edges(ref: np.ndarray, bins: int = 10) -> np.ndarray:
    """Inner bin edges at reference quantiles (duplicates removed, so ties give fewer bins)."""
    q = np.quantile(ref[np.isfinite(ref)], np.linspace(0, 1, bins + 1)[1:-1])
    return np.unique(q)


def bin_props(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Share of ``x`` in each of ``len(edges) + 1`` bins; bin i is (edge[i-1], edge[i]]."""
    x = x[np.isfinite(x)]
    idx = np.searchsorted(edges, x, side="left")
    return np.bincount(idx, minlength=len(edges) + 1) / max(len(x), 1)


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI of ``cur`` against ``ref`` using ``bins`` reference-quantile bins."""
    edges = quantile_edges(np.asarray(ref, float), bins)
    return psi_from_props(
        bin_props(np.asarray(ref, float), edges), bin_props(np.asarray(cur, float), edges)
    )


def _kolmogorov_sf(lam: float) -> float:
    """Survival function of the Kolmogorov distribution: 2 sum (-1)^(k-1) exp(-2 k^2 lam^2)."""
    if lam <= 0:
        return 1.0
    k = np.arange(1, 101)
    return float(min(max(2.0 * np.sum((-1.0) ** (k - 1) * np.exp(-2.0 * k**2 * lam**2)), 0.0), 1.0))


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample KS statistic and asymptotic p-value (scipy kstwobign)."""
    a = np.sort(np.asarray(a, float)[np.isfinite(a)])
    b = np.sort(np.asarray(b, float)[np.isfinite(b)])
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0, 1.0
    allv = np.concatenate([a, b])
    d = float(
        np.max(
            np.abs(
                np.searchsorted(a, allv, side="right") / n
                - np.searchsorted(b, allv, side="right") / m
            )
        )
    )
    return d, _kolmogorov_sf(d * np.sqrt(n * m / (n + m)))


def status_from_psi(value: float) -> str:
    """green below 0.1, amber 0.1-0.2, red above 0.2."""
    return "red" if value > PSI_RED else "amber" if value >= PSI_AMBER else "green"


def worst(statuses: list[str]) -> str:
    """Worst of a list of green/amber/red."""
    order = {"green": 0, "amber": 1, "red": 2}
    return max(statuses, key=order.__getitem__, default="green")
