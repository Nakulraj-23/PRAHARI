"""Drift statistics (against scipy and a hand-computed case) and run_drift on a small simulation."""

import numpy as np
import pytest
from scipy import stats

from prahari.config import Settings
from prahari.monitoring.drift import FEATURES, _rolling_mean, build_reference, run_drift
from prahari.monitoring.stats import (
    bin_props,
    ks_2samp,
    psi,
    psi_from_props,
    quantile_edges,
    status_from_psi,
    worst,
)


def test_psi_hand_computed():
    e, a = np.array([0.5, 0.5]), np.array([0.25, 0.75])
    # (0.25-0.5) ln(0.5) + (0.75-0.5) ln(1.5)
    assert psi_from_props(e, a) == pytest.approx(-0.25 * np.log(0.5) + 0.25 * np.log(1.5), abs=1e-9)
    assert psi_from_props(e, e) == pytest.approx(0.0, abs=1e-12)
    assert psi_from_props(e, np.array([1.0, 0.0])) > 4  # empty bin floored, not infinite


def test_psi_on_samples_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 20000)
    same = psi(ref, rng.normal(0, 1, 20000))
    small = psi(ref, rng.normal(0.3, 1, 20000))
    big = psi(ref, rng.normal(1.0, 1, 20000))
    assert same < 0.01 < small < 0.2 < big
    assert status_from_psi(same) == "green" and status_from_psi(big) == "red"
    assert status_from_psi(0.15) == "amber" and status_from_psi(0.1) == "amber"
    assert worst(["green", "amber"]) == "amber" and worst([]) == "green"


def test_bin_props_edges():
    x = np.array([0.0, 1.0, 1.0, 2.0, 5.0, np.nan])
    assert bin_props(x, np.array([1.0, 3.0])).tolist() == [0.6, 0.2, 0.2]
    assert len(quantile_edges(np.zeros(100))) == 1  # ties collapse to one edge


@pytest.mark.parametrize("shift,n,m", [(0.0, 500, 700), (0.2, 2000, 1500), (1.0, 60, 80)])
def test_ks_matches_scipy(shift, n, m):
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, n), rng.normal(shift, 1, m)
    d, p = ks_2samp(a, b)
    ref = stats.ks_2samp(a, b, method="asymp")
    assert d == pytest.approx(ref.statistic, abs=1e-12)
    # classical asymptotic limit distribution; scipy's asymp applies a finite-n correction
    assert p == pytest.approx(stats.kstwobign.sf(d * np.sqrt(n * m / (n + m))), rel=1e-6, abs=1e-12)
    assert p == pytest.approx(ref.pvalue, rel=0.5, abs=0.02)


def test_ks_with_ties_and_nan():
    a = np.array([1.0, 1.0, 2.0, 3.0, np.nan])
    b = np.array([1.0, 2.0, 2.0, 5.0])
    assert ks_2samp(a, b)[0] == pytest.approx(stats.ks_2samp(a[:4], b).statistic)


def test_rolling_mean_ignores_nan():
    x = np.array([[1.0], [np.nan], [3.0], [5.0]])
    out = _rolling_mean(x, 2)[:, 0]
    assert out.tolist() == [1.0, 1.0, 3.0, 4.0]


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    from prahari.pipeline.splits import make_splits
    from prahari.sim.generate import build_dataset, write_dataset

    d = tmp_path_factory.mktemp("mon")
    s = Settings(db_url=f"sqlite:///{d / 'm.db'}", artifacts_dir=d / "art")
    write_dataset(build_dataset(days=5, seed=42), s)
    make_splits(s)
    build_reference(s, n_days=3, with_predictions=False)
    return s


def test_no_drift_on_normal_window_and_report_schema(sim):
    from prahari.schemas import DriftReportOut

    rep = run_drift(("2026-10-04 06:00", "2026-10-04 18:00"), sim)
    out = DriftReportOut(**rep)
    assert out.status == "green" and {f.feature for f in out.features} == set(FEATURES)
    assert max(f.psi for f in out.features) < 0.1 and out.id > 0


def test_firmware_drift_is_red_and_names_latency(sim):
    from prahari.monitoring.drift import latest_report
    from prahari.sim.live import LiveSimulator

    info = LiveSimulator(sim).scenario("firmware_drift", vendor="Orion", minutes=360)
    rep = run_drift((info["start_ts"], info["end_ts"]), sim)
    assert rep["status"] == "red"
    red = {f["feature"] for f in rep["features"] if f["status"] == "red"}
    assert "latency_ms" in red and "latency_ms_ma15" in red
    assert (
        max(rep["extra"]["vendor_latency_psi"], key=rep["extra"]["vendor_latency_psi"].get)
        == "Orion"
    )
    assert latest_report(sim)["id"] == rep["id"]
