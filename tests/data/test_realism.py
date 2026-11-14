"""Realism targets from the A1 brief, measured on the 30-day seed-42 plan (no KPI write)."""

import numpy as np
import pandas as pd

from prahari.sim.constants import FAULT_TYPES
from prahari.sim.faults import FAULT_MIX, RAMP_DOWN


def test_noise_ratio_in_brief_range(plan30):
    noise = 1 - plan30.alarms["label_is_true"].mean()
    assert 0.20 <= noise <= 0.30


def test_incident_count_and_fault_mix(plan30):
    assert 5 * 30 <= len(plan30.faults) <= 8 * 30  # 5-8 incidents per day
    mix = pd.Series([f.fault_type for f in plan30.faults]).value_counts(normalize=True)
    for ft, p in zip(FAULT_TYPES, FAULT_MIX, strict=True):
        assert abs(mix[ft] - p) < 0.07, (ft, mix[ft], p)  # ~180 incidents: sampling noise


def test_root_alarm_missing_rate_near_15pct(plan30):
    a = plan30.alarms
    roots = {f.incident_id: plan30.net.node_ids[f.root_idx] for f in plan30.faults}
    t = a[a["label_is_true"]]
    has = t.groupby("label_incident_id").apply(
        lambda d: (d["node_id"] == roots[d.name]).any(), include_groups=False
    )
    assert len(has) == len(roots)  # every incident has at least one alarm
    assert 0.10 <= 1 - has.mean() <= 0.20


def test_overlap_rate_near_10pct(plan30):
    fs = sorted(plan30.faults, key=lambda f: f.start)
    starts_inside = sum(
        any(g.start <= f.start < g.end for g in fs[max(0, i - 5) : i]) for i, f in enumerate(fs)
    )
    assert 0.06 <= starts_inside / len(fs) <= 0.14


def test_network_mostly_healthy(plan30):
    minutes = np.zeros(plan30.n_minutes, bool)
    for f in plan30.faults:
        i0 = int((f.start - plan30.start) / pd.Timedelta(minutes=1))
        minutes[i0 : i0 + f.duration_min + RAMP_DOWN] = True
    assert minutes.mean() < 0.25, f"{minutes.mean():.1%} of minutes inside an incident"


def test_kpis_physically_plausible(ds7):
    _, store = ds7
    v = store.values
    assert 0.01 <= np.isnan(v).mean() <= 0.05  # a few % of samples are missing
    ok = ~np.isnan(v)
    assert (v[..., 0][ok[..., 0]] > 0).all() and (v[..., 2][ok[..., 2]] >= 0).all()
    assert ((v[..., 1] >= 0) & (v[..., 1] <= 100))[ok[..., 1]].all()
    assert ((v[..., 3:5] >= 0) & (v[..., 3:5] <= 100))[ok[..., 3:5]].all()
    # daily seasonality: evening throughput clearly above morning
    hours = (store.t0 + pd.to_timedelta(np.arange(v.shape[0]), unit="min")).hour
    thr = pd.Series(np.nanmean(v[:, :, 2], axis=1)).groupby(hours).mean()
    assert thr[20] > 1.8 * thr[8]
