# EXPERIMENTS

Append-only log of every approach tried, including failures. Format per entry:
date, what was tried, result, what changed. Real measured numbers only.


## 2026-11-14 — A1 simulator and pipeline
- Tried: first topology build crashed (`KeyError ('agg','core')`, capacity lookup was order-dependent). Fix: order link endpoints upstream-first. No effect on data.
- Tried: raw per-alarm KPI z-scores. Result: packet-loss z reached ~7000 because the healthy loss baseline has tiny variance. Changed: std floored at 2% of mean and z clipped to [-50, 50] in both alarm and node features.
- Realism check (seed 42, 30 days): target root-alarm-missing was 15%, measured 13.4% (the "keep at least one alarm" rule re-adds a root alarm to otherwise silent incidents). Target 10% overlap: scheduled 10% of incidents to start during the previous one; 19.5% of incidents intersect another in time (counts both members). Left as is.
- Window-feature serving query (one 15 min window, 120 min baseline) takes ~0.66 s on the 5.2M-row SQLite DB; full KPI load for training takes ~19 s. Not optimised; A4 may want a ts index (db.py is A0's).

## 2026-11-14 — Q1 data audit
- Tried: corrupting a KPI value to NULL to test the null checks. Result: SQLite raised IntegrityError (schema NOT NULL), so the null checks can never fire. Changed: test asserts the schema rejects NULLs instead.
- Tried: off-grid KPI timestamp via `ts = '... 10:01:00'`. Result: 0 rows matched (stored as `...00.000000`). Changed: LIKE match; corruption then goes undetected by quality checks (load_kpi floors it silently).
- Leakage probes (depth-2 trees, chronological 70/30 alarm split; 70/15/15 incident split): kpi_z_max alone AUC 0.976 vs 0.998 for HistGB on all alarm features; d_latency_ms alone finds the root top-1 in 82.9% of test incidents vs 98.1% HistGB. Details in docs/handoffs/Q1.md.

## 2026-11-14 — A1 fixes after Q1 audit (F1-F7)
- Before: kpi_z_max depth-2 AUC 0.976; argmax d_latency finds root 82.9%; 694 incidents (60% of minutes faulty); white KPI noise; 8/16 corruptions undetected.
- Changed: per-incident fault magnitude and per-node gain (40% of dependents ~KPI-silent), 30% of noise alarms get a KPI blip, noise/true alarms use all 15 types and 3 severities, 5-8 incidents/day, AR(1) noise, 2% missing samples, 11 new quality checks, window jitter, live root-drop and random tiers.
- First pass (30% silent, blip mag 0.3-1.1): AUC 0.871, top-1 d_latency 57.9% (d_packet_loss 64%). Raised silent share to 40%, dependent gain to U(0.5,2.2), blip mag 0.4-1.4: AUC 0.846, top-1 52.8% / 61.8%.
- Not fixed: HistGB on all alarm features still AUC 0.995 (flap/neighbour counts); packet-loss delta still finds the root 62%.
