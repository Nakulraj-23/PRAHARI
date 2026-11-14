# EXPERIMENTS

Append-only log of every approach tried, including failures. Format per entry:
date, what was tried, result, what changed. Real measured numbers only.


## 2026-11-14 — A1 simulator and pipeline
- Tried: first topology build crashed (`KeyError ('agg','core')`, capacity lookup was order-dependent). Fix: order link endpoints upstream-first. No effect on data.
- Tried: raw per-alarm KPI z-scores. Result: packet-loss z reached ~7000 because the healthy loss baseline has tiny variance. Changed: std floored at 2% of mean and z clipped to [-50, 50] in both alarm and node features.
- Realism check (seed 42, 30 days): target root-alarm-missing was 15%, measured 13.4% (the "keep at least one alarm" rule re-adds a root alarm to otherwise silent incidents). Target 10% overlap: scheduled 10% of incidents to start during the previous one; 19.5% of incidents intersect another in time (counts both members). Left as is.
- Window-feature serving query (one 15 min window, 120 min baseline) takes ~0.66 s on the 5.2M-row SQLite DB; full KPI load for training takes ~19 s. Not optimised; A4 may want a ts index (db.py is A0's).
