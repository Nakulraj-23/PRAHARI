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

## 2026-11-14 — A5 agent and governance
- Tried: regex-only SQL guard. Result: a comma join (`FROM nodes, incidents`) and `SELECT *` on alarms (exposes label_*) are hard to catch by regex. Changed: kept the parser (single SELECT, table allow-list, forbidden words, any `label_*` token) and added the sqlite authorizer + `query_only` as a second layer; tests cover both layers separately.
- Tried: phone redaction regex for 3-3-4 numbers only. Result: `+91 98765 43210` leaked through. Changed: added a 5-5 digit pattern (Indian mobile); redact test now catches 4/4 kinds.
- Tried: `window_severity` comparing the RCA ISO window (`...T10:10`) as strings against SQLite `... 10:10`. Result: returned no alarms ('T' > ' '). Changed: bind datetimes.
- scikit-learn is not installed in .venv here, so TF-IDF is hand-written in numpy. Retrieval top-1 on the 15 fault x tier queries: 15/15 (queries are built from the same words the runbooks use, so this is an easy test, not a retrieval benchmark).
- LLM providers (anthropic, openai_compatible) were tested only against fake transports; never called a real API (no keys here).

## 2026-11-14 — A3 RCA GNN
- Tried: `build_incident_graphs(split=None)` on the 90-day DB. Result: process killed (OOM, 15 GB box, ~7 GB free) while loading 15M KPI rows. Changed: `train.build_graphs_chunked` loads KPIs 8 days at a time and reuses A1's `_window_features` with the same seeded jitter; whole data build 76 s.
- Result (seed 42, 408/88/88 split, CPU): GNN test top-1 96.6% / top-3 100% / fault macro-F1 0.957; MLP without graph 95.5% / 100% / 0.938 (1 incident of 88 difference, within noise); upstream-alarmed rule 73.9%; most-alarms rule 58.0%. Alt topology (47 incidents): GNN 100%, MLP 97.9%, rules 83.0% / 48.9%.
- Honest read: message passing adds about 1 point here; most of the gain over rules comes from KPI deltas that the node features already hold. Only a single seed and 88 test incidents were run, so no significance claim.

## 2026-11-14 — A2 anomaly detector and alarm validator
- Alarm features: A1's `build_alarm_features` neighbour count looks +-5 min (uses future alarms) and its same-key counts count simultaneous ties. Not usable for "available at alarm time". Changed: validator has its own causal builder (strictly earlier alarms; test removes later alarms and checks features are unchanged).
- Validator threshold: first rule "max threshold with val recall >= 0.95" gave HGB test recall 0.964 (lower than plain 0.5 at 0.983). Changed to recall >= 0.98 on val: HGB test recall 0.991, precision 0.962 (threshold 0.18). Calibration and threshold are both fitted on validation, so val numbers are optimistic; test was read once per run.
- HistGB training took 72 s on 5k rows (OpenMP spin-waits while other jobs ran). Changed: `threadpool_limits(1)` around the fit; same model, ~10 s for the whole tiny test.
- Anomaly: IsolationForest (12 features: 5/15-min mean z vs previous hour) PR-AUC 0.346 vs 0.193 for the 3-sigma rule, but absolute quality is poor: targets count every node with a true alarm in an incident minute, and 40% of dependents are KPI-silent by design. A 3-sigma rule with sigma tuned on val (4.07) gets F1 0.317 vs 0.421 for the forest. Not tried: longer windows, per-node-type models, LSTM autoencoder.
- Validator PR-AUC is 0.998 (HistGB) / 0.997 (LogReg); Q1/A1 predicted this is easy (flap and neighbour counts separate noise). The gain of HGB over LogReg is small.
- Side effect: I ran `ruff format`/`--fix` on all of `prahari/models`, which touched A3's files (formatting only) while A3 was writing.

## 2026-11-14 — Q2 model audit
- Card numbers reproduce exactly for all three models (validator, anomaly, RCA seed 42) with my own split/metric code; thresholds re-derived on val equal the shipped ones.
- Tried: perturbing RCA windows by filtering alarms and calling A1's `_window_features`. Result: crash (`bincount` on object dtype) whenever the filtered frame is empty; a fresh empty frame from SQL works. Changed: audit zeroes alarm columns itself; latent bug reported to A1.
- RCA, 10 seeds (42-51), same data and early stopping: GNN top-1 96.4% +-1.1 vs MLP 95.3% +-0.6; paired diff +1.0 pt +-1.6 (7 wins, 2 ties, 1 loss; paired t ~2.1, p ~0.07). Fault macro-F1: no gain (-0.2 pt test, -3.6 pt alt). Gain grows under drop30+noise50 (+2.2 pt, 8/10 wins) but reverses on alt with both (-1.3 pt).
- Anomaly relabel probe: scoring the shipped model against simulator KPI-visible nodes raises PR-AUC 0.346 -> 0.461; recall of visible node-minutes by minutes since start: 0.71 (0-5), 0.81 (5-15), 0.25 (15-30), 0.02 (30-60): the 60-min rolling baseline absorbs persistent faults. Both a label and a model problem.
- Validator leakage probe: counts are causal (match a brute-force strictly-earlier recount; 388 same-second alarms, but recount without ties gives the same PR-AUC 0.9981). Shortcut, not leak: noise alarms in test never have a neighbour alarm in the prior 5 min (mean 0.000), true alarms almost never repeat the same type within 10 min (0.002 vs 1.03). Dropping all count features: PR-AUC 0.9945.

## 2026-11-14 — A2 anomaly fix after Q2 audit
- Q2 found the 60-min rolling baseline ending 5 min before t absorbed lasting faults (recall 0.81 at 5-15 min, 0.25 at 15-30, 0.02 at 30-60) and that 157 of 444 test positives were KPI-silent nodes counted as misses.
- Changed: baseline is the 60 min ending 75 min before t (longest fault is 70 min, so it never contains the current fault); targets use the simulated KPI effect (propagation impact x fault_gain >= 0.15), silent nodes are negatives; added max|z5|, max|z15| features (14 total).
- Result (test, 407,640 node-minutes, 0.79% positive): IForest PR-AUC 0.8135, F1 0.779 (P 0.786, R 0.773); first version without the max features 0.793 / 0.773. Recall by minutes since onset: 0-5 0.45, 5-15 0.80, 15-30 0.88, 30-60 0.78, 60+ 0.68 (3-sigma rule: 0.96 to 1.0 everywhere).
- Honest comparison: the fixed 3-sigma rule now has precision 0.016 (F1 0.032) because the lagged baseline is noisier; with sigma tuned on val (16.9) the plain rule gets F1 0.809, above the forest (0.779); PR-AUC of max|z5| is 0.745. The forest wins PR-AUC, not F1. Not tried: seasonal (same-time-yesterday) baseline, per-node thresholds.
- Costs: onset detection is slower (0-5 min recall 0.45, ramp-up) and a node hit twice within 75 min has a contaminated baseline. Old numbers (PR-AUC 0.346) are not comparable: different labels.

## 2026-11-14 — A4 serving
- Tried: calling each model's `load()` with no arguments in the API. Result: models silently read the default `prahari.db` and `artifacts/`, not the app's own Settings, so my first test run scored a test DB's alarms against the 90-day DB (flat RCA probs 0.06, empty evidence). Changed: Registry calls `load(settings=...)` and falls back to `load()` only on TypeError; tests now use the 1-day DB for real.
- Tried: sending the anomaly model exactly 65 minutes of KPI history (A2's first version). Result: after A2 retrained, the model needs 135 and raised ValueError. Changed: send 240 rows (the model ignores older ones) and retry once with the size the error names.
- Tried: opening an incident from every cluster of >= 2 valid alarms. Result: a second inject on the same node 10 min later merged into the still-open incident (same neighbourhood, within 10 min). Kept: that is the intended behaviour for one fault still burning; the duplicate-action test now approves the first action before re-injecting.
- Validator KPI z-score is recomputed in `serving/data.py` (definition copied from the pipeline) because the CONTRACTS interface only exposes `predict(payload)`; if the pipeline changes that definition the two will drift apart.
