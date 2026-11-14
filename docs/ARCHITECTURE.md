# PRAHARI architecture

PRAHARI is a human-in-the-loop network-operations copilot. It runs on synthetic data from its own simulator.
This file lists the components, how data flows between them, and the main design decisions.
Every number here comes from a handoff note in `docs/handoffs/` or a model card in `artifacts/`.

## 1. Components

| # | Component | Code | Job |
|---|---|---|---|
| 1 | Simulator | `prahari/sim/` | Builds a 120-node tiered topology (core, agg, access, cell), 6 KPIs per node per minute, faults that spread downstream, true and noise alarms. Also a live mode for the demo (`live.py`). |
| 2 | Database | `prahari/db.py` | SQLAlchemy 2 models. SQLite by default. 9 contract tables plus `detected_incidents`. |
| 3 | Feature pipeline | `prahari/pipeline/` | Loads KPIs as a (time, node, 6) array, builds alarm and node-window features, runs 60 data-quality checks, makes chronological splits. |
| 4 | M1 anomaly detector | `prahari/models/anomaly/` | IsolationForest on 14 z-score features per node-minute. |
| 5 | M2 alarm validator | `prahari/models/validator/` | HistGradientBoosting + isotonic calibration. Real alarm or noise. |
| 6 | M3 RCA GNN | `prahari/models/rca/` | Hand-written 3-layer residual GCN in PyTorch. Root node + fault type. Exported to ONNX. |
| 7 | Agent | `prahari/agent/` | Tool loop: read-only SQL, get RCA, TF-IDF runbook search, propose action. Writes an English or Hindi narrative. Default provider `none` = deterministic templates, no LLM. |
| 8 | Governance gate | `prahari/governance/` | 10 checks (G1-G10) from `policy.yaml`. Decision: `auto_approved`, `needs_approval` or `blocked`. |
| 9 | Serving API | `prahari/serving/` | FastAPI. All REST endpoints, a background incident detector, approve/reject flow, Prometheus `/metrics`. |
| 10 | Monitoring | `prahari/monitoring/` | PSI and KS in NumPy, drift reports, retrain-and-promote script, ONNX benchmark, load test. |
| 11 | Dashboard | `web/` | React + Vite + Tailwind + Cytoscape + Recharts. Live Ops, Incident, Model health, Audit. |
| 12 | Tests and CI | `tests/`, `web/tests/`, `.github/workflows/ci.yml` | pytest (unit, API, data audit, model quality gates), Playwright. CI runs ruff and pytest. |

## 2. Data flow

```
 Simulator (sim/) ──writes──► SQLite: nodes, links, kpi_samples, alarms, incidents (ground truth)
                                   │
                    label-free reads only (pipeline/store.py, serving/data.py)
                                   ▼
 New alarms ──► M2 validator ──► valid alarms ──► cluster (≤5 min apart, ≤2 hops, ≥2 alarms)
                                                        │
                                                        ▼
                     15-min window (starts 5 min before first valid alarm)
                                                        │
                    node features (alarm counts, KPI deltas, z-scores, tier)
                                                        ▼
                                        M3 RCA GNN ──► top-5 roots + fault type
                                                        ▼
                     Agent: get_rca → search_runbooks → propose_action → narrative (en/hi)
                                                        ▼
                     Governance G1-G10 ──► decision ──► actions row + audit_log row
                                                        ▼
         needs_approval: human approves/rejects in the dashboard
         approve: all 10 checks re-run; any fail → 409 and status "blocked"
         pass: simulator applies the action; network recovers
 Drift monitor (PSI/KS vs reference.json) ──► drift_reports ──► G6 (red = blocked)
 M1 anomaly ──► /v1/anomaly/score (a triage signal; not in the incident path)
```

The detector runs every `detector_interval_s` (10 s by default) inside the API process (`serving/incidents.py`).
It keeps a cursor (last alarm id) in `audit_log`, so a restart resumes where it stopped.

## 3. Ground-truth firewall

`incidents` and every `label_*` column are ground truth. Only training code reads them.

- `pipeline/store.py:load_alarms` uses an allow-list of columns.
- The agent SQL guard rejects `incidents`, any `label_` token, `SELECT *` on alarms, comma joins, and more (`agent/sqlguard.py`).
  It has a second layer: a SQLite authorizer and `PRAGMA query_only`.
- Tests: a SQL-capture test and a source scan in `tests/serving`. Q3 renamed the ground-truth table and columns; every serving endpoint still worked.

## 4. Key design decisions

| Decision | Chosen | Rejected alternative | Why |
|---|---|---|---|
| Data | Own simulator with noise, missing root alarms, overlapping incidents | Public dataset | No public dataset has topology + KPIs + alarms + root-cause labels together. Cost: all results are synthetic. |
| GNN library | GCN written by hand in PyTorch (`models/rca/model.py`) | torch-geometric | Small graphs (120 nodes). Fewer install problems. Shows the maths. ONNX export is simple. |
| GNN baseline | Same network with message passing switched off (`use_graph=False`) | Only rule baselines | A fair test of whether the graph helps. Answer: only a little (+1.0 pt, p about 0.07). |
| Validator model | HistGradientBoosting + isotonic calibration, threshold tuned for recall ≥ 0.98 on validation | Plain 0.5 threshold | Dropping a real alarm costs more than keeping a noise alarm. |
| Anomaly baseline | Lagged hour (ends 75 min before t) | Rolling hour ending 5 min before t | Q2 found the rolling hour absorbed lasting faults (recall 0.02 at 30-60 min). |
| Database | SQLite + WAL; Postgres via `PRAHARI_DB_URL` | Postgres only | Zero setup for a demo. Postgres path is untested. |
| Retrieval | TF-IDF in NumPy over 15 runbooks | Vector database + embeddings | 15 documents. No service to run. |
| Agent | Deterministic `none` provider by default; `anthropic` and `openai_compatible` adapters | LLM required | Demo works offline and is reproducible. Real LLM providers were tested only with fake transports. |
| Narrative safety | G3 groundedness: every node id and number in the text must come from the RCA output | Trust the LLM | Hallucinated numbers are blocked, not shown. |
| Governance | 10 small `Check` subclasses, thresholds in YAML, fail closed on exceptions, re-run at approve time | One big if-else | Each check is testable alone. Adding a check is one class. Re-run fixed a Q3 bypass. |
| Optimisation | ONNX Runtime fp32 and dynamic int8 | TorchScript, GPU | Measured 2.0x and 2.2x faster forward pass. But the forward pass is about 0.1-0.3 ms; the DB query (about 42 ms end to end) dominates. |
| Drift | PSI + KS written in NumPy; reference per hour-of-day x weekday/weekend | One global reference | A global reference gave false reds because KPIs follow a daily cycle. |
| Concurrency | One worker, SQLite WAL, busy timeout, 503 + `Retry-After` on lock, 30 s RCA cache | Many workers | Simple. Cost: `/v1/alarms/validate` slows under load (CPU-bound). |

## 5. Deployment

- Local: `make setup`, `make seed train`, `make serve`, `make web` (see README).
- Docker: `Dockerfile` (API, CPU torch), `web/Dockerfile` (nginx), `docker-compose.yml`. **Never built or run** in this project (disk space). Postgres profile is optional and untested.

## 6. Known weak points

See the Limitations section of `README.md` and section 9 of `docs/INTERVIEW_GUIDE.md`.
