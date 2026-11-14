# CONTRACTS — shared by every agent

Every agent reads this file and its own brief, nothing else from `docs/plan/`.
If a contract must change, do not change it silently: write the proposed change in your handoff note.

## Repo layout and ownership

```
prahari/
  pyproject.toml  Makefile  docker-compose.yml  .github/workflows/ci.yml   # A0
  prahari/
    config.py  db.py  schemas.py          # A0 (settings, SQLAlchemy models, Pydantic schemas)
    sim/         # A1  topology, KPI and fault simulator
    pipeline/    # A1  ingestion, features, data-quality checks
    models/
      anomaly/   # A2
      validator/ # A2
      rca/       # A3
    agent/       # A5  tool loop, LLM adapter, runbook retrieval
    governance/  # A5  checks + policy.yaml
    serving/     # A4  FastAPI app
    monitoring/  # A6  drift, metrics, onnx benchmark
  runbooks/      # A5  markdown runbooks
  artifacts/     # model files (git-ignored) + model cards *.card.json (committed)
  fixtures/      # A0  example JSON for every API response (frontend and tests use these)
  web/           # A7
  tests/         # each builder adds unit tests for its own package; Q-agents add the rest
  docs/          # A9; docs/handoffs/ (all), docs/EXPERIMENTS.md (all, append-only)
```

Write only inside the directories you own. Shared files (`schemas.py`, `db.py`, `fixtures/`) are changed only by A0 or the orchestrator.

## Conventions

- Python 3.11, `ruff` clean, type hints, docstrings on public functions. No notebooks.
- Every random process takes a `seed`. Default seed 42.
- Config through `prahari.config.Settings` (env vars, prefix `PRAHARI_`). No hard-coded paths.
- `make setup | seed | train | serve | web | test | bench` must keep working.
- Record every failed approach in `docs/EXPERIMENTS.md`: date, what was tried, result, what changed. This is used in the interview; be honest.
- Report real measured numbers only. Never write a metric you did not run.
- Finish by writing `docs/handoffs/<agent-id>.md`, 30 lines at most: what exists, how to run it, measured numbers, known gaps, contract changes requested.

## Database (SQLAlchemy models in `prahari/db.py`)

| Table | Columns |
|---|---|
| `nodes` | node_id PK, node_type (`core`/`agg`/`access`/`cell`), region, vendor |
| `links` | link_id PK, src FK, dst FK, capacity_mbps |
| `kpi_samples` | ts, node_id FK, latency_ms, packet_loss_pct, throughput_mbps, cpu_pct, mem_pct, temp_c — index (node_id, ts) |
| `alarms` | alarm_id PK, ts, node_id FK, alarm_type, severity (`minor`/`major`/`critical`), raw_text, **label_is_true**, **label_incident_id** — index (ts) |
| `incidents` | incident_id PK, start_ts, end_ts, root_node_id FK, fault_type — **ground truth** |
| `predictions` | pred_id PK, ts, kind, model_name, model_version, input_ref, output_json, latency_ms |
| `actions` | action_id PK, incident_ref, proposed_action, target_node_id, status (`proposed`/`approved`/`rejected`/`executed`/`blocked`), governance_json, decided_by, decided_at |
| `audit_log` | id PK, ts, actor, event, payload_json |
| `drift_reports` | id PK, ts, status (`green`/`amber`/`red`), report_json |

Columns starting `label_` and the `incidents` table are ground truth. **They must never be used as model input features or read by the serving path.**

Fault types: `fibre_cut`, `power_failure`, `congestion`, `hardware_degradation`, `config_error`.

## Model interface

Each model package exposes:

```python
def train(settings) -> dict            # trains, saves artifact + card, returns metrics
def load(version: str | None = None)   # returns a predictor object
predictor.predict(payload) -> dict     # pure function of its input
```

Artifacts: `artifacts/<name>/<version>/model.*` and `artifacts/<name>/<version>.card.json`
(card: name, version, trained_at, data_hash, features, metrics, baseline_metrics, intended_use, limitations).

RCA output (the contract A3, A4, A5 and A7 share):

```json
{
  "window": {"start": "...", "end": "..."},
  "root_candidates": [{"node_id": "AGG-07", "prob": 0.83}, {"node_id": "CORE-02", "prob": 0.09}],
  "fault_type": {"label": "fibre_cut", "prob": 0.91},
  "evidence": {"alarmed_nodes": ["..."], "valid_alarm_count": 41, "noise_alarm_count": 12,
               "kpi_deltas": {"AGG-07": {"packet_loss_pct": 38.2}}},
  "model": {"name": "rca_gnn", "version": "..."},
  "latency_ms": 7.4
}
```

## REST API (FastAPI, prefix `/v1` except health/metrics)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/metrics` | liveness; Prometheus |
| GET | `/v1/topology` | nodes + links |
| GET | `/v1/alarms?since=&limit=` | alarms with `is_valid_pred`, `valid_prob` |
| POST | `/v1/alarms/validate` | score a batch of alarms |
| POST | `/v1/anomaly/score` | anomaly scores for nodes in a window |
| POST | `/v1/rca` | body `{start, end}` → RCA output above |
| GET | `/v1/incidents`, `/v1/incidents/{id}` | detected incidents with RCA, action, governance report |
| POST | `/v1/incidents/{id}/explain?lang=en\|hi` | narrative + cited runbook |
| POST | `/v1/actions/{id}/approve`, `/reject` | human decision, body `{user, note}` |
| GET | `/v1/models` | versions + model cards |
| GET | `/v1/drift/latest`, POST `/v1/drift/run` | drift report |
| POST | `/v1/sim/inject` | body `{fault_type, node_id?}` — demo only |
| POST | `/v1/sim/scenario` | body `{name: "firmware_drift"}` — demo only |
| GET | `/v1/audit?limit=` | audit trail |

`fixtures/<endpoint>.json` holds one realistic example response for each endpoint.

## Governance report (A5 produces, A4 stores, A7 renders)

```json
{"decision": "needs_approval", "checks": [
  {"id": "G1", "name": "confidence_threshold", "status": "pass", "detail": "0.83 >= 0.70"}]}
```
`decision` ∈ `auto_approved`, `needs_approval`, `blocked`. `status` ∈ `pass`, `warn`, `fail`.
