# PRAHARI interview guide

For: the Tech Mahindra Digital Catalyst AI Engineer interview (campus hire).
Goal: you can explain, defend and change every part of this repo.

Rules this guide follows:

- Every number comes from a file in this repo. The source is named next to it.
- Where something is weak, it says so. Say the weak parts yourself, before the interviewer finds them.
- Company facts come only from the public links in section 12. Nothing here is knowledge of Tech Mahindra's internal systems.

---

## 0. How to use this guide in 7 days

| Day | Do | Done when |
|---|---|---|
| 1 | Read sections 1-3. Run `make setup`, `make seed train`, `make serve`, `VITE_MOCK=0 make web`. Click the demo path once. | You can say the 2-minute pitch without notes. |
| 2 | Section 4 (components) with the code open. Read `models/rca/model.py`, `governance/checks.py`, `monitoring/stats.py` line by line. | You can draw the data flow on paper. |
| 3 | Section 5 (concepts). Do the worked examples on paper. Do exercises 1 and 2 (section 14). | You can compute A_hat and PSI by hand. |
| 4 | Sections 6-9 (results, audits, failures, limits). Learn the numbers sheet (section 15). | You can state each weakness in one sentence. |
| 5 | Question bank 13.1-13.3 out loud. Exercises 3-5. | You answer without reading. |
| 6 | Question bank 13.4-13.7. Company sheet (section 12). Read every link yourself. | You have 5 questions to ask. |
| 7 | Two timed demo runs (section 11). Practise the fallback. Rest. | Demo under 5 min twice. |

If you only have 3 days: day 1 = sections 1-4 and the demo; day 2 = sections 5-9 and exercises 1-2; day 3 = question bank, company sheet, two demo runs.

---

## 1. Pitches

### 30 seconds

"PRAHARI is a copilot for a telecom network operations centre. When a fault causes an alarm storm, it first filters noise alarms with a calibrated gradient-boosting model. Then a graph neural network, written by hand in PyTorch, finds the root-cause node on the network topology. An agent proposes a fix from a runbook and writes an explanation in English or Hindi. Ten governance checks run before any action, and a human approves it. It is served with FastAPI, has a React dashboard, drift monitoring and tests. The data is synthetic, from a simulator I designed."

### 2 minutes

Add these points to the 30-second version, in this order:

1. **Problem.** One fibre cut can raise dozens of alarms on many nodes. Operators must find the one node that caused it. A quarter of alarms in my simulator are noise.
2. **Data.** No public dataset has topology, KPIs, alarms and root-cause labels together. So I built a simulator: 120 nodes in 4 tiers, 6 KPIs per minute, 5 fault types, 90 days, 584 incidents (`docs/handoffs/A1.md`).
3. **Models.** Validator: precision 0.962, recall 0.991. RCA GNN: root top-1 96.6% on 88 test incidents, top-3 100%.
4. **Honesty.** I ran independent audits. The first data was too easy; I fixed it. The GNN beats a no-graph MLP by only 1 point over 10 seeds, which is not significant. I report that.
5. **Production parts.** Versioned models with model cards, REST API, 10-check governance with an audit log, PSI/KS drift that blocks actions when red, ONNX export (2x faster forward pass), load test, CI.
6. **Close.** "It is a small version of a problem Tech Mahindra has said publicly it is working on: alarm validation, root-cause analysis and governed resolution with a human in the loop."

### Why I built this for Tech Mahindra (one paragraph)

"Telecom is a large part of Tech Mahindra's business. In March 2026 it announced, with NVIDIA, agents for communication service providers that do autonomous alarm validation, root-cause analysis and resolution with a human in the loop ([press release](https://www.techmahindra.com/insights/press-releases/techm-advances-ai-driven-autonomous-network-operations-csps/)). At DTW Ignite 2026 it showed 'GNN-Healing Networks' ([event page](https://www.techmahindra.com/insights/events/tech-mahindra-platinum-sponsor-tm-forum-dtw-ignite-2026/)). TechM Orion talks about configurable governance checks per agent ([Orion](https://www.techmahindra.com/insights/press-releases/tech-mahindra-unveils-techm-orion/)). So I built the same three stages at small scale, with a GNN for RCA and a governance gate. I do not know how your systems work inside. This is my own version, on synthetic data."

---

## 2. Say these facts first (honesty block)

Say them early. An interviewer trusts the rest more when you do.

1. **All data is synthetic.** It comes from `prahari/sim/`. Accuracy here is not real-network accuracy.
2. **The first data was too easy.** Audit Q1 found one feature (`kpi_z_max`) separated real from noise alarms with AUC 0.976. I changed the simulator; it fell to 0.846 (`docs/handoffs/Q1.md`, `docs/EXPERIMENTS.md`).
3. **The graph helps only a little.** Over 10 seeds, GNN root top-1 is 96.4% ± 1.1 and the no-graph MLP is 95.3% ± 0.6. Paired difference +1.0 pt, p about 0.07. No gain on fault type (`docs/handoffs/Q2.md`).
4. **Validator 0.998 PR-AUC is a simulator shortcut.** In test, noise alarms never had a neighbour alarm in the previous 5 minutes. The honest number is noise-class AP 0.9675 (`Q2.md`, validator card).
5. **A tuned 3-sigma rule beats my IsolationForest on F1** (0.809 vs 0.779). The forest wins PR-AUC only (0.8135 vs 0.745) (anomaly card `20261114T100804`).
6. **2 of 10 live injected faults were not detected** (leaf nodes with fewer than 2 valid alarms) (`A4.md`).
7. **Validate latency degrades under load** on one worker: p50 56 ms at 1 client, about 4.3 s at 32 (`Q3.md`).
8. **Docker was never built or run. Postgres is untested.** (`Q3.md`, `docker-compose.yml`)

### "Was this built with AI tools?"

Answer truthfully. Model answer:

"Yes. I used an AI-assisted, multi-agent workflow. I wrote the plan: the problem, the scope, the architecture, the contracts between components and the quality gates (`docs/plan/`). Builder agents wrote code against those contracts. Separate audit agents checked the data, the models and the whole system, and their findings drove fixes. That is why the repo has handoff notes and an experiments log. My job now is to own it: I can explain every file, I re-ran training, and I changed parts myself, for example [name the exercise you did from section 14]. I think this is how AI engineering work is done now, and the audits are the part I am most proud of, because they caught real problems."

Do not hide it. Do not say you typed every line. Do show you understand every line: offer to open any file and explain it.

---

## 3. System map

```
 Simulator ──► SQLite (nodes, links, kpi_samples, alarms, incidents = ground truth)
                     │   label-free reads only
                     ▼
              Feature pipeline (pandas/NumPy)
     ┌───────────────┼────────────────┐
     ▼               ▼                ▼
 M1 Anomaly     M2 Alarm validator   M3 RCA GNN (PyTorch → ONNX)
 IsolationForest HistGB + isotonic   root node + fault type
 (triage only)       │                ▲
                     └─ valid alarms ─┘ (clustered into incidents)
                                      ▼
                 Agent (read-only SQL, RCA, runbook search, propose action)
                                      ▼
                 Governance G1-G10 (policy.yaml) ──► audit_log
                                      ▼
                 Human approve/reject (checks re-run) ──► simulator applies action
 Drift monitor (PSI/KS) ──► G6        FastAPI serves all ◄── React dashboard
```

One incident, step by step (what the code does):

1. The live simulator writes new alarms and KPI rows (`sim/live.py`).
2. Every 10 s the detector reads alarms after its cursor (`serving/incidents.py`).
3. The validator scores them. Alarms above threshold 0.18 are "valid".
4. Valid alarms within 5 minutes and 2 hops of each other are joined (union-find). A cluster needs at least 2 valid alarms.
5. RCA runs on a 15-minute window that starts 5 minutes before the first valid alarm.
6. The agent picks a runbook and an action. Governance runs 10 checks.
7. An `actions` row is written with the governance report. Every step goes to `audit_log`.
8. A human approves. All 10 checks run again. If any fails, the API returns 409 and marks the action blocked.

---

## 4. Component walkthrough

For each: what it does, how, why this way, file.

### 4.1 Simulator — `prahari/sim/`
- **What.** A fake but structured telecom network and 90 days of data.
- **How.** `topology.py` builds 6 core, 18 agg, 30 access, 66 cell nodes (120). Cores form a ring with chords. Some nodes get a second uplink (redundancy). 3 regions, 3 vendors (Nexa, Orion, Zenith; made-up names).
  `kpi.py` makes 6 KPIs per minute with daily seasonality and AR(1) noise (phi 0.7), 2% missing samples.
  `faults.py` schedules 5-8 faults per day. `propagate` spreads impact downstream tier by tier: impact 1 at the root, times 0.8 per hop, shared across parents (a node with two uplinks loses less). `fault_gain` gives each incident its own magnitude, and 40% of dependents almost no KPI change.
  `alarms.py` adds noise alarms (25%) with vendor-style raw text. The root alarm is missing in about 13.5% of incidents.
- **Why.** Real labelled data with topology is not public. The simulator was made harder after audit Q1 (section 7).
- **Numbers.** 584 incidents, 6393 alarms, 15,241,049 KPI rows, DB 2.3 GB (`A1.md`).

### 4.2 Database — `prahari/db.py`
- Tables: `nodes`, `links`, `kpi_samples`, `alarms`, `incidents`, `predictions`, `actions`, `audit_log`, `drift_reports`, `detected_incidents`.
- Index `(node_id, ts)` on `kpi_samples` and `ts` on `alarms`.
- Ground truth: the `incidents` table and `label_*` columns. Never read by serving.
- SQLite in WAL mode with a 30 s busy timeout (A4 concurrency fix).

### 4.3 Feature pipeline — `prahari/pipeline/`
- `store.py`: `load_kpi` returns a (time, node, 6) array. `load_alarms` reads an allow-list of columns, so labels cannot leak.
- `features.py`: alarm features and node-window features for RCA. Per node: 15 alarm-type counts, total, critical count, first-alarm minute, KPI deltas vs a baseline, z-scores, tier one-hot (34 features, see the RCA card).
- `quality.py`: 60 data-quality checks (completeness, grid timestamps, flatlines, time bounds, label consistency). Fails loudly with `DataQualityError`.
- `splits.py`: chronological train/val/test split of incidents (408/88/88).
- Why chronological: a random split would let the model see the future and similar overlapping incidents.

### 4.4 M2 alarm validator — `prahari/models/validator/model.py`
- **What.** Probability that an alarm is real.
- **Features (33).** Severity, alarm type, node type, vendor one-hots; same-node and same-type alarm counts in the last 10 min; neighbour alarms in the last 5 min; 6 KPI z-scores at alarm time and their max.
- **How.** Baseline LogisticRegression. Main model HistGradientBoosting (300 iterations, learning rate 0.05, 15 leaves). Isotonic calibration fitted on validation. Threshold = the largest value that keeps validation recall ≥ 0.98 (0.1765).
- **Why.** Dropping a real alarm is worse than passing a noise alarm, so the threshold favours recall.
- **Important detail.** The first feature builder (from the pipeline) counted alarms up to 5 minutes in the future. A2 wrote a causal builder: only strictly earlier alarms. A test removes later alarms and checks features do not change.
- **Numbers.** Test 826 alarms, 78.1% true. P 0.962, R 0.991, F1 0.976, PR-AUC 0.998, Brier 0.021. LogReg: P 0.958, R 0.988, PR-AUC 0.997 (card).

### 4.5 M1 anomaly detector — `prahari/models/anomaly/model.py`
- **What.** Flags node-minutes whose KPIs depart from the node's own past.
- **Features (14).** For each KPI: z-score of the 5-minute and 15-minute mean against a baseline hour; plus max |z5| and max |z15|.
- **Baseline hour.** The 60 minutes ending 75 minutes before t. Longest fault is 70 minutes, so the baseline never contains the current fault.
- **How.** IsolationForest, 200 trees, 512 samples per tree, fitted on up to 300,000 training rows. Threshold maximises F1 on validation.
- **Not in the incident path.** It is served at `/v1/anomaly/score` as a triage signal.
- **Numbers.** PR-AUC 0.8135, F1 0.779. Tuned 3-sigma rule F1 0.809. Recall 0.45 at 0-5 min after onset, 0.88 at 15-30 (card `20261114T100804`).

### 4.6 M3 RCA GNN — `prahari/models/rca/`
- **What.** Given a 15-minute window, a probability for each node being the root, and a fault type.
- **Model (`model.py`).** Input: (nodes × 34 features). Signed log1p, then standardise (stored in buffers so ONNX takes raw input). Linear to 64 dims. 3 residual GCN layers: `h = h + relu(A_hat · (h W))`, dropout 0.2. Root head: small MLP per node → one logit per node → softmax over nodes. Fault head: mean-pool and max-pool over nodes → MLP → 5 logits.
- **Training (`train.py`).** Adam, lr 2e-3, weight decay 1e-5, up to 80 epochs, early stopping with patience 12 on validation top-1. Loss = cross-entropy(root) + 0.5 × cross-entropy(fault). The MLP baseline is the same network with `use_graph=False`.
- **Why a GNN.** A fault spreads along links, so a node's neighbours carry evidence. Three layers = each node sees 3 hops.
- **Why hand-written.** 120 nodes, so a dense matrix multiply is fine. No torch-geometric install. ONNX export is trivial.
- **Numbers.** Test top-1 96.6%, top-3 100%, fault macro-F1 0.957. Root alarm missing: top-1 81.8% (11 incidents). Unseen 80-node topology: 100% on 47 incidents (card).

### 4.7 Agent — `prahari/agent/`
- **Tools (`tools.py`).** `sql_query` (read-only, guarded), `get_rca`, `search_runbooks`, `propose_action`. Tool errors are returned to the model, not raised.
- **Providers (`llm.py`).** `none` (default, deterministic: fixed tool order and templates), `anthropic`, `openai_compatible` (covers NIM, Azure, Ollama-style endpoints). Max 6 steps. The real providers were tested only with fake transports.
- **Retrieval (`retrieval.py`).** TF-IDF written in NumPy (sublinear tf, smoothed idf, L2 norm), cosine similarity over sections of 15 runbooks. Retrieval top-1 15/15, but the queries use the same words as the runbooks, so it is an easy test.
- **SQL guard (`sqlguard.py`).** Layer 1: parser (one SELECT, table allow-list, forbidden words, no `label_`, no `incidents`, no `SELECT *` on alarms, no comma join, no WITH/PRAGMA). Layer 2: SQLite authorizer + `PRAGMA query_only` + 2 s progress-handler timeout. 19 injection probes rejected in Q3.
- **Templates (`templates.py`).** English and Hindi narratives.

### 4.8 Governance — `prahari/governance/`
- `checks.py`: abstract class `Check` with `run(ctx)`. One subclass per check, added by a `@register` decorator to `REGISTRY`. `evaluate()` wraps `run()`: any exception becomes `fail` (fail closed).
- `policy.yaml`: thresholds.
- `decide()`: any fail → `blocked`; any warn (or G10 not pass) → `needs_approval`; else `auto_approved`.

| Check | Rule (policy.yaml) |
|---|---|
| G1 confidence | root prob pass ≥ 0.70, fail < 0.50; fault-type prob pass ≥ 0.60, fail < 0.40; worst of the two |
| G2 margin | top-1 minus top-2: pass ≥ 0.15, fail < 0.05 |
| G3 groundedness | every node id and number in the narrative must come from the RCA output |
| G4 allow-list + blast radius | action in allow-list; downstream nodes warn > 5, fail > 40 |
| G5 redaction | no email, phone, IP or key left in text |
| G6 drift | green pass, amber warn, red fail, unknown warn |
| G7 model card | card file exists for the model version |
| G8 schema | RCA and agent output validate against Pydantic schemas |
| G9 duplicate action | no other action on the same node in the last 30 min |
| G10 severity | major or critical → human required |

### 4.9 Serving API — `prahari/serving/`
- `app.py`: FastAPI app, all endpoints, models loaded once at startup (lifespan). A missing model gives 503 only on its endpoints.
- `incidents.py`: detector and approve/reject. Uses a lock so parallel approvals are safe (Q3: 5 approves + 5 rejects in parallel gave one 200, nine 409).
- `obs.py`: request ids, JSON logs, per-endpoint latency histograms on `/metrics`, typed error body `{"error": {type, message, request_id}}`.
- Limits (Q3 fixes): RCA window ≤ 6 h, 422 `no_data` without KPIs, validate ≤ 1000 alarms, no duplicate ids.
- `/v1/rca` result cached 30 s per window.

### 4.10 Monitoring — `prahari/monitoring/`
- `stats.py`: PSI and two-sample KS in NumPy. PSI < 0.1 green, 0.1-0.2 amber, > 0.2 red.
- `drift.py`: compares a window (default last 6 h) against `artifacts/drift/reference.json`. Reference is grouped by hour-of-day × weekday/weekend. Features: 6 KPIs, 15-min means of latency and loss, validator output, RCA confidence (outputs only with ≥ 100 values).
- `retrain.py`: trains into a staging folder; promotes only if no metric is worse by more than 0.002.
- `bench.py`: PyTorch vs ONNX fp32 vs ONNX int8. `loadtest.py`: closed-loop async load test.

### 4.11 Dashboard — `web/`
- React 18, Vite, Tailwind, Cytoscape (topology), Recharts (charts). One API client in `src/api.js`. `VITE_MOCK=1` serves fixtures.
- Views: Live Ops, Incident, Model health, Audit. Playwright tests: mock smoke and one real-API test.

### 4.12 Tests and CI
- 168 tests pass locally (Q3). Clean clone without data: 147 pass, 21 skip.
- Kinds: unit (A_hat, PSI, checks), API tests on a 1-day DB, data-audit tests (`tests/data`), model quality gates (`tests/models/test_quality_gates.py`), a ground-truth scan, Playwright.
- CI (`.github/workflows/ci.yml`): ruff + pytest on push. Card quality gates run in CI because the cards are committed.

---

## 5. Concepts from first principles

### 5.1 GCN and the normalised adjacency (worked example)

A graph has an adjacency matrix A: `A[i][j] = 1` if nodes i and j are linked.
A GCN layer mixes each node's vector with its neighbours' vectors:

`H_next = relu( A_hat · H · W )`, with `A_hat = D^-1/2 (A + I) D^-1/2`.

- `A + I` adds a self-loop, so a node keeps its own features.
- `D` is the degree matrix of `A + I` (row sums).
- Dividing by `sqrt(d_i · d_j)` stops high-degree nodes from getting huge sums. It keeps the scale stable across layers.

**Example: 3 nodes in a line, A — B — C.**

```
A     = [[0,1,0],[1,0,1],[0,1,0]]
A + I = [[1,1,0],[1,1,1],[0,1,1]]      degrees d = [2, 3, 2]
A_hat[i][j] = (A+I)[i][j] / sqrt(d_i * d_j)

A_hat = [[0.500, 0.408, 0    ],
         [0.408, 0.333, 0.408],
         [0,     0.408, 0.500]]
```
(0.5 = 1/√(2·2); 0.408 = 1/√(2·3); 0.333 = 1/√(3·3).)

Give one feature "has alarm": x = [1, 0, 0] (only A alarmed).

- One layer (ignoring W and relu): `A_hat · x = [0.5, 0.408, 0]`. B now "knows" about A's alarm. C does not.
- Two layers: `A_hat · A_hat · x = [0.417, 0.340, 0.167]`. Now C has signal too.

So k layers = k hops of information. PRAHARI uses 3 layers. The code is `normalised_adjacency()` in `prahari/models/rca/model.py`; the test is in `tests/test_rca.py`.
PRAHARI adds a residual: `h = h + relu(A_hat · (h W))`. The node keeps its own vector and adds the neighbour message. This helps training and stops over-smoothing (all nodes looking the same).

Cost: dense `A_hat · M` is O(N² · hidden). At 120 nodes that is nothing. At 100k nodes you need a sparse matrix (O(E · hidden)).

### 5.2 IsolationForest

Idea: an outlier is easy to isolate.

1. Take a random sample (512 points here).
2. Build a tree: pick a random feature and a random split value between its min and max. Repeat until each point is alone.
3. An outlier gets isolated after few splits (short path). A normal point needs many.
4. Average the path length over 200 trees. Score `s = 2^(-E[h(x)] / c(n))`, where c(n) is the average path length in a random tree of n points. Near 1 = anomaly.

PRAHARI uses `-model.score_samples(x)` as the score and picks the threshold that gives the best F1 on validation.
It is unsupervised: labels are used only to choose the threshold and to evaluate.

### 5.3 Gradient boosting (HistGradientBoosting)

1. Start with one constant prediction (the log-odds of the base rate). Example: 78% true → log(0.78/0.22) ≈ 1.27.
2. For each alarm, compute the gradient of the log loss: roughly (true label − predicted probability).
3. Fit a small tree (here ≤ 15 leaves) to those residuals.
4. Add the tree times the learning rate (0.05) to the prediction.
5. Repeat 300 times. Each tree fixes what the previous ones got wrong.

"Hist" = features are binned into at most 256 buckets first, so finding splits is fast.
L2 regularisation (1.0) and the small learning rate fight overfitting.

### 5.4 Calibration

A model is calibrated if, among alarms scored 0.8, about 80% are real.
Boosted trees are often not. Isotonic regression fits a non-decreasing step function from raw score to observed frequency, on validation data.
Measure it with the Brier score (mean squared error of probabilities) and a reliability table.
PRAHARI: Brier 0.021 (0.020 before calibration). Q2's reliability table: 624 alarms at mean predicted 0.997, observed 0.997; 142 at 0.035 / 0.035. Calibration matters because governance thresholds (G1) read probabilities as confidence.

### 5.5 Precision, recall and the threshold (worked example)

- Precision = of the alarms I kept, how many are real. TP / (TP + FP).
- Recall = of the real alarms, how many I kept. TP / (TP + FN).

Validator test set: 826 alarms, 78.1% true → about 645 true and 181 noise.
At threshold 0.18: recall 0.991 → about 639 kept, 6 lost. Precision 0.962 → about 25 noise alarms pass (derived; A2 says "about 25 of 181").
At threshold 0.5: precision 0.980, recall 0.983.
Lowering the threshold saved about 5 more real alarms and let in more noise. That is the trade-off, chosen on purpose.

**PR-AUC vs ROC-AUC.** With a 78% positive rate, PR-AUC on the true class looks great even for a weak model. So Q2 reported noise-class AP (0.9675): how well the model finds the minority class. For the anomaly model (0.79% positive) PR-AUC is the right metric; ROC-AUC would look high because of the huge number of easy negatives.

### 5.6 PSI and KS (worked examples)

**PSI (population stability index).** Bin the reference data into quantile bins (10 here). Measure the share of the reference (e) and the current window (a) in each bin.
`PSI = Σ (a − e) · ln(a / e)`.

Example with 4 bins. Reference e = [0.25, 0.25, 0.25, 0.25]. Current a = [0.10, 0.20, 0.30, 0.40].

| bin | a − e | ln(a/e) | product |
|---|---|---|---|
| 1 | −0.15 | −0.916 | 0.137 |
| 2 | −0.05 | −0.223 | 0.011 |
| 3 | +0.05 | +0.182 | 0.009 |
| 4 | +0.15 | +0.470 | 0.071 |

PSI = 0.228 → red (> 0.2). Code: `psi_from_props` in `prahari/monitoring/stats.py`. Empty bins are floored at 1e-4 so the log stays finite.

**KS (Kolmogorov–Smirnov).** The largest vertical gap between the two empirical CDFs.
Example: a = [1, 2, 3, 4], b = [3, 4, 5, 6]. At x = 2: F_a = 0.5, F_b = 0 → gap 0.5. At x = 3: 0.75 vs 0.25 → 0.5. At x = 4: 1.0 vs 0.5 → 0.5. D = 0.5.
p-value: λ = D·√(nm/(n+m)) = 0.5·√2 = 0.707 → p ≈ 0.70 (not significant with 4 points). Code: `ks_2samp`.
KS p-values become tiny with large samples even for small shifts, so PRAHARI uses PSI for status and KS as extra evidence.

PRAHARI result: `firmware_drift` (+25% latency on one vendor) gave latency PSI 0.44, vendor Nexa 4.99, others < 0.01. A 72 h window hid it (PSI 0.005), so the default window is 6 h (`A6.md`).

### 5.7 Quantisation

Store weights as 8-bit integers instead of 32-bit floats: `w ≈ scale × (q − zero_point)`.
Dynamic quantisation: weights are quantised ahead of time; activations are quantised on the fly at run time.
Effect here: model 113 KB → 43 KB, forward pass p50 0.244 ms (PyTorch) → 0.135 ms (ONNX fp32) → 0.122 ms (int8). Accuracy unchanged on 88 test incidents (`docs/bench.md`).
Honest read: most of the speed-up comes from ONNX Runtime, not int8. And end-to-end predict is about 42 ms because of the DB query. So the forward pass was never the bottleneck.

### 5.8 REST semantics (as used in this API)

- GET reads and has no side effects (`/v1/incidents`). It is safe to retry.
- POST creates or triggers (`/v1/rca`, `/v1/actions/{id}/approve`).
- Approve is not idempotent: the second approve returns **409 Conflict** ("action already executed").
- **422** = valid JSON but invalid content (unknown alarm type, window > 6 h, no data).
- **404** = unknown id. **503** = model missing or database busy, with `Retry-After: 1`.
- Pydantic schemas in `prahari/schemas.py` give typed requests and responses and OpenAPI docs at `/docs`.
- Versioned prefix `/v1` so a breaking change can go to `/v2`.

### 5.9 Database: normalisation and indexing

- **Normalisation.** Node facts (type, region, vendor) live once in `nodes`. `links`, `kpi_samples`, `alarms` refer to them by `node_id` (foreign key). No repeated vendor strings in 15M KPI rows. That is roughly third normal form.
- **Deliberate denormalisation.** Reports (`governance_json`, `rca_json`, `report_json`) are stored as JSON. They are written once and read whole, so splitting them into tables gains nothing.
- **Indexing.** `kpi_samples` has a composite index on `(node_id, ts)`. A query "node X between t1 and t2" seeks to X, then scans a small range of ts. Without it, SQLite scans 15M rows.
- **Real lesson.** Even with the index, SQLite chose a bad join order and scanned the whole table (1.4 s per window). Writing `CROSS JOIN` forces SQLite to drive from `nodes` and use the index: 1078 ms → 30 ms for a 75-minute window (`docs/bench.md`). Check with `EXPLAIN QUERY PLAN`.

### 5.10 Graph traversal (BFS/DFS) and complexity

Three traversals in the code:

1. `NetworkModel._hop_distances` (`sim/topology.py`): BFS from all nodes at once. The frontier is a boolean matrix; one step = frontier × adjacency. Each step is a dense N×N by N×N product, O(N³), times the graph diameter. Fine for 120 nodes. Classic BFS with an adjacency list is O(V + E) per source.
2. `dependent_count` (`agent/flow.py`): DFS with a stack, only moving to deeper tiers. Counts downstream nodes for the blast-radius check G4. O(V + E).
3. `propagate` (`sim/faults.py`): tier by tier, impact = 0.8 × average impact of parents. O(depth × N²) with dense matrices.

BFS vs DFS: both visit every reachable node once. BFS gives shortest hop counts. DFS uses less memory on wide graphs. For "how many nodes are downstream" either works.

### 5.11 Union-find (alarm clustering)

`Incidents._clusters` (`serving/incidents.py`) joins two alarms if they are ≤ 5 min apart and ≤ 2 hops apart. It uses union-find with path halving. Every pair is compared: O(n²) for n new alarms. Fine for a few dozen alarms per 10 s tick. For a storm of thousands, bucket alarms by time first.

### 5.12 TF-IDF

- tf = how often a word appears in a section (sublinear: 1 + log tf).
- idf = log((1 + n) / (1 + df)) + 1: rare words count more.
- Each section becomes a vector, L2-normalised. Search = cosine similarity with the query vector.

### 5.13 Evaluation hygiene

- Chronological split: train on the past, test on the future.
- Thresholds and calibration fitted on validation; test read once.
- Baselines tuned the same way (MLP has the same early stopping; 3-sigma is tuned on validation).
- Paired comparison over seeds: same data, same seed for both models, look at the difference. Mean difference 1.0, sd 1.6, n = 10 → t ≈ 1.0 / (1.6/√10) ≈ 2.0 → p about 0.07. Not below 0.05.

---

## 6. Results and how to read them

| Model | Test metric | Value | Baseline | Source |
|---|---|---|---|---|
| RCA GNN | root top-1 / top-3 (n = 88) | 96.6% / 100% | MLP 95.5% / 100%; upstream rule 73.9%; most-alarms 58.0% | RCA card |
| RCA GNN | fault macro-F1 | 0.957 | MLP 0.938 | RCA card |
| RCA GNN | 10 seeds top-1 | 96.4% ± 1.1 | MLP 95.3% ± 0.6 | Q2 |
| RCA GNN | alt topology top-1 (n = 47) | 100% | MLP 97.9% | RCA card |
| Validator | P / R / F1 | 0.962 / 0.991 / 0.976 | LogReg 0.958 / 0.988 / 0.973 | validator card |
| Validator | noise-class AP | 0.9675 | | Q2 |
| Anomaly | PR-AUC / F1 | 0.8135 / 0.779 | tuned 3-sigma 0.745 / 0.809 | anomaly card |

How to read them:

- **RCA.** Top-3 is 100%, so the true root is always in the operator's shortlist. Errors happen when two incidents overlap: 17 windows with a concurrent incident score 88.2%, the other 71 score 98.6% (Q2). When the root alarm is missing, top-1 drops to 81.8%.
- **RCA errors (Q2).** The 3 test errors were 1, 4 and 5 hops off. The 4- and 5-hop errors picked the root of a concurrent incident. A 1-hop error picked the child of a silent root: the model does not infer a silent upstream node.
- **RCA calibration (Q2).** 62 windows with confidence ≥ 0.99 were all right. 4 windows at 0.5-0.8 were 50% right: slightly overconfident at the low end. G1 sends those to a human.
- **Robustness (Q2).** Dropping 30% of alarms and adding 50% noise: shipped top-1 94.3%, fault F1 0.906. Injected noise had no KPI blip, so this is optimistic.
- **Validator.** Near ceiling because of the simulator shortcut. Without KPI z-scores, recall falls to 0.836 (Q2), so the serving path must compute them.
- **Anomaly.** The forest ranks better (PR-AUC) but a tuned simple rule has better F1. In an interview, say: "I would ship the rule as the baseline and keep the forest only if it beats the rule on the metric the operator cares about."

---

## 7. The audit-and-fix loop (a strength)

Three independent audits checked the work. Each found real problems. Each problem was fixed or written down.

### Q1 — data audit (`docs/handoffs/Q1.md`)
| Finding | Fix |
|---|---|
| One feature (`kpi_z_max`) gave AUC 0.976 for alarm validity: noise alarms ignored KPIs | 30% of noise alarms got a real KPI blip; low-impact true alarms show almost no KPI change. AUC fell to 0.846. |
| 9 of 15 alarm types appeared in one class only; critical was 331/331 true | Every type and severity appears in both classes |
| Largest latency delta = root in 82.9% of incidents | Per-incident magnitude and per-node gain; now 52.8% |
| Windows started exactly at fault start | Window start jittered 0-10 min earlier |
| 23 incidents/day, white noise, no gaps | 5-8/day, AR(1) noise, 2% missing samples |
| Quality checks missed 8 of 16 corruptions | 11 new checks (completeness, grid, flatline, time bounds, label consistency) |

Not fixed, and said so: gradient boosting on all alarm features still gets AUC 0.995.

### Q2 — model audit (`docs/handoffs/Q2.md`)
| Finding | Fix |
|---|---|
| GNN vs MLP: +1.0 pt over 10 seeds, p about 0.07 | Reported in the card and README. Claim reduced. |
| Validator 0.998 is a simulator shortcut (neighbour counts) | Report noise-class AP 0.9675; written in the card |
| Anomaly PR-AUC 0.346: the rolling baseline absorbed lasting faults (recall 0.02 at 30-60 min) and KPI-silent nodes were labelled positive | Lagged baseline (ends 75 min before t), labels from simulated KPI effect, 2 new features: PR-AUC 0.8135. The tuned rule still wins F1; card says so. |
| All card metrics reproduce to 1e-4 | Quality gates added in `tests/models/test_quality_gates.py` |

### Q3 — system test (`docs/handoffs/Q3.md`)
| Finding | Fix |
|---|---|
| HIGH: approve did not re-run governance; an action approved after drift went red was executed | Approve re-runs all 10 checks; 409 on any fail; audit event `approval_blocked_by_governance` |
| HIGH: model cards were git-ignored, so CI quality gates silently skipped | `.gitignore` fixed; cards committed |
| HIGH: low fault-type confidence (0.36) passed governance | G1 now also checks fault-type probability |
| MED: one fault gave two incidents | A new cluster joins an open incident if its RCA root matches |
| MED: unbounded work (87-day RCA window took 74 s; agent SQL ran 100 s) | Window ≤ 6 h, batch ≤ 1000, 2 s SQL timeout |
| MED: no-data RCA returned 200 with flat probabilities | 422 `no_data` |
| LOW: empty user name accepted, duplicate ids accepted | Validation added |
| Load test: `/v1/rca` at 32 clients returned 500 `database is locked` (found by A6) | WAL, busy timeout, 503 + Retry-After, 30 s cache: 0 errors at 50 clients |

Not fixed: Docker never run; Postgres untested; validate latency under load.

How to say it: "I did not trust my own numbers. I had independent audits try to break the data, the models and the system. They found that the data was too easy, that the governance gate could be bypassed, and that CI gates were silently skipping. I fixed those and I report what is still weak."

---

## 8. What failed and what I learned (from `docs/EXPERIMENTS.md`)

| Tried | What happened | Learned |
|---|---|---|
| Raw KPI z-scores | Packet-loss z reached about 7000 (tiny baseline variance) | Floor the std (2% of mean) and clip z to ±50 |
| Building all 90 days of RCA graphs at once | Out of memory on a 15 GB box | Load KPIs 8 days at a time |
| Pipeline's alarm neighbour count | Looked 5 min into the future | Write causal features; test them by deleting the future |
| Validator threshold for recall ≥ 0.95 | Test recall 0.964, lower than at 0.5 | Use recall ≥ 0.98 on validation |
| HistGB fit | 72 s on 5k rows (OpenMP spin-waits) | `threadpool_limits(1)` |
| Regex-only SQL guard | Comma joins and `SELECT *` slipped through | Parser + SQLite authorizer as two layers |
| Phone redaction for 3-3-4 numbers | `+91 98765 43210` leaked | Add 5-5 digit pattern |
| Comparing ISO and SQLite timestamps as strings | No rows ('T' > ' ') | Bind datetimes, never compare strings |
| Calling model `load()` with no settings in the API | Scored a test DB against the 90-day DB | Pass settings explicitly |
| Drift reference per 4-hour block | False reds on normal windows | Reference per hour-of-day × weekday/weekend |
| 72 h drift window | Hid a 6 h drift (PSI 0.005) | Default window 6 h |
| Plain JOIN for KPI query | Full scan of 15M rows | `CROSS JOIN` fixes join order: 36x faster |
| Rolling anomaly baseline | Absorbed the fault | Lag the baseline past the longest fault |

---

## 9. Honest limitations and what production would need

Limitations:

- Synthetic data. Real networks have mixed vendors, missing topology, alarm floods of thousands, and changing configs.
- Scale: 120 nodes, one SQLite file, one worker. Dense matrices.
- Small test sets (88 incidents, 47 on the alt topology). No significance for GNN vs MLP.
- 2 of 10 live faults missed (rule: ≥ 2 valid alarms to open an incident).
- The serving path recomputes validator KPI z-scores in `serving/data.py`, a copy of the pipeline definition. They can drift apart.
- Real LLM providers never called. G3 blocks any number not in the RCA output, so a real LLM narrative quoting a time would be blocked.
- Docker never run; Postgres untested; no authentication on the API.
- Drift thresholds 0.1/0.2 are conventional, not tuned. Retraining was run once, as a dry run on the validator.

What production would need:

1. Real data: alarms, KPIs, inventory/topology, and trouble tickets as root-cause labels.
2. Streaming ingest (e.g. Kafka) instead of polling a table.
3. Postgres (or a time-series DB for KPIs), several API workers, a model server.
4. Sparse GNN and per-region subgraphs for 100k nodes.
5. Auth (OAuth/SSO), role-based approvals, secrets management.
6. Shadow mode first: the model proposes, humans decide, compare for weeks.
7. Scheduled drift checks that trigger retraining via `monitoring/retrain.py`, with human sign-off before promotion.

---

## 10. JD line → evidence

| JD line | Evidence in repo |
|---|---|
| Python | Whole backend, typed, ruff-clean |
| ML models as per design | 3 models with baselines: `prahari/models/` |
| PyTorch | Hand-written GCN, training loop, ONNX export: `prahari/models/rca/` |
| NumPy / pandas / scikit-learn | PSI/KS and TF-IDF in NumPy; pipeline in pandas; IsolationForest, HistGB, isotonic in sklearn |
| SQL / DBMS | Schema and indexes `prahari/db.py`; CROSS JOIN fix; read-only SQL guard |
| REST APIs | FastAPI `prahari/serving/app.py`, Pydantic schemas, status codes |
| Git | Commit history by wave; CI on push |
| Deployment | Dockerfile + compose (written, not run); Makefile; model versioning |
| Drift monitoring | `prahari/monitoring/drift.py`, G6 blocks under red drift |
| Testing | pytest (168 pass), quality gates, Playwright, load test |
| Performance | ONNX benchmark `docs/bench.md`; load test `docs/loadtest.md`; query profiling |
| Responsible AI | Governance gate, groundedness, PII redaction, audit log, model cards, human approval |
| Data pipelines / reliability | `prahari/pipeline/quality.py` (60 checks) |
| Debugging | `docs/EXPERIMENTS.md` |
| JavaScript | React dashboard `web/` |
| Statistics / linear algebra | PSI, KS, calibration, A_hat |
| DSA / OOP | BFS/DFS, union-find; `Check` class hierarchy with registry |

---

## 11. Demo script (5 minutes)

Before: copy the DB (`cp prahari.db prahari.db.bak`). Start `make serve`, then `VITE_MOCK=0 make web`. Open http://localhost:5173. Wait 30 s.

| Time | Do | Say |
|---|---|---|
| 0:00 | Live Ops, healthy topology | "120 nodes, 4 tiers. Synthetic, from my simulator." |
| 0:30 | Inject fault → fibre cut on an AGG node | "This simulates a fibre cut. Watch the alarms." |
| 0:45 | Alarm feed fills; noise greyed | "The validator marks noise. Threshold favours recall." |
| 1:15 | Root ringed on the map | "The GNN picks the root. Top-3 was 100% on test." |
| 1:45 | Incident tab: root bars, fault type, evidence | "Probabilities, evidence, KPI deltas." |
| 2:15 | Governance list (10 checks) | "G10: critical severity needs a human. G4 warns when many nodes are downstream." |
| 2:45 | Approve with a note | "Approve re-runs all 10 checks. Then the simulator applies the fix." |
| 3:15 | EN → HI | "Same facts, Hindi template (not an LLM)." |
| 3:40 | Model health → Run drift scenario | "Firmware update adds 25% latency for one vendor. PSI goes red." |
| 4:10 | Inject again → new incident blocked by G6 | "Red drift blocks actions until retraining." |
| 4:30 | Audit tab | "Every step is logged with actor and time." |
| 4:50 | Close | "Weak points: synthetic data, small GNN gain. I can show the audits." |

Notes:

- The detector runs every 10 s by default. A4 measured median 1.0 s detection (max 25.4 s) with the detector at 1 s. For the demo, start the API with `PRAHARI_DETECTOR_INTERVAL_S=2 make serve`.
- If nothing appears in 30 s, inject on an AGG node, not a cell (leaf faults may not be detected).
- There is no reset for the drift scenario, so do it last. To reset: stop the API, `cp prahari.db.bak prahari.db`.

Fallback if the demo breaks:

1. Show `docs/screens/01`-`08` in order (same story).
2. Or run `make web` (mock mode, no backend needed).
3. Or `curl localhost:8000/v1/models` and open `/docs` to show the API.
Say: "Live demos fail; here is the same flow in screenshots from the system test."

---

## 12. Company sheet

Caution: these summaries were collected during planning (`docs/plan/PLAN.md` section 1 and 9). I could not re-open the pages while writing this guide. **Open every link yourself before the interview and quote only what you read.**

| Topic | What the public source says (per the plan) | Link |
|---|---|---|
| TechM + NVIDIA (Mar 2026) | Reasoning agents for CSPs: autonomous alarm validation, root-cause analysis and resolution, human-in-the-loop, aiming at Level 4+ autonomous networks | [press release](https://www.techmahindra.com/insights/press-releases/techm-advances-ai-driven-autonomous-network-operations-csps/) |
| TechM + Microsoft (Mar 2026) | Ontology-driven agentic AI platform for telecom | [press release](https://www.techmahindra.com/insights/press-releases/techm-collaborates-with-microsoft-to-launch-ontology-driven-agentic-ai-platform/) |
| DTW Ignite 2026 | Showcased "GNN-Healing Networks" (AI-led autonomous network healing) | [event page](https://www.techmahindra.com/insights/events/tech-mahindra-platinum-sponsor-tm-forum-dtw-ignite-2026/) |
| TechM Orion (Jul 2025) | Agentic AI platform, 200+ agents, 30+ configurable governance checks per agent, cloud/on-prem/hybrid | [press release](https://www.techmahindra.com/insights/press-releases/tech-mahindra-unveils-techm-orion/) |
| VerifAI | Validation of AI across discovery, development and production (data quality, model testing, hallucination detection, monitoring) | [service page](https://www.techmahindra.com/services/artificial-intelligence/verifai/) |
| Project Indus | Makers Lab Indic LLM, Hindi + 37 dialects | [press release](https://www.techmahindra.com/insights/press-releases/tech-mahindra-launches-project-indus-large-language-model-llm/) |
| AI Delivered Right | Strategy for responsible, production-grade AI (April 2025, per the plan) | No link in the plan. Find the official page before quoting it, or do not quote it. |
| FY26 AI investment | Increased AI investment in FY26 (news article) | [Entrepreneur India](https://india.entrepreneur.com/technology/tech-mahindra-increases-ai-investment-in-fy26-working-on/498829) |

How PRAHARI maps (say "mirrors", never "matches their system"):

- NVIDIA launch → validation, RCA, governed resolution with a human.
- GNN-Healing → GNN for RCA.
- Orion governance checks → 10 YAML checks.
- VerifAI → data-quality checks, quality gates, groundedness, drift.
- Project Indus → Hindi narratives (templates, not an LLM).

5 questions to ask the interviewer:

1. "In your network-operations work, how do teams get root-cause labels to train on? Trouble tickets, or expert review?"
2. "How does a Digital Catalyst AI Engineer split time between building models and making them production-ready?"
3. "When an agent proposes an action for a customer network, who approves it, and how are governance checks configured per customer?"
4. "How do you monitor models after deployment: drift dashboards, scheduled retraining, or both?"
5. "What would a new AI engineer usually work on in the first three months?"

---

## 13. Question bank

### 13.1 Project-specific

#### Q1. Walk me through what happens when a fibre cut occurs.
The simulator raises alarms on the node and its downstream nodes. The detector reads new alarms every 10 s. The validator scores them; noise drops out. Valid alarms close in time (5 min) and space (2 hops) form a cluster. RCA runs on a 15-minute window. The agent picks a runbook and action. Ten governance checks run. A human approves; checks re-run; the simulator applies the fix. Every step is in `audit_log`.

#### Q2. Why a GNN for root cause?
Faults spread along links. A node's neighbours carry evidence: if all children of AGG-07 alarm, AGG-07 is likely the cause. A GCN mixes neighbour information. But be honest: on my data the GNN beat a no-graph MLP by only 1 point over 10 seeds (p about 0.07). Most of the signal is in node features like KPI deltas.

#### Q3. Then why keep the GNN?
It is never worse on average, it generalises to the 80-node topology (100% vs 97.9%), and its gain grows with noisy, dropped alarms (+2.2 pt in Q2). On real data, where node features are weaker, the graph should matter more. That is a hypothesis; I would test it on real data.

#### Q4. How do you know the model does not cheat with labels?
Labels live only in `incidents` and `label_*` columns. Loaders use column allow-lists. The SQL guard rejects them. Q1 shuffled labels and features did not change. Q3 renamed the ground-truth table and columns and every serving endpoint still worked.

#### Q5. What is the input to the GNN?
For each node in a 15-minute window: 15 alarm-type counts, total, critical count, first-alarm minute, 6 KPI deltas, 6 KPI z-scores, 4 tier one-hots. 34 features. Plus A_hat for the topology.

#### Q6. How was the RCA model trained?
408 train incidents, chronological. Adam, lr 2e-3, up to 80 epochs, early stopping on validation top-1 (patience 12). Loss = root cross-entropy + 0.5 × fault cross-entropy. It stopped after 25 epochs. Train time 92 s on CPU.

#### Q7. What does top-3 = 100% mean for an operator?
The true root is always in the 3 nodes shown. The operator checks at most 3 nodes. That is useful even when top-1 is wrong.

#### Q8. Where does the model fail?
Overlapping incidents (88.2% top-1 vs 98.6% without) and missing root alarms (81.8%). It does not infer a silent upstream node.

#### Q9. Why does the validator threshold sit at 0.18, not 0.5?
I chose the largest threshold that keeps validation recall ≥ 0.98. A dropped real alarm can hide an outage. Cost: precision 0.962 instead of 0.980.

#### Q10. Your validator has 0.998 PR-AUC. Is that suspicious?
Yes. Q2 found a shortcut: in the simulator, test noise alarms never have a neighbour alarm in the previous 5 minutes. Without the count features it still gets 0.9945. I report noise-class AP 0.9675. On real data I would expect lower.

#### Q11. Why does a 3-sigma rule beat your IsolationForest on F1?
The faults in my simulator are big KPI shifts, so a single max-z feature with a tuned threshold works well (F1 0.809). The forest ranks better overall (PR-AUC 0.8135 vs 0.745) but its F1 is 0.779. I would keep the simple rule as baseline. The forest is not on the incident path.

#### Q12. What was the anomaly model's first problem?
The baseline was the rolling hour before t. A lasting fault entered its own baseline, so the z-score shrank: recall 0.02 at 30-60 min. I lagged the baseline to end 75 minutes before t.

#### Q13. Why 10 governance checks? Why these?
They cover the ways an automated action goes wrong: low confidence (G1, G2), made-up text (G3), dangerous action (G4), leaked data (G5), model out of date (G6), unregistered model (G7), malformed output (G8), repeated action (G9), high severity (G10).

#### Q14. What did the Q3 governance bypass teach you?
Checks ran once at proposal time. Drift turned red later and approve still executed. Conditions change between proposal and approval, so approval must re-check. Now approve re-runs all checks and returns 409 on any fail.

#### Q15. How does the groundedness check work?
Regex finds node ids and numbers in the narrative. Each id must be in the RCA evidence or candidates. Each number must be derived from RCA output (probabilities as %, counts, KPI deltas). Anything else → fail → blocked.

#### Q16. What does the agent do without an LLM?
With provider `none` it runs a fixed plan: get RCA, search runbooks with "fault + tier" words, propose the runbook's action, fill a template in English or Hindi. Deterministic and offline.

#### Q17. Why TF-IDF and not embeddings?
15 runbooks. TF-IDF is enough, needs no model download or vector DB, and is explainable. Honest: my retrieval test (15/15) is easy because queries use runbook words.

#### Q18. How does drift block actions?
The latest `drift_reports` status feeds G6. Red = fail = blocked. The dashboard shows it. Retraining and a new reference are needed to clear it.

#### Q19. How do you retrain safely?
`python -m prahari.monitoring.retrain <model>` trains into staging, compares card metrics, and promotes only if no metric is worse by more than 0.002. Then rebuild the drift reference. Honest: run once, dry run, on the validator only.

#### Q20. Why did ONNX not matter much?
The forward pass is 0.1-0.3 ms. End-to-end predict is about 42 ms, mostly the DB query. I profiled and fixed the query (36x) — that was the real win.

#### Q21. What does the load test show?
No errors up to 50 clients after the fixes. `/v1/alarms/validate` p50 rises from 56 ms (1 client) to 4.3 s (32). It is CPU-bound on one worker. Fix: more workers with Postgres, batch requests, cache z-scores.

#### Q22. Why were 2 of 10 live faults missed?
An incident needs ≥ 2 valid alarms. Leaf faults (a cell) can raise only 1. Lowering to 1 would open many incidents from single noise alarms that pass the validator. A better fix: open single-alarm incidents only when the validator is very confident, or when the anomaly detector agrees.

#### Q23. What does the data look like?
90 days, 120 nodes, 15.2M KPI rows, 6393 alarms, 584 incidents, 25% noise alarms, root alarm missing 13.5%, about 10% of incidents start during another.

#### Q24. What is in a model card?
Name, version, train time, data hash, features, test metrics, baseline metrics, intended use, limitations. G7 blocks actions from a model with no card.

#### Q25. How do the tests prove quality?
Quality gates fail CI if card metrics fall below set lines. Data tests check leakage and realism. API tests run the real detector on a 1-day DB. A ground-truth test checks serving never reads labels.

### 13.2 ML fundamentals

#### Q26. Precision vs recall — which matters more here?
For alarm validation, recall: missing a real alarm is worse. For auto-executing an action, precision: a wrong action harms the network. That is why actions need high confidence and a human.

#### Q27. Why PR-AUC and not accuracy for anomaly detection?
Only 0.79% of node-minutes are positive. Predicting "normal" always gives 99.2% accuracy. PR-AUC focuses on the positives.

#### Q28. What is overfitting and how did you control it?
Learning noise in training data so test results drop. Controls: early stopping on validation, dropout 0.2, weight decay, small trees and L2 in boosting, and a held-out chronological test set read once.

#### Q29. What is data leakage? Give an example from this project.
Information at training time that is not available at prediction time. Example: the pipeline's neighbour count looked 5 minutes into the future. A2 rebuilt it with strictly earlier alarms and tested it.

#### Q30. Why a chronological split?
Production predicts the future from the past. A random split lets overlapping incidents sit in both train and test. Q1 checked: all 77 overlapping pairs stay in one split.

#### Q31. Explain calibration and why you need it.
Calibrated probabilities match frequencies. Governance uses probabilities as confidence thresholds, so they must mean what they say. I used isotonic regression on validation; Brier 0.021.

#### Q32. Logistic regression vs gradient boosting?
LogReg is linear in features, fast, easy to explain. Boosting learns interactions and non-linear thresholds. Here boosting gained little (F1 0.976 vs 0.973), a sign the task is easy.

#### Q33. What is softmax and where is it used?
Turns logits into probabilities that sum to 1. Over nodes for the root (one root per window) and over 5 classes for fault type.

#### Q34. Why cross-entropy loss?
It is the negative log-likelihood of the correct class. It strongly punishes confident wrong answers.

#### Q35. What is macro-F1?
F1 per class, then averaged without weights. Rare fault types count the same as common ones.

#### Q36. What does dropout do?
Randomly zeros units during training, so the network cannot rely on one path. Off at inference.

#### Q37. What is a residual connection?
`h = h + f(h)`. The layer learns a change, not a replacement. Gradients flow better, and in GCNs it reduces over-smoothing.

#### Q38. What is over-smoothing in GNNs?
After many layers every node averages over most of the graph and looks the same. That is why 3 layers plus residuals.

#### Q39. How would you test whether the GNN gain is real?
More seeds, more test incidents, a paired test, and harder data (overlaps, dropped alarms). Q2 did 10 seeds: p about 0.07. I would need a bigger test set.

#### Q40. Unsupervised vs supervised anomaly detection?
Unsupervised (IsolationForest) needs no labels and can find new failure kinds. Supervised needs labels but usually scores better on known kinds. I used labels only to pick the threshold and evaluate.

### 13.3 DSA, OOP, DBMS tied to the code

#### Q41. What is the complexity of your hop-distance computation?
Level-synchronous BFS with dense matrix products: O(D × N³), D = diameter. For 120 nodes it is instant. An adjacency-list BFS from every node is O(N(N + E)).

#### Q42. How do you count downstream nodes?
DFS with a stack over strictly deeper tiers (`dependent_count`, `agent/flow.py`). O(V + E). BFS would give the same count.

#### Q43. How are alarms clustered?
Union-find over pairs within 5 min and 2 hops. O(n²) pair checks, near-constant find with path halving.

#### Q44. Where is OOP used well?
Governance: abstract `Check` with `run()`, one subclass per rule, a `@register` decorator, and `evaluate()` in the base class that turns exceptions into fail. Open for extension: adding G11 is one class. LLM providers share one interface too.

#### Q45. Is your schema normalised?
Yes for core data: node attributes once in `nodes`, foreign keys elsewhere. Reports are JSON on purpose (written once, read whole).

#### Q46. Why the (node_id, ts) index?
Every KPI query asks "these nodes, this time range". A composite index on (node_id, ts) seeks to the node, then a time range. Order matters: (ts, node_id) would be worse for one node over time.

#### Q47. Tell me about a slow query you fixed.
The KPI window query took about 1.1 s. `EXPLAIN QUERY PLAN` showed SQLite scanning 15M rows because it chose the wrong join order. `CROSS JOIN` forces it to drive from `nodes` and use the index: 30 ms (36x).

#### Q48. Write SQL: number of alarms per node type in the last hour.
`SELECT n.node_type, COUNT(*) FROM alarms a JOIN nodes n ON n.node_id = a.node_id WHERE a.ts >= :since GROUP BY n.node_type;` The `ix_alarms_ts` index serves the WHERE.

#### Q49. How do you prevent SQL injection in the agent's SQL tool?
Two layers. Parser: one SELECT, allow-listed tables, forbidden words, no labels. Then SQLite authorizer + `query_only` + timeout. Elsewhere the code binds parameters. Honest: one helper (`_node_type`) builds a query with an f-string from a node id; it still passes the guard, but I would parameterise it.

#### Q50. How do you handle two people approving at once?
A lock around the decision plus a status check: only `proposed` (or `blocked` for a reject) can change. Q3 test: 10 parallel decisions → one 200, nine 409, one audit event.

### 13.4 System design

#### Q51. Scale this to 100,000 nodes.
- Ingest: Kafka topics for alarms and KPIs; stream processing for features.
- Storage: time-series DB for KPIs, Postgres for state.
- Validator: stateless, horizontally scaled; features from a streaming store.
- RCA: run on the subgraph around the alarm cluster (k-hop neighbourhood), not the whole graph. Sparse adjacency. Batch on GPU if needed.
- Detector: partition by region.
- API: several workers behind a load balancer; cache; queue long jobs.

#### Q52. How would you serve models in production?
Containerised API with versioned artifacts from a registry, health checks, canary rollout, metrics per version. ONNX Runtime for the GNN.

#### Q53. How do you monitor in production?
Latency and error metrics (Prometheus `/metrics` exists), input drift (PSI/KS), output drift (validator probability, RCA confidence), and outcome feedback (operator approvals vs rejections as labels).

#### Q54. SQLite in production?
No. One writer. I used WAL and busy timeouts to survive the load test, but production needs Postgres. Note: my SQL guard's second layer is SQLite-only, so Postgres needs a read-only DB role instead.

#### Q55. How would you make the API secure?
Auth (OAuth2/JWT), role-based approval rights, rate limits, input limits (done: window ≤ 6 h, batch ≤ 1000), no secrets in code, audit log (done), TLS.

### 13.5 Responsible AI

#### Q56. Why a human in the loop?
Network actions can cause outages. The model is right most of the time, not always. A human approves high-severity or uncertain cases (G10, G1, G2). The audit log shows who approved what.

#### Q57. How do you stop hallucinated explanations?
G3: every node id and number in the text must come from the RCA output. Otherwise the action is blocked. The default provider uses templates, so it cannot hallucinate.

#### Q58. What about privacy?
Redaction of emails, phones (incl. Indian mobile format), IPs and keys before text is stored or shown; G5 fails if any remains. Serving never reads labels. Serving never reads ground-truth labels either.

#### Q59. What is a model card for?
Tells users what the model is for, how it was tested, and where it fails. Mine list the honest limits (e.g. the shortcut, the small GNN gain).

#### Q60. Could the model be biased?
Yes, by vendor or region if data differs. The drift report has per-vendor latency PSI; the firmware scenario showed one vendor drifting (PSI 4.99). I would report accuracy per vendor and per tier before deployment.

### 13.6 What would you do differently

#### Q61. What would you change first?
Make the simulator harder before training anything: overlapping incidents, true-alarm flaps, noise near incidents. I would also set the honest metric (noise-class AP) at the start.

#### Q62. What would you add with one more week?
Postgres and two workers, the single-alarm detection rule, a seasonal anomaly baseline, a real LLM behind G3, and actually running Docker.

#### Q63. What did you learn?
Audits find what you do not want to see. Simple baselines must be tuned fairly. Bottlenecks are rarely where you expect (the DB, not the model). State the weak numbers yourself.

### 13.7 Behavioural and AI-tools

#### Q64. Was this built with AI tools?
Yes — see section 2 for the full answer. Short version: "AI-assisted multi-agent workflow. I designed the problem, plan, contracts and audits. I can explain and change every file. Ask me to open any one."

#### Q65. How do you know the AI-written code is correct?
Same way as any code: tests (168), independent audits that reproduced every metric to 1e-4, a real-API end-to-end test, a load test, and reading it. The audits found real bugs, which were fixed.

#### Q66. Tell me about a time you found a problem in your own work.
The Q3 governance bypass: an action was executed after drift turned red. I traced it to `decide()` not re-running checks, fixed it, and added a test.

#### Q67. How do you handle a result that looks too good?
Suspect it. The 0.976 single-feature AUC and the 0.998 validator PR-AUC were both too good. Probes with one-feature trees found why.

---

## 14. Exercises to do by hand

Do at least 3 before the interview. Each takes 20-60 minutes. Commit each on a branch.

1. **Add an 11th governance check.** In `prahari/governance/checks.py`, add `class MaintenanceWindow(Check)` with `id, name = "G11", "maintenance_window"` and `@register`. Fail if the target node is in a list from `policy.yaml` (add a `G11: {nodes: [AGG-01]}` block). Add a test in `tests/test_governance.py`. Run `.venv/bin/pytest tests/test_governance.py`. Note: `evaluate()` sorts by the number after "G", so G11 sorts after G10.
2. **Change the GNN depth.** In `prahari/models/rca/model.py`, change `range(3)` to `range(1)` and then `range(5)`. Run `.venv/bin/python -m prahari.models.rca.train` (about 95 s). Compare `root_top1` in the new card with the MLP. Expect little change; explain why (features carry most signal). Afterwards put the old version id back in `artifacts/rca_gnn/latest` (the predictor loads the version named there), and delete the new folder.
3. **A_hat and PSI by hand.** Compute A_hat for the 3-node line (section 5.1) on paper. Check with `python -c "from prahari.models.rca.model import normalised_adjacency; import numpy as np; print(normalised_adjacency(np.array([[0,1,0],[1,0,1],[0,1,0]])))"`. Then check the PSI example with `psi_from_props` from `prahari.monitoring.stats`.
4. **Move the validator threshold.** In `prahari/models/validator/model.py`, change `TARGET_RECALL` from 0.98 to 0.95. Retrain (`python -m prahari.models.validator.train`, about 64 s). Compare precision and recall in the new card. Explain the trade-off. Revert.
5. **SQL and the guard.** Run `EXPLAIN QUERY PLAN` on a KPI query with plain `JOIN` vs `CROSS JOIN` in `sqlite3 prahari.db`. Then call `validate_sql` from `prahari/agent/sqlguard.py` with `SELECT label_is_true FROM alarms` and with `SELECT * FROM nodes, incidents`. Explain each rejection.

---

## 15. Numbers sheet (memorise)

| Number | Meaning | Source |
|---|---|---|
| 120 / 80 | nodes in main / alt topology | A1, A3 |
| 584; 408/88/88 | incidents; train/val/test | A1 |
| 25% / 13.5% | noise alarms / root alarm missing | A1 |
| 2.3 GB, ~10 min | DB size, `make seed train` time (9 min 41 s) | A1, Q3 |
| 0.976 → 0.846 | Q1 one-feature AUC before / after fix | Q1, A1 |
| 96.6% / 100% / 0.957 | RCA top-1 / top-3 / fault F1 | RCA card |
| 95.5% | MLP top-1 | RCA card |
| +1.0 pt, p ≈ 0.07 | GNN − MLP over 10 seeds | Q2 |
| 0.962 / 0.991 | validator precision / recall at 0.18 | validator card |
| 0.998 → 0.9675 | validator PR-AUC → honest noise-class AP | Q2 |
| 0.8135 / 0.779 vs 0.809 | anomaly PR-AUC / F1 vs tuned 3-sigma F1 | anomaly card |
| 8/10 | live faults detected; 8/8 correct root | A4 |
| 0.244 / 0.135 / 0.122 ms | PyTorch / ONNX / int8 forward p50 | bench.md |
| 42.2 ms | end-to-end RCA predict p50 | bench.md |
| 1078 → 30 ms | KPI query, plain JOIN → CROSS JOIN | bench.md |
| 56 ms → 4.3 s | validate p50, 1 → 32 clients | Q3 |
| 0.44 | latency PSI in drift scenario (0.37 in Q3 run) | A6, Q3 |
| 168 | tests passing | Q3 |
