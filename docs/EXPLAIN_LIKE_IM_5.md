# PRAHARI explained like you are 5 (and then like you are 25)

The whole project in plain language: what it does, how it was built and tested, and where it is weak.

---

## 1. The story (30 seconds)

Imagine a big city with 120 playgrounds connected by roads. Phones are like kids that need roads to reach the playgrounds.

One day a road is cut. Now **lots of playgrounds start shouting "I can't reach anyone!"** at the same time. A person watching the city gets 20 shouts in one minute. Some shouts are real. Some are just kids who shout all the time (noise). Which road is actually broken?

**PRAHARI is a helper robot that does 4 things:**

1. **Ignores the fake shouts.** (Alarm validator)
2. **Works out which road is really broken**, by looking at who is shouting and how the roads connect. (Graph neural network)
3. **Suggests a fix and says why**, in English or Hindi. (Agent)
4. **Asks a grown-up before doing anything risky.** A strict checklist of 10 rules runs first, then a human presses Approve. (Governance)

Plus a **smoke alarm for the robot itself**: if the world changes (new equipment behaves differently), it says "I might be getting wrong, retrain me." (Drift monitor)

Why this project? Tech Mahindra has said publicly that it builds exactly this kind of thing for phone companies (validate alarms, find the root cause, human-in-the-loop). PRAHARI is a **small practice version** of that idea. The data is **made up by a simulator**. Say that first, every time.

---

## 2. The pieces, one by one

| Piece | Like I'm 5 | Real name | Where in the repo |
|---|---|---|---|
| Fake city | A toy city that breaks roads on purpose, so we know the right answer | Simulator | `prahari/sim/` |
| Cleaner | Turns messy shouts into neat tables of numbers | Feature pipeline (pandas/NumPy) | `prahari/pipeline/` |
| Spotter | "This node looks odd compared to normal" | Anomaly detector (IsolationForest) | `prahari/models/anomaly/` |
| Fake-shout filter | "Real shout or crybaby?" | Alarm validator (HistGradientBoosting, baseline LogisticRegression) | `prahari/models/validator/` |
| Detective | Finds the broken road by passing notes between neighbours | Root-cause GNN (PyTorch) | `prahari/models/rca/` |
| Speaker | Explains the answer and picks a fix from a short menu | Agent + runbooks | `prahari/agent/`, `runbooks/` |
| Strict teacher | 10 rules every suggestion must pass | Governance gate | `prahari/governance/` |
| Smoke alarm | "The world changed!" | Drift monitor (PSI and KS tests) | `prahari/monitoring/` |
| Front desk | One door where the dashboard and tools ask questions | REST API (FastAPI) | `prahari/serving/` |
| Screen | The picture you see | React dashboard | `web/` |
| Notebook | Remembers everything | SQL database | `prahari/db.py` |

### The detective, in one picture
Each node (playground) has a few numbers: how many alarms, how bad its latency is, and so on. A **GCN layer** lets each node average the numbers of its neighbours with its own: `new = ReLU( normalised_neighbours × numbers × weights )`. Do that 3 times and each node "knows" about things 3 hops away. A node whose neighbours are all shouting, but which is itself the *source* of the trouble, gets the highest score. Softmax turns the scores into probabilities that add to 1.

**Honest line:** the same model *without* the graph (a plain MLP) scores almost as well (95.3% vs 96.4%, 10 seeds). The graph helps a little, not a lot. Say this before they ask.

---

## 3. How it was built (the honest version)

I (Claude) acted as the **manager**. I split the work into small jobs and gave each to a **separate worker agent** with its own short instruction sheet and a shared rulebook (`docs/plan/CONTRACTS.md`). Workers did not read the whole chat; they talked through small hand-off notes in `docs/handoffs/`.

- **Planning:** Fable (the planner model). Read the JD, researched Tech Mahindra's public announcements, chose the project, wrote the plan.
- **Building:** Sonnet agents. Scaffold, simulator, models, agent, API, monitoring, dashboard.
- **Checking:** Opus agents. They did **not** build; they **audited** (see section 5).

**Tools used:** Python 3.11, NumPy, pandas, scikit-learn, PyTorch, ONNX Runtime, networkx, SQLAlchemy + SQLite, FastAPI + Pydantic, React + Vite + Cytoscape.js + Recharts, pytest, Playwright, ruff, Git/GitHub, Docker files (written, **not run**).

**AI-assisted build:** the author chose the problem and owns the design; AI agents wrote and audited code under that direction.

---

## 4. How each part was tested

| Part | How it was tested | Result |
|---|---|---|
| Simulator + pipeline | Unit tests (determinism with a seed, faults spread along the real graph, no `label_*` column ever used as a feature); 60 data-quality checks that fail loudly on bad data | Pass |
| Validator | Time-based split (train on the past, test on the future); baseline vs better model; calibration + threshold chosen on validation only | F1 0.976, but see "weak spots" |
| Anomaly detector | Same split; compared against a simple 3-sigma rule | PR-AUC 0.81, F1 0.78 |
| GNN | 3 baselines; 10 seeds; a never-seen second city layout; dropping 30% of alarms + adding 50% more noise | 96.6% top-1, 100% top-3; held-out city 100% |
| ONNX / speed | Compared PyTorch vs ONNX vs int8 on the same 88 incidents | Same accuracy; model 113 KB → 43 KB; forward pass about 0.27 ms → 0.12 ms |
| Agent + governance | One passing and one failing test per check; SQL tool must refuse DROP/UPDATE/multi-statement/ground-truth reads (19 attack strings all rejected) | Pass |
| Drift monitor | Normal windows must stay green; a simulated firmware change must turn red | Green on normal, red on drift (latency PSI 0.44) |
| API | Every endpoint, good and bad input; inject fault → incident → approve → recovered; race of 5 approves + 5 rejects on one action | One wins, nine get 409 |
| Dashboard | Browser tests (Playwright) in mock mode and against the real API | Pass |
| Whole system | Fresh clone from GitHub, `make setup`, `make seed train`, run all tests; 50 concurrent users | 0 errors |
| All together | `make test` | **175 passed**, lint clean |

---

## 5. The checkers found problems and we fixed them

This is real evidence of engineering discipline. Tell it as a story.

1. **Data audit.** "My first dataset was too easy: one KPI number alone separated real alarms from noise (AUC 0.976). A reviewer caught it, I made the simulator less obvious (now 0.846) and rebuilt." 
2. **Model audit.** "The anomaly detector scored PR-AUC 0.35. The audit found its 'normal' baseline slowly learned the fault as normal. I lagged the baseline by 75 minutes and fixed labels: now 0.81." Also: "GNN beats the no-graph model by only about 1 point."
3. **System audit.** "A fault with low fault-type confidence (0.36) got through and the wrong fix ran. Fixed: governance now checks fault-type confidence. Approve also re-runs governance (drift may have turned red since). Database locked under 32 users: fixed with WAL mode and caching; now 0 errors."

---

## 6. Weak spots

- **All data is synthetic.** Real networks are messier. Real data would need privacy review, labelling by engineers, and probably retraining.
- **Validator 0.998 PR-AUC is inflated** by a simulator shortcut (noise alarms never have a neighbour alarm). Quote **0.9675** (noise-class average precision).
- **GNN gain is small** (+1.0 point, p about 0.07).
- **A simple tuned 3-sigma rule beats the forest on F1** (0.809 vs 0.779).
- **2 of 10 live injected faults were missed** (leaf nodes with fewer than 2 valid alarms).
- **Slow under heavy load.** `/v1/alarms/validate` slows down with many users on one worker.
- **Docker and Postgres were never run.** SQLite is the tested path.
- **Scale:** 120 nodes. For 100,000 nodes you would shard by region, use sparse adjacency and neighbour sampling, and stream alarms.

---

## 7. How to run it (what I did in this session)

```bash
make setup          # once, about 2 min
make seed train     # once, about 10 min; builds a 2.3 GB git-ignored database + models
make serve          # API on :8000
cd web && VITE_MOCK=0 VITE_API_URL=http://localhost:8000 npx vite --port 5173
```
Note: plain `make web` starts in **mock mode** (reads fixture files). Use the `VITE_MOCK=0` line above for the real API.

**Live run result in this session** (fibre cut injected on AGG-01):
- Incident found after about 12 seconds.
- Root cause: AGG-01 (probability 0.9999). Fault type: fibre_cut (0.92).
- 17 valid alarms, 3 noise alarms ignored.
- Proposed action: reroute_traffic. Governance: **needs_approval** (G4 warned: affects 9 downstream nodes; G10: critical severity needs a human). 8 checks passed.
- English and Hindi explanations both produced.
- Approved → status `executed`.
- Then the firmware-drift scenario turned the drift status **red**.

---

## 8. Ten common questions (short answers)

1. **What does it do?** Filters noise alarms, finds the root cause with a graph model, proposes a fix, asks a human, with 10 safety checks in between.
2. **Why a graph model?** Faults spread along connections. The graph tells each node who its neighbours are.
3. **Is the data real?** No. Simulated, on purpose, so I know the true cause. I would validate on real data next.
4. **Why is it not 100% accurate?** Missing alarms, overlapping incidents, noise. Errors: 2 of 3 test misses picked the root of a *simultaneous* incident.
5. **What is drift?** The live data stops looking like the training data. I measure it with PSI and the KS test, written in NumPy.
6. **Why governance?** AI suggestions can be wrong. Checks catch low confidence, high blast radius, duplicates, ungrounded text. A human decides risky ones.
7. **How did you test it?** Unit tests, model quality gates in CI, API tests, browser tests, load test, 3 independent audits.
8. **What went wrong?** Data too easy, anomaly baseline absorbing the fault, a governance gap, DB locking. All found by audits and fixed.
9. **What would you do differently?** Real data, more seeds, per-region models, a proper queue for alarms, Postgres in production.
10. **Was it built with AI?** Yes, an AI-assisted multi-agent workflow. I designed the problem, I can explain every file and change it, and I reviewed the audits.

