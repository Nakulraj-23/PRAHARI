# PRAHARI — Master Plan

**PRAHARI** (प्रहरी, "sentinel"): a human-in-the-loop network-operations copilot for telecom operators.
It validates alarms, finds the root cause of an alarm storm with a graph neural network, proposes a fix,
and passes every proposal through a governance gate before a human approves it.

Target role: Tech Mahindra — Digital Catalyst, AI Engineer.

---

## 1. Why this project (the research behind the choice)

| What Tech Mahindra has said publicly | Where PRAHARI mirrors it |
|---|---|
| Telecom is its largest vertical. In March 2026 it launched, with NVIDIA, reasoning agents for CSPs covering **autonomous alarm validation, root-cause analysis and resolution**, human-in-the-loop, aiming at Level 4+ autonomous networks. | The exact three stages: alarm validation → RCA → governed resolution with human approval. |
| At TM Forum DTW Ignite 2026 it showcased **"GNN-Healing Networks"** (AI-led autonomous network healing). | RCA model is a graph neural network over the network topology, written in PyTorch. |
| **TechM Orion** (July 2025): agentic AI platform, 200+ agents, **30+ configurable governance checks per agent**, cloud/on-prem/hybrid. | A tool-using RCA agent, plus 10 YAML-configurable governance checks run before any action. Docker = runs anywhere. |
| **VerifAI**: validation of AI across discovery → development → production (data quality, model testing, hallucination detection, monitoring). | Data-quality tests, model quality gates in CI, groundedness check on agent text, drift monitoring. |
| **"AI Delivered Right"** strategy (April 2025): responsible, production-grade AI. | Audit log, model cards, PII redaction, approval workflow. |
| **Project Indus** (Makers Lab): Indic LLM, Hindi + 37 dialects. | Incident explanations available in English and Hindi. |
| Promise to customers: **"scale at speed"**. | Latency benchmark before/after ONNX export + quantization; load test. |

How to phrase it in the interview: *"I built a small-scale version of a problem you have publicly said you are
solving."* Never claim knowledge of their internal systems. The data is synthetic and you say so up front.

## 2. JD line → where it shows up

| JD requirement | In PRAHARI |
|---|---|
| Implement ML/AI models as per design | 3 models: anomaly detector, alarm validator, RCA GNN |
| Convert models into production-ready components | Versioned artifacts + model cards, loaded by a FastAPI service |
| Integrate models with APIs, apps, data pipelines | REST API ← feature pipeline ← SQL; React dashboard on top |
| Deploy on cloud or on-prem | Docker Compose (same images run on a laptop or a VM) |
| Monitor performance and basic drift indicators | PSI + KS drift monitor written with NumPy; Prometheus metrics |
| Feature pipelines, preprocessing, data reliability | pandas pipeline, schema + data-quality checks |
| Functional testing of AI components | pytest: unit, API, model-quality gates, e2e browser test |
| Optimize performance, scalability, latency | ONNX + quantization benchmark; p50/p95 under load |
| Debug inference and integration issues | `EXPERIMENTS.md` logs every failure and the fix |
| Documentation | README, architecture doc, model cards, API docs |
| Security guidelines, responsible AI, data privacy | Governance gate, read-only SQL tool, PII redaction, audit log |
| Python; JavaScript an advantage | Python backend; React (JavaScript) frontend |
| DSA, OOP, DBMS | Graph traversal for fault propagation; class hierarchy for checks; indexed relational schema |
| Statistics, probability, linear algebra | PSI/KS, calibration, GCN normalised adjacency |
| NumPy, pandas, scikit-learn, PyTorch | All four, each with a clear job |
| SQL, Git, REST | SQLite/PostgreSQL, conventional commits + CI, FastAPI |

## 3. Architecture

```
 Simulator ──► SQL (nodes, links, kpi_samples, alarms, incidents)
 (topology,          │
  KPIs, faults)      ▼
              Feature pipeline (pandas/NumPy)
                     │
     ┌───────────────┼────────────────┐
     ▼               ▼                ▼
 M1 Anomaly     M2 Alarm         M3 RCA GNN
 IsolationForest validator       (PyTorch → ONNX)
 (scikit-learn) (scikit-learn)   root node + fault type
     └───────────────┼────────────────┘
                     ▼
            RCA Agent (tools: read-only SQL, RCA, runbook search)
                     ▼
            Governance gate (10 checks, YAML) ──► audit_log
                     ▼
            Human approves / rejects  ──► action "executed" in simulator
                     
 FastAPI serves all of the above ◄── React dashboard
 Drift monitor (PSI/KS) ──► feeds governance check G6 + dashboard
```

## 4. Stack

| Layer | Choice | Reason to give if asked |
|---|---|---|
| Language | Python 3.11 | JD |
| Data | NumPy, pandas, networkx | networkx for topology and propagation |
| DB | SQLite by default, PostgreSQL via `DATABASE_URL` (SQLAlchemy 2) | Zero-setup demo, real DB in Compose |
| Classical ML | scikit-learn (IsolationForest, LogisticRegression baseline, HistGradientBoosting) | Baseline first, then a stronger model |
| Deep learning | PyTorch; GCN layers written by hand (no torch-geometric) | Small graphs; shows the maths is understood; fewer install problems |
| Optimisation | ONNX Runtime + dynamic quantisation | Measurable latency story |
| Serving | FastAPI, Pydantic v2, Uvicorn | Typed REST, auto OpenAPI docs |
| Agent | Small tool loop with an LLM adapter: `none` (deterministic templates), `anthropic`, `openai_compatible` | Demo works offline; `openai_compatible` covers NVIDIA NIM / Azure / Ollama |
| Retrieval | TF-IDF (scikit-learn) over markdown runbooks | Enough for ~20 runbooks; no vector DB to maintain |
| Monitoring | Hand-written PSI/KS, `prometheus-client` | Shows statistics, not a black-box library |
| Frontend | React + Vite (JavaScript), Tailwind, Cytoscape.js, Recharts | JD lists JavaScript |
| Tests | pytest, httpx, Playwright, a small asyncio load script | |
| Ops | Docker Compose, Makefile, GitHub Actions | |

## 5. Scope tiers

- **MVP (must work in the demo):** simulator, pipeline, M2, M3, RCA endpoint, governance gate, approve/reject, dashboard live view, tests, docs.
- **Full:** M1, drift monitor + drift scenario, ONNX benchmark, runbook retrieval, Hindi explanations, load test.
- **Stretch (only if time):** LLM-written narratives, generalisation to an unseen topology, deploy to a free host.

## 6. Build waves

| Wave | Agents (model) | Output | Gate (Opus) |
|---|---|---|---|
| 0 | A0 Scaffold (Sonnet) | Repo skeleton, DB models, config, Makefile, CI | — |
| 1 | A1 Data (Sonnet) | Simulator, pipeline, seeded DB | Q1: data realism + label-leakage audit |
| 2 (parallel) | A2 Classical ML, A3 GNN, A5 Agent + governance, A7 Frontend (Sonnet) | Models, agent, UI against fixtures | Q2: independent model evaluation |
| 3 (parallel) | A4 Serving, A6 Monitoring + optimisation (Sonnet) | Integrated API, drift, ONNX | Q3: functional, e2e, load, security review |
| 4 | A9 Docs + Interview Guide (Opus) | README, architecture, `INTERVIEW_GUIDE.md` | — |

Details: `ORCHESTRATION.md`, `CONTRACTS.md`, `briefs/`.

## 7. Demo script (5 minutes)

1. Dashboard shows a healthy network. Click **Inject fault → fibre cut on an aggregation node**.
2. Alarm storm appears (dozens of alarms). Validator greys out the noise alarms.
3. RCA card: root node highlighted on the topology, fault type, top-3 candidates with probabilities, evidence.
4. Proposed action with the governance report (10 checks). High severity → needs human approval. Approve; network recovers.
5. Switch explanation to Hindi.
6. Trigger the **drift scenario** (firmware update shifts latency). Drift panel turns red; governance now blocks auto-approval. Explain retraining.
7. Show CI: tests, model quality gates, latency benchmark table.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Synthetic data is too easy → suspicious 99% scores | Noise alarms, missing alarms, overlapping incidents, held-out topology; Opus gate Q1 checks for leakage |
| Interviewer: "is this real data?" | Say synthetic first, explain how the simulator was designed and what real data would change |
| Interviewer: "did AI write this?" | Answer honestly: AI-assisted build, your design choices; be able to explain every file. Read the code, re-run training, change one thing yourself |
| Live demo fails | Recorded video + screenshots in the repo |
| No LLM key | `LLM_PROVIDER=none` is the default and fully functional |

## 9. Sources

- TechM + NVIDIA autonomous network operations (Mar 2026): https://www.techmahindra.com/insights/press-releases/techm-advances-ai-driven-autonomous-network-operations-csps/
- TechM + Microsoft ontology-driven agentic AI for telecom (Mar 2026): https://www.techmahindra.com/insights/press-releases/techm-collaborates-with-microsoft-to-launch-ontology-driven-agentic-ai-platform/
- DTW Ignite 2026, GNN-Healing Networks: https://www.techmahindra.com/insights/events/tech-mahindra-platinum-sponsor-tm-forum-dtw-ignite-2026/
- TechM Orion: https://www.techmahindra.com/insights/press-releases/tech-mahindra-unveils-techm-orion/
- VerifAI: https://www.techmahindra.com/services/artificial-intelligence/verifai/
- Project Indus: https://www.techmahindra.com/insights/press-releases/tech-mahindra-launches-project-indus-large-language-model-llm/
- FY26 AI investment: https://india.entrepreneur.com/technology/tech-mahindra-increases-ai-investment-in-fy26-working-on/498829
