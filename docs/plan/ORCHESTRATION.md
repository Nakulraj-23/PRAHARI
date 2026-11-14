# ORCHESTRATION — runbook for the session that drives the build

You are the orchestrator. You dispatch agents and check handoffs. You do not write project code yourself
and you do not read source files unless a gate fails.

## Token rules

1. Every agent is spawned fresh (Agent tool, `subagent_type: "general-purpose"`) with an explicit `model`:
   builders `sonnet`, Q-gates and A9 `opus`. An agent never inherits this chat.
2. The spawn prompt is only this (fill in the id):
   > Work in `/home/user/prahari`. Read `docs/plan/CONTRACTS.md` and `docs/plan/briefs/<ID>.md`, plus the
   > handoff notes your brief lists. Do the work, run it, and finish by writing `docs/handoffs/<ID>.md`
   > (30 lines max). Reply with five lines at most: done/not done, measured numbers, blockers.
3. Agents talk to each other through files only: `CONTRACTS.md`, `fixtures/`, `docs/handoffs/`.
4. After each agent, read only its handoff note. Run `make test` yourself as the single check.
5. Agents in the same wave run in parallel (one message, several Agent calls). They own different directories.
6. Commit after each wave: `git add -A && git commit -m "wave N: ..."`.

## Waves

| Wave | Spawn | Model | Then |
|---|---|---|---|
| 0 | A0 | sonnet | `make setup && make test` |
| 1 | A1 | sonnet | Q1 (opus). Fix findings by sending them back to A1 with SendMessage |
| 2 | A2, A3, A5, A7 in parallel | sonnet | Q2 (opus) on A2 + A3 |
| 3 | A4, A6 in parallel | sonnet | Q3 (opus) |
| 4 | A9 | opus | Send `docs/INTERVIEW_GUIDE.md` to the user |

Fixes found by a Q-gate go back to the builder that owns the code (SendMessage to that agent, so its context is reused).
The Q-agent re-checks only the items it raised.

## Escalation to the advisor (Fable)

Only when the same problem has survived two fix attempts, or a contract has to change in a way that affects
more than one agent. The stuck agent writes `docs/escalations/<n>.md`, one page: goal, what was tried,
exact error or number, the question. Spawn one agent with `model: "fable"` whose prompt is: read
`CONTRACTS.md` and that one file, answer in `docs/escalations/<n>.answer.md`. The advisor never reads the code
base and never builds. Record each escalation in `docs/EXPERIMENTS.md`.

## Definition of done

- `make setup seed train serve` works from a clean clone; `docker compose up` works.
- `make test` green; model quality gates pass; `make bench` prints the latency table.
- Demo script in `PLAN.md` §7 runs end to end with `LLM_PROVIDER=none`.
- `docs/INTERVIEW_GUIDE.md` contains only numbers that appear in handoff notes or model cards.
