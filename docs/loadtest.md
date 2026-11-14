# Load test

Current numbers, after the A4 concurrency fixes (SQLite WAL, busy timeout, 503 on lock, 30 s RCA cache).
They replace the first run by A6, where `/v1/rca` at 32 clients failed with HTTP 500 `database is locked`. That failure is fixed.

Setup: script `prahari/monitoring/loadtest.py` (closed-loop clients), one uvicorn worker, 4 shared CPU cores, scratch copy of the 90-day DB, detector running.
Client and server on the same machine, which was shared with other jobs. Treat the numbers as rough.
Note: re-running the script with `--base-url` overwrites this file with a raw table.

## Run by A4 (12 s per level), source `docs/handoffs/A4.md`

| endpoint | clients | errors | req/s | p50 ms | p95 ms |
|---|---|---|---|---|---|
| /v1/rca | 1 | 0 | n/a | 95.7 | 243 |
| /v1/rca | 8 | 0 | n/a | 3.7 | 1581 |
| /v1/rca | 32 | 0 | 116 | 8.6 | 2354 |
| /v1/alarms/validate | 1 | 0 | 17.6 | 54.9 | 74.5 |
| /v1/alarms/validate | 8 | 0 | n/a | 1060 | 1987 |
| /v1/alarms/validate | 32 | 0 | 6.5 | 4513 | 6986 |

`/v1/rca` numbers at 8 and 32 clients are mostly cache hits: the clients reuse windows. An uncached RCA costs about 100-250 ms.

## Run by Q3 (12 s per level, 1/8/32/50 clients), source `docs/handoffs/Q3.md`

- 0 errors at every level.
- `/v1/alarms/validate` p50/p95 ms: 56/75 (1 client), 1067/2099 (8), 4300/7174 (32), 6174/12274 (50). Throughput falls from 17.3 to 6.4 req/s.
- `/v1/rca`: 28.5 req/s at 1 client (cache hits), 178 req/s at 32, then 10.3 req/s with p50 1673 ms at 50.
- Mixed load, 50 clients on 10 endpoints plus fault injection: 248/248 HTTP 200, p95 30.3 s.
- Race test, 5 approves and 5 rejects in parallel on one action: one 200, nine 409, one audit event.

## How to read it

- No errors, but latency grows with clients. `/v1/alarms/validate` is not cached and is CPU-bound, so throughput drops as clients rise. One worker cannot serve 32 clients well.
- Fixes not done: more workers (needs Postgres or a read replica, because SQLite has one writer), batching validate calls, caching KPI z-scores.
