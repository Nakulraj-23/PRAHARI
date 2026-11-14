# Load test

Target http://127.0.0.1:8123, closed-loop clients, 15 s per level, client and server on one x86_64 machine. Latency in ms, successful requests only; errors counted separately.

| endpoint | clients | requests | errors | req/s | p50 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| /v1/rca | 1 | 11 | 0 | 0.7 | 891.8 | 4326.5 | 4329.8 |
| /v1/rca | 8 | 53 | 0 | 3.4 | 1935.0 | 6255.9 | 7337.4 |
| /v1/rca | 32 | 0 | 32 | 0.0 | nan | nan | nan |
| /v1/alarms/validate | 1 | 230 | 0 | 15.3 | 56.2 | 95.2 | 285.2 |
| /v1/alarms/validate | 8 | 117 | 0 | 7.7 | 957.5 | 1838.2 | 2050.5 |
| /v1/alarms/validate | 32 | 125 | 0 | 7.3 | 3947.5 | 6894.0 | 8431.0 |
