# Benchmarks (`make bench`)

Machine: x86_64, 4 cores, torch set to 1 thread, ONNX Runtime 1 intra-op thread, Python 3.11.18, torch 2.15.0+cu132, onnxruntime 1.32.0.

## RCA runtimes (model `rca_gnn` 20261114093826, not retrained)

One graph of 120 nodes, forward pass only, 500 timed runs after 30 warm-up. Accuracy on the 88 test incidents of the shipped split.

| runtime | p50 ms | p95 ms | size KB | root top-1 | root top-3 | fault macro-F1 | argmax agrees with PyTorch |
|---|---|---|---|---|---|---|---|
| PyTorch fp32 | 0.244 | 0.348 | 113 | 0.966 | 1.000 | 0.957 | 1.000 |
| ONNX Runtime fp32 | 0.135 | 0.174 | 112 | 0.966 | 1.000 | 0.957 | 1.000 |
| ONNX Runtime dynamic int8 | 0.122 | 0.154 | 43 | 0.966 | 1.000 | 0.957 | 1.000 |

Accuracy change of int8 vs PyTorch: top-1 +0.000, fault F1 +0.000.
End-to-end `RCAPredictor.predict` for one window (DB query + features + forward): p50 42.2 ms, p95 142.9 ms, so the forward pass is the minor cost.

## Feature pipeline profile (serving path, `build_node_window_features`)

30 distinct 15-minute windows from the last 5 days of the DB, profiled with cProfile (top rows by own time):

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       60    1.994    0.033    1.994    0.033 {method 'fetchall' of 'sqlite3.Cursor' objects}
       60    1.513    0.025    3.508    0.058 /home/user/prahari/.venv/lib/python3.11/site-packages/sqlalchemy/engine/result.py:920(fetchall)
       60    0.216    0.004    0.468    0.008 /home/user/prahari/.venv/lib/python3.11/site-packages/pandas/io/sql.py:186(_convert_arrays_to_dataframe)
      378    0.182    0.000    0.207    0.001 /home/user/prahari/.venv/lib/python3.11/site-packages/pandas/core/internals/construction.py:1032(convert)
       53    0.169    0.003    0.200    0.004 /home/user/prahari/.venv/lib/python3.11/site-packages/pandas/core/arrays/_mixins.py:210(_cast_pointwise_result)
       53    0.067    0.001    0.078    0.001 /home/user/prahari/.venv/lib/python3.11/site-packages/pandas/core/tools/datetimes.py:507(_array_strptime_with_fallback)
```

Whole call: p50 122.1 ms, p95 314.5 ms.
Top hotspot: the KPI query in `load_kpi` (SQLite `fetchall` plus SQLAlchemy row objects).

Speed-up (A1 has already applied it in `load_kpi`: `CROSS JOIN` so SQLite drives the join from `nodes` and uses the `(node_id, ts)` index instead of scanning all 15M rows). Before = the same query with a plain `JOIN`, measured through a raw cursor here; after = stock `load_kpi`. A raw cursor with vectorised parsing (`_fast_load_kpi`, `prahari/monitoring/bench.py`) is shown as a further variant. Outputs identical to `load_kpi`: True.

| window | plain JOIN p50 / p95 ms | stock `load_kpi` p50 / p95 ms | raw cursor + CROSS JOIN p50 / p95 ms | speed-up plain to stock |
|---|---|---|---|---|
| 75 min (serving window incl. baseline) | 1078 / 1099 | 30 / 146 | 20 / 21 | 35.7x |
| 24 h (drift window) | 1355 / 1421 | 751 / 845 | 470 / 478 | 1.8x |
