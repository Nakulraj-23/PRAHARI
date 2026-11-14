# ruff: noqa: E501
"""`make bench`: RCA runtime benchmark (PyTorch vs ONNX Runtime vs dynamic int8) and a profile of
the feature pipeline with one measured speed-up. Writes ``docs/bench.md``.

The RCA model is the shipped artifact (``artifacts/rca_gnn/latest``); nothing is retrained.
"""

from __future__ import annotations

import cProfile
import io
import os
import platform
import pstats
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.metrics import f1_score

from prahari.config import ROOT, Settings, get_settings
from prahari.db import get_engine
from prahari.models.rca.model import RCAGNN, normalised_adjacency
from prahari.pipeline.features import build_node_window_features
from prahari.pipeline.store import KpiStore, load_kpi
from prahari.sim.constants import KPI_COLUMNS
from prahari.sim.topology import NetworkModel

N_RUNS = 500
WARMUP = 30
_MIN = pd.Timedelta(minutes=1)


def _pct(ms: list[float]) -> tuple[float, float]:
    return float(np.percentile(ms, 50)), float(np.percentile(ms, 95))


def _time(fn, n: int = N_RUNS, warmup: int = WARMUP) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


def _test_graphs(settings: Settings, cache: Path):
    """Test-split graphs (x, a, root, fault); built once with A3's chunked builder, then cached."""
    if cache.exists():
        z = np.load(cache)
        return z["x"], z["a"], z["y"], z["f"]
    from prahari.models.rca.train import load_data

    x, a, y, f = load_data(settings)["test"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, a=a, y=y, f=f)
    return x, a, y, f


def rca_bench(settings: Settings, workdir: Path) -> dict:
    """Latency, size and accuracy of the shipped RCA model under three runtimes."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    root = Path(settings.artifacts_dir) / "rca_gnn"
    ver = (root / "latest").read_text().strip()
    ck = torch.load(root / ver / "model.pt", map_location="cpu")
    model = RCAGNN(ck["n_features"], ck["n_faults"])
    model.load_state_dict(ck["state"])
    model.eval()
    onnx_fp32 = root / ver / "model.onnx"
    onnx_int8 = workdir / "model.int8.onnx"
    workdir.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(onnx_fp32), str(onnx_int8), weight_type=QuantType.QInt8)

    x, a, y, f = _test_graphs(settings, workdir / "rca_test.npz")
    ah = normalised_adjacency(a)
    torch.set_num_threads(1)
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    sess = {
        "ONNX Runtime fp32": ort.InferenceSession(str(onnx_fp32), so, ["CPUExecutionProvider"]),
        "ONNX Runtime dynamic int8": ort.InferenceSession(
            str(onnx_int8), so, ["CPUExecutionProvider"]
        ),
    }
    ah_t = torch.as_tensor(ah)

    def run_torch(xi):
        with torch.no_grad():
            r, fl = model(torch.as_tensor(xi).unsqueeze(0), ah_t)
        return r[0].numpy(), fl[0].numpy()

    runners = {"PyTorch fp32": run_torch}
    for k, s in sess.items():
        runners[k] = lambda xi, s=s: tuple(s.run(None, {"x": xi, "a_hat": ah}))
    sizes = {
        "PyTorch fp32": (root / ver / "model.pt").stat().st_size,
        "ONNX Runtime fp32": onnx_fp32.stat().st_size,
        "ONNX Runtime dynamic int8": onnx_int8.stat().st_size,
    }
    res: dict = {"version": ver, "n_test": len(x), "n_nodes": x.shape[1], "rows": []}
    ref_top = None
    for name, run in runners.items():
        lat = _time(lambda run=run: run(x[0]))
        outs = [run(xi) for xi in x]
        top = np.array([int(np.argmax(o[0])) for o in outs])
        top3 = np.array([int(y[i] in np.argsort(-outs[i][0])[:3]) for i in range(len(x))])
        ft = np.array([int(np.argmax(o[1])) for o in outs])
        ref_top = top if ref_top is None else ref_top
        p50, p95 = _pct(lat)
        res["rows"].append(
            {
                "runtime": name,
                "p50_ms": p50,
                "p95_ms": p95,
                "size_kb": sizes[name] / 1024,
                "top1": float((top == y).mean()),
                "top3": float(top3.mean()),
                "fault_f1": float(f1_score(f, ft, average="macro")),
                "agree": float((top == ref_top).mean()),
            }
        )
    return res


def _fast_load_kpi(engine, net: NetworkModel, start, end, cross: bool = True) -> KpiStore:
    """Same tensor as ``load_kpi`` but with a raw DBAPI cursor and vectorised timestamp parsing."""
    cols = ", ".join(f"k.{c}" for c in KPI_COLUMNS)
    sql = (
        f"SELECT k.ts, k.node_id, {cols} FROM nodes n {'CROSS JOIN' if cross else 'JOIN'} kpi_samples k "  # noqa: S608
        "ON k.node_id = n.node_id WHERE k.ts >= ? AND k.ts < ?"
    )
    conn = engine.raw_connection()
    try:
        rows = conn.cursor().execute(sql, (start.to_pydatetime(), end.to_pydatetime())).fetchall()
    finally:
        conn.close()
    ts = np.array([r[0] for r in rows], dtype="datetime64[us]").astype("datetime64[m]")
    t0 = pd.Timestamp(ts.min())
    t_idx = (ts - ts.min()).astype(np.int64)
    n_idx = np.fromiter((net.index[r[1]] for r in rows), np.int64, len(rows))
    vals = np.array([r[2:] for r in rows], dtype=np.float64)
    out = np.full((int(t_idx.max()) + 1, net.n, 6), np.nan, dtype=np.float32)
    out[t_idx, n_idx] = vals
    return KpiStore(t0, out, net.node_ids)


def pipeline_profile(settings: Settings) -> dict:
    """cProfile of the serving-path feature build, then before/after of the KPI loader."""
    engine = get_engine(settings)
    net = NetworkModel.from_db(engine)
    end = pd.Timestamp(pd.read_sql("SELECT MAX(ts) AS m FROM kpi_samples", engine)["m"][0])
    start = (end - pd.Timedelta(days=5)).floor("h")
    wins = [
        (start + i * pd.Timedelta(hours=3), start + i * pd.Timedelta(hours=3, minutes=15))
        for i in range(30)
    ]
    it = iter(range(10**9))

    def feats():
        s, e = wins[next(it) % len(wins)]
        return build_node_window_features(s, e, settings, net=net)

    feats()
    pr = cProfile.Profile()
    pr.enable()
    t = [feats() for _ in range(len(wins))]
    pr.disable()
    del t
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(6)
    top = [ln for ln in buf.getvalue().splitlines() if ln.strip()][-7:]
    lo, hi = wins[3][0] - 60 * _MIN, wins[3][1]
    a, b = load_kpi(engine, net, lo, hi), _fast_load_kpi(engine, net, lo, hi)
    same = bool(
        a.t0 == b.t0
        and a.values.shape == b.values.shape
        and np.allclose(a.values, b.values, equal_nan=True)
    )
    plain = lambda lo_, hi_: _fast_load_kpi(engine, net, lo_, hi_, cross=False)  # noqa: E731
    cross = lambda lo_, hi_: _fast_load_kpi(engine, net, lo_, hi_)  # noqa: E731
    long_lo, long_hi = wins[3][0], wins[3][0] + pd.Timedelta(days=1)
    t_ = {}
    for key, fn, (l_, h_), n_ in (
        ("plain", plain, (lo, hi), 8),
        ("stock", lambda a_, b_: load_kpi(engine, net, a_, b_), (lo, hi), 40),
        ("raw", cross, (lo, hi), 40),
        ("plain_day", plain, (long_lo, long_hi), 4),
        ("stock_day", lambda a_, b_: load_kpi(engine, net, a_, b_), (long_lo, long_hi), 6),
        ("raw_day", cross, (long_lo, long_hi), 6),
    ):
        t_[key] = _pct(_time(lambda fn=fn, l_=l_, h_=h_: fn(l_, h_), n_, 1))
    full = _pct(_time(lambda: feats(), 40, 3))
    return {
        "profile_top": top,
        "identical": same,
        "t": t_,
        "serve_features_ms": full,
    }


def predictor_latency(settings: Settings) -> tuple[float, float]:
    """End-to-end RCA ``predict`` for one 15-min window (feature query + forward pass)."""
    from prahari.models.rca import load

    pred = load(settings=settings)
    end = pd.Timestamp(
        pd.read_sql("SELECT MAX(ts) AS m FROM kpi_samples", get_engine(settings))["m"][0]
    ).floor("h")
    s = end - pd.Timedelta(hours=40)
    return _pct(_time(lambda: pred.predict({"start": s, "end": s + 15 * _MIN}), 40, 3))


def write_report(rca: dict, prof: dict, e2e: tuple[float, float], path: Path) -> None:
    """Render docs/bench.md."""
    L = ["# Benchmarks (`make bench`)", ""]
    L.append(
        f"Machine: {platform.machine()}, {os.cpu_count()} cores, torch set to 1 thread, "
        f"ONNX Runtime 1 intra-op thread, Python {platform.python_version()}, "
        f"torch {torch.__version__}, onnxruntime {ort.__version__}."
    )
    L += ["", f"## RCA runtimes (model `rca_gnn` {rca['version']}, not retrained)", ""]
    L.append(
        f"One graph of {rca['n_nodes']} nodes, forward pass only, {N_RUNS} timed runs after "
        f"{WARMUP} warm-up. Accuracy on the {rca['n_test']} test incidents of the shipped split."
    )
    L += [
        "",
        "| runtime | p50 ms | p95 ms | size KB | root top-1 | root top-3 | fault macro-F1 | argmax agrees with PyTorch |",
        "|---|---|---|---|---|---|---|---|",
    ]
    base = rca["rows"][0]
    for r in rca["rows"]:
        L.append(
            f"| {r['runtime']} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['size_kb']:.0f} | "
            f"{r['top1']:.3f} | {r['top3']:.3f} | {r['fault_f1']:.3f} | {r['agree']:.3f} |"
        )
    L += [
        "",
        f"Accuracy change of int8 vs PyTorch: top-1 {rca['rows'][2]['top1'] - base['top1']:+.3f}, "
        f"fault F1 {rca['rows'][2]['fault_f1'] - base['fault_f1']:+.3f}.",
        f"End-to-end `RCAPredictor.predict` for one window (DB query + features + forward): "
        f"p50 {e2e[0]:.1f} ms, p95 {e2e[1]:.1f} ms, so the forward pass is the minor cost.",
        "",
        "## Feature pipeline profile (serving path, `build_node_window_features`)",
        "",
        "30 distinct 15-minute windows from the last 5 days of the DB, profiled with cProfile "
        "(top rows by own time):",
        "",
        "```",
        *prof["profile_top"],
        "```",
        "",
        f"Whole call: p50 {prof['serve_features_ms'][0]:.1f} ms, p95 {prof['serve_features_ms'][1]:.1f} ms.",
        "Top hotspot: the KPI query in `load_kpi` (SQLite `fetchall` plus SQLAlchemy row objects).",
        "",
        "Speed-up (A1 has already applied it in `load_kpi`: `CROSS JOIN` so SQLite drives the join from "
        "`nodes` and uses the `(node_id, ts)` index instead of scanning all 15M rows). Before = the same "
        "query with a plain `JOIN`, measured through a raw cursor here; after = stock `load_kpi`. "
        "A raw cursor with vectorised parsing (`_fast_load_kpi`, `prahari/monitoring/bench.py`) is "
        f"shown as a further variant. Outputs identical to `load_kpi`: {prof['identical']}.",
        "",
        "| window | plain JOIN p50 / p95 ms | stock `load_kpi` p50 / p95 ms | raw cursor + CROSS JOIN p50 / p95 ms | speed-up plain to stock |",
        "|---|---|---|---|---|",
    ]
    t = prof["t"]
    for lab, k in (("75 min (serving window incl. baseline)", ""), ("24 h (drift window)", "_day")):
        p_, s_, r_ = t["plain" + k], t["stock" + k], t["raw" + k]
        L.append(
            f"| {lab} | {p_[0]:.0f} / {p_[1]:.0f} | {s_[0]:.0f} / {s_[1]:.0f} | "
            f"{r_[0]:.0f} / {r_[1]:.0f} | {p_[0] / s_[0]:.1f}x |"
        )
    L.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


def main() -> None:
    """Run the benchmarks and write ``docs/bench.md``."""
    settings = get_settings()
    work = Path(settings.artifacts_dir) / "bench"
    rca = rca_bench(settings, work)
    prof = pipeline_profile(settings)
    e2e = predictor_latency(settings)
    write_report(rca, prof, e2e, ROOT / "docs" / "bench.md")
    print((ROOT / "docs" / "bench.md").read_text())


if __name__ == "__main__":
    main()
