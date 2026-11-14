"""Q2 audit: rebuild RCA incident windows (same seeded jitter as A3's trainer) plus perturbed
variants for robustness. Ground truth is read here only to build targets and to pick noise-alarm
types for the injected-noise variant (evaluation only, never a model input).

Variants per incident window:
  clean   - exactly what ``prahari.models.rca.train.build_graphs_chunked`` produces
  drop30  - each alarm in the window removed with p=0.3
  noise50 - extra noise alarms = ceil(0.5 * alarms in window), random node, (type, severity)
            drawn from real noise alarms, time uniform in the window (KPIs untouched)
  both    - drop30 then noise50

Run: ``.venv/bin/python -m tests.models.audit.rca_data`` -> artifacts/q2_audit/rca_<db>.npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prahari.config import Settings, get_settings
from prahari.db import get_engine
from prahari.pipeline.features import BASELINE_MINUTES, _window_features, feature_names
from prahari.pipeline.splits import load_splits
from prahari.pipeline.store import load_alarms, load_ground_truth_incidents, load_kpi
from prahari.sim.constants import FAULT_TYPES
from prahari.sim.topology import NetworkModel

VARIANTS = ("clean", "drop30", "noise50", "both")
_ALARM_COLS = [i for i, c in enumerate(feature_names()) if c.startswith("alarm_")]
OUT = Path(__file__).resolve().parents[3] / "artifacts" / "q2_audit"


def _perturb(a: pd.DataFrame, start, rng, pool, net, drop: bool, noise: bool) -> pd.DataFrame:
    if drop:
        a = a[rng.random(len(a)) >= 0.3]
    k = int(np.ceil(0.5 * len(a))) if noise else 0
    if k:
        pick = pool.iloc[rng.integers(0, len(pool), k)]
        extra = pd.DataFrame(
            {
                "alarm_id": [f"FAKE-{i}" for i in range(k)],
                "ts": start + pd.to_timedelta(rng.uniform(0, 15 * 60, k), unit="s"),
                "node_id": np.array(net.node_ids)[rng.integers(0, net.n, k)],
                "alarm_type": pick["alarm_type"].to_numpy(),
                "severity": pick["severity"].to_numpy(),
                "raw_text": "",
            }
        )
        a = pd.concat([a, extra], ignore_index=True)
    return a


def build(settings: Settings, window_minutes=15, jitter_min=10.0, chunk=8, pert_seed=2024):
    """Return dict of arrays: X_<variant> (M,N,F), adj, y, f, ids, root_present_<variant>."""
    engine = get_engine(settings)
    net = NetworkModel.from_db(engine)
    inc = load_ground_truth_incidents(engine).sort_values("start_ts").reset_index(drop=True)
    pool = pd.read_sql("SELECT alarm_type, severity FROM alarms WHERE label_is_true = 0", engine)
    names = feature_names()
    rng = np.random.default_rng([settings.seed, 8008])  # identical to A3's jitter stream
    starts = [
        (r.start_ts - pd.Timedelta(seconds=float(rng.uniform(0, jitter_min * 60)))).floor("s")
        for r in inc.itertuples()
    ]
    prng = np.random.default_rng(pert_seed)
    xs = {v: [] for v in VARIANTS}
    present = {v: [] for v in VARIANTS}
    n_alarms = []
    i, n = 0, len(inc)
    while i < n:
        j = i
        while j < n and starts[j] < starts[i] + pd.Timedelta(days=chunk):
            j += 1
        lo = starts[i] - pd.Timedelta(minutes=BASELINE_MINUTES + 1)
        hi = max(starts[i:j]) + pd.Timedelta(minutes=window_minutes)
        store, alarms = load_kpi(engine, net, lo, hi), load_alarms(engine, lo, hi)
        for k in range(i, j):
            s = starts[k]
            e = s + pd.Timedelta(minutes=window_minutes)
            win = alarms[(alarms["ts"] >= s) & (alarms["ts"] < e)]
            n_alarms.append(len(win))
            root = inc.root_node_id[k]
            for v in VARIANTS:
                a = (
                    _perturb(
                        win, s, prng, pool, net, v in ("drop30", "both"), v in ("noise50", "both")
                    )
                    if v != "clean"
                    else alarms
                )
                if len(a):
                    x = _window_features(net, store, a, s, e)[names].to_numpy(np.float32)
                else:  # A1's builder crashes on a filtered-empty frame (object dtype in bincount)
                    x = _window_features(net, store, alarms, s, e)[names].to_numpy(np.float32)
                    x[:, _ALARM_COLS] = 0.0
                    x[:, names.index("first_alarm_min")] = window_minutes
                xs[v].append(x)
                aw = a[(a["ts"] >= s) & (a["ts"] < e)]
                present[v].append(bool((aw["node_id"] == root).any()))
        i = j
    out = {f"X_{v}": np.stack(xs[v]) for v in VARIANTS}
    out.update({f"root_present_{v}": np.array(present[v]) for v in VARIANTS})
    out["adj"] = net.adj.astype(np.float32)
    out["dist"] = net.dist
    out["y"] = np.array([net.index[r] for r in inc.root_node_id])
    out["f"] = np.array([FAULT_TYPES.index(t) for t in inc.fault_type])
    out["ids"] = inc.incident_id.to_numpy().astype(str)
    out["node_ids"] = np.array(net.node_ids)
    out["n_window_alarms"] = np.array(n_alarms)
    out["starts"] = np.array([str(s) for s in starts])
    return out


def main() -> None:
    s = get_settings()
    alt_db = Path(str(s.db_url).replace("sqlite:///", "")).with_name("prahari_alt.db")
    alt = s.model_copy(update={"db_url": f"sqlite:///{alt_db}"})
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, st in (("main", s), ("alt", alt)):
        t0 = time.time()
        d = build(st)
        if tag == "main":
            sp = load_splits(s)
            d["split"] = np.array(
                [
                    "train"
                    if i in set(sp["train"])
                    else "val"
                    if i in set(sp["val"])
                    else "test"
                    if i in set(sp["test"])
                    else "none"
                    for i in d["ids"]
                ]
            )
        np.savez_compressed(OUT / f"rca_{tag}.npz", **d)
        print(tag, len(d["y"]), f"{time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
