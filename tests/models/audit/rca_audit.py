"""Q2 audit of the RCA GNN on windows from ``rca_data``: reproduce the card, multi-seed GNN vs
no-graph MLP, per fault type, error hop distance, root alarm missing, alt topology, robustness
(drop 30% alarms / +50% noise alarms), reliability table, worst mistakes, ONNX parity.

Run after rca_data: ``.venv/bin/python -m tests.models.audit.rca_audit [n_seeds]``
-> artifacts/q2_audit/rca.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from prahari.config import get_settings
from prahari.models.rca.model import RCAGNN, normalised_adjacency
from prahari.models.rca.train import _fit, _predict
from prahari.pipeline.features import feature_names
from prahari.sim.constants import FAULT_TYPES, NODE_TYPES
from tests.models.audit.rca_data import OUT, VARIANTS

NAMES = feature_names()
TOT = NAMES.index("alarm_total")


def load_npz(tag):
    return dict(np.load(OUT / f"rca_{tag}.npz", allow_pickle=False))


def scores(model, x, adj):
    r, f = _predict(model, x, torch.as_tensor(normalised_adjacency(adj)))
    return r, f


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def summarize(r, f, y, ft, dist, present) -> dict:
    top = np.argsort(-r, 1)
    p1 = top[:, 0]
    ok = p1 == y
    out = {
        "n": int(len(y)),
        "top1": round(float(ok.mean()), 4),
        "top3": round(float((top[:, :3] == y[:, None]).any(1).mean()), 4),
        "fault_macro_f1": round(float(f1_score(ft, f.argmax(1), average="macro")), 4),
        "root_missing_n": int((~present).sum()),
        "root_missing_top1": round(float(ok[~present].mean()), 4) if (~present).any() else None,
    }
    hops = dist[p1[~ok], y[~ok]]
    out["error_hops"] = {int(h): int((hops == h).sum()) for h in np.unique(hops)}
    out["by_fault"] = {
        FAULT_TYPES[k]: [
            int((ft == k).sum()),
            round(float(ok[ft == k].mean()), 3),
            round(float((f.argmax(1) == ft)[ft == k].mean()), 3),
        ]
        for k in range(len(FAULT_TYPES))
        if (ft == k).any()
    }
    return out


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    s = get_settings()
    D, A = load_npz("main"), load_npz("alt")
    sp = D["split"]
    tr, va, te = sp == "train", sp == "val", sp == "test"
    adj, adj_alt = D["adj"], A["adj"]
    sets = {
        f"test_{v}": (
            D[f"X_{v}"][te],
            adj,
            D["y"][te],
            D["f"][te],
            D["dist"],
            D[f"root_present_{v}"][te],
        )
        for v in VARIANTS
    }
    sets.update(
        {
            f"alt_{v}": (A[f"X_{v}"], adj_alt, A["y"], A["f"], A["dist"], A[f"root_present_{v}"])
            for v in VARIANTS
        }
    )
    res: dict = {
        "n": {
            "train": int(tr.sum()),
            "val": int(va.sum()),
            "test": int(te.sum()),
            "alt": int(len(A["y"])),
        }
    }
    res["perturb_alarm_counts_test"] = {
        v: float(D[f"X_{v}"][te][:, :, TOT].sum(1).mean()) for v in VARIANTS
    }

    # ---- shipped model
    root = Path(s.artifacts_dir) / "rca_gnn"
    ver = (root / "latest").read_text().strip()
    ck = torch.load(root / ver / "model.pt", map_location="cpu")
    m = RCAGNN(ck["n_features"], ck["n_faults"])
    m.load_state_dict(ck["state"])
    m.eval()
    res["shipped_version"] = ver
    res["shipped"] = {}
    worst = []
    for name, (x, a, y, ft, dist, pres) in sets.items():
        r, f = scores(m, x, a)
        res["shipped"][name] = summarize(r, f, y, ft, dist, pres)
        if name == "test_clean":
            pr = softmax(r)
            conf, ok = pr.max(1), r.argmax(1) == y
            bins = [0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0001]
            res["shipped"]["reliability_test_top1"] = [
                [
                    f"{lo:.2f}-{min(hi, 1):.2f}",
                    int(sel.sum()),
                    round(float(conf[sel].mean()), 3),
                    round(float(ok[sel].mean()), 3),
                ]
                for lo, hi in zip(bins[:-1], bins[1:], strict=True)
                if (sel := (conf >= lo) & (conf < hi)).any()
            ]
        ids = D["ids"][te] if name.startswith("test") else A["ids"]
        nids = D["node_ids"] if name.startswith("test") else A["node_ids"]
        pr = softmax(r)
        for i in np.flatnonzero(r.argmax(1) != y):
            p, t = int(r[i].argmax()), int(y[i])
            z0, z1 = NAMES.index("z_latency_ms"), NAMES.index("z_temp_c") + 1

            def ntype(j, x=x, i=i):
                return NODE_TYPES[int(x[i, j, -4:].argmax())]

            def kz(j, x=x, i=i, z0=z0, z1=z1):
                return float(np.abs(x[i, j, z0:z1]).max())

            worst.append(
                {
                    "set": name,
                    "incident": str(ids[i]),
                    "fault": FAULT_TYPES[int(ft[i])],
                    "true": f"{nids[t]} ({ntype(t)})",
                    "pred": f"{nids[p]} ({ntype(p)})",
                    "p_pred": round(float(pr[i, p]), 3),
                    "p_true": round(float(pr[i, t]), 4),
                    "rank_true": int((r[i] > r[i, t]).sum()) + 1,
                    "hops": int(dist[p, t]),
                    "pred_is_upstream": bool(x[i, p, -4:].argmax() < x[i, t, -4:].argmax()),
                    "alarms_true": int(x[i, t, TOT]),
                    "alarms_pred": int(x[i, p, TOT]),
                    "max|z|_true": round(kz(t), 1),
                    "max|z|_pred": round(kz(p), 1),
                }
            )
    worst.sort(key=lambda w: (-w["rank_true"], -w["p_pred"]))
    res["worst_mistakes"] = worst[:25]
    res["n_mistakes_all_sets"] = len(worst)

    # ---- ONNX parity on every clean test + alt graph
    import onnxruntime as ort

    sess = ort.InferenceSession(str(root / ver / "model.onnx"), providers=["CPUExecutionProvider"])
    mx, agree, n = 0.0, 0, 0
    for x, a in ((D["X_clean"][te], adj), (A["X_clean"], adj_alt)):
        ah = normalised_adjacency(a)
        rt, ftt = scores(m, x, a)
        for i in range(len(x)):
            ro, fo = sess.run(None, {"x": x[i], "a_hat": ah})
            mx = max(mx, float(np.abs(ro - rt[i]).max()), float(np.abs(fo - ftt[i]).max()))
            agree += int(ro.argmax() == rt[i].argmax() and fo.argmax() == ftt[i].argmax())
            n += 1
    res["onnx"] = {"graphs": n, "max_abs_diff": mx, "argmax_agree": agree}

    # ---- multi-seed GNN vs MLP (same data, same early stopping on val)
    xt, yt, ftr = D["X_clean"][tr], D["y"][tr], D["f"][tr]
    xv, yv = D["X_clean"][va], D["y"][va]
    runs = {"gnn": [], "mlp": []}
    for sd in range(42, 42 + n_seeds):
        torch.set_num_threads(4)
        for kind, g in (("gnn", True), ("mlp", False)):
            mm, info = _fit(xt, adj, yt, ftr, xv, yv, adj, use_graph=g, seed=sd)
            row = {"seed": sd, "val_top1": round(info["val_top1"], 4), "epochs": info["epochs"]}
            for name, (x, a, y, ft, dist, pres) in sets.items():
                r, f = scores(mm, x, a)
                sm = summarize(r, f, y, ft, dist, pres)
                row[name] = {
                    k: sm[k] for k in ("top1", "top3", "fault_macro_f1", "root_missing_top1")
                }
            runs[kind].append(row)
        print(sd, runs["gnn"][-1]["test_clean"], runs["mlp"][-1]["test_clean"], file=sys.stderr)
    res["seeds"] = runs
    agg = {}
    for name in sets:
        for met in ("top1", "fault_macro_f1", "root_missing_top1"):
            g = np.array([r[name][met] for r in runs["gnn"]], float)
            b = np.array([r[name][met] for r in runs["mlp"]], float)
            agg[f"{name}.{met}"] = {
                "gnn_mean": round(g.mean(), 4),
                "gnn_sd": round(g.std(ddof=1), 4),
                "mlp_mean": round(b.mean(), 4),
                "mlp_sd": round(b.std(ddof=1), 4),
                "diff_mean": round((g - b).mean(), 4),
                "diff_sd": round((g - b).std(ddof=1), 4),
                "gnn_wins": int((g > b).sum()),
                "ties": int((g == b).sum()),
                "mlp_wins": int((g < b).sum()),
            }
    res["seed_summary"] = agg
    (OUT / "rca.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("seeds",)}, indent=1))


if __name__ == "__main__":
    main()
