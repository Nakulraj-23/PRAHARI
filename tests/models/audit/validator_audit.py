"""Q2 audit of the alarm validator: reproduce card metrics, reliability table, leakage probes on
flap/neighbour counts (independent brute-force recount, tie-break check, ablations, single
feature AUCs), and train/serve parity of ``predict_window``.

Run: ``.venv/bin/python -m tests.models.audit.validator_audit`` -> artifacts/q2_audit/validator.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from prahari.config import get_settings
from prahari.db import get_engine
from prahari.models.validator import model as V
from prahari.pipeline.splits import load_splits
from prahari.pipeline.store import load_alarms, load_kpi
from prahari.sim.topology import NetworkModel
from tests.models.audit.rca_data import OUT


def pr_auc(y, p) -> float:
    """Own average precision (step-wise, ties grouped) to cross-check sklearn."""
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    last = np.r_[np.flatnonzero(np.diff(ps) != 0), len(ps) - 1]  # end of each tie group
    prec, rec = tp[last] / (last + 1), tp[last] / ys.sum()
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def prf(y, pred) -> dict:
    tp = int((pred & (y == 1)).sum())
    p, r = tp / max(pred.sum(), 1), tp / max((y == 1).sum(), 1)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4)}


def reliability(y, p, bins=10) -> list:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            out.append(
                [
                    f"{lo:.1f}-{hi:.1f}",
                    int(m.sum()),
                    round(float(p[m].mean()), 3),
                    round(float(y[m].mean()), 3),
                ]
            )
    return out


def brute_counts(al: pd.DataFrame, net: NetworkModel) -> pd.DataFrame:
    """Independent recount: alarms with ts STRICTLY earlier (ties excluded)."""
    sec = ((al["ts"] - pd.Timestamp("2000-01-01")) / pd.Timedelta(seconds=1)).to_numpy()
    node = al["node_id"].map(net.index).to_numpy()
    typ = al["alarm_type"].to_numpy()
    fn, ft, nb = (np.zeros(len(al), int) for _ in range(3))
    for i in range(len(al)):
        w10 = (sec < sec[i]) & (sec >= sec[i] - 600)
        w5 = (sec < sec[i]) & (sec >= sec[i] - 300)
        fn[i] = int((w10 & (node == node[i])).sum())
        ft[i] = int((w10 & (node == node[i]) & (typ == typ[i])).sum())
        nb[i] = int((w5 & net.adj[node[i]][node]).sum())
    return pd.DataFrame({"flap_node_10m": fn, "flap_type_10m": ft, "neigh_5m": nb})


def main() -> None:
    s = get_settings()
    eng = get_engine(s)
    net = NetworkModel.from_db(eng)
    pred = V.load(settings=s)
    al = load_alarms(eng)
    lab = pd.read_sql("SELECT alarm_id, label_is_true FROM alarms", eng)
    y = al[["alarm_id"]].merge(lab, on="alarm_id")["label_is_true"].to_numpy().astype(int)
    sp = load_splits(s)
    t_tr, t_va = (pd.Timestamp(sp["boundaries"][k]) for k in ("train_end", "val_end"))
    store = load_kpi(
        eng, net, al.ts.min() - pd.Timedelta(minutes=70), al.ts.max() + pd.Timedelta(minutes=1)
    )
    x = V.build_features(al, net, V.alarm_kpi_z(al, net, store))
    del store
    tr = (al.ts < t_tr).to_numpy()
    va = ((al.ts >= t_tr) & (al.ts < t_va)).to_numpy()
    te = (al.ts >= t_va).to_numpy()
    res: dict = {"n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}}
    p_te, p_va = pred.proba(x[te]), pred.proba(x[va])
    res["shipped"] = {
        "threshold": pred.threshold,
        "test": {
            **prf(y[te], p_te >= pred.threshold),
            "pr_auc_sklearn": round(float(average_precision_score(y[te], p_te)), 4),
            "pr_auc_own": round(pr_auc(y[te], p_te), 4),
            "roc_auc": round(float(roc_auc_score(y[te], p_te)), 4),
            "brier": round(float(np.mean((p_te - y[te]) ** 2)), 4),
        },
        "val": prf(y[va], p_va >= pred.threshold),
        "reliability_test": reliability(y[te], p_te),
        "n_distinct_test_probs": int(len(np.unique(p_te))),
    }
    # threshold re-derived on val with the documented rule
    cand = np.unique(p_va[y[va] == 1])
    ok = [c for c in cand if (p_va[y[va] == 1] >= c).mean() >= V.TARGET_RECALL]
    res["shipped"]["threshold_rederived_on_val"] = float(max(ok))

    # --- leakage probes on the counting features
    cnt = brute_counts(al, net)
    same = {c: float((cnt[c].to_numpy() == x[c].to_numpy()).mean()) for c in cnt}
    sec = al.ts.astype("int64").to_numpy()
    tie = pd.Series(sec).duplicated(keep=False).to_numpy()
    res["counts_vs_bruteforce_agreement"] = same
    res["alarms_sharing_exact_ts"] = {
        "n": int(tie.sum()),
        "true_rate": round(float(y[tie].mean()), 3) if tie.any() else None,
    }
    res["single_feature_test_auc"] = {
        c: round(float(roc_auc_score(y[te], x[c].to_numpy()[te])), 4)
        for c in ["flap_node_10m", "flap_type_10m", "neigh_5m", "kpi_z_max", "sev_num"]
    }
    res["class_means_test"] = {
        c: {
            "true": round(float(x[c][te][y[te] == 1].mean()), 3),
            "noise": round(float(x[c][te][y[te] == 0].mean()), 3),
        }
        for c in ["flap_node_10m", "flap_type_10m", "neigh_5m", "kpi_z_max"]
    }
    groups = {
        "all": V.feature_names(),
        "no_counts": [f for f in V.feature_names() if f not in cnt.columns],
        "counts_only": list(cnt.columns),
        "no_neigh": [f for f in V.feature_names() if f != "neigh_5m"],
        "no_flap": [f for f in V.feature_names() if not f.startswith("flap")],
        "bruteforce_counts_all": V.feature_names(),
        "cat_only": [
            f for f in V.feature_names() if f.split("_")[0] in ("sev", "type", "ntype", "vendor")
        ],
    }
    abl = {}
    for g, cols in groups.items():
        xx = x.copy()
        if g == "bruteforce_counts_all":
            for c in cnt:
                xx[c] = cnt[c].to_numpy()
        with threadpool_limits(1):
            m = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=42,
            )
            m.fit(xx[cols][tr], y[tr])
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
            m.predict_proba(xx[cols][va])[:, 1], y[va]
        )
        p = cal.predict(m.predict_proba(xx[cols][te])[:, 1])
        abl[g] = {
            "pr_auc": round(float(average_precision_score(y[te], p)), 4),
            "roc_auc": round(float(roc_auc_score(y[te], p)), 4),
        }
    res["ablation_test"] = abl
    res["chance_pr_auc"] = round(float(y[te].mean()), 4)

    # --- train/serve parity: predict_window on 3 test windows vs offline features
    diffs = []
    for st in pd.date_range(t_va + pd.Timedelta(hours=6), periods=3, freq="37h"):
        out = pred.predict_window(st, st + pd.Timedelta(hours=2), s)
        if out.empty:
            continue
        idx = al.reset_index().set_index("alarm_id").loc[out.alarm_id, "index"].to_numpy()
        diffs.append(float(np.abs(out.valid_prob.to_numpy() - pred.proba(x.iloc[idx])).max()))
    res["serve_vs_offline_max_prob_diff"] = diffs
    (OUT / "validator.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
