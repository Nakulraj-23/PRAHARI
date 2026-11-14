"""Q2 audit of the anomaly detector: reproduce the card, then score the SAME model under
alternative ground-truth labellings rebuilt from the simulator's per-node KPI effect
(``propagate`` impact x ``fault_gain``, deterministic in seed and incident_id). Evaluation only.

Labellings (positive = node n at grid minute t):
  card       t in [start, end] and n is root or raised a true alarm in the incident (A2's target)
  visible    t in [start, end] and the node's simulated KPI effect >= VIS (alarm or not)
  card_mask  card positives, but KPI-silent positives (effect < VIS) are dropped from scoring
             instead of counted as misses
  visible_tail  visible, with the window extended by the 3-min ramp-down + 5-min feature lag

Run: ``.venv/bin/python -m tests.models.audit.anomaly_audit`` -> artifacts/q2_audit/anomaly.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from prahari.config import get_settings
from prahari.db import get_engine
from prahari.models.anomaly import model as M
from prahari.pipeline.splits import load_splits
from prahari.pipeline.store import load_ground_truth_incidents, load_kpi
from prahari.sim.faults import fault_gain, propagate
from prahari.sim.topology import NetworkModel
from tests.models.audit.rca_data import OUT

VIS = 0.15  # silent dependents have gain <= 0.12 (x impact <= 1); visible ones >= 0.5 x 0.12


def prf(y, s, thr) -> dict:
    pred = s >= thr
    tp = int((pred & y).sum())
    p, r = tp / max(pred.sum(), 1), tp / max(y.sum(), 1)
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(2 * p * r / max(p + r, 1e-12), 4),
    }


def main() -> None:
    s = get_settings()
    eng = get_engine(s)
    net = NetworkModel.from_db(eng)
    pred = M.load(settings=s)
    sp = load_splits(s)
    t_tr, t_va = (pd.Timestamp(sp["boundaries"][k]) for k in ("train_end", "val_end"))
    store = load_kpi(eng, net)
    idx = np.arange(M.NEED - 1, store.n_minutes, M.STEP)
    ts = store.t0 + pd.to_timedelta(idx, unit="min")
    va, te = np.asarray((ts >= t_tr) & (ts < t_va)), np.asarray(ts >= t_va)
    sel = va | te
    score, zmax = pred.score_matrix(store.values, idx[sel])
    t0 = store.t0
    del store
    gidx = idx[sel]
    is_te = te[sel]

    inc = load_ground_truth_incidents(eng)
    al = pd.read_sql(
        "SELECT DISTINCT label_incident_id AS iid, node_id FROM alarms WHERE label_is_true = 1", eng
    )
    alarm_nodes = al.groupby("iid")["node_id"].apply(set).to_dict()
    shape = (len(gidx), net.n)
    lab = {k: np.zeros(shape, bool) for k in ("card", "visible", "visible_tail", "strong")}
    since = np.full(shape, -1, np.int32)  # minutes since incident start for visible positives
    silent_pos = np.zeros(shape, bool)
    per_type = {}
    stats = {
        "card_pos_node_incidents": 0,
        "card_pos_silent": 0,
        "visible_no_alarm": 0,
        "visible_total": 0,
    }
    for r in inc.itertuples():
        ri = net.index[r.root_node_id]
        eff = propagate(net, ri, r.fault_type) * fault_gain(net, r.incident_id, ri, s.seed)
        vis = set(np.flatnonzero(eff >= VIS))
        card = {net.index[n] for n in alarm_nodes.get(r.incident_id, set())} | {ri}
        a_min = int((r.start_ts - t0) / pd.Timedelta(minutes=1))
        b_min = int((r.end_ts - t0) / pd.Timedelta(minutes=1))
        a = int(np.searchsorted(gidx, a_min, "left"))
        b = int(np.searchsorted(gidx, b_min, "right"))
        b2 = int(np.searchsorted(gidx, b_min + 8, "right"))
        if b <= a:
            continue
        lab["card"][a:b, list(card)] = True
        if vis:
            lab["visible"][a:b, list(vis)] = True
            lab["visible_tail"][a:b2, list(vis)] = True
            since[a:b, list(vis)] = (gidx[a:b] - a_min)[:, None]
        strong = list(np.flatnonzero(eff >= 0.5))
        if strong:
            lab["strong"][a:b, strong] = True
        silent = [n for n in card if n not in vis]
        silent_pos[a:b, silent] = True
        if is_te[a]:
            stats["card_pos_node_incidents"] += len(card)
            stats["card_pos_silent"] += len(silent)
            stats["visible_total"] += len(vis)
            stats["visible_no_alarm"] += len(vis - card)
            per_type.setdefault(r.fault_type, []).append((a, b, sorted(vis)))
    res: dict = {"vis_threshold": VIS, "test_stats": stats, "threshold": pred.threshold}

    def block(y, mask, thr_src="val"):
        out = {}
        for name, sc in (("iforest", score), ("max_abs_z5", zmax)):
            yv, sv = y[~is_te][mask[~is_te]], sc[~is_te][mask[~is_te]]
            yt, st = y[is_te][mask[is_te]], sc[is_te][mask[is_te]]
            thr = M._best_f1_threshold(yv, sv)
            out[name] = {
                "pr_auc": round(float(average_precision_score(yt, st)), 4),
                "pos_rate": round(float(yt.mean()), 4),
                "at_val_f1_thr": prf(yt, st, thr),
            }
            if name == "iforest":
                out[name]["at_shipped_thr"] = prf(yt, st, pred.threshold)
        return out

    full = np.ones(shape, bool)
    res["card"] = block(lab["card"], full)
    res["card_mask_silent"] = block(lab["card"], ~silent_pos)
    res["visible"] = block(lab["visible"], full)
    res["visible_tail"] = block(lab["visible_tail"], full)
    res["strong_effect_0.5"] = block(lab["strong"], full)
    lagr = {}
    for lo, hi in ((0, 5), (5, 15), (15, 30), (30, 60), (60, 10**6)):
        msk = lab["visible"] & (since >= lo) & (since < hi) & is_te[:, None]
        lagr[f"{lo}-{hi}"] = [
            int(msk.sum()),
            round(float((score[msk] >= pred.threshold).mean()), 3) if msk.any() else None,
        ]
    res["visible_recall_by_minutes_since_start"] = lagr
    # recall at shipped threshold of KPI-visible node-minutes, by fault type (test)
    rt = {}
    for ft, spans in per_type.items():
        hit = tot = 0
        for a, b, v in spans:
            if v:
                hit += int((score[a:b, v] >= pred.threshold).sum())
                tot += (b - a) * len(v)
        rt[ft] = round(hit / max(tot, 1), 3)
    res["visible_recall_by_fault_type"] = rt
    (OUT / "anomaly.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
