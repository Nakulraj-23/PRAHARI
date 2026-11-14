"""Model quality gates (Q2). Thresholds sit just below the numbers Q2 re-measured on the test
split (seed 42, 90-day DB, model versions in docs/handoffs/Q2.md), so a regression fails CI.

Two layers:
* card gates read the committed ``artifacts/*/*.card.json`` (always run, also in CI);
* live gates re-score the shipped artifacts on the real test period; they skip when the DB or
  the model files are absent (CI has neither) and take ~30 s locally.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prahari.config import get_settings

S = get_settings()
ART = Path(S.artifacts_dir)
DB = Path(str(S.db_url).replace("sqlite:///", ""))
RCA_NPZ = ART / "q2_audit" / "rca_main.npz"  # built by tests/models/audit/rca_data.py

# verified value -> gate (comment)
GATES = {
    "alarm_validator": {
        "pr_auc": 0.995,
        "recall": 0.98,
        "precision": 0.95,
        "f1": 0.97,
    },  # .998 .991 .962 .976
    "anomaly_iforest": {"pr_auc": 0.33, "f1": 0.40},  # .346 .421
    "rca_gnn": {"root_top1": 0.95, "root_top3": 0.99, "fault_macro_f1": 0.94},  # .966 1.0 .957
}
MAX_BRIER = 0.03  # validator .021


def _card(name: str) -> dict:
    root = ART / name
    if name == "rca_gnn" and (root / "latest").exists():
        return json.loads((root / f"{(root / 'latest').read_text().strip()}.card.json").read_text())
    cards = sorted(root.glob("*.card.json"))
    if not cards:
        pytest.skip(f"no card for {name}")
    return json.loads(cards[-1].read_text())


# ---------------------------------------------------------------- card gates
def test_card_validator_gates():
    c = _card("alarm_validator")
    for k, v in GATES["alarm_validator"].items():
        assert c["metrics"][k] >= v, (k, c["metrics"][k])
    assert c["metrics"]["brier"] <= MAX_BRIER
    assert c["metrics"]["pr_auc"] >= c["baseline_metrics"]["pr_auc"] - 0.002
    assert "val" in c["metrics"]["threshold_rule"]


def test_card_anomaly_gates():
    c = _card("anomaly_iforest")
    for k, v in GATES["anomaly_iforest"].items():
        assert c["metrics"][k] >= v, (k, c["metrics"][k])
    assert c["metrics"]["pr_auc"] > c["baseline_metrics"]["pr_auc"] + 0.1  # 3-sigma .193
    assert c["metrics"]["f1"] > c["extra"]["baseline_sigma_tuned_on_val"]["f1"]  # .317


def test_card_rca_gates():
    c = _card("rca_gnn")
    t = c["metrics"]["test"]
    for k, v in GATES["rca_gnn"].items():
        assert t[k] >= v, (k, t[k])
    assert t["root_alarm_missing"]["root_top1"] >= 0.80  # 9/11
    assert c["metrics"]["alt_topology"]["root_top1"] >= 0.97  # 1.0 on 47
    assert c["metrics"]["onnx_max_abs_diff"] < 1e-4
    rules = c["baseline_metrics"]
    assert t["root_top1"] >= rules["upstream_alarmed"]["test"]["root_top1"] + 0.15  # .739
    assert t["root_top1"] >= rules["mlp_no_graph"]["test"]["root_top1"] - 0.02  # no regression


# ---------------------------------------------------------------- live gates
needs_db = pytest.mark.skipif(not DB.exists(), reason="no seeded DB")


def _has(name: str, f: str) -> bool:
    return any((ART / name).glob(f"*/{f}"))


@needs_db
@pytest.mark.skipif(not _has("alarm_validator", "model.joblib"), reason="no validator artifact")
def test_live_validator_test_split():
    from sklearn.metrics import average_precision_score

    from prahari.db import get_engine
    from prahari.models.validator import model as V
    from prahari.pipeline.splits import load_splits
    from prahari.pipeline.store import load_alarms, load_kpi
    from prahari.sim.topology import NetworkModel

    eng, p = get_engine(S), V.load(settings=S)
    net = NetworkModel.from_db(eng)
    tva = pd.Timestamp(load_splits(S)["boundaries"]["val_end"])
    al = load_alarms(eng, tva - pd.Timedelta(minutes=10))
    inw = (al.ts >= tva).to_numpy()
    lab = pd.read_sql("SELECT alarm_id, label_is_true FROM alarms", eng).set_index("alarm_id")
    y = lab.loc[al.alarm_id[inw], "label_is_true"].to_numpy().astype(int)
    st = load_kpi(eng, net, tva - pd.Timedelta(minutes=80), al.ts.max() + pd.Timedelta(minutes=1))
    q = p.proba(V.build_features(al[inw], net, V.alarm_kpi_z(al[inw], net, st), al[~inw]))
    pred = q >= p.threshold
    assert len(y) == 826
    assert average_precision_score(y, q) >= 0.995
    assert average_precision_score(1 - y, 1 - q) >= 0.95  # noise class, verified .9675
    assert pred[y == 1].mean() >= 0.98
    assert y[pred].mean() >= 0.95
    assert np.mean((q - y) ** 2) <= MAX_BRIER


@pytest.mark.skipif(not RCA_NPZ.exists(), reason="run tests/models/audit/rca_data.py first")
@pytest.mark.skipif(not (ART / "rca_gnn" / "latest").exists(), reason="no RCA artifact")
def test_live_rca_test_alt_robustness_onnx():
    import onnxruntime as ort
    import torch

    from prahari.models.rca.model import normalised_adjacency
    from prahari.models.rca.predictor import load
    from tests.models.audit.rca_audit import load_npz, scores

    m = load(settings=S).model
    d = load_npz("main")
    te = d["split"] == "test"
    y = d["y"][te]
    top1 = {}
    for v in ("clean", "drop30", "noise50", "both"):
        r, _ = scores(m, d[f"X_{v}"][te], d["adj"])
        top1[v] = float((r.argmax(1) == y).mean())
    assert top1["clean"] >= 0.95  # .966
    assert top1["drop30"] >= 0.95  # .966
    assert top1["noise50"] >= 0.95  # .966
    assert top1["both"] >= 0.93  # .943
    alt = load_npz("alt")
    r, _ = scores(m, alt["X_clean"], alt["adj"])
    assert (r.argmax(1) == alt["y"]).mean() >= 0.97  # 1.0
    ver = (ART / "rca_gnn" / "latest").read_text().strip()
    sess = ort.InferenceSession(
        str(ART / "rca_gnn" / ver / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    ah = normalised_adjacency(d["adj"])
    x = d["X_clean"][te][:20]
    with torch.no_grad():
        rt, _ = m(torch.as_tensor(x), torch.as_tensor(ah))
    for i in range(len(x)):
        ro, _ = sess.run(None, {"x": x[i], "a_hat": ah})
        assert np.abs(ro - rt[i].numpy()).max() < 1e-4


@needs_db
@pytest.mark.skipif(not _has("anomaly_iforest", "model.joblib"), reason="no anomaly artifact")
def test_live_anomaly_test_split():
    from sklearn.metrics import average_precision_score

    from prahari.db import get_engine
    from prahari.models.anomaly import model as M
    from prahari.pipeline.splits import load_splits
    from prahari.pipeline.store import load_kpi
    from prahari.sim.topology import NetworkModel

    eng, p = get_engine(S), M.load(settings=S)
    net = NetworkModel.from_db(eng)
    tva = pd.Timestamp(load_splits(S)["boundaries"]["val_end"])
    st = load_kpi(eng, net, tva - pd.Timedelta(minutes=M.NEED + 10))
    first = pd.read_sql("SELECT MIN(ts) AS m FROM kpi_samples", eng, parse_dates=["m"])["m"][0]
    off = int((st.t0 - first.floor("min")) / pd.Timedelta(minutes=1))  # same grid as training
    idx = np.arange(M.NEED - 1, off + st.n_minutes, M.STEP) - off
    idx = idx[(idx >= M.NEED - 1) & np.asarray(st.t0 + pd.to_timedelta(idx, unit="min") >= tva)]
    s, _ = p.score_matrix(st.values, idx)
    y = M._targets(eng, net, st, idx)
    assert y.size == 407_640
    assert average_precision_score(y.ravel(), s.ravel()) >= 0.33  # .346
    pred = s >= p.threshold
    prec, rec = (pred & y).sum() / pred.sum(), (pred & y).sum() / y.sum()
    assert 2 * prec * rec / (prec + rec) >= 0.40  # .421
