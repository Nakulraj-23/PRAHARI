"""Q2: validator robustness on the test period. drop30 = remove 30% of alarms at random (thins
flap/neighbour context); noise50 = add synthetic noise alarms equal to 50% of the real noise count
(random node/time, type+severity drawn from real noise alarms, KPI z = 0, as for ~70% of noise).
Run: ``.venv/bin/python -m tests.models.audit.validator_robust``"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from prahari.config import get_settings
from prahari.db import get_engine
from prahari.models.validator import model as V
from prahari.pipeline.splits import load_splits
from prahari.pipeline.store import load_alarms, load_kpi
from prahari.sim.topology import NetworkModel
from tests.models.audit.rca_data import OUT


def main() -> None:
    s = get_settings()
    e = get_engine(s)
    net = NetworkModel.from_db(e)
    p = V.load(settings=s)
    tva = pd.Timestamp(load_splits(s)["boundaries"]["val_end"])
    al = load_alarms(e, tva - pd.Timedelta(minutes=10))
    lab = pd.read_sql("SELECT alarm_id, label_is_true FROM alarms", e).set_index("alarm_id")
    al["y"] = lab.loc[al.alarm_id, "label_is_true"].to_numpy().astype(int)
    st = load_kpi(e, net, tva - pd.Timedelta(minutes=80), al.ts.max() + pd.Timedelta(minutes=1))
    al["z"] = list(V.alarm_kpi_z(al, net, st))
    rng = np.random.default_rng(7)
    pool = al[al.y == 0]
    out = {}
    for name in ("clean", "drop30", "noise50", "both"):
        a = al.copy()
        if name in ("drop30", "both"):
            a = a[rng.random(len(a)) >= 0.3]
        if name in ("noise50", "both"):
            k = int(0.5 * (a.y == 0).sum())
            pk = pool.iloc[rng.integers(0, len(pool), k)]
            span = (a.ts.max() - a.ts.min()).total_seconds()
            fake = pd.DataFrame(
                {
                    "alarm_id": [f"F{i}" for i in range(k)],
                    "ts": (a.ts.min() + pd.to_timedelta(rng.uniform(0, span, k), unit="s")).floor(
                        "s"
                    ),
                    "node_id": np.array(net.node_ids)[rng.integers(0, net.n, k)],
                    "alarm_type": pk.alarm_type.to_numpy(),
                    "severity": pk.severity.to_numpy(),
                    "raw_text": "",
                    "y": 0,
                    "z": [np.zeros(6)] * k,
                }
            )
            a = pd.concat([a, fake]).sort_values("ts", kind="stable").reset_index(drop=True)
        inw = (a.ts >= tva).to_numpy()
        x = V.build_features(a[inw], net, np.stack(a.z[inw].to_numpy()), a[~inw])
        q, y = p.proba(x), a.y[inw].to_numpy()
        pred = q >= p.threshold
        tp = int((pred & (y == 1)).sum())
        out[name] = {
            "n": int(len(y)),
            "noise": int((y == 0).sum()),
            "precision": round(tp / pred.sum(), 4),
            "recall": round(tp / y.sum(), 4),
            "ap_true": round(float(average_precision_score(y, q)), 4),
            "ap_noise": round(float(average_precision_score(1 - y, 1 - q)), 4),
        }
    (OUT / "validator_robust.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
