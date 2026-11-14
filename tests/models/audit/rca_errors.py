"""Q2: shipped RCA model, error context (concurrent incidents) and fault-type mistakes on test.
Run after rca_data: ``.venv/bin/python -m tests.models.audit.rca_errors``"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prahari.config import get_settings
from prahari.db import get_engine
from prahari.models.rca.model import RCAGNN
from prahari.pipeline.store import load_ground_truth_incidents
from prahari.sim.constants import FAULT_TYPES
from tests.models.audit.rca_audit import load_npz, scores, softmax
from tests.models.audit.rca_data import OUT


def main() -> None:
    s = get_settings()
    root = Path(s.artifacts_dir) / "rca_gnn"
    ver = (root / "latest").read_text().strip()
    ck = torch.load(root / ver / "model.pt", map_location="cpu")
    m = RCAGNN(ck["n_features"], ck["n_faults"])
    m.load_state_dict(ck["state"])
    m.eval()
    D = load_npz("main")
    inc = load_ground_truth_incidents(get_engine(s)).set_index("incident_id")
    te = D["split"] == "test"
    ids, starts = D["ids"][te], pd.to_datetime(D["starts"][te])
    r, f = scores(m, D["X_clean"][te], D["adj"])
    pr, pf = softmax(r), softmax(f)
    y, ft = D["y"][te], D["f"][te]
    out = {"overlap": {}, "root_errors": [], "fault_errors": []}
    conc = []
    for i, iid in enumerate(ids):
        w0, w1 = starts[i], starts[i] + pd.Timedelta(minutes=15)
        o = inc[
            (inc.index != iid) & (inc.start_ts < w1) & (inc.end_ts + pd.Timedelta(minutes=3) > w0)
        ]
        conc.append(list(o.root_node_id))
    ok = r.argmax(1) == y
    has = np.array([len(c) > 0 for c in conc])
    out["overlap"] = {
        "n_with_concurrent": int(has.sum()),
        "top1_with": round(float(ok[has].mean()), 3) if has.any() else None,
        "top1_without": round(float(ok[~has].mean()), 3),
    }
    nid = D["node_ids"]
    for i in np.flatnonzero(~ok):
        p = int(r[i].argmax())
        out["root_errors"].append(
            {
                "incident": str(ids[i]),
                "pred": str(nid[p]),
                "true": str(nid[y[i]]),
                "p": round(float(pr[i, p]), 3),
                "pred_is_concurrent_root": str(nid[p]) in conc[i],
                "concurrent_roots": conc[i],
            }
        )
    for i in np.flatnonzero(pf.argmax(1) != ft):
        out["fault_errors"].append(
            {
                "incident": str(ids[i]),
                "true": FAULT_TYPES[ft[i]],
                "pred": FAULT_TYPES[int(pf[i].argmax())],
                "p": round(float(pf[i].max()), 3),
                "root_ok": bool(ok[i]),
                "root": str(nid[y[i]]),
                "root_alarms": int(D["X_clean"][te][i, y[i], 15]),
                "concurrent_roots": conc[i],
            }
        )
    (OUT / "rca_errors.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
