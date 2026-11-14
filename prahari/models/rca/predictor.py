"""RCA predictor: window (or precomputed features) -> shared RCA output contract."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from prahari.config import Settings, get_settings
from prahari.models.rca.model import RCAGNN, normalised_adjacency
from prahari.pipeline.features import build_node_window_features, feature_names
from prahari.sim.constants import FAULT_TYPES, KPI_COLUMNS
from prahari.sim.topology import NetworkModel

NAME = "rca_gnn"


class RCAPredictor:
    """Pure function of its input.

    ``predict({"start", "end"})`` builds node features from the DB; ``predict({"features": df,
    "start", "end"})`` uses a node-feature frame (index node_id, ``feature_names()`` columns).
    Optional ``valid_alarm_count`` / ``noise_alarm_count`` come from the alarm validator (A2/A5);
    without them every alarm counts as valid.
    """

    def __init__(
        self, model: RCAGNN, version: str, settings: Settings, net: NetworkModel | None = None
    ):
        self.model, self.version, self.settings = model.eval(), version, settings
        self._net = net
        self._ahat: torch.Tensor | None = None

    def _network(self) -> NetworkModel:
        if self._net is None:
            from prahari.db import get_engine

            self._net = NetworkModel.from_db(get_engine(self.settings))
        return self._net

    def predict(self, payload: dict) -> dict:
        """Return the RCA contract dict for the window in ``payload``."""
        t0 = time.perf_counter()
        net = self._network()
        start, end = pd.Timestamp(payload["start"]), pd.Timestamp(payload["end"])
        feats = payload.get("features")
        if feats is None:
            feats = build_node_window_features(start, end, self.settings, net=net)
        feats = feats.reindex(net.node_ids)
        x = feats[feature_names()].to_numpy(np.float32)
        if self._ahat is None or self._ahat.shape[0] != net.n:
            self._ahat = torch.as_tensor(normalised_adjacency(net.adj))
        with torch.no_grad():
            r, f = self.model(torch.as_tensor(x).unsqueeze(0), self._ahat)
        pr = torch.softmax(r[0], 0).numpy()
        pf = torch.softmax(f[0], 0).numpy()
        top = np.argsort(-pr, kind="stable")[:5]
        names = feature_names()
        total = x[:, names.index("alarm_total")]
        alarmed = [net.node_ids[i] for i in np.flatnonzero(total > 0)]
        kd = {}
        for i in top:
            d = {k: round(float(x[i, names.index(f"d_{k}")]), 3) for k in KPI_COLUMNS}
            kd[net.node_ids[i]] = {k: v for k, v in d.items() if abs(v) > 1e-6}
        n_alarms = int(total.sum())
        valid = int(payload.get("valid_alarm_count", n_alarms))
        noise = int(payload.get("noise_alarm_count", n_alarms - valid))
        fi = int(pf.argmax())
        return {
            "window": {"start": start.isoformat(), "end": end.isoformat()},
            "root_candidates": [
                {"node_id": net.node_ids[i], "prob": round(float(pr[i]), 4)} for i in top
            ],
            "fault_type": {"label": FAULT_TYPES[fi], "prob": round(float(pf[fi]), 4)},
            "evidence": {
                "alarmed_nodes": alarmed,
                "valid_alarm_count": valid,
                "noise_alarm_count": noise,
                "kpi_deltas": kd,
            },
            "model": {"name": NAME, "version": self.version},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }


def load(version: str | None = None, settings: Settings | None = None) -> RCAPredictor:
    """Load ``artifacts/rca_gnn/<version>/model.pt`` (latest if version is None)."""
    settings = settings or get_settings()
    root = Path(settings.artifacts_dir) / NAME
    version = version or (root / "latest").read_text().strip()
    ck = torch.load(root / version / "model.pt", map_location="cpu")
    model = RCAGNN(ck["n_features"], ck["n_faults"])
    model.load_state_dict(ck["state"])
    return RCAPredictor(model, version, settings)
