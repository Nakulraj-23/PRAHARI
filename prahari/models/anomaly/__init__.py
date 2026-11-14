"""M1 anomaly detector: IsolationForest on per-node rolling KPI features."""

from prahari.models.anomaly.model import AnomalyPredictor, load, train

__all__ = ["AnomalyPredictor", "load", "train"]
