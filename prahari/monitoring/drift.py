"""Data and prediction drift: reference at training time, PSI/KS per feature, ``run_drift``.

Reference (``artifacts/drift/reference.json``) is built from the *train* period of the DB
(feature distributions) and the train+validation periods (model outputs, which are bimodal and
so not very sensitive to over-fitting).
KPIs are strongly seasonal, so reference proportions are kept per (hour of day, weekday/weekend)
group and a window is compared with the mixture of groups it actually covers.
Only the public KPI/alarm loaders and the model ``predict`` APIs are used; no label column is read.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from prahari.config import Settings, get_settings
from prahari.db import Base, DriftReport, get_engine
from prahari.monitoring.stats import (
    bin_props,
    ks_2samp,
    psi_from_props,
    quantile_edges,
    status_from_psi,
    worst,
)
from prahari.pipeline.splits import load_splits
from prahari.pipeline.store import load_alarms, load_kpi
from prahari.sim.constants import KPI_COLUMNS, SIM_START, VENDORS
from prahari.sim.topology import NetworkModel

MA_MIN = 15
FEATURES = [*KPI_COLUMNS, "latency_ms_ma15", "packet_loss_pct_ma15"]
PRED_FEATURES = ["pred_validator_prob", "pred_rca_top1_conf"]
N_GROUPS = 48
GRID = 200  # quantile points kept per (group, feature) for the KS reference sample
MIN_PRED_N = 100
MIN_ROWS = 500
_MIN = pd.Timedelta(minutes=1)


def reference_path(settings: Settings) -> Path:
    """Where the reference statistics live."""
    return Path(settings.artifacts_dir) / "drift" / "reference.json"


def _groups(ts: pd.DatetimeIndex) -> np.ndarray:
    return (ts.hour.to_numpy() + 24 * (ts.day_of_week.to_numpy() >= 5)).astype(int)


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over ``w`` samples along axis 0, ignoring NaN (NaN if none in window)."""
    t = x.shape[0]
    ok = ~np.isnan(x)
    c = np.vstack([np.zeros((1, *x.shape[1:])), np.cumsum(np.where(ok, x, 0.0), 0)])
    n = np.vstack([np.zeros((1, *x.shape[1:])), np.cumsum(ok, 0)])
    lo = np.maximum(np.arange(t) + 1 - w, 0)
    cnt = n[1:] - n[lo]
    with np.errstate(all="ignore"):
        return np.where(cnt > 0, (c[1:] - c[lo]) / np.maximum(cnt, 1), np.nan)


def _rows(engine, net: NetworkModel, start: pd.Timestamp, end: pd.Timestamp):
    """Feature matrix (R, len(FEATURES)) with row group and vendor index, for [start, end)."""
    store = load_kpi(engine, net, start - MA_MIN * _MIN, end)
    v = store.values.astype(np.float64)
    if v.shape[0] == 0:
        return np.zeros((0, len(FEATURES))), np.zeros(0, int), np.zeros(0, int)
    ts = store.t0 + pd.to_timedelta(np.arange(v.shape[0]), unit="min")
    keep = np.asarray((ts >= start) & (ts < end))
    ma = [_rolling_mean(v[:, :, 0], MA_MIN), _rolling_mean(v[:, :, 1], MA_MIN)]
    cols = [v[:, :, k] for k in range(6)] + ma
    x = np.stack([c[keep] for c in cols], axis=-1)  # (T', N, F)
    g = np.repeat(_groups(ts[keep]), net.n)
    ven = np.tile(net.vendor_idx, int(keep.sum()))
    return x.reshape(-1, len(FEATURES)), g, ven


def _mixture_props(props: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Expected bin proportions for group weights ``w``; props (G, B). Empty groups are skipped."""
    ok = props.sum(1) > 0
    ww = np.where(ok, w, 0.0)
    return (ww[:, None] * props).sum(0) / ww.sum() if ww.sum() > 0 else props[ok].mean(0)


def _ref_sample(grid: np.ndarray, w: np.ndarray, m: int = 5000) -> np.ndarray:
    """Mixture sample: each group contributes round(w_g * m) evenly spaced quantiles."""
    parts = []
    for g in range(len(w)):
        k = int(round(w[g] * m))
        if k and len(grid[g]):
            parts.append(grid[g][np.linspace(0, len(grid[g]) - 1, k).astype(int)])
    return np.concatenate(parts) if parts else np.zeros(0)


def _days(start: pd.Timestamp, end: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    first, last = start.normalize() + pd.Timedelta(days=1), end.normalize() - pd.Timedelta(days=1)
    if last < first:  # tiny databases: fall back to the first day
        return [start.normalize()]
    span = (last - first).days
    idx = np.unique(np.linspace(0, span, n).round().astype(int))
    return [first + pd.Timedelta(days=int(i)) for i in idx]


def _model_outputs(settings: Settings, net: NetworkModel, start, end) -> dict:
    """Validator probabilities and RCA top-1 confidences over [start, end) (labels never read)."""
    out: dict = {"validator": None, "rca": None, "versions": {}}
    engine = get_engine(settings)
    try:
        from prahari.models.validator import load as load_validator

        v = load_validator(settings=settings)
        out["versions"]["alarm_validator"] = v.version
        out["validator"] = v.predict_window(start, end, settings)["valid_prob"].to_numpy(float)
    except Exception as exc:  # noqa: BLE001 - model absent or unreadable: skip that feature
        out["validator_error"] = repr(exc)
    try:
        from prahari.models.rca import load as load_rca
        from prahari.pipeline.features import build_node_window_features

        r = load_rca(settings=settings)
        out["versions"]["rca_gnn"] = r.version
        store = load_kpi(engine, net, start - pd.Timedelta(minutes=70), end)
        al = load_alarms(engine, start, end)
        conf = []
        for t in pd.date_range(start, end - 15 * _MIN, freq="15min"):
            te = t + 15 * _MIN
            if ((al["ts"] >= t) & (al["ts"] < te)).sum() < 3:
                continue
            f = build_node_window_features(t, te, settings, net=net, store=store, alarms=al)
            res = r.predict({"start": t, "end": te, "features": f})
            conf.append(res["root_candidates"][0]["prob"])
        out["rca"] = np.array(conf, float)
    except Exception as exc:  # noqa: BLE001
        out["rca_error"] = repr(exc)
    return out


def build_reference(
    settings: Settings | None = None,
    n_days: int = 20,
    n_output_days: int = 30,
    with_predictions=True,
) -> dict:
    """Compute and save reference statistics from the train (features) and val (outputs) periods."""
    settings = settings or get_settings()
    engine = get_engine(settings)
    net = NetworkModel.from_db(engine)
    sp = load_splits(settings)["boundaries"]
    t_end, v_end = pd.Timestamp(sp["train_end"]), pd.Timestamp(sp["val_end"])
    xs, gs, vs = [], [], []
    for d in _days(pd.Timestamp(SIM_START), t_end, n_days):
        x, g, v = _rows(engine, net, d, d + pd.Timedelta(days=1))
        xs.append(x), gs.append(g), vs.append(v)
    x, g, ven = np.vstack(xs), np.concatenate(gs), np.concatenate(vs)
    feats: dict = {}
    for j, name in enumerate(FEATURES):
        col = x[:, j]
        ok = np.isfinite(col)
        edges = quantile_edges(col[ok])
        nb = len(edges) + 1
        props = np.zeros((N_GROUPS, nb))
        vprops = np.zeros((len(VENDORS), N_GROUPS, nb))
        grid = []
        for gi in range(N_GROUPS):
            m = ok & (g == gi)
            props[gi] = bin_props(col[m], edges) if m.any() else 0
            grid.append(
                np.quantile(col[m], np.linspace(0, 1, GRID)).round(5).tolist() if m.any() else []
            )
            for vi in range(len(VENDORS)):
                mv = m & (ven == vi)
                vprops[vi, gi] = bin_props(col[mv], edges) if mv.any() else 0
        feats[name] = {
            "edges": edges.tolist(),
            "props": props.tolist(),
            "vendor_props": vprops.tolist(),
            "grid": grid,
        }
    ref = {
        "built_at": datetime.now(UTC).isoformat(),
        "features": feats,
        "n_rows": int(len(x)),
        "feature_period": [str(SIM_START), sp["train_end"]],
        "predictions": {},
        "meta": {"n_days": n_days, "seed_free": True},
    }
    if with_predictions:
        vdays = _days(pd.Timestamp(SIM_START), v_end, n_output_days)
        vp, rp, versions = [], [], {}
        for d in vdays:
            o = _model_outputs(settings, net, d, d + pd.Timedelta(days=1))
            versions.update(o["versions"])
            if o["validator"] is not None:
                vp.append(o["validator"])
            if o["rca"] is not None:
                rp.append(o["rca"])
        edges = [0.2, 0.4, 0.6, 0.8]
        for name, arrs in (("pred_validator_prob", vp), ("pred_rca_top1_conf", rp)):
            if arrs:
                a = np.concatenate(arrs)
                ref["predictions"][name] = {
                    "edges": edges,
                    "props": bin_props(a, np.array(edges)).tolist(),
                    "sample": np.quantile(a, np.linspace(0, 1, 1000)).round(5).tolist(),
                    "n": int(len(a)),
                }
        ref["output_period"] = [str(SIM_START), v_end.isoformat()]
        ref["model_versions"] = versions
    path = reference_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref))
    return ref


def load_reference(settings: Settings | None = None) -> dict:
    """Load the saved reference, building it first if it does not exist yet."""
    settings = settings or get_settings()
    p = reference_path(settings)
    if not p.exists():
        return build_reference(settings)
    return json.loads(p.read_text())


def _window(window, engine) -> tuple[pd.Timestamp, pd.Timestamp]:
    if window is None:
        end = pd.Timestamp(pd.read_sql("SELECT MAX(ts) AS m FROM kpi_samples", engine)["m"][0])
        return end.floor("min") + _MIN - pd.Timedelta(hours=6), end.floor("min") + _MIN
    if isinstance(window, dict):
        return pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
    s, e = window
    return pd.Timestamp(s), pd.Timestamp(e)


def run_drift(window=None, settings: Settings | None = None, store: bool = True) -> dict:
    """Compare ``window`` with the reference and return a ``DriftReportOut``-shaped dict.

    ``window`` is ``(start, end)``, ``{"start", "end"}`` or None (last 6 h; long windows dilute a
    short drift).
    Per-feature PSI (status green < 0.1 <= amber <= 0.2 < red) and KS p-value; prediction drift
    for validator probabilities and RCA top-1 confidence. The report is saved to ``drift_reports``
    when ``store`` is true and its row id is returned in ``id``.
    """
    settings = settings or get_settings()
    engine = get_engine(settings)
    Base.metadata.create_all(engine, tables=[DriftReport.__table__])
    start, end = _window(window, engine)
    net = NetworkModel.from_db(engine)
    ref = load_reference(settings)
    x, g, ven = _rows(engine, net, start, end)
    w = np.bincount(g, minlength=N_GROUPS) / max(len(g), 1)
    rows, ks_stats = [], {}
    for j, name in enumerate(FEATURES):
        r = ref["features"][name]
        edges = np.array(r["edges"])
        cur = x[:, j]
        exp = _mixture_props(np.array(r["props"]), w)
        value = psi_from_props(exp, bin_props(cur, edges)) if len(cur) >= MIN_ROWS else 0.0
        d, p = ks_2samp(_ref_sample([np.array(q) for q in r["grid"]], w), cur)
        ks_stats[name] = round(d, 4)
        rows.append(
            {
                "feature": name,
                "psi": round(value, 4),
                "ks_pvalue": float(p),
                "status": status_from_psi(value),
            }
        )
    notes: list[str] = []
    vendor_psi: dict[str, float] = {}
    if len(x) >= MIN_ROWS:
        r = ref["features"]["latency_ms"]
        edges = np.array(r["edges"])
        vp = np.array(r["vendor_props"])
        for vi, vname in enumerate(VENDORS):
            m = ven == vi
            if m.sum() >= MIN_ROWS:
                vendor_psi[vname] = round(
                    psi_from_props(
                        _mixture_props(vp[vi], np.bincount(g[m], minlength=N_GROUPS) / m.sum()),
                        bin_props(x[m, 0], edges),
                    ),
                    4,
                )
    outs = _model_outputs(settings, net, start, end) if ref.get("predictions") else {}
    n_pred = {}
    for name, key in (("pred_validator_prob", "validator"), ("pred_rca_top1_conf", "rca")):
        pr, cur = ref.get("predictions", {}).get(name), outs.get(key)
        if pr is None or cur is None:
            continue
        n_pred[name] = int(len(cur))
        ok = len(cur) >= MIN_PRED_N
        value = (
            psi_from_props(np.array(pr["props"]), bin_props(cur, np.array(pr["edges"])))
            if ok
            else 0.0
        )
        d, p = ks_2samp(np.array(pr["sample"]), cur)
        ks_stats[name] = round(d, 4)
        rows.append(
            {
                "feature": name,
                "psi": round(value, 4),
                "ks_pvalue": float(p),
                "status": status_from_psi(value),
            }
        )
        if not ok:
            notes.append(f"{name}: only {len(cur)} values (<{MIN_PRED_N}), not scored")
    for mname, ver in outs.get("versions", {}).items():
        if ref.get("model_versions", {}).get(mname) not in (None, ver):
            notes.append(
                f"reference outputs came from {mname} {ref['model_versions'][mname]}, "
                f"serving {ver}: rebuild the reference after retraining"
            )
    status = worst([r["status"] for r in rows]) if len(x) >= MIN_ROWS else "green"
    bad = sorted((r for r in rows if r["status"] != "green"), key=lambda r: -r["psi"])
    if len(x) < MIN_ROWS:
        summary = f"insufficient data in window ({len(x)} KPI rows); not scored"
    elif not bad:
        top = max(rows, key=lambda r: r["psi"])
        summary = f"No drift: largest PSI {top['psi']:.3f} ({top['feature']})."
    else:
        summary = f"{status.upper()}: " + ", ".join(
            f"{r['feature']} PSI {r['psi']:.2f}" for r in bad[:4]
        )
        if vendor_psi:
            tv = max(vendor_psi, key=vendor_psi.__getitem__)
            if vendor_psi[tv] >= 0.2:
                summary += f"; latency shift concentrated on vendor {tv} (PSI {vendor_psi[tv]:.2f})"
    report = {
        "id": 0,
        "ts": end.to_pydatetime(),
        "status": status,
        "features": rows,
        "summary": summary,
        "extra": {
            "window": {"start": start.isoformat(), "end": end.isoformat()},
            "n_kpi_rows": int(len(x)),
            "ks_statistic": ks_stats,
            "vendor_latency_psi": vendor_psi,
            "n_predictions": n_pred,
            "notes": notes,
            "thresholds": {"amber": 0.1, "red": 0.2},
            "reference_built_at": ref["built_at"],
            "reference_model_versions": ref.get("model_versions", {}),
        },
    }
    if store:
        from sqlalchemy.orm import Session

        stored = json.loads(json.dumps(report, default=lambda o: o.isoformat()))
        with Session(engine) as s:
            row = DriftReport(ts=report["ts"], status=status, report_json=stored)
            s.add(row)
            s.flush()
            report["id"] = row.id
            row.report_json = {**stored, "id": row.id}
            s.commit()
    return report


def latest_report(settings: Settings | None = None) -> dict | None:
    """Most recent stored report (as written by :func:`run_drift`), or None."""
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    engine = get_engine(settings or get_settings())
    Base.metadata.create_all(engine, tables=[DriftReport.__table__])
    with Session(engine) as s:
        row = s.scalars(select(DriftReport).order_by(DriftReport.id.desc()).limit(1)).first()
        return None if row is None else {**row.report_json, "id": row.id}


def main() -> None:
    """CLI: ``build-reference`` or ``run --start S --end E``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-reference", "run"])
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    if a.cmd == "build-reference":
        r = build_reference()
        print(f"reference saved: {r['n_rows']} rows, outputs {list(r['predictions'])}")
    else:
        rep = run_drift((a.start, a.end) if a.start else None)
        print(rep["status"], rep["summary"])


if __name__ == "__main__":
    main()
