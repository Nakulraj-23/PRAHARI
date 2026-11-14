"""Retrain a model into a staging directory, compare with the current card, promote if not worse.

``retrain("alarm_validator")`` calls the model package's ``train(settings)`` with
``artifacts_dir`` pointed at a staging folder (so the live artifact is never touched while
training), compares the new card's metrics with the current card's, and copies the new version
into the live artifacts directory only when no tracked metric got worse by more than ``tol``.
The comparison is written to ``docs/retrain/<name>-<new version>.md`` and ``.json`` either way.

Note: models train on the chronological splits in ``artifacts/splits.json`` of the current DB, so
"recent data" means whatever is in the DB now; the test split (and so the metrics) stays the same
for old and new versions, which is what makes the comparison fair.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from prahari.config import ROOT, Settings, get_settings

# metric name -> (path inside card["metrics"], higher_is_better)
SPECS: dict[str, dict] = {
    "alarm_validator": {
        "module": "prahari.models.validator",
        "metrics": {
            "pr_auc": (("pr_auc",), True),
            "f1": (("f1",), True),
            "recall": (("recall",), True),
            "brier": (("brier",), False),
        },
    },
    "anomaly_iforest": {
        "module": "prahari.models.anomaly",
        "metrics": {"pr_auc": (("pr_auc",), True), "f1": (("f1",), True)},
    },
    "rca_gnn": {
        "module": "prahari.models.rca",
        "metrics": {
            "root_top1": (("test", "root_top1"), True),
            "root_top3": (("test", "root_top3"), True),
            "fault_macro_f1": (("test", "fault_macro_f1"), True),
        },
    },
}


def _dig(d: dict, path: tuple[str, ...]) -> float:
    for k in path:
        d = d[k]
    return float(d)


def compare_cards(old: dict, new: dict, name: str, tol: float = 0.002) -> dict:
    """Metric-by-metric comparison. ``promote`` is true when none is worse by more than ``tol``."""
    rows = []
    for m, (path, higher) in SPECS[name]["metrics"].items():
        o, n = _dig(old["metrics"], path), _dig(new["metrics"], path)
        delta = n - o if higher else o - n  # positive = new is better
        rows.append(
            {
                "metric": m,
                "old": o,
                "new": n,
                "higher_is_better": higher,
                "delta_better": round(delta, 6),
                "worse": delta < -tol,
            }
        )
    worse = [r["metric"] for r in rows if r["worse"]]
    return {
        "model": name,
        "old_version": old["version"],
        "new_version": new["version"],
        "tolerance": tol,
        "metrics": rows,
        "worse_metrics": worse,
        "promote": not worse,
    }


def _current_version(settings: Settings, name: str) -> str:
    root = Path(settings.artifacts_dir) / name
    if (root / "latest").exists():
        return (root / "latest").read_text().strip()
    return sorted(p.name for p in root.glob("*") if (p / "model.joblib").exists())[-1]


def _new_version(staging: Path, name: str) -> str:
    cards = sorted((staging / name).glob("*.card.json"), key=lambda p: p.stat().st_mtime)
    return cards[-1].name.removesuffix(".card.json")


def render(cmp: dict, promoted: bool) -> str:
    """Markdown for the comparison."""
    L = [
        f"# Retrain {cmp['model']}: {cmp['old_version']} -> {cmp['new_version']}",
        "",
        f"Decision: **{'promoted' if promoted else 'NOT promoted'}** "
        f"(promote when no metric is worse than the tolerance {cmp['tolerance']}).",
        "",
        "| metric | current | retrained | change (+ is better) | worse |",
        "|---|---|---|---|---|",
    ]
    for r in cmp["metrics"]:
        L.append(
            f"| {r['metric']} | {r['old']:.4f} | {r['new']:.4f} | {r['delta_better']:+.4f} | "
            f"{'yes' if r['worse'] else 'no'} |"
        )
    return "\n".join(L) + "\n"


def retrain(
    name: str,
    settings: Settings | None = None,
    tol: float = 0.002,
    train_kwargs: dict | None = None,
    dry_run: bool = False,
    out_dir: Path | None = None,
) -> dict:
    """Retrain ``name`` into staging, compare with the current card, promote if not worse."""
    settings = settings or get_settings()
    live = Path(settings.artifacts_dir)
    out_dir = out_dir or ROOT / "docs" / "retrain"
    mod = importlib.import_module(SPECS[name]["module"])
    old_ver = _current_version(settings, name)
    old_card = json.loads((live / name / f"{old_ver}.card.json").read_text())
    with tempfile.TemporaryDirectory(prefix="retrain_", dir=live) as tmp:
        staging = Path(tmp)
        for f in live.glob("splits.json"):
            shutil.copy(f, staging / f.name)
        t0 = datetime.now(UTC)
        mod.train(settings.model_copy(update={"artifacts_dir": staging}), **(train_kwargs or {}))
        new_ver = _new_version(staging, name)
        new_card = json.loads((staging / name / f"{new_ver}.card.json").read_text())
        cmp = compare_cards(old_card, new_card, name, tol)
        cmp["trained_at"] = t0.isoformat()
        promoted = cmp["promote"] and not dry_run
        if promoted:
            shutil.copytree(staging / name / new_ver, live / name / new_ver)
            shutil.copy(staging / name / f"{new_ver}.card.json", live / name)
            if (live / name / "latest").exists():
                (live / name / "latest").write_text(new_ver)
    cmp["promoted"] = promoted
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}-{new_ver}.json").write_text(json.dumps(cmp, indent=1))
    (out_dir / f"{name}-{new_ver}.md").write_text(render(cmp, promoted))
    if promoted:  # model outputs changed, so the drift reference for outputs is stale
        from prahari.monitoring.drift import build_reference

        build_reference(settings)
    return cmp


def main() -> None:
    """CLI: ``python -m prahari.monitoring.retrain alarm_validator [--dry-run]``."""
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(SPECS))
    ap.add_argument("--tol", type=float, default=0.002)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    c = retrain(a.model, tol=a.tol, dry_run=a.dry_run)
    print(render(c, c["promoted"]))


if __name__ == "__main__":
    main()
