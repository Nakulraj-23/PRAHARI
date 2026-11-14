"""Static check: feature/serving-side pipeline code never selects label_* or incidents."""

import ast
from pathlib import Path

import prahari.pipeline as pipeline_pkg
from prahari.pipeline.store import ALARM_COLUMNS

PKG = Path(pipeline_pkg.__file__).parent


def _strings(path: Path) -> list[str]:
    return [
        n.value
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def test_alarm_allow_list_has_no_labels():
    assert not any(c.startswith("label_") for c in ALARM_COLUMNS)


def test_features_module_has_no_label_literals_except_drop_filter():
    # docstrings/prose contain spaces; code literals do not. The only code literal is the
    # prefix used by _drop_labels to remove label columns.
    code_lits = {s for s in _strings(PKG / "features.py") if "label_" in s and " " not in s}
    assert code_lits == {"label_"}, code_lits


def test_no_select_star_on_alarms_in_pipeline():
    for py in PKG.glob("*.py"):
        for s in _strings(py):
            assert "SELECT * FROM alarms" not in s.upper().replace("  ", " "), py.name


def test_incidents_table_only_read_by_ground_truth_loader_and_splits():
    readers = {
        py.name
        for py in PKG.glob("*.py")
        if any("FROM incidents" in s for s in _strings(py)) and py.name != "quality.py"
    }
    assert readers <= {"store.py", "splits.py"}, readers
