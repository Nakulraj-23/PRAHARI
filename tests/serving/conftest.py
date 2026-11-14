"""Fixtures for the API tests: a 1-day simulated DB (seed 42, same topology as the trained models)
copied per test, and a temp artifacts dir that links the real model artifacts."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prahari.config import ROOT, Settings
from prahari.serving.app import create_app

MODELS = ("alarm_validator", "anomaly_iforest", "rca_gnn")


@pytest.fixture(scope="session")
def base_db(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("apidb") / "base.db"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "prahari.sim.generate",
            "--days",
            "1",
            "--no-checks",
            "--db-url",
            f"sqlite:///{path}",
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="session")
def models_ok() -> bool:
    return all((ROOT / "artifacts" / m).exists() for m in MODELS)


def make_settings(tmp: Path, base_db: Path) -> Settings:
    db = tmp / "api.db"
    shutil.copy(base_db, db)
    art = tmp / "artifacts"
    art.mkdir()
    for m in MODELS:
        if (ROOT / "artifacts" / m).exists():
            (art / m).symlink_to(ROOT / "artifacts" / m)
    return Settings(db_url=f"sqlite:///{db}", artifacts_dir=art, detector_enabled=False)


@pytest.fixture
def settings(tmp_path, base_db, models_ok):
    if not models_ok:
        pytest.skip("trained model artifacts not present")
    return make_settings(tmp_path, base_db)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        c.settings = settings
        yield c


@pytest.fixture
def bare_client(tmp_path, base_db):
    """App without models or simulator (503 paths)."""
    s = make_settings(tmp_path, base_db)
    with TestClient(create_app(s, load_models=False, start_sim=False)) as c:
        yield c
