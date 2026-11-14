"""Central settings, read from environment variables prefixed PRAHARI_."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings. Override with e.g. PRAHARI_DB_URL, PRAHARI_SEED."""

    model_config = SettingsConfigDict(env_prefix="PRAHARI_", extra="ignore")

    seed: int = 42
    db_url: str = f"sqlite:///{ROOT / 'prahari.db'}"
    artifacts_dir: Path = ROOT / "artifacts"
    runbooks_dir: Path = ROOT / "runbooks"
    fixtures_dir: Path = ROOT / "fixtures"
    policy_path: Path = ROOT / "prahari" / "governance" / "policy.yaml"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    llm_provider: str = "none"  # none | anthropic | openai_compatible (A5 adapter)
    llm_model: str = ""
    confidence_threshold: float = 0.70
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]
    detector_enabled: bool = True
    detector_interval_s: float = 10.0


def get_settings() -> Settings:
    """Return a fresh Settings instance."""
    return Settings()
