"""Runtime settings from environment variables (see .env.example)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from power_core.config import find_repo_root
from power_core.deployment import Deployment, load_deployment


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # SQLite by default (zero-config local dev); PostgreSQL in production, e.g.
    # postgresql+asyncpg://power:power@db:5432/power  (pip extra: power-api[postgres])
    database_url: str = Field(default="", alias="DATABASE_URL")
    source: str = Field(default="palo_alto", alias="POWER_SOURCE")
    model_version: str = Field(default="", alias="MODEL_VERSION")  # empty = configs/models.yaml version
    champion: str = Field(default="M4_lightgbm_quantile", alias="POWER_CHAMPION")
    sim_mode: str = Field(default="REPLAY", alias="SIM_MODE")  # REPLAY | LIVE_SHADOW
    sim_start: str = Field(default="", alias="SIM_START")  # empty = first hour of the test split
    tick_seconds: float = Field(default=10.0, alias="SIM_TICK_SECONDS")  # wall seconds per simulated hour
    embedded_worker: bool = Field(default=True, alias="POWER_EMBEDDED_WORKER")
    admin_token: str = Field(default="", alias="ADMIN_TOKEN")  # empty = admin endpoints open (dev only)
    history_hours: int = Field(default=720, alias="POWER_HISTORY_HOURS")
    cors_origins: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")
    root: Path | None = Field(default=None, alias="POWER_ROOT")
    # Deployment profile in configs/deployments/ (the live console's city); "none" = the training sites.
    deployment_name: str = Field(default="pune", alias="POWER_DEPLOYMENT")
    # A8: look up other public chargers on OpenStreetMap at start-up (at most once a day).
    discovery_on_start: bool = Field(default=True, alias="POWER_DISCOVERY")

    @property
    def deployment(self) -> Deployment | None:
        return load_deployment(self.repo_root, self.deployment_name)

    @property
    def repo_root(self) -> Path:
        return self.root or find_repo_root()

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        path = self.repo_root / "data" / "power.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{path.as_posix()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
