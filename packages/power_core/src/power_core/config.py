"""Typed loading of the YAML files in ``configs/``."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (or CWD) to the first directory containing ``configs/``."""
    env = os.environ.get("POWER_ROOT")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for p in (here, *here.parents):
        if (p / "configs").is_dir() and (p / "pyproject.toml").is_file():
            return p
    raise FileNotFoundError("Could not locate the repository root (a folder with configs/ and pyproject.toml)")


# ---------------------------------------------------------------- data.yaml
class PaloAltoCfg(BaseModel):
    raw_path: str
    download_url: str | None = None
    site_regex: str
    site_prefix: str = "PA_"
    port_kw: dict[str, float]
    default_port_kw: float = 6.6
    poi_default: str = "downtown_public"
    poi_overrides: dict[str, str] = Field(default_factory=dict)


class AcnCfg(BaseModel):
    raw_path: str
    site_id: str = "caltech"
    port_kw: float = 6.6
    latitude: float = 34.1377
    longitude: float = -118.1253
    poi: str = "workplace"


class WindowCfg(BaseModel):
    start: str
    end: str
    ood_start: str
    ood_end: str


class SplitCfg(BaseModel):
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    embargo_hours: int = 168

    @field_validator("test_frac")
    @classmethod
    def _sum_to_one(cls, v: float, info: Any) -> float:
        total = info.data.get("train_frac", 0) + info.data.get("val_frac", 0) + v
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"split fractions must sum to 1, got {total}")
        return v


class CleaningCfg(BaseModel):
    max_duration_hours: float = 48
    min_failed_duration_minutes: float = 5
    power_tolerance: float = 1.10
    min_sessions_per_day_public: float = 1.0


class WeatherCfg(BaseModel):
    archive_url: str
    forecast_url: str
    hourly: list[str]
    cache_path: str
    interpolate_max_gap_hours: int = 3


class DataConfig(BaseModel):
    timezone: str = "America/Los_Angeles"
    palo_alto: PaloAltoCfg
    acn: AcnCfg
    window: WindowCfg
    split: SplitCfg
    cleaning: CleaningCfg
    weather: WeatherCfg
    pricing_changes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------- features.yaml
class WeatherAugCfg(BaseModel):
    enabled: bool = True
    temp_sigma_base_c: float = 0.8
    temp_sigma_per_hour_c: float = 0.05
    precip_log_sigma: float = 0.3
    seed: int = 42


class FeatureConfig(BaseModel):
    horizons: tuple[int, int] = (1, 24)
    origin_stride_hours: int = 1
    warmup_hours: int = 672
    ewma_alpha: float = 0.3
    cold_stress_c: float = 10.0
    heat_stress_c: float = 28.0
    rain_threshold_mm: float = 0.2
    weather_augmentation: WeatherAugCfg = Field(default_factory=WeatherAugCfg)

    @property
    def horizon_list(self) -> list[int]:
        lo, hi = self.horizons
        if not 1 <= lo <= hi <= 24:
            raise ValueError("horizons must satisfy 1 <= lo <= hi <= 24 (seasonal lags assume h <= 24)")
        return list(range(lo, hi + 1))


# ---------------------------------------------------------------- thresholds.yaml
class AlertCfg(BaseModel):
    warning_raise: float = 0.70
    warning_clear: float = 0.65
    critical_raise: float = 0.85
    critical_clear: float = 0.80
    clear_consecutive: int = 2


class MapColoursCfg(BaseModel):
    green_below: float = 0.50
    red_at_or_above: float = 0.85


class RerouteCfg(BaseModel):
    radius_km: float = 3.0
    divertible_share: float = 0.30
    target_safe_ratio: float = 0.70


class AciCfg(BaseModel):
    gamma: float = 0.005
    alpha_min: float = 0.01
    alpha_max: float = 0.50


class DriftCfg(BaseModel):
    psi_threshold: float = 0.20
    page_hinkley_delta: float = 0.05
    page_hinkley_lambda: float = 50.0


class ThresholdConfig(BaseModel):
    diversity_factor: float = 0.70
    interval_level: float = 0.90
    horizon_buckets: list[tuple[int, int]] = Field(default_factory=lambda: [(1, 3), (4, 12), (13, 24)])
    alert: AlertCfg = Field(default_factory=AlertCfg)
    map_colours: MapColoursCfg = Field(default_factory=MapColoursCfg)
    reroute: RerouteCfg = Field(default_factory=RerouteCfg)
    aci: AciCfg = Field(default_factory=AciCfg)
    drift: DriftCfg = Field(default_factory=DriftCfg)


def _load(name: str, root: Path | None) -> dict[str, Any]:
    path = (root or find_repo_root()) / "configs" / name
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@lru_cache(maxsize=4)
def data_config(root: Path | None = None) -> DataConfig:
    return DataConfig.model_validate(_load("data.yaml", root))


@lru_cache(maxsize=4)
def feature_config(root: Path | None = None) -> FeatureConfig:
    return FeatureConfig.model_validate(_load("features.yaml", root))


@lru_cache(maxsize=4)
def threshold_config(root: Path | None = None) -> ThresholdConfig:
    return ThresholdConfig.model_validate(_load("thresholds.yaml", root))
