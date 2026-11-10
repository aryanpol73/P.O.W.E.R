"""Weather-derived features and forecast-error augmentation (flaw F6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from power_core.config import FeatureConfig


def derive_weather(temp_c: pd.Series, precip_mm: pd.Series, cfg: FeatureConfig) -> pd.DataFrame:
    precip = precip_mm.clip(lower=0)
    return pd.DataFrame(
        {
            "temperature_2m_c": temp_c,
            "precipitation_mm": precip,
            "is_rain": (precip >= cfg.rain_threshold_mm).astype(np.int8).where(precip.notna()),
            "cold_stress": (cfg.cold_stress_c - temp_c).clip(lower=0),
            "heat_stress": (temp_c - cfg.heat_stress_c).clip(lower=0),
        },
        index=temp_c.index,
    )


def augment_frame(frame: pd.DataFrame, cfg: FeatureConfig, seed: int | None = None) -> pd.DataFrame:
    """Simulate forecast error that grows with lead time (training rows only).

    Training uses observed (archive) weather but serving uses forecasts; adding noise with
    sigma(h) = base + slope * h to temperature, and log-normal noise to precipitation, keeps
    the model from over-trusting weather. Derived weather features are recomputed.
    """
    a = cfg.weather_augmentation
    rng = np.random.default_rng(a.seed if seed is None else seed)
    out = frame.copy()
    h = out["h"].to_numpy(dtype=float)
    temp = out["temperature_2m_c"].to_numpy(dtype=float)
    precip = out["precipitation_mm"].to_numpy(dtype=float)
    temp = temp + rng.normal(0.0, 1.0, size=len(out)) * (a.temp_sigma_base_c + a.temp_sigma_per_hour_c * h)
    precip = precip * np.exp(rng.normal(0.0, a.precip_log_sigma, size=len(out)))
    w = derive_weather(pd.Series(temp, index=out.index), pd.Series(precip, index=out.index), cfg)
    for c in w.columns:
        out[c] = w[c].astype(out[c].dtype) if c in out else w[c]
    return out
