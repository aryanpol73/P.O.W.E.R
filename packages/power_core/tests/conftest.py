from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from power_core.config import FeatureConfig

TZ = "America/Los_Angeles"


def make_panel(
    n_days: int = 60, stations: tuple[str, ...] = ("S_A", "S_B", "S_C"), seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2019-01-01 08:00", tz="UTC")  # 00:00 local
    ts = pd.date_range(start, periods=24 * n_days, freq="h", tz="UTC")
    local_hour = ts.tz_convert(TZ).hour.to_numpy()
    frames, meta = [], []
    for i, sid in enumerate(stations):
        ports = 4 + 2 * i
        base = np.where((local_hour >= 8) & (local_hour <= 19), 1.0, 0.0) * (6 + 3 * i)
        y = np.clip(base + rng.normal(0, 1.5, len(ts)), 0, None)
        y[rng.random(len(ts)) < 0.15] = 0.0
        occupied = np.clip(np.round(y / 3 + rng.integers(0, 2, len(ts))), 0, ports)
        active = np.minimum(occupied, np.round(y / 4))
        frames.append(pd.DataFrame({
            "station_id": sid, "ts": ts, "energy_kwh": y,
            "occ_ratio": occupied / ports, "active_sessions": active, "occupied_ports": occupied,
            "arrivals": rng.integers(0, 3, len(ts)), "is_saturated": occupied >= ports,
            "temperature_2m_c": 12 + 8 * np.sin(2 * np.pi * (local_hour - 9) / 24) + rng.normal(0, 1, len(ts)),
            "precipitation_mm": np.where(rng.random(len(ts)) < 0.1, rng.exponential(1.0, len(ts)), 0.0),
        }))
        meta.append({"station_id": sid, "total_ports": ports, "rated_kw_total": ports * 6.6,
                     "poi_category": "workplace" if i == 0 else "downtown_public",
                     "first_seen": start, "latitude": 37.44 + 0.005 * i, "longitude": -122.16})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(meta)


@pytest.fixture
def panel_and_stations() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_panel()


@pytest.fixture
def fcfg() -> FeatureConfig:
    return FeatureConfig(warmup_hours=24 * 29)
