"""Demand-derived features: target-anchored seasonal lags and origin-anchored state.

All functions take a *dense* hourly series (one row per hour, no gaps) for one station.
Shifts are therefore in rows == hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MAX_HORIZON = 24
SEASONAL_LAGS = (24, 48, 168)


def target_lags(y: pd.Series) -> pd.DataFrame:
    """Seasonal lags evaluated at the target hour t.

    Every lag used here is >= 24 hours, so for any horizon h <= 24 the value y(t - lag)
    satisfies t - lag = T + h - lag <= T: it is observed at the forecast origin.
    """
    assert min(SEASONAL_LAGS) >= MAX_HORIZON
    out = pd.DataFrame(index=y.index)
    out["lag_24h"] = y.shift(24)
    out["lag_48h"] = y.shift(48)
    out["lag_168h"] = y.shift(168)
    out["same_hour_7d_avg"] = pd.concat([y.shift(24 * k) for k in range(1, 8)], axis=1).mean(axis=1, skipna=True)
    out["same_how_4w_mean"] = pd.concat([y.shift(168 * k) for k in range(1, 5)], axis=1).mean(axis=1, skipna=True)
    return out


def _hours_since_nonzero(y: np.ndarray, cap: int = 8760) -> np.ndarray:
    idx = np.arange(len(y), dtype=float)
    last = np.where(np.nan_to_num(y, nan=0.0) > 0, idx, np.nan)
    last = pd.Series(last).ffill().to_numpy()
    out = idx - last
    return np.where(np.isnan(out), cap, np.minimum(out, cap))


def origin_state(st: pd.DataFrame, total_ports: float, ewma_alpha: float = 0.3) -> pd.DataFrame:
    """Features describing the station at the end of each hour T, using only rows <= T."""
    y = st["energy_kwh"]
    out = pd.DataFrame(index=st.index)
    out["y_org_0"] = y
    out["y_org_1"] = y.shift(1)
    out["y_org_2"] = y.shift(2)
    out["hours_since_nonzero"] = _hours_since_nonzero(y.to_numpy())
    r24 = y.rolling(24, min_periods=24)
    out["rolling_mean_24h"] = r24.mean()
    out["rolling_std_24h"] = r24.std(ddof=0)
    out["rolling_max_24h"] = r24.max()
    out["ewma_6h"] = y.ewm(alpha=ewma_alpha, adjust=False, ignore_na=True).mean()
    out["level_28d"] = y.rolling(672, min_periods=168).mean()
    out["level_ratio"] = out["rolling_mean_24h"] / (out["level_28d"] + 0.1)

    def col(name: str) -> pd.Series:
        return st[name] if name in st else pd.Series(np.nan, index=st.index)

    active = col("active_sessions")
    occupied = col("occupied_ports")
    out["active_sessions_T"] = active
    out["occupied_ports_T"] = occupied
    out["idle_ports_T"] = (occupied - active).clip(lower=0)
    out["arrivals_T"] = col("arrivals")
    out["headroom_ports_T"] = (total_ports - occupied).clip(lower=0)
    sat = col("is_saturated").astype(float)
    out["saturation_hours_24h"] = sat.rolling(24, min_periods=1).sum()
    return out
