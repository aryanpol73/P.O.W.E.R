"""Calendar features for target hours. Computed in local time; input index is UTC."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from functools import lru_cache

import holidays as holidays_lib
import numpy as np
import pandas as pd

TWO_PI = 2.0 * np.pi


@lru_cache(maxsize=32)
def _holiday_set_cached(years: tuple[int, ...], country: str, subdiv: str | None) -> frozenset[date]:
    return frozenset(holidays_lib.country_holidays(country, subdiv=subdiv, years=list(years)).keys())


def _holiday_set(years: Sequence[int], country: str = "US", subdiv: str | None = "CA") -> frozenset[date]:
    """Public holidays (default US + California, the training data); cached because building the calendar
    costs ~0.1 s per call. A deployment profile passes its own country, e.g. ("IN", "MH")."""
    return _holiday_set_cached(tuple(years), country, subdiv)


def calendar_features(
    ts_utc: pd.DatetimeIndex,
    tz: str,
    pricing_changes: Sequence[str] = (),
    holidays: tuple[str, str | None] = ("US", "CA"),
) -> pd.DataFrame:
    """Return calendar features indexed by the given UTC hours (local hour/day computed in ``tz``)."""
    if ts_utc.tz is None:
        raise ValueError("calendar_features expects a tz-aware UTC index")
    loc = ts_utc.tz_convert(tz)
    hour = loc.hour.to_numpy()
    dow = loc.dayofweek.to_numpy()  # Monday=0
    doy = loc.dayofyear.to_numpy()
    how = dow * 24 + hour

    days = pd.Index(loc.date)
    years = range(min(d.year for d in days) - 1, max(d.year for d in days) + 2) if len(days) else range(2000, 2001)
    hol = _holiday_set(years, *holidays)
    is_hol = np.fromiter((d in hol for d in days), dtype=bool, count=len(days))
    is_wkend = dow >= 5

    def _is_bridge(d: date) -> bool:
        if d.weekday() >= 5 or d in hol:
            return False
        prev, nxt = d - timedelta(days=1), d + timedelta(days=1)
        prev_off = prev in hol or prev.weekday() >= 5
        next_off = nxt in hol or nxt.weekday() >= 5
        return (prev in hol and next_off) or (nxt in hol and prev_off)

    unique_days = {d: _is_bridge(d) for d in set(days)}
    is_bridge = np.fromiter((unique_days[d] for d in days), dtype=bool, count=len(days))

    out = pd.DataFrame(
        {
            "hour_sin": np.sin(TWO_PI * hour / 24),
            "hour_cos": np.cos(TWO_PI * hour / 24),
            "dow_sin": np.sin(TWO_PI * dow / 7),
            "dow_cos": np.cos(TWO_PI * dow / 7),
            "doy_sin": np.sin(TWO_PI * doy / 365.25),
            "doy_cos": np.cos(TWO_PI * doy / 365.25),
            "is_weekend": is_wkend.astype(np.int8),
            "is_holiday": is_hol.astype(np.int8),
            "is_bridge_day": is_bridge.astype(np.int8),
            "how": how.astype(np.int16),
        },
        index=ts_utc,
    )
    for k in (1, 2, 3):
        out[f"how_f{k}_sin"] = np.sin(TWO_PI * k * how / 168)
        out[f"how_f{k}_cos"] = np.cos(TWO_PI * k * how / 168)
    if pricing_changes:
        changes = pd.DatetimeIndex([pd.Timestamp(c) for c in pricing_changes]).tz_convert("UTC").sort_values()
        out["pricing_era"] = np.searchsorted(changes.asi8, ts_utc.asi8, side="right").astype(np.int8)
    return out
