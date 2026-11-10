"""Time helpers. Storage is always UTC; calendar features are always local time."""

from __future__ import annotations

import pandas as pd

HOUR = pd.Timedelta(hours=1)


def to_utc(ts: pd.Series | pd.DatetimeIndex | pd.Timestamp | str) -> pd.Series | pd.DatetimeIndex | pd.Timestamp:
    """Convert to tz-aware UTC. Naive inputs are rejected so local times are never misread as UTC."""
    if isinstance(ts, str):
        ts = pd.Timestamp(ts)
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is None:
            raise ValueError(f"naive timestamp {ts!r}: attach a timezone before converting")
        return ts.tz_convert("UTC")
    tz = ts.dt.tz if isinstance(ts, pd.Series) else ts.tz
    if tz is None:
        raise ValueError("naive timestamps: attach a timezone before converting")
    return ts.dt.tz_convert("UTC") if isinstance(ts, pd.Series) else ts.tz_convert("UTC")


def floor_hour(ts: pd.Series) -> pd.Series:
    return ts.dt.floor("h")


def hourly_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Inclusive, gap-free UTC hourly grid. DST-safe because it is built in UTC."""
    start = to_utc(start).floor("h")  # type: ignore[union-attr]
    end = to_utc(end).floor("h")  # type: ignore[union-attr]
    return pd.date_range(start, end, freq="h", tz="UTC")


def local(ts: pd.Series | pd.DatetimeIndex, tz: str) -> pd.Series | pd.DatetimeIndex:
    return ts.dt.tz_convert(tz) if isinstance(ts, pd.Series) else ts.tz_convert(tz)
