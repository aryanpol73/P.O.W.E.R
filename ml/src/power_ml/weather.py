"""Open-Meteo historical weather back-fill and the weather merge into the hourly panel."""

from __future__ import annotations

from collections.abc import Mapping

import httpx
import numpy as np
import pandas as pd

VAR_MAP = {"temperature_2m": "temperature_2m_c", "precipitation": "precipitation_mm"}


def fetch_archive(coords: Mapping[str, tuple[float, float]], start: pd.Timestamp, end: pd.Timestamp,
                  url: str, hourly: list[str], client: httpx.Client | None = None,
                  chunk_days: int = 366) -> pd.DataFrame:
    """Hourly archive weather for each station (UTC). One batched request per date chunk."""
    sids = list(coords)
    lat = ",".join(f"{coords[s][0]:.5f}" for s in sids)
    lon = ",".join(f"{coords[s][1]:.5f}" for s in sids)
    own = client is None
    client = client or httpx.Client(timeout=60.0, headers={"User-Agent": "power-ml/0.1"})
    frames = []
    try:
        cur = pd.Timestamp(start).tz_convert("UTC").normalize()
        stop = pd.Timestamp(end).tz_convert("UTC").normalize()
        while cur <= stop:
            nxt = min(cur + pd.Timedelta(days=chunk_days - 1), stop)
            params = {"latitude": lat, "longitude": lon, "start_date": cur.date().isoformat(),
                      "end_date": nxt.date().isoformat(), "hourly": ",".join(hourly), "timezone": "GMT"}
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
            items = payload if isinstance(payload, list) else [payload]
            if len(items) != len(sids):
                raise ValueError(f"Open-Meteo returned {len(items)} locations for {len(sids)} requested")
            for sid, item in zip(sids, items, strict=True):
                h = item["hourly"]
                df = pd.DataFrame({VAR_MAP.get(k, k): v for k, v in h.items() if k != "time"})
                df["ts"] = pd.to_datetime(h["time"], utc=True)
                df["station_id"] = sid
                frames.append(df)
            cur = nxt + pd.Timedelta(days=1)
    finally:
        if own:
            client.close()
    return pd.concat(frames, ignore_index=True)


def interpolate_short_gaps(s: pd.Series, max_gap: int) -> pd.Series:
    """Linear interpolation of interior NaN runs no longer than ``max_gap``; longer runs stay NaN."""
    na = s.isna()
    if not na.any():
        return s
    run_id = (na != na.shift()).cumsum()
    run_len = na.groupby(run_id).transform("sum")
    filled = s.interpolate(limit_area="inside")
    return s.where(~(na & (run_len <= max_gap)), filled)


def merge_weather(panel: pd.DataFrame, weather: pd.DataFrame, *, max_gap_hours: int = 3,
                  climatology_until: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Left-join weather on (station_id, ts); interpolate short gaps; fill long gaps with an
    hour-of-day x month climatology computed only from rows <= ``climatology_until``."""
    cols = [c for c in VAR_MAP.values() if c in weather]
    out = panel.drop(columns=[c for c in cols if c in panel]).merge(
        weather[["station_id", "ts", *cols]], on=["station_id", "ts"], how="left")
    report = {"missing_before": int(out[cols].isna().any(axis=1).sum())}
    parts = []
    for _, g in out.groupby("station_id", sort=False):
        g = g.sort_values("ts").copy()
        for c in cols:
            g[c] = interpolate_short_gaps(g[c], max_gap_hours)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    report["missing_after_interp"] = int(out[cols].isna().any(axis=1).sum())

    if out[cols].isna().any().any():
        ref = out if climatology_until is None else out.loc[out["ts"] <= pd.Timestamp(climatology_until)]
        key_ref = [ref["station_id"], ref["ts"].dt.month, ref["ts"].dt.hour]
        clim = ref.groupby(key_ref)[cols].mean()
        key = pd.MultiIndex.from_arrays([out["station_id"], out["ts"].dt.month, out["ts"].dt.hour])
        filled = clim.reindex(key).to_numpy()
        vals = out[cols].to_numpy()
        out[cols] = np.where(np.isnan(vals), filled, vals)
    report["missing_final"] = int(out[cols].isna().any(axis=1).sum())
    return out, report
