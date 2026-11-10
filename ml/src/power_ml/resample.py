"""Session log -> gap-free hourly panel per site (report Section 4.3).

* energy is prorated over each session's *charging* window [connect, charge_end);
* occupancy and idle port-hours use the connection window [connect, disconnect) and the
  idle window [charge_end, disconnect);
* end-of-hour counts (active sessions, occupied ports) are instantaneous states at tau + 1h;
* every site is reindexed to a complete hourly grid from its first session onwards and
  hours without activity are true zeros.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from power_core.timeutils import HOUR, hourly_grid

NS_PER_HOUR = 3_600_000_000_000


def _to_ns(s: pd.Series) -> np.ndarray:
    return s.dt.tz_convert("UTC").astype("int64").to_numpy()


def interval_hour_overlap(station: np.ndarray, start_ns: np.ndarray, end_ns: np.ndarray,
                          weight: np.ndarray | None = None) -> pd.DataFrame:
    """Split intervals into hour buckets. Returns (station_id, hour_ns, value) where value is
    overlap hours, multiplied by ``weight`` per hour if given (e.g. kW -> kWh)."""
    ok = end_ns > start_ns
    station, start_ns, end_ns = station[ok], start_ns[ok], end_ns[ok]
    w = np.ones(ok.sum()) if weight is None else np.asarray(weight, float)[ok]
    h0 = (start_ns // NS_PER_HOUR) * NS_PER_HOUR
    h1 = -((-end_ns) // NS_PER_HOUR) * NS_PER_HOUR  # ceil
    n = ((h1 - h0) // NS_PER_HOUR).astype(np.int64)
    rep = np.repeat(np.arange(len(h0)), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    hs = h0[rep] + offs * NS_PER_HOUR
    ov = (np.minimum(end_ns[rep], hs + NS_PER_HOUR) - np.maximum(start_ns[rep], hs)) / NS_PER_HOUR
    df = pd.DataFrame({"station_id": station[rep], "hour_ns": hs, "value": np.clip(ov, 0, 1) * w[rep]})
    return df.groupby(["station_id", "hour_ns"], sort=False, as_index=False)["value"].sum()


def _state_at_hour_end(starts: np.ndarray, ends: np.ndarray, hour_ns: np.ndarray) -> np.ndarray:
    """Number of intervals with start < e < end at e = hour + 1h (intervals with end > start)."""
    e = hour_ns + NS_PER_HOUR
    s_sorted, e_sorted = np.sort(starts), np.sort(ends)
    return np.searchsorted(s_sorted, e, side="left") - np.searchsorted(e_sorted, e, side="right")


def sessions_to_hourly(sessions: pd.DataFrame, stations: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """Build the hourly panel. ``end`` is the last hour (inclusive) of the grid."""
    sid = sessions["station_id"].astype(str).to_numpy()
    c, b, d = _to_ns(sessions["connect_time"]), _to_ns(sessions["charge_end_time"]), _to_ns(sessions["disconnect_time"])
    e_kwh = sessions["energy_kwh"].to_numpy(float)
    ch_h = (b - c) / NS_PER_HOUR
    rate = np.divide(e_kwh, ch_h, out=np.zeros_like(e_kwh), where=ch_h > 0)

    energy = interval_hour_overlap(sid, c, b, rate).rename(columns={"value": "energy_kwh"})
    occ = interval_hour_overlap(sid, c, d).rename(columns={"value": "occupied_port_hours"})
    idle = interval_hour_overlap(sid, b, d).rename(columns={"value": "idle_port_hours"})
    arr = (pd.DataFrame({"station_id": sid, "hour_ns": (c // NS_PER_HOUR) * NS_PER_HOUR})
           .groupby(["station_id", "hour_ns"], as_index=False).size().rename(columns={"size": "arrivals"}))

    meta = stations.set_index("station_id")
    frames = []
    for s in sorted(set(sid)):
        first = pd.Timestamp(meta.loc[s, "first_seen"]).tz_convert("UTC").floor("h")
        grid = hourly_grid(first, end)
        if len(grid) == 0:
            continue
        g = pd.DataFrame({"station_id": s, "hour_ns": grid.asi8})
        for part in (energy, occ, idle, arr):
            g = g.merge(part.loc[part["station_id"] == s], on=["station_id", "hour_ns"], how="left")
        g = g.fillna({"energy_kwh": 0.0, "occupied_port_hours": 0.0, "idle_port_hours": 0.0, "arrivals": 0})
        m = sid == s
        hour_ns = g["hour_ns"].to_numpy()
        act = ch_h[m] > 0
        g["active_sessions"] = _state_at_hour_end(c[m][act], b[m][act], hour_ns)
        g["occupied_ports"] = _state_at_hour_end(c[m], d[m], hour_ns)
        n_ports = float(meta.loc[s, "total_ports"])
        g["occ_ratio"] = (g["occupied_port_hours"] / n_ports).clip(0, 1)
        g["idle_ratio"] = np.where(g["occupied_port_hours"] > 0,
                                   (g["idle_port_hours"] / g["occupied_port_hours"]).clip(0, 1), np.nan)
        g["is_saturated"] = g["occ_ratio"] * n_ports >= n_ports - 0.5
        frames.append(g)

    panel = pd.concat(frames, ignore_index=True)
    panel["ts"] = pd.to_datetime(panel["hour_ns"], utc=True)
    panel["arrivals"] = panel["arrivals"].astype(int)
    cols = ["station_id", "ts", "energy_kwh", "occ_ratio", "idle_ratio", "active_sessions", "occupied_ports",
            "arrivals", "is_saturated", "occupied_port_hours", "idle_port_hours"]
    return panel[cols].sort_values(["station_id", "ts"]).reset_index(drop=True)


def restrict(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    s, e = pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")
    return panel.loc[(panel["ts"] >= s) & (panel["ts"] <= e)].reset_index(drop=True)


__all__ = ["HOUR", "interval_hour_overlap", "restrict", "sessions_to_hourly"]
