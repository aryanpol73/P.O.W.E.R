"""Grid-risk logic shared by the alert agent, the API and the UI.

Transformer capacity is a SYNTHETIC planning assumption: ``C_s = DF * sum(port kW)`` (flaw F12).
Hourly kWh equals the mean kW over the hour, so alerts use the upper quantile q95.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from power_core.config import AlertCfg, RerouteCfg
from power_core.geo import haversine_km


class Risk(StrEnum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


def transformer_kw(rated_kw_total: float | np.ndarray, diversity_factor: float) -> float | np.ndarray:
    return np.asarray(rated_kw_total, float) * diversity_factor


def load_ratio(kwh: float | np.ndarray, capacity_kw: float | np.ndarray) -> np.ndarray:
    return np.asarray(kwh, float) / np.asarray(capacity_kw, float)


def risk_from_ratio(ratio_q95: float, cfg: AlertCfg) -> Risk:
    """Stateless classification (used for map snapshots)."""
    if ratio_q95 >= cfg.critical_raise:
        return Risk.CRITICAL
    if ratio_q95 >= cfg.warning_raise:
        return Risk.WARNING
    return Risk.NORMAL


@dataclass
class AlertHysteresis:
    """Stateful per-station risk with separate raise/clear thresholds to stop flapping.

    Raising is immediate; lowering a level requires ``clear_consecutive`` consecutive
    origins below that level's clear threshold.
    """

    cfg: AlertCfg
    level: Risk = Risk.NORMAL
    _below: int = field(default=0, init=False)

    def update(self, ratio_q95: float) -> Risk:
        c = self.cfg
        if ratio_q95 >= c.critical_raise:
            self.level, self._below = Risk.CRITICAL, 0
            return self.level
        if ratio_q95 >= c.warning_raise and self.level is Risk.NORMAL:
            self.level, self._below = Risk.WARNING, 0
            return self.level
        clear = c.critical_clear if self.level is Risk.CRITICAL else c.warning_clear
        if self.level is not Risk.NORMAL and ratio_q95 < clear:
            self._below += 1
            if self._below >= c.clear_consecutive:
                self._below = 0
                if self.level is Risk.CRITICAL:
                    self.level = Risk.WARNING if ratio_q95 >= c.warning_clear else Risk.NORMAL
                else:
                    self.level = Risk.NORMAL
        else:
            self._below = 0
        return self.level


@dataclass(frozen=True)
class Reroute:
    from_station: str
    to_station: str
    distance_km: float
    shifted_kwh: float
    to_headroom_kwh: float


def plan_reroutes(snapshot: pd.DataFrame, cfg: RerouteCfg, critical_ratio: float) -> list[Reroute]:
    """Greedy, headroom-aware reroute plan for one target hour.

    ``snapshot`` columns: station_id, lat, lon, q50, q95, capacity_kw, is_modeled.
    Assignments to a receiving station reduce its remaining headroom, so overflow is never
    pushed into a station that would itself become critical.
    """
    df = snapshot.loc[snapshot["is_modeled"].astype(bool)].copy()
    df["headroom"] = cfg.target_safe_ratio * df["capacity_kw"] - df["q50"]
    remaining = dict(zip(df["station_id"], df["headroom"].clip(lower=0), strict=True))
    out: list[Reroute] = []
    critical = df.loc[df["q95"] >= critical_ratio * df["capacity_kw"]].sort_values("q95", ascending=False)
    for _, s in critical.iterrows():
        need = cfg.divertible_share * (s["q95"] - critical_ratio * s["capacity_kw"])
        if need <= 0:
            continue
        others = df.loc[df["station_id"] != s["station_id"]].copy()
        others["dist"] = haversine_km(s["lat"], s["lon"], others["lat"].to_numpy(), others["lon"].to_numpy())
        others = others.loc[(others["dist"] <= cfg.radius_km) & (others["q95"] < critical_ratio * others["capacity_kw"])]
        others["avail"] = others["station_id"].map(remaining)
        others = others.loc[others["avail"] > 0]
        others["score"] = others["avail"] / (1 + others["dist"])
        for _, o in others.sort_values("score", ascending=False).iterrows():
            if need <= 1e-9:
                break
            move = float(min(need, remaining[o["station_id"]]))
            if move <= 0:
                continue
            remaining[o["station_id"]] -= move
            need -= move
            out.append(Reroute(str(s["station_id"]), str(o["station_id"]), float(o["dist"]), move,
                               float(remaining[o["station_id"]])))
    return out
