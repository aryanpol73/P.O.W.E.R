"""Single source of truth for every model feature: its name, anchor class and model membership.

Anchor classes (see docs/01_audit_and_refinement.md, flaw F1):

* ``TGT`` - known in advance for the *target* hour ``t = T + h`` (calendar, weather forecast,
  seasonal lags whose lag is >= h).
* ``ORG`` - state observed at the forecast *origin* ``T`` (recent demand, rolling stats, occupancy).
* ``STA`` - static station metadata.
* ``HOR`` - the horizon ``h`` itself.

The builder computes features exactly according to these classes and the leakage tests in
``tests/test_leakage.py`` prove that no ORG/TGT feature reads demand after ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Anchor(StrEnum):
    HOR = "HOR"
    TGT = "TGT"
    ORG = "ORG"
    STA = "STA"


RIDGE, RF, GBDT, LSTM, PRIOR = "ridge", "rf", "gbdt", "lstm", "prior"
TABULAR = frozenset({RIDGE, RF, GBDT})
TREES = frozenset({RF, GBDT})
ALL = frozenset({RIDGE, RF, GBDT, LSTM, PRIOR})


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    anchor: Anchor
    group: str
    models: frozenset[str]
    categorical: bool = False
    description: str = ""


def _f(name: str, anchor: Anchor, group: str, models: frozenset[str] | set[str], desc: str, cat: bool = False) -> FeatureSpec:
    return FeatureSpec(name, anchor, group, frozenset(models), cat, desc)


FEATURES: tuple[FeatureSpec, ...] = (
    _f("h", Anchor.HOR, "horizon", TABULAR, "forecast horizon in hours (1..24)"),
    # --- calendar (target hour, local time)
    _f("hour_sin", Anchor.TGT, "calendar", ALL, "sin(2*pi*hour/24)"),
    _f("hour_cos", Anchor.TGT, "calendar", ALL, "cos(2*pi*hour/24)"),
    _f("dow_sin", Anchor.TGT, "calendar", ALL, "sin(2*pi*dow/7)"),
    _f("dow_cos", Anchor.TGT, "calendar", ALL, "cos(2*pi*dow/7)"),
    _f("doy_sin", Anchor.TGT, "calendar", TABULAR | {PRIOR}, "sin(2*pi*doy/365.25)"),
    _f("doy_cos", Anchor.TGT, "calendar", TABULAR | {PRIOR}, "cos(2*pi*doy/365.25)"),
    _f("is_weekend", Anchor.TGT, "calendar", ALL, "Saturday or Sunday"),
    _f("is_holiday", Anchor.TGT, "calendar", ALL, "US + California public holiday"),
    _f("is_bridge_day", Anchor.TGT, "calendar", TREES | {PRIOR}, "weekday between a holiday and a weekend"),
    _f("how", Anchor.TGT, "calendar", {RIDGE}, "hour-of-week 0..167 (one-hot encoded inside the Ridge pipeline)"),
    *(
        _f(f"how_f{k}_{fn}", Anchor.TGT, "calendar", {RIDGE}, f"{fn}(2*pi*{k}*how/168)")
        for k in (1, 2, 3)
        for fn in ("sin", "cos")
    ),
    _f("pricing_era", Anchor.TGT, "calendar", TABULAR, "number of documented pricing changes before t (optional)"),
    # --- seasonal lags anchored at the target hour (valid because lag >= h for h <= 24)
    _f("lag_24h", Anchor.TGT, "lags", TABULAR, "y(t-24)"),
    _f("lag_48h", Anchor.TGT, "lags", TREES, "y(t-48)"),
    _f("lag_168h", Anchor.TGT, "lags", TABULAR, "y(t-168)"),
    _f("same_hour_7d_avg", Anchor.TGT, "lags", TABULAR, "mean of y(t-24k), k=1..7"),
    _f("same_how_4w_mean", Anchor.TGT, "lags", TABULAR, "mean of y(t-168k), k=1..4"),
    # --- recent state at the origin
    _f("y_org_0", Anchor.ORG, "recent", TABULAR, "y(T)"),
    _f("y_org_1", Anchor.ORG, "recent", TABULAR, "y(T-1)"),
    _f("y_org_2", Anchor.ORG, "recent", TABULAR, "y(T-2)"),
    _f("hours_since_nonzero", Anchor.ORG, "recent", TREES, "T minus the last hour with y > 0"),
    _f("rolling_mean_24h", Anchor.ORG, "rolling", TABULAR, "mean y(T-23..T)"),
    _f("rolling_std_24h", Anchor.ORG, "rolling", TREES, "std y(T-23..T)"),
    _f("rolling_max_24h", Anchor.ORG, "rolling", TREES, "max y(T-23..T)"),
    _f("ewma_6h", Anchor.ORG, "rolling", TABULAR, "EWMA of y up to T, alpha=0.3"),
    _f("level_28d", Anchor.ORG, "rolling", TABULAR, "mean y over the last 672 h"),
    _f("level_ratio", Anchor.ORG, "rolling", TREES, "rolling_mean_24h / (level_28d + 0.1)"),
    # --- occupancy state at the origin
    _f("active_sessions_T", Anchor.ORG, "occupancy", TREES, "sessions charging at the end of hour T"),
    _f("occupied_ports_T", Anchor.ORG, "occupancy", TREES, "ports occupied at the end of hour T"),
    _f("idle_ports_T", Anchor.ORG, "occupancy", TREES, "occupied but not charging at the end of hour T"),
    _f("arrivals_T", Anchor.ORG, "occupancy", TREES, "sessions that plugged in during hour T"),
    _f("headroom_ports_T", Anchor.ORG, "occupancy", TREES, "total ports minus occupied ports at T"),
    _f("saturation_hours_24h", Anchor.ORG, "occupancy", TREES, "saturated hours in T-23..T (censoring)"),
    # --- weather at the target hour (archive in training, forecast in serving)
    _f("temperature_2m_c", Anchor.TGT, "weather", ALL, "air temperature at 2 m"),
    _f("precipitation_mm", Anchor.TGT, "weather", ALL, "hourly precipitation"),
    _f("is_rain", Anchor.TGT, "weather", TABULAR | {PRIOR}, "precipitation >= 0.2 mm"),
    _f("cold_stress", Anchor.TGT, "weather", ALL, "max(0, 10 - temperature)"),
    _f("heat_stress", Anchor.TGT, "weather", ALL, "max(0, temperature - 28)"),
    # --- static metadata
    _f("station_id", Anchor.STA, "static", {RIDGE, RF, GBDT, LSTM}, "site identifier", cat=True),
    _f("total_ports", Anchor.STA, "static", TABULAR | {PRIOR}, "number of ports"),
    _f("poi_category", Anchor.STA, "static", {GBDT, PRIOR}, "point-of-interest category", cat=True),
    _f("rated_kw_total", Anchor.STA, "static", {PRIOR}, "sum of port nameplate kW"),
    _f("station_age_days", Anchor.TGT, "static", TREES | {PRIOR}, "days since the site's first session at t"),
)

BY_NAME: dict[str, FeatureSpec] = {f.name: f for f in FEATURES}
ID_COLUMNS = ("station_id", "origin", "target_time", "h")
TARGET_COLUMNS = ("y", "occ_target")


def feature_names(model: str | None = None, *, include_optional: bool = True) -> list[str]:
    """Feature names for a model tag (``ridge``, ``rf``, ``gbdt``, ``lstm``, ``prior``) or all."""
    out = [f.name for f in FEATURES if model is None or model in f.models]
    if not include_optional:
        out = [n for n in out if n != "pricing_era"]
    return out


def categorical_features(model: str | None = None) -> list[str]:
    return [f.name for f in FEATURES if f.categorical and (model is None or model in f.models)]


def by_anchor(anchor: Anchor) -> list[str]:
    return [f.name for f in FEATURES if f.anchor is anchor]
