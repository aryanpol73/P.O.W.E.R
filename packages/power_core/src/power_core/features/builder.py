"""Direct multi-horizon feature frames, identical for training and serving.

One output row per (station, origin T, horizon h). Features follow the anchor classes in
``registry.py``:

* ORG features are read at row T (data up to and including hour T only);
* TGT features are read at row T + h (calendar, weather, seasonal lags with lag >= h);
* STA features come from the station table.

Training and serving call the same function. For serving, the panel holds observed demand up
to T and rows for T+1..T+24 with demand set to NaN and *forecast* weather filled in.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from power_core.config import FeatureConfig
from power_core.features import registry
from power_core.features.calendar import calendar_features
from power_core.features.demand import origin_state, target_lags
from power_core.features.weather import derive_weather
from power_core.timeutils import HOUR

PANEL_REQUIRED = ("station_id", "ts", "energy_kwh")
STATION_REQUIRED = ("station_id", "total_ports", "rated_kw_total", "poi_category", "first_seen")
WEATHER_RAW = ("temperature_2m_c", "precipitation_mm")


class PanelError(ValueError):
    pass


def _check(panel: pd.DataFrame, stations: pd.DataFrame) -> None:
    missing = [c for c in PANEL_REQUIRED if c not in panel]
    if missing:
        raise PanelError(f"panel is missing columns {missing}")
    missing = [c for c in STATION_REQUIRED if c not in stations]
    if missing:
        raise PanelError(f"stations table is missing columns {missing}")
    if getattr(panel["ts"].dt, "tz", None) is None:
        raise PanelError("panel.ts must be tz-aware UTC")
    if panel.duplicated(["station_id", "ts"]).any():
        raise PanelError("panel has duplicate (station_id, ts) rows")


def _dense_station(panel: pd.DataFrame, sid: str) -> pd.DataFrame:
    st = panel.loc[panel["station_id"] == sid].set_index("ts").sort_index()
    if len(st) > 1:
        gaps = st.index.to_series().diff().dropna()
        if (gaps != HOUR).any():
            raise PanelError(f"station {sid}: panel is not a gap-free hourly grid (reindex with zeros first)")
    for c in WEATHER_RAW:
        if c not in st:
            st[c] = np.nan
    return st


def build_direct_frame(
    panel: pd.DataFrame,
    stations: pd.DataFrame,
    cfg: FeatureConfig,
    *,
    tz: str = "America/Los_Angeles",
    pricing_changes: Sequence[str] = (),
    horizons: Iterable[int] | None = None,
    origins: pd.DatetimeIndex | None = None,
    origin_stride_hours: int | None = None,
    apply_warmup: bool = True,
    drop_missing_target: bool = True,
) -> pd.DataFrame:
    """Build the (station, origin, horizon) design matrix.

    Parameters
    ----------
    origins:
        Restrict to these forecast origins (e.g. a single T when serving).
    drop_missing_target:
        Drop rows whose target y(T+h) is unknown. Set False for serving.

    Weather forecast-error augmentation for training rows is applied afterwards with
    ``power_core.features.weather.augment_frame``.
    """
    _check(panel, stations)
    hs = list(horizons) if horizons is not None else cfg.horizon_list
    if any(h < 1 or h > 24 for h in hs):
        raise ValueError("horizons must lie in 1..24; seasonal lags are only causal for h <= 24")
    stride = origin_stride_hours or cfg.origin_stride_hours

    st_meta = stations.set_index("station_id")
    station_cats = pd.CategoricalDtype(sorted(stations["station_id"].astype(str).unique()))
    poi_cats = pd.CategoricalDtype(sorted(stations["poi_category"].astype(str).unique()))
    want = set(registry.feature_names()) | {"how"}

    frames: list[pd.DataFrame] = []
    for sid in sorted(panel["station_id"].unique()):
        meta = st_meta.loc[sid]
        st = _dense_station(panel, sid)
        if st.empty:
            continue
        n_ports = float(meta["total_ports"])
        first_seen = pd.Timestamp(meta["first_seen"])
        if first_seen.tzinfo is None:
            raise PanelError(f"station {sid}: first_seen must be tz-aware")

        org = origin_state(st, n_ports, cfg.ewma_alpha)
        tgt = pd.concat(
            [calendar_features(st.index, tz, pricing_changes), target_lags(st["energy_kwh"])],
            axis=1,
        )
        tgt["station_age_days"] = (st.index - first_seen.tz_convert("UTC")) / pd.Timedelta(days=1)
        y = st["energy_kwh"]
        occ = st["occ_ratio"] if "occ_ratio" in st else pd.Series(np.nan, index=st.index)
        temp, precip = st["temperature_2m_c"], st["precipitation_mm"]

        idx = st.index
        mask = np.ones(len(idx), dtype=bool)
        if apply_warmup:
            # History available to the features starts at the later of the site's first session and the
            # first panel row (the panel may be cut to a modelling window that starts after first_seen).
            history_start = max(first_seen.tz_convert("UTC").ceil("h"), idx[0])
            mask &= idx >= history_start + pd.Timedelta(hours=cfg.warmup_hours)
        if origins is not None:
            mask &= idx.isin(origins)
        if stride > 1:
            mask &= ((idx - idx[0]) // HOUR) % stride == 0
        if not mask.any():
            continue
        org_T = org.loc[mask]

        for h in hs:
            g = tgt.shift(-h).loc[mask]
            w = derive_weather(temp.shift(-h).loc[mask], precip.shift(-h).loc[mask], cfg)
            block = pd.concat([org_T, g, w], axis=1)
            block.insert(0, "h", np.int8(h))
            block.insert(0, "target_time", org_T.index + h * HOUR)
            block.insert(0, "origin", org_T.index)
            block.insert(0, "station_id", sid)
            block["total_ports"] = n_ports
            block["rated_kw_total"] = float(meta["rated_kw_total"])
            block["poi_category"] = str(meta["poi_category"])
            block["y"] = y.shift(-h).loc[mask].to_numpy()
            block["occ_target"] = occ.shift(-h).loc[mask].to_numpy()
            # Cast now (not after the concat) so peak memory stays at float32 size.
            num = [c for c in block.columns if c not in registry.ID_COLUMNS and c != "poi_category"
                   and block[c].dtype.kind in "iufb"]
            block[num] = block[num].astype(np.float32)
            frames.append(block.reset_index(drop=True))

    if not frames:
        return pd.DataFrame(columns=[*registry.ID_COLUMNS, *registry.feature_names(), *registry.TARGET_COLUMNS])

    out = pd.concat(frames, ignore_index=True)
    if drop_missing_target:
        out = out.loc[out["y"].notna()].reset_index(drop=True)
    out["station_id"] = out["station_id"].astype(str).astype(station_cats)
    out["poi_category"] = out["poi_category"].astype(poi_cats)

    feats = [c for c in registry.feature_names() if c in out.columns and c in want and c not in registry.ID_COLUMNS]
    # Fixed float32 schema for every numeric feature, so dtypes never depend on whether the
    # input happened to contain NaNs (int vs float) - training and serving see identical types.
    float_cols = [c for c in feats if c not in registry.categorical_features() and out[c].dtype.kind in "iufb"]
    out[float_cols] = out[float_cols].astype(np.float32)
    ordered = [*registry.ID_COLUMNS, *feats, *registry.TARGET_COLUMNS]
    return out[ordered].sort_values(["origin", "station_id", "h"], kind="stable").reset_index(drop=True)


def feature_matrix(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Select the columns a given model family consumes (ignoring absent optional features)."""
    cols = [c for c in registry.feature_names(model) if c in frame.columns]
    return frame[cols]
