"""Leakage proofs (report Section 4.7 / audit Section 2.6). Run with ``pytest -m leakage``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from power_core.features import Anchor, build_direct_frame
from power_core.features.registry import FEATURES, by_anchor
from power_core.splits import PanelTimeSeriesSplit, assign_split, chronological_bounds

TZ = "America/Los_Angeles"

pytestmark = pytest.mark.leakage

OBSERVED_AT_ORIGIN = ("energy_kwh", "occ_ratio", "active_sessions", "occupied_ports", "arrivals", "is_saturated")


def _poison_after(panel: pd.DataFrame, T: pd.Timestamp, value: float) -> pd.DataFrame:
    """Overwrite every observed (non-forecastable) quantity after T."""
    p = panel.copy()
    future = p["ts"] > T
    for c in OBSERVED_AT_ORIGIN:
        if p[c].dtype == bool:
            p.loc[future, c] = True
        else:
            p.loc[future, c] = value
    return p


@pytest.mark.parametrize("value", [np.nan, 1e9])
@pytest.mark.parametrize("offset_days", [31, 40, 52])
def test_future_poisoning_does_not_change_features(panel_and_stations, fcfg, value, offset_days):
    panel, stations = panel_and_stations
    T = panel["ts"].min() + pd.Timedelta(days=offset_days, hours=13)
    origins = pd.DatetimeIndex([T])
    kw = dict(tz=TZ, origins=origins, drop_missing_target=False)
    clean = build_direct_frame(panel, stations, fcfg, **kw)
    dirty = build_direct_frame(_poison_after(panel, T, value), stations, fcfg, **kw)

    assert len(clean) == len(stations) * 24
    feature_cols = [f.name for f in FEATURES if f.name in clean.columns and f.name != "station_id"]
    pd.testing.assert_frame_equal(clean[feature_cols], dirty[feature_cols], check_exact=True)
    # The target itself *must* change: it is the future.
    assert not clean["y"].equals(dirty["y"])


def test_every_feature_has_an_anchor_and_seasonal_lags_are_causal():
    assert all(isinstance(f.anchor, Anchor) for f in FEATURES)
    lag_names = {"lag_24h", "lag_48h", "lag_168h", "same_hour_7d_avg", "same_how_4w_mean"}
    assert lag_names <= set(by_anchor(Anchor.TGT))


def test_horizons_beyond_24_are_rejected(panel_and_stations, fcfg):
    panel, stations = panel_and_stations
    with pytest.raises(ValueError):
        build_direct_frame(panel, stations, fcfg, tz=TZ, horizons=[25])


def test_chronological_split_purity():
    start = pd.Timestamp("2019-01-01", tz="UTC")
    end = pd.Timestamp("2019-12-31 23:00", tz="UTC")
    b = chronological_bounds(start, end, 0.70, 0.15, embargo_hours=168)
    assert b.train_end + pd.Timedelta(hours=168) < b.val_start
    assert b.val_end + pd.Timedelta(hours=168) < b.test_start
    origins = pd.Series(pd.date_range(start, end, freq="h", tz="UTC"))
    for h in (1, 12, 24):
        target = origins + pd.Timedelta(hours=h)
        lab = assign_split(origins, target, b)
        train_targets = target[lab == "train"]
        val_origins = origins[lab == "val"]
        test_origins = origins[lab == "test"]
        assert train_targets.max() < val_origins.min()
        assert target[lab == "val"].max() < test_origins.min()
        assert target[lab == "test"].max() <= b.test_end


def test_panel_cv_splits_on_time_not_rows(panel_and_stations, fcfg):
    panel, stations = panel_and_stations
    frame = build_direct_frame(panel, stations, fcfg, tz=TZ, origin_stride_hours=3)
    cv = PanelTimeSeriesSplit(n_splits=4, embargo_hours=48)
    n = 0
    for tr, va in cv.split(frame):
        n += 1
        tr_targets = frame.loc[tr, "target_time"]
        va_origins = frame.loc[va, "origin"]
        assert tr_targets.max() <= va_origins.min() - pd.Timedelta(hours=48)
        # all stations of a validation timestamp are on the same side
        assert set(frame.loc[va, "origin"]).isdisjoint(set(frame.loc[tr, "origin"]))
    assert n == 4
