from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from power_core.features import PanelError, build_direct_frame, feature_matrix, feature_names
from power_core.features.calendar import calendar_features
from power_core.features.weather import augment_frame

TZ = "America/Los_Angeles"


def test_values_match_definitions(panel_and_stations, fcfg):
    panel, stations = panel_and_stations
    T = panel["ts"].min() + pd.Timedelta(days=35, hours=10)
    f = build_direct_frame(panel, stations, fcfg, tz=TZ, origins=pd.DatetimeIndex([T]))
    y = panel.loc[panel["station_id"] == "S_B"].set_index("ts")["energy_kwh"]
    rows = f.loc[f["station_id"] == "S_B"].set_index("h")
    H = pd.Timedelta(hours=1)
    for h in (1, 7, 24):
        r = rows.loc[h]
        t = T + h * H
        assert r["target_time"] == t
        assert r["y"] == pytest.approx(y[t])
        assert r["y_org_0"] == pytest.approx(y[T], rel=1e-6)
        assert r["y_org_2"] == pytest.approx(y[T - 2 * H], rel=1e-6)
        assert r["lag_24h"] == pytest.approx(y[t - 24 * H], rel=1e-6)
        assert r["lag_168h"] == pytest.approx(y[t - 168 * H], rel=1e-6)
        assert r["same_hour_7d_avg"] == pytest.approx(np.mean([y[t - 24 * k * H] for k in range(1, 8)]), rel=1e-5)
        assert r["rolling_mean_24h"] == pytest.approx(y[T - 23 * H: T].mean(), rel=1e-5)
    assert rows.index.tolist() == list(range(1, 25))


def test_serving_frame_with_future_nan_demand(panel_and_stations, fcfg):
    panel, stations = panel_and_stations
    T = panel["ts"].min() + pd.Timedelta(days=40)
    serve = panel.copy()
    serve.loc[serve["ts"] > T, "energy_kwh"] = np.nan
    f = build_direct_frame(serve, stations, fcfg, tz=TZ, origins=pd.DatetimeIndex([T]), drop_missing_target=False)
    assert len(f) == 3 * 24
    assert f["y"].isna().all()
    assert f[["lag_24h", "y_org_0", "temperature_2m_c"]].notna().all().all()


def test_gap_in_panel_is_rejected(panel_and_stations, fcfg):
    panel, stations = panel_and_stations
    broken = panel.drop(index=panel.index[100])
    with pytest.raises(PanelError):
        build_direct_frame(broken, stations, fcfg, tz=TZ)


def test_naive_timestamps_rejected(panel_and_stations, fcfg):
    panel, stations = panel_and_stations
    p = panel.copy()
    p["ts"] = p["ts"].dt.tz_localize(None)
    with pytest.raises(PanelError):
        build_direct_frame(p, stations, fcfg, tz=TZ)


def test_warmup_and_dtypes(panel_and_stations, fcfg):
    panel, stations = panel_and_stations
    f = build_direct_frame(panel, stations, fcfg, tz=TZ, origin_stride_hours=6)
    assert f["origin"].min() >= panel["ts"].min() + pd.Timedelta(hours=fcfg.warmup_hours)
    assert f["station_id"].dtype.name == "category"
    assert f["lag_24h"].dtype == np.float32
    for model in ("ridge", "rf", "gbdt", "prior"):
        X = feature_matrix(f, model)
        assert set(X.columns) <= set(feature_names(model))
    assert "lag_24h" not in feature_matrix(f, "prior").columns


def test_warmup_counts_from_panel_start_when_window_is_cut(panel_and_stations, fcfg):
    panel, stations = panel_and_stations
    cut = panel.loc[panel["ts"] >= panel["ts"].min() + pd.Timedelta(days=5)]   # first_seen is 5 days earlier
    f = build_direct_frame(cut, stations, fcfg, tz=TZ, origin_stride_hours=6)
    assert f["origin"].min() >= cut["ts"].min() + pd.Timedelta(hours=fcfg.warmup_hours)
    core = ["lag_24h", "lag_168h", "same_how_4w_mean", "rolling_mean_24h", "level_28d", "y_org_2"]
    assert f[core].notna().all().all()


def test_calendar_is_local_time_and_dst_safe():
    # 2019-03-10 is the US spring-forward day: local 02:00 does not exist.
    ts = pd.date_range("2019-03-10 07:00", periods=6, freq="h", tz="UTC")
    cal = calendar_features(ts, TZ)
    local_hours = ts.tz_convert(TZ).hour
    assert list(local_hours) == [23, 0, 1, 3, 4, 5]
    assert cal["hour_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * 23 / 24))
    hol = calendar_features(pd.date_range("2019-11-28 20:00", periods=1, freq="h", tz="UTC"), TZ)
    assert hol["is_holiday"].iloc[0] == 1  # Thanksgiving
    # The day after Thanksgiving is itself a California holiday, so it is not a bridge day.
    fri_tg = calendar_features(pd.date_range("2019-11-29 20:00", periods=1, freq="h", tz="UTC"), TZ)
    assert fri_tg["is_holiday"].iloc[0] == 1 and fri_tg["is_bridge_day"].iloc[0] == 0
    # 4 July 2019 was a Thursday: Friday 5 July sits between a holiday and the weekend.
    jul5 = calendar_features(pd.date_range("2019-07-05 19:00", periods=1, freq="h", tz="UTC"), TZ)
    assert jul5["is_holiday"].iloc[0] == 0 and jul5["is_bridge_day"].iloc[0] == 1


def test_pricing_era_steps():
    ts = pd.date_range("2018-10-31 12:00", periods=3, freq="D", tz="UTC")
    cal = calendar_features(ts, TZ, pricing_changes=["2018-11-01T00:00:00-07:00"])
    assert cal["pricing_era"].tolist() == [0, 1, 1]


def test_weather_augmentation_grows_with_horizon(panel_and_stations, fcfg):
    panel, stations = panel_and_stations
    f = build_direct_frame(panel, stations, fcfg, tz=TZ, origin_stride_hours=2)
    a = augment_frame(f, fcfg, seed=1)
    d = (a["temperature_2m_c"] - f["temperature_2m_c"]).groupby(f["h"]).std()
    assert d.loc[24] > d.loc[1]
    assert (a["cold_stress"] >= 0).all() and (a["heat_stress"] >= 0).all()
    pd.testing.assert_series_equal(a["y"], f["y"])
