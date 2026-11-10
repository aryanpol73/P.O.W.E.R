from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from power_core import metrics as m
from power_core.stats import diebold_mariano, holm


def test_point_metrics_known_values():
    y = np.array([0.0, 2.0, 4.0, 0.0])
    f = np.array([1.0, 2.0, 2.0, 0.0])
    assert m.mae(y, f) == pytest.approx(0.75)
    assert m.rmse(y, f) == pytest.approx(np.sqrt(5 / 4))
    assert m.wape(y, f) == pytest.approx(3 / 6)
    assert m.mape_filtered(y, f, min_y=1.0) == pytest.approx(100 * (0 + 0.5) / 2)
    assert m.r2(y, y) == pytest.approx(1.0)


def test_wape_all_zero_is_nan_not_crash():
    assert np.isnan(m.wape([0, 0], [1, 1]))


def test_pinball_and_intervals():
    y = np.array([1.0, 2.0, 3.0])
    assert m.pinball(y, y, 0.9) == 0
    assert m.pinball([1.0], [0.0], 0.9) == pytest.approx(0.9)
    assert m.pinball([0.0], [1.0], 0.9) == pytest.approx(0.1)
    lo, hi = np.array([0.0, 2.5, 2.0]), np.array([2.0, 3.0, 4.0])
    assert m.picp(y, lo, hi) == pytest.approx(2 / 3)
    assert m.mpiw(lo, hi) == pytest.approx((2 + 0.5 + 2) / 3)
    assert m.winkler(y, lo, hi, 0.1) == pytest.approx((2 + (0.5 + 20 * 0.5) + 2) / 3)


def test_mase_scale_is_seasonal_naive():
    ts = pd.date_range("2019-01-01", periods=72, freq="h", tz="UTC")
    y = np.tile(np.arange(24, dtype=float), 3)
    y[-1] += 24
    train = pd.DataFrame({"station_id": "A", "ts": ts, "energy_kwh": y})
    assert m.seasonal_naive_scale(train, 24) == pytest.approx(24 / 48)


def test_dm_detects_better_model_and_holm():
    rng = np.random.default_rng(0)
    e_good = rng.normal(0, 1.0, 2000)
    e_bad = rng.normal(0, 1.5, 2000)
    r = diebold_mariano(e_good, e_bad, h=1)
    assert r.statistic < 0 and r.p_value < 1e-6
    same = diebold_mariano(e_good, e_good + rng.normal(0, 1e-9, 2000), h=6)
    assert same.p_value > 0.05
    adj = holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])
