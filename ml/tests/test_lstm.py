from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from power_core.config import FeatureConfig  # noqa: E402
from power_core.features import build_direct_frame  # noqa: E402
from power_core.splits import time_holdout  # noqa: E402
from power_ml.models.lstm import LSTMModel  # noqa: E402

from .test_models import TZ, synthetic_panel  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    panel, stations = synthetic_panel(n_days=70)
    frame = build_direct_frame(panel, stations, FeatureConfig(warmup_hours=24 * 29), tz=TZ, origin_stride_hours=3)
    fi, hi = time_holdout(frame["origin"], frame["target_time"], 0.25, 24)
    return panel, frame.iloc[fi].reset_index(drop=True), frame.iloc[hi].reset_index(drop=True)


def _small(**kw):
    return LSTMModel(lookback=48, hidden=16, layers=1, batch_size=128, max_epochs=kw.pop("epochs", 4), patience=2, **kw)


def test_lstm_learns_and_predicts_all_horizons(setup):
    panel, fit, hold = setup
    m = _small(epochs=6).prepare(panel, fit).fit(fit, hold)
    assert m.history_[-1]["train_l1"] < m.history_[0]["train_l1"]
    p = m.predict(hold)
    assert p.shape == (len(hold),) and np.isfinite(p).all() and (p >= 0).all()
    # A tiny net on toy data: only require it to beat a constant forecast (catches a broken network).
    constant = np.abs(hold["y"] - fit["y"].mean()).sum() / hold["y"].sum()
    assert np.abs(hold["y"] - p).sum() / hold["y"].sum() < constant


def test_lstm_never_reads_demand_after_the_origin(setup):
    panel, fit, hold = setup
    m = _small(epochs=1).prepare(panel, fit).fit(fit, None)
    T = hold["origin"].iloc[len(hold) // 2]
    rows = hold.loc[hold["origin"] == T]
    before = m.predict(rows)
    poisoned = panel.copy()
    for c in ("energy_kwh", "occ_ratio", "active_sessions"):
        poisoned.loc[poisoned["ts"] > T, c] = 1e6
    m.attach_panel(poisoned)
    np.testing.assert_array_equal(before, m.predict(rows))


def test_lstm_pickles_without_panel(setup, tmp_path):
    import joblib

    panel, fit, hold = setup
    m = _small(epochs=1).prepare(panel, fit).fit(fit, None)
    joblib.dump(m, tmp_path / "m.joblib")
    m2 = joblib.load(tmp_path / "m.joblib")
    assert not hasattr(m2, "_enc")
    np.testing.assert_allclose(m2.attach_panel(panel).predict(hold.head(48)), m.predict(hold.head(48)), rtol=1e-5)


def test_window_guard_rejects_origins_without_history(setup):
    panel, fit, _ = setup
    m = _small().prepare(panel, fit)
    early = pd.DataFrame({"station_id": ["S_A"], "origin": [panel["ts"].min() + pd.Timedelta(hours=5)], "h": [1]})
    with pytest.raises(ValueError):
        m._global_index(early["station_id"], early["origin"])
