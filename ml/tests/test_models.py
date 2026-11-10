from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from power_core import metrics
from power_core.config import FeatureConfig
from power_core.features import build_direct_frame
from power_core.splits import assign_split, chronological_bounds, time_holdout
from power_ml.models import LGBMModel, LGBMQuantile, RFModel, RidgeModel, WeightedEnsemble, XGBModel
from power_ml.models.baselines import Climatology, default_baselines

TZ = "America/Los_Angeles"
REPO = Path(__file__).resolve().parents[2]


def synthetic_panel(n_days: int = 75, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2019-01-01 08:00", periods=24 * n_days, freq="h", tz="UTC")
    loc = ts.tz_convert(TZ)
    frames, meta = [], []
    for i, sid in enumerate(("S_A", "S_B")):
        ports = 6
        weekday = (loc.dayofweek < 5).astype(float)
        profile = np.exp(-((loc.hour - 11) ** 2) / 8) * (8 + 4 * i) * (0.4 + 0.6 * weekday)
        y = np.clip(profile + rng.normal(0, 0.8, len(ts)), 0, None)
        occ = np.clip(np.round(y / 2), 0, ports)
        frames.append(pd.DataFrame({
            "station_id": sid, "ts": ts, "energy_kwh": y, "occ_ratio": occ / ports, "active_sessions": occ,
            "occupied_ports": occ, "arrivals": rng.integers(0, 3, len(ts)), "is_saturated": occ >= ports,
            "temperature_2m_c": 14 + rng.normal(0, 3, len(ts)), "precipitation_mm": 0.0,
        }))
        meta.append({"station_id": sid, "total_ports": ports, "rated_kw_total": ports * 6.6, "poi_category": "x",
                     "first_seen": ts[0], "latitude": 37.44, "longitude": -122.16})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(meta)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    panel, stations = synthetic_panel()
    return build_direct_frame(panel, stations, FeatureConfig(warmup_hours=24 * 29), tz=TZ, origin_stride_hours=2)


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_idx, hold_idx = time_holdout(frame["origin"], frame["target_time"], 0.25, 24)
    return frame.iloc[fit_idx], frame.iloc[hold_idx]


def test_time_holdout_is_causal(frame):
    fit_idx, hold_idx = time_holdout(frame["origin"], frame["target_time"], 0.25, 24)
    assert frame.iloc[fit_idx]["target_time"].max() <= frame.iloc[hold_idx]["origin"].min() - pd.Timedelta(hours=24)


def test_baselines_read_causal_columns(frame):
    b0, b1, b2, b3 = default_baselines()
    np.testing.assert_allclose(b1.predict(frame), np.maximum(frame["lag_24h"], 0))
    np.testing.assert_allclose(b0.predict(frame), np.maximum(frame["y_org_0"], 0))
    clim = Climatology().fit(frame)
    unseen = frame.head(3).assign(station_id=pd.Categorical(["S_A"] * 3))
    assert np.isfinite(clim.predict(unseen)).all()


def test_learned_models_beat_seasonal_naive(frame):
    fit, hold = _split(frame)
    naive = metrics.wape(hold["y"], np.maximum(hold["lag_24h"], 0))
    small = {"n_estimators": 200, "learning_rate": 0.1}
    models = [
        RidgeModel(alphas=(1.0, 10.0)),
        RFModel(params={"n_estimators": 30, "min_samples_leaf": 5, "max_features": 0.5}, n_jobs=2),
        XGBModel(params={**small, "objective": "reg:tweedie", "max_depth": 4}, early_stopping_rounds=20, n_jobs=2),
        LGBMModel(params={**small, "objective": "tweedie", "num_leaves": 15}, early_stopping_rounds=20, n_jobs=2),
    ]
    for m in models:
        m.fit(fit, hold if not isinstance(m, RFModel) else None)
        p = m.predict(hold)
        assert p.shape == (len(hold),) and np.isfinite(p).all() and (p >= 0).all()
        assert metrics.wape(hold["y"], p) < naive, m.name


def test_quantile_heads_are_ordered_on_average(frame):
    fit, hold = _split(frame)
    q = LGBMQuantile(params={"n_estimators": 150, "learning_rate": 0.1, "num_leaves": 15}, n_jobs=2).fit(fit, hold)
    qs = q.predict_quantiles(hold)
    assert qs[0.05].mean() < qs[0.50].mean() < qs[0.95].mean()
    assert 0.6 < metrics.picp(hold["y"], qs[0.05], qs[0.95]) <= 1.0


def test_ensemble_recovers_known_weights():
    rng = np.random.default_rng(0)
    a, b = rng.normal(5, 2, 2000), rng.normal(5, 2, 2000)
    y = 0.7 * a + 0.3 * b
    ens = WeightedEnsemble(("a", "b")).fit({"a": a, "b": b}, y)
    assert ens.weights_["a"] == pytest.approx(0.7, abs=1e-3)
    assert sum(ens.weights_.values()) == pytest.approx(1.0)


def test_train_and_evaluate_end_to_end(tmp_path):
    """Full ``train_all`` -> ``evaluate`` on a tiny synthetic root with a fast config."""
    from power_ml.evaluation import evaluate
    from power_ml.training import train_all

    root = tmp_path
    shutil.copytree(REPO / "configs", root / "configs")
    (root / "pyproject.toml").write_text("[project]\nname='t'\n")
    cfg = yaml.safe_load((root / "configs" / "models.yaml").read_text())
    cfg["n_jobs"] = 2
    cfg["origin_stride"] = {k: 1 for k in cfg["origin_stride"]}
    cfg["early_stopping"] = {"holdout_frac": 0.2, "embargo_hours": 24, "rounds": 20}
    cfg["rf"].update(n_estimators=20)
    for k in ("xgb", "lgbm"):
        cfg[k].update(n_estimators=150, learning_rate=0.1)
    cfg["lgbm_quantile"].update(n_estimators=100, learning_rate=0.1)
    (root / "configs" / "models.yaml").write_text(yaml.safe_dump(cfg))

    panel, stations = synthetic_panel(n_days=90)
    proc = root / "data" / "processed"
    proc.mkdir(parents=True)
    panel.to_parquet(proc / "panel_palo_alto.parquet")
    b = chronological_bounds(panel["ts"].min(), panel["ts"].max(), 0.6, 0.2, 24)
    (proc / "splits_palo_alto.json").write_text(json.dumps(b.to_dict()))
    (root / "data" / "manifest.json").write_text(json.dumps({"palo_alto": {"sites": 2}}))
    f = build_direct_frame(panel, stations, FeatureConfig(warmup_hours=24 * 29), tz=TZ, origin_stride_hours=2)
    lab = assign_split(f["origin"], f["target_time"], b)
    for s in ("train", "val", "test"):
        f.loc[lab == s].to_parquet(proc / f"features_palo_alto_{s}.parquet")

    manifest = train_all(root, "palo_alto", log=lambda *_: None)
    assert set(manifest["calibrators"]) >= {"M4_lightgbm", "E1_ensemble", "M4_lightgbm_quantile_cqr"}
    summary = evaluate(root, "palo_alto", "test", log=lambda *_: None)
    board = summary["leaderboard"]
    assert board["M4_lightgbm"]["wape"] < board["B1_seasonal_naive_24h"]["wape"]
    assert 0.0 < board["M4_lightgbm_quantile"]["interval_cqr"]["picp"] <= 1.0
    for name in ("metrics_test.json", "metrics_by_horizon_test.csv", "dm_tests_test.csv", "test_evaluations.log"):
        assert (root / "reports" / name).exists()
    by_h = pd.read_csv(root / "reports" / "metrics_by_horizon_test.csv")
    assert set(by_h["h"]) == set(range(1, 25))
