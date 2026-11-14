"""A complete, tiny P.O.W.E.R. root: synthetic data -> trained models -> seeded SQLite -> app."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from power_core.config import FeatureConfig
from power_core.features import build_direct_frame
from power_core.splits import assign_split, chronological_bounds

REPO = Path(__file__).resolve().parents[3]
TZ = "America/Los_Angeles"
N_DAYS = 100


def _panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(1)
    ts = pd.date_range("2019-01-01 08:00", periods=24 * N_DAYS, freq="h", tz="UTC")
    loc = ts.tz_convert(TZ)
    frames, meta = [], []
    coords = {"S_A": (37.4446, -122.1608), "S_B": (37.4463, -122.1622), "S_C": (37.4482, -122.1580)}
    for i, (sid, (lat, lon)) in enumerate(coords.items()):
        ports = 4 + 2 * i
        rated = ports * 6.6
        wk = (loc.dayofweek < 5).astype(float)
        prof = np.exp(-((loc.hour - 11) ** 2) / 8) * rated * (0.55 + 0.3 * i) * (0.3 + 0.7 * wk)
        y = np.clip(prof + rng.normal(0, 1.0, len(ts)), 0, rated)
        occ = np.clip(np.round(y / 5), 0, ports)
        frames.append(pd.DataFrame({
            "station_id": sid, "ts": ts, "energy_kwh": y, "occ_ratio": occ / ports, "idle_ratio": 0.2,
            "active_sessions": occ, "occupied_ports": occ, "arrivals": rng.integers(0, 3, len(ts)),
            "is_saturated": occ >= ports, "temperature_2m_c": 15 + 6 * np.sin(2 * np.pi * (loc.hour - 9) / 24),
            "precipitation_mm": 0.0,
        }))
        meta.append({"station_id": sid, "latitude": lat, "longitude": lon, "total_ports": ports, "rated_kw_total": rated,
                     "poi_category": "downtown_public", "first_seen": ts[0]})
    return pd.concat(frames, ignore_index=True), pd.DataFrame(meta)


def build_root(root: Path) -> None:
    from power_ml.evaluation import evaluate
    from power_ml.training import train_all

    shutil.copytree(REPO / "configs", root / "configs")
    (root / "pyproject.toml").write_text("[project]\nname='t'\n")
    cfg = yaml.safe_load((root / "configs" / "models.yaml").read_text())
    cfg["n_jobs"] = 2
    cfg["origin_stride"] = {k: 2 for k in cfg["origin_stride"]}
    cfg["early_stopping"] = {"holdout_frac": 0.2, "embargo_hours": 24, "rounds": 20}
    cfg["rf"].update(n_estimators=15)
    for k in ("xgb", "lgbm"):
        cfg[k].update(n_estimators=120, learning_rate=0.1)
    cfg["lgbm_quantile"].update(n_estimators=80, learning_rate=0.1)
    (root / "configs" / "models.yaml").write_text(yaml.safe_dump(cfg))

    panel, stations = _panel()
    (root / "data" / "interim").mkdir(parents=True)
    proc = root / "data" / "processed"
    proc.mkdir(parents=True)
    stations.to_parquet(root / "data" / "interim" / "stations_palo_alto.parquet")
    panel.to_parquet(proc / "panel_palo_alto.parquet")
    b = chronological_bounds(panel["ts"].min(), panel["ts"].max(), 0.6, 0.2, 24)
    (proc / "splits_palo_alto.json").write_text(json.dumps(b.to_dict()))
    (root / "data" / "manifest.json").write_text(json.dumps({"palo_alto": {"sites": 3}}))
    f = build_direct_frame(panel, stations, FeatureConfig(warmup_hours=24 * 29), tz=TZ, origin_stride_hours=2)
    lab = assign_split(f["origin"], f["target_time"], b)
    for s in ("train", "val", "test"):
        f.loc[lab == s].to_parquet(proc / f"features_palo_alto_{s}.parquet")
    quiet = lambda *_: None  # noqa: E731
    train_all(root, "palo_alto", log=quiet)
    evaluate(root, "palo_alto", "val", log=quiet)  # predictions_val.parquet -> network calibration


@pytest.fixture(scope="session")
def power_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("power_root")
    build_root(root)
    return root


@pytest.fixture
def client(power_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    from power_api.app import create_app
    from power_api.db.session import make_sync_engine
    from power_api.seed import seed
    from power_api.settings import Settings

    # POWER_TEST_DATABASE_URL runs the suite on PostgreSQL, e.g.
    # postgresql+asyncpg://power:power@127.0.0.1:5433/power_test (docker compose up -d db)
    url = os.environ.get("POWER_TEST_DATABASE_URL") or f"sqlite+aiosqlite:///{(tmp_path / 'power.db').as_posix()}"
    seed(make_sync_engine(url), power_root, "palo_alto")
    settings = Settings(DATABASE_URL=url, POWER_ROOT=power_root, POWER_EMBEDDED_WORKER=False, ADMIN_TOKEN="secret",
                        POWER_HISTORY_HOURS=720, POWER_DEPLOYMENT="none",
                        POWER_DISCOVERY=False)
    with TestClient(create_app(settings)) as c:
        c.headers.update({"Authorization": "Bearer secret"})
        yield c
