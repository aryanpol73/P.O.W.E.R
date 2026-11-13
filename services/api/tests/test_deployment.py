"""The Pune-style digital twin: deployed site ids, IST wall-clock alignment, local weather, no donor ids leak."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

SITES = {"S_A": ("TW_ALPHA", 18.5623, 73.8069), "S_B": ("TW_BETA", 18.5639, 73.8071), "S_C": ("TW_GAMMA", 18.5536, 73.8066)}


def _fake_weather(coords: dict[str, tuple[float, float]], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ts = pd.date_range(start.floor("h"), end.ceil("h"), freq="h", tz="UTC")
    return pd.concat(
        [pd.DataFrame({"station_id": sid, "ts": ts, "temperature_2m_c": 30.0, "precipitation_mm": 0.0}) for sid in coords],
        ignore_index=True,
    )


@pytest.fixture
def twin(power_root: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    from power_api.app import create_app
    from power_api.db.session import make_sync_engine
    from power_api.seed import seed
    from power_api.settings import Settings
    from power_core.deployment import load_deployment

    cfg = {
        "name": "twin",
        "display_name": "Test twin",
        "city": "Pune",
        "country": "India",
        "timezone": "Asia/Kolkata",
        "holidays": {"country": "IN", "subdiv": "MH"},
        "model_source": "palo_alto",
        "donor_timezone": "America/Los_Angeles",
        "weather_cache": str((tmp_path / "wx.parquet").as_posix()),
        "sites": [
            {"id": sid, "name": sid.title(), "latitude": lat, "longitude": lon, "donor": donor}
            for donor, (sid, lat, lon) in SITES.items()
        ],
    }
    (power_root / "configs" / "deployments").mkdir(parents=True, exist_ok=True)
    (power_root / "configs" / "deployments" / "twin.yaml").write_text(yaml.safe_dump(cfg))
    dep = load_deployment(power_root, "twin")
    url = f"sqlite+aiosqlite:///{(tmp_path / 'twin.db').as_posix()}"
    seed(make_sync_engine(url), power_root, "palo_alto", dep, weather_fetcher=_fake_weather)
    settings = Settings(DATABASE_URL=url, POWER_ROOT=power_root, POWER_EMBEDDED_WORKER=False, ADMIN_TOKEN="secret",
                        POWER_HISTORY_HOURS=720, POWER_DEPLOYMENT="twin")
    with TestClient(create_app(settings)) as c:
        c.headers.update({"Authorization": "Bearer secret"})
        yield c


def test_deployment_endpoint(twin: TestClient) -> None:
    d = twin.get("/api/v1/deployment").json()
    assert d["is_twin"] and d["timezone"] == "Asia/Kolkata" and d["model_trained_on"] == "palo_alto"


def test_only_deployed_sites_are_exposed(twin: TestClient) -> None:
    ids = {s["station_id"] for s in twin.get("/api/v1/stations", params={"modeled": False}).json()}
    assert ids == {v[0] for v in SITES.values()}
    snap = twin.get("/api/v1/network/snapshot", params={"h": 1}).json()
    assert {s["station_id"] for s in snap["stations"]} <= ids
    for a in twin.get("/api/v1/alerts", params={"status": "all"}).json():
        assert a["station_id"] in ids and "S_" not in a["message"]


def test_forecast_runs_on_the_ist_grid(twin: TestClient) -> None:
    f = twin.get("/api/v1/stations/TW_ALPHA/forecast").json()
    origin = pd.Timestamp(f["forecast_origin"])
    assert origin.minute == 30  # IST hours start at :30 UTC
    assert origin.tz_convert("Asia/Kolkata").minute == 0
    q = np.array([p["q50"] for p in f["points"]])
    assert len(q) == 24 and np.isfinite(q).all() and (q >= 0).all()
    r = twin.post("/api/v1/sim/tick")
    assert r.status_code == 200, r.text
    e = twin.get("/api/v1/stations/TW_BETA/explain").json()
    assert e["station_id"] == "TW_BETA"


def test_demand_keeps_its_local_wall_clock(power_root: Path) -> None:
    from power_core.deployment import load_deployment

    dep = load_deployment(power_root, "twin")
    assert dep is not None
    panel = pd.read_parquet(power_root / "data" / "processed" / "panel_palo_alto.parquet")
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    out = dep.retime_panel(panel, [c for c in panel.columns if c not in ("station_id", "ts")])
    src = panel.loc[panel["station_id"] == "S_A"].set_index("ts")["energy_kwh"]
    dst = out.loc[out["station_id"] == "TW_ALPHA"].set_index("ts")["energy_kwh"]
    # A donor hour lands on the same local wall-clock hour in IST.
    t = src.index[500]
    wall = t.tz_convert("America/Los_Angeles").tz_localize(None)
    assert dst.loc[wall.tz_localize("Asia/Kolkata").tz_convert("UTC")] == src.loc[t]
    gaps = dst.index.to_series().diff().dropna()
    assert (gaps == pd.Timedelta(hours=1)).all()
