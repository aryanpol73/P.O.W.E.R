from __future__ import annotations

import pandas as pd
import pytest


def _tick(client, n: int = 1) -> list[dict]:  # type: ignore[no-untyped-def]
    return [client.post("/api/v1/sim/tick").json() for _ in range(n)]


def test_health_and_models(client):
    h = client.get("/api/v1/health").json()
    assert h["status"] == "ok" and h["champion"] == "M4_lightgbm_quantile"
    assert {"B1_seasonal_naive_24h", "M4_lightgbm", "E1_ensemble"} <= set(h["models"])
    assert h["sim"]["mode"] == "REPLAY" and h["sim"]["paused"] is True
    assert client.get("/api/v1/models").json()["version"]


def test_ticks_advance_clock_and_store_every_forecast(client):
    t0 = pd.Timestamp(client.get("/api/v1/sim/state").json()["sim_now"])
    res = _tick(client, 3)
    t3 = pd.Timestamp(client.get("/api/v1/sim/state").json()["sim_now"])
    assert t3 - t0 == pd.Timedelta(hours=3) and res[-1]["sim_now"] == t3.isoformat()
    f = client.get("/api/v1/stations/S_A/forecast").json()
    assert f["forecast_origin"] == t3.isoformat() and len(f["points"]) == 24
    assert all(p["q05"] <= p["q50"] + 1e-9 and p["q50"] <= p["q95"] + 1e-9 for p in f["points"])
    assert [p["h"] for p in f["points"]] == list(range(1, 25))
    assert len(f["history"]) == 48
    # the buffer never holds data past the forecast origin
    assert pd.Timestamp(f["history"][-1]["ts"]) == t3


def test_every_model_forecasts_every_station(client):
    _tick(client)
    h = client.get("/api/v1/health").json()
    for model in h["models"]:
        f = client.get("/api/v1/stations/S_B/forecast", params={"model": model, "history_hours": 0}).json()
        assert len(f["points"]) == 24, model


def test_explanation_adds_up_to_the_champion_forecast(client):
    _tick(client)
    f = client.get("/api/v1/stations/S_C/forecast").json()
    e = client.get("/api/v1/stations/S_C/explain", params={"h": 6}).json()
    q50_h6 = next(p["q50"] for p in f["points"] if p["h"] == 6)
    assert e["prediction"] == pytest.approx(q50_h6, abs=1e-2)
    assert e["narrative"].startswith("Base ") and e["contributions"][-1]["feature"] == "__other__"
    peak = client.get("/api/v1/stations/S_C/explain").json()  # default: the forecast peak
    assert peak["h"] == max(f["points"], key=lambda p: p["q50"])["h"]


def test_network_kpis_and_snapshot(client):
    _tick(client)
    k = client.get("/api/v1/network/kpis").json()
    assert k["next_hour"]["lower"] <= k["next_hour"]["q50"] <= k["next_hour"]["upper"]
    assert sum(k["stations_by_risk"].values()) == 3 and k["risk_status"] in {"NORMAL", "WARNING", "CRITICAL"}
    snap = client.get("/api/v1/network/snapshot", params={"h": 12}).json()
    assert {s["station_id"] for s in snap["stations"]} == {"S_A", "S_B", "S_C"}
    assert all(s["load_ratio_q95"] >= 0 for s in snap["stations"])


def test_live_arena_scores_observed_hours(client):
    _tick(client, 4)
    arena = client.get("/api/v1/models/live-arena", params={"hours": 24}).json()
    assert arena["n"] > 0 and "M4_lightgbm" in arena["models"]
    assert all(0 <= v["wape"] < 5 for v in arena["models"].values())


def test_scenario_flags_out_of_range_and_leaves_live_state_alone(client):
    _tick(client)
    before = client.get("/api/v1/stations/S_A/forecast").json()
    sc = client.post("/api/v1/scenarios/simulate", json={"temperature_delta_c": 20, "force_holiday": True}).json()
    assert sc["ood_flags"] and sc["ood_flags"][0]["feature"] == "temperature_2m_c"
    assert len(sc["baseline"]) == len(sc["scenario"]) == 3 * 24
    assert any("holiday" in a for a in sc["assumptions"])
    assert client.get("/api/v1/stations/S_A/forecast").json() == before


def test_alert_flow_and_reroutes_are_consistent(client):
    _tick(client, 6)
    alerts = client.get("/api/v1/alerts", params={"status": "all"}).json()
    for a in alerts:
        assert a["level"] in {"WARNING", "CRITICAL"} and a["load_ratio_q95"] >= 0.70
    if alerts:
        aid = alerts[0]["alert_id"]
        assert client.post(f"/api/v1/alerts/{aid}/ack").status_code == 200
    for r in client.get("/api/v1/reroutes").json():
        assert r["shifted_kwh"] > 0 and r["distance_km"] <= 3.0


def test_admin_endpoints_require_the_token(client):
    r = client.post("/api/v1/sim/tick", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert client.get("/api/v1/stations").status_code == 200  # reads stay public


def test_sim_control_pause_speed_reset(client):
    s = client.post("/api/v1/sim/control", json={"action": "set_speed", "speed_ticks_per_min": 30}).json()
    assert s["speed_ticks_per_min"] == 30
    start = client.get("/api/v1/sim/state").json()["sim_now"]
    _tick(client, 2)
    s = client.post("/api/v1/sim/control", json={"action": "reset"}).json()
    assert s["sim_now"] == start and s["paused"] is True


def test_pipeline_is_idempotent_per_origin(client):
    _tick(client)
    sim = client.app.state.sim
    from power_api.db.session import to_db

    T = to_db(pd.Timestamp(client.get("/api/v1/sim/state").json()["sim_now"]))
    # Run on the app's own event loop: pooled asyncpg connections are bound to it.
    assert client.portal.call(sim._process, T) == {"skipped": True}


def test_websocket_streams_pipeline_events(client):
    with client.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["event"] == "hello"
        ws.send_json({"op": "subscribe", "channels": ["forecast", "sim"]})
        assert ws.receive_json()["event"] == "subscribed"
        client.post("/api/v1/sim/tick")
        events = [ws.receive_json()["event"] for _ in range(2)]
        assert events == ["forecast.updated", "sim.clock"]


def test_insight_endpoints_report_missing_files_clearly(client):
    # the synthetic root has no analysis outputs: endpoints must say how to produce them
    r = client.get("/api/v1/insights/shap")
    assert r.status_code == 404 and "power-ml analyze" in r.json()["detail"]
    rows = client.get("/api/v1/models/horizons", params={"split": "val"}).json()
    assert {r["h"] for r in rows} == set(range(1, 25))
