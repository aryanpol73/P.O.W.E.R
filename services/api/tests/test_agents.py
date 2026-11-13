"""A5 drift, A7 champion-challenger retrain, A8 discovery and the ops status endpoint."""

from __future__ import annotations

import asyncio

import httpx
from fastapi.testclient import TestClient

from power_core.config import RetrainCfg


def test_status_lists_every_agent(client: TestClient) -> None:
    st = client.get("/api/v1/agents/status").json()
    assert set(st["agents"]) == {"pipeline", "drift", "retrain", "discovery"}
    assert st["agents"]["pipeline"]["runs"] >= 1
    assert st["buffer"]["healthy"] is True
    assert st["live_champion"]["run_id"] is None


def test_drift_check_logs_signals(client: TestClient) -> None:
    for _ in range(30):  # cross at least one local midnight so the scheduled check runs too
        assert client.post("/api/v1/sim/tick").status_code == 200
    rep = client.post("/api/v1/agents/drift/run").json()
    assert {"coverage", "seasonal_psi"} <= set(rep["signals"])  # wape_ratio needs reports/metrics_test.json
    for s in rep["signals"].values():
        assert isinstance(s["is_drift"], bool) and s["threshold"] > 0
    d = client.get("/api/v1/drift", params={"days": 7}).json()
    assert d["checks"] >= 1 and d["latest"]
    assert client.get("/api/v1/agents/status").json()["agents"]["drift"]["runs"] >= 1


def test_admin_required_for_agent_actions(client: TestClient) -> None:
    c = TestClient(client.app)
    assert c.post("/api/v1/agents/retrain").status_code == 401
    assert c.post("/api/v1/agents/drift/run").status_code == 401


def test_retrain_gate_promote_and_rollback(client: TestClient) -> None:
    from power_api.agents import retrain as retrain_agent

    app = client.app
    eng = app.state.engine
    cfg = RetrainCfg(window_days=20, calib_days=5, shadow_days=3, wape_gain=10.0, picp_low=0.0, picp_high=1.0,
                     max_station_regression=10.0)  # permissive gate: this test checks the mechanics
    res = client.portal.call(lambda: retrain_agent.run(app.state.maker, eng, cfg, 24 * 7, True))
    assert res["status"] == "ok", res
    assert set(res["gate"]) == {"wape", "picp", "stations"}
    assert res["shadow"]["challenger"]["wape"] > 0 and res["shadow"]["rows"] > 0
    assert res["passed"] and res["promoted"]
    assert eng.live_run == res["run_id"]

    # Live forecasts now carry the challenger tag; the frozen leaderboard is untouched.
    client.post("/api/v1/sim/tick")
    f = client.get("/api/v1/stations/S_A/forecast").json()
    assert f["model_version"].endswith(res["run_id"])
    runs = client.get("/api/v1/agents/retrain/runs").json()
    assert runs["active"] == res["run_id"]

    rb = client.post("/api/v1/agents/retrain/rollback").json()
    assert rb["rolled_back"] == res["run_id"] and eng.live_run is None
    assert client.post(f"/api/v1/agents/retrain/{res['run_id']}/activate").status_code == 200
    assert eng.live_run == res["run_id"]
    client.post("/api/v1/agents/retrain/rollback")


def test_retrain_gate_rejects_a_worse_challenger(client: TestClient) -> None:
    from power_api.agents import retrain as retrain_agent

    app = client.app
    cfg = RetrainCfg(window_days=20, calib_days=5, shadow_days=3, wape_gain=0.0)  # nothing can beat 0 x champion
    client.post("/api/v1/sim/tick")
    res = client.portal.call(lambda: retrain_agent.run(app.state.maker, app.state.engine, cfg, 24 * 7, True))
    assert res["status"] == "ok" and not res["passed"] and not res["promoted"]
    assert app.state.engine.live_run is None


def test_discovery_adds_unmodelled_stations(client: TestClient) -> None:
    from power_api.agents import discovery

    payload = {"elements": [
        {"type": "node", "id": 1, "lat": 37.40, "lon": -122.10, "tags": {"name": "Mall charger", "capacity": "4"}},
        {"type": "node", "id": 2, "lat": 37.4446, "lon": -122.1608, "tags": {}},  # same place as S_A: skipped
        {"type": "way", "id": 3, "center": {"lat": 37.41, "lon": -122.12}, "tags": {"operator": "Tata Power"}},
    ]}
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json=payload))

    async def go() -> dict:
        async with client.app.state.maker() as s, httpx.AsyncClient(transport=transport) as http:
            res = await discovery.run(s, client=http)
            await s.commit()
            return res

    res = client.portal.call(go)
    assert res == {**res, "found": 3, "added": 2}
    sts = {x["station_id"]: x for x in client.get("/api/v1/stations", params={"modeled": False}).json()}
    assert sts["OSM_N1"]["is_modeled"] is False and sts["OSM_N1"]["total_ports"] == 4
    assert "OSM_N2" not in sts and "OSM_W3" in sts
    snap = client.get("/api/v1/network/snapshot", params={"h": 1}).json()
    assert any(x["station_id"] == "OSM_N1" and x["risk"] is None for x in snap["stations"])
    asyncio.run(asyncio.sleep(0))
