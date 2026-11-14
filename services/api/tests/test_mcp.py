"""MCP server tools against a live (in-process) API: schemas, numbers and guard-rails."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

srv = pytest.importorskip("power_mcp.server")
from power_mcp.guards import RateLimiter  # noqa: E402


@pytest.fixture
def tools(client: TestClient, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    class Prefixed:  # the MCP server's httpx client, routed into the app (same event loop as the fixture)
        def request(self, method: str, path: str, **kw):  # type: ignore[no-untyped-def]
            return client.request(method, "/api/v1" + path, **kw)

    monkeypatch.setattr(srv, "_client", Prefixed())
    monkeypatch.setattr(srv, "limiter", RateLimiter(1000, 60))
    return srv


def test_tools_are_registered_with_schemas(tools) -> None:  # type: ignore[no-untyped-def]
    names = {t.name for t in client_run(tools.mcp.list_tools())}
    assert {"list_stations", "get_network_status", "get_station_forecast", "explain_peak_shap",
            "simulate_weather_shock", "compare_models", "get_alerts", "get_drift_report", "get_agent_status"} <= names
    schema = next(t for t in client_run(tools.mcp.list_tools()) if t.name == "get_station_forecast").input_schema
    assert schema["properties"]["horizon_hours"]["maximum"] == 24


def client_run(coro):  # type: ignore[no-untyped-def]
    import anyio

    async def go():  # type: ignore[no-untyped-def]
        return await coro

    return anyio.run(go)


def test_forecast_and_explain_match_the_api(tools, client: TestClient) -> None:  # type: ignore[no-untyped-def]
    f = tools.get_station_forecast("S_A", horizon_hours=6)
    api = client.get("/api/v1/stations/S_A/forecast").json()
    assert len(f["points"]) == 6
    assert f["points"][0]["q50"] == api["points"][0]["q50"]
    e = tools.explain_peak_shap("s_a", top_k=3)  # name/id matching is case-insensitive
    assert len(e["top_contributions"]) == 3
    total = e["base_value_kwh"] + sum(c["kwh"] for c in e["top_contributions"]) + e["all_other_features_kwh"]
    assert total == pytest.approx(e["prediction_kwh"], abs=0.05)


def test_scenario_and_leaderboards(tools, client: TestClient) -> None:  # type: ignore[no-untyped-def]
    for _ in range(3):  # observed hours for the live leaderboard
        client.post("/api/v1/sim/tick")
    r = tools.simulate_weather_shock(temperature_delta_c=12.0)
    assert r["requested_by"] == "mcp" and r["station_peaks_kwh"]
    assert any(f["feature"] == "temperature_2m_c" for f in r["ood_flags"])  # synthetic training max is ~21 C
    live = tools.compare_models(source="live")
    assert live and all("wape" in row for row in live)
    st = tools.get_network_status()
    assert "kpis" in st and "deployment" in st


def test_call_tool_path_and_errors(tools) -> None:  # type: ignore[no-untyped-def]
    out = client_run(tools.mcp.call_tool("list_stations", {}))
    assert not out.is_error
    data = json.loads(out.content[0].text)
    rows = data if isinstance(data, list) else [data]
    assert {r["station_id"] for r in rows} >= {"S_A"}
    with pytest.raises(ValueError, match="no station matches"):
        tools.resolve_station("nowhere")
