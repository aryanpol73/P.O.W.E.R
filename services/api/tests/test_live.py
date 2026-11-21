"""LIVE_SHADOW: the clock follows real time, closes an hour only once it has ended, and never runs out."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from power_api import simulator as sim_mod
from power_api.agents import weather as weather_agent

HOUR = timedelta(hours=1)


@pytest.fixture
def clock(client: TestClient, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """A settable wall clock two weeks into the test period; weather sync stubbed (no network)."""
    sim = client.app.state.sim
    start, _ = sim.test_window()
    now = {"t": start + timedelta(days=14, hours=3, minutes=20)}
    monkeypatch.setattr(sim_mod, "utcnow", lambda: now["t"])

    async def no_sync(*_a, **_k) -> int:
        return 0

    monkeypatch.setattr(weather_agent, "sync", no_sync)
    return now


def test_live_starts_at_the_latest_finished_hour(client: TestClient, clock: dict[str, datetime]) -> None:
    st = client.post("/api/v1/sim/control", json={"action": "reset", "mode": "LIVE_SHADOW"}).json()
    T = datetime.fromisoformat(st["sim_now"]).replace(tzinfo=None)
    now = clock["t"]
    assert T + HOUR <= now < T + 2 * HOUR  # hour T has ended, the one after it has not
    assert st["mode"] == "LIVE_SHADOW" and st["paused"] is False
    assert st["week_offset_hours"] % 168 == 0 and st["week_offset_hours"] > 0
    start, end = client.app.state.sim.test_window()
    assert start <= T - timedelta(hours=st["week_offset_hours"]) <= end  # demand comes from held-out data
    assert datetime.fromisoformat(st["next_update"]).replace(tzinfo=None) == T + 2 * HOUR
    fc = client.get("/api/v1/stations/S_A/forecast")
    assert fc.status_code == 200, fc.text  # forecast made for the hours ahead of the real clock


def test_live_waits_for_the_real_hour_then_closes_it(client: TestClient, clock: dict[str, datetime]) -> None:
    T = datetime.fromisoformat(
        client.post("/api/v1/sim/control", json={"action": "reset", "mode": "LIVE_SHADOW"}).json()["sim_now"]
    ).replace(tzinfo=None)
    assert client.post("/api/v1/sim/tick").json()["waiting"] is True  # too early: nothing changes
    assert datetime.fromisoformat(client.get("/api/v1/sim/state").json()["sim_now"]).replace(tzinfo=None) == T
    clock["t"] += HOUR
    res = client.post("/api/v1/sim/tick").json()
    assert "waiting" not in res and datetime.fromisoformat(res["sim_now"]).replace(tzinfo=None) == T + HOUR


def test_live_wraps_when_the_test_period_runs_out(client: TestClient, clock: dict[str, datetime]) -> None:
    sim = client.app.state.sim
    start, end = sim.test_window()
    T = start + timedelta(days=14)
    off = sim.live_offset(T)
    assert sim.live_offset(T + HOUR, off) == off  # continuous while the source hour is in the test period
    far = end + timedelta(hours=off) + 5 * HOUR  # source hour would be past the end
    new = sim.live_offset(far, off)
    assert new != off and new % 168 == 0
    assert start <= far - timedelta(hours=new) < start + timedelta(days=7)  # back to the first week, same weekday


def test_live_origin_sits_on_the_deployment_grid(client: TestClient) -> None:
    sim = client.app.state.sim
    assert sim.live_origin(datetime(2026, 11, 21, 10, 45)) == datetime(2026, 11, 21, 9, 0)  # grid minute 0 here
    sim.engine.deployment = type("D", (), {"grid_minute": 30})()  # IST hours start at :30 UTC
    try:
        assert sim.live_origin(datetime(2026, 11, 21, 10, 45)) == datetime(2026, 11, 21, 9, 30)
        assert sim.live_origin(datetime(2026, 11, 21, 10, 15)) == datetime(2026, 11, 21, 8, 30)
    finally:
        sim.engine.deployment = None
