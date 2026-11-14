"""Web Push: subscription lifecycle, coalesced alert payloads, level filtering, dead-endpoint clean-up."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from power_api import push

SUB = {"endpoint": "https://push.example.test/send/abc", "keys": {"p256dh": "B" * 87, "auth": "A" * 22}}
SUB2 = {"endpoint": "https://push.example.test/send/def", "keys": {"p256dh": "C" * 87, "auth": "D" * 22}}


def test_subscribe_validate_unsubscribe(client: TestClient) -> None:
    key = client.get("/api/v1/push/vapid-public-key").json()
    assert len(key["public_key"]) == 87  # base64url of a 65-byte uncompressed P-256 point
    assert client.post("/api/v1/push/subscribe", json={"subscription": SUB}).status_code == 201
    bad = {**SUB, "endpoint": "http://insecure.example/x"}
    assert client.post("/api/v1/push/subscribe", json={"subscription": bad}).status_code == 422
    assert client.post("/api/v1/push/unsubscribe", json={"endpoint": SUB["endpoint"]}).json() == {"removed": 1}


def test_notify_coalesces_filters_and_prunes(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    client.post("/api/v1/push/subscribe", json={"subscription": SUB, "min_level": "CRITICAL"})
    client.post("/api/v1/push/subscribe", json={"subscription": SUB2, "min_level": "WARNING"})
    calls: list[tuple[str, dict]] = []
    gone = {SUB2["endpoint"]}

    def fake_send(keys, info, payload):  # type: ignore[no-untyped-def]
        calls.append((info["endpoint"], json.loads(payload)))
        return 410 if info["endpoint"] in gone else 201

    monkeypatch.setattr(push, "_send_one", fake_send)
    app = client.app
    raised = [
        {"station_id": "S_A", "level": "WARNING", "message": "WARNING: S_A at 72%"},
        {"station_id": "S_B", "level": "CRITICAL", "message": "CRITICAL: S_B at 91%"},
        {"station_id": "S_C", "level": "CRITICAL", "message": "CRITICAL: S_C at 88%"},
    ]

    async def go() -> dict:
        async with app.state.maker() as s:
            r = await push.notify(s, app.state.sim.vapid(), raised)
            await s.commit()
            return r

    res = client.portal.call(go)
    assert res == {"sent": 1, "removed": 1}
    by = dict(calls)
    assert by[SUB["endpoint"]]["level"] == "CRITICAL" and by[SUB["endpoint"]]["title"].endswith("+1 more")
    assert len(calls) == 2  # one coalesced notification per subscription, not one per alert
    # the 410 subscription was deleted
    assert client.post("/api/v1/push/unsubscribe", json={"endpoint": SUB2["endpoint"]}).json() == {"removed": 0}

    # warnings alone do not reach a CRITICAL-only subscription
    calls.clear()
    client.portal.call(lambda: _notify(app, raised[:1]))
    assert calls == []


async def _notify(app, raised):  # type: ignore[no-untyped-def]
    async with app.state.maker() as s:
        return await push.notify(s, app.state.sim.vapid(), raised)
