"""Copilot tool loop against a fake Claude API (SSE at the HTTP layer) and the in-process REST API."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

srv = pytest.importorskip("power_mcp.server")
cp = pytest.importorskip("power_mcp.copilot")
anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")
from power_mcp.guards import RateLimiter  # noqa: E402


def _sse(events: list[dict]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def _msg_start(i: int) -> dict:
    return {"type": "message_start", "message": {
        "id": f"msg_{i}", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}}


TOOL_TURN = [
    _msg_start(1),
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Checking the sites. "}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1,
     "content_block": {"type": "tool_use", "id": "toolu_1", "name": "list_stations", "input": {}}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None}, "usage": {"output_tokens": 20}},
    {"type": "message_stop"},
]
ANSWER_TURN = [
    _msg_start(2),
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Three sites are modelled."}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 8}},
    {"type": "message_stop"},
]


@pytest.fixture
def copilot(client: TestClient, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    class Prefixed:  # MCP tools -> in-process API on the fixture's event loop
        def request(self, method: str, path: str, **kw):  # type: ignore[no-untyped-def]
            return client.request(method, "/api/v1" + path, **kw)

    monkeypatch.setattr(srv, "_client", Prefixed())
    monkeypatch.setattr(srv, "limiter", RateLimiter(1000, 60))
    sent: list[dict] = []
    turns = [TOOL_TURN, ANSWER_TURN]

    def handler(req: httpx2.Request) -> httpx2.Response:
        sent.append({"body": json.loads(req.content), "beta": req.headers.get("anthropic-beta", "")})
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=_sse(turns[len(sent) - 1]))

    fake = anthropic.AsyncAnthropic(api_key="test", http_client=anthropic.DefaultAsyncHttpxClient(
        transport=httpx2.MockTransport(handler)))
    return cp.Copilot(srv.mcp, client=fake), sent


def _run(client: TestClient, agen) -> list[dict]:  # type: ignore[no-untyped-def]
    async def collect() -> list[dict]:
        return [e async for e in agen]

    return client.portal.call(collect)


def test_tool_loop_streams_and_keeps_history(client: TestClient, copilot) -> None:  # type: ignore[no-untyped-def]
    bot, sent = copilot
    events = _run(client, bot.chat(None, "Which sites are modelled?"))
    kinds = [e["type"] for e in events]
    assert kinds[0] == "conversation" and kinds[-1] == "done"
    assert {"type": "tool_call", "name": "list_stations", "input": {}} in events
    assert {"type": "tool_result", "name": "list_stations", "ok": True} in events
    assert "".join(e["text"] for e in events if e["type"] == "text") == "Checking the sites. Three sites are modelled."

    first, second = sent
    assert first["body"]["model"] == "claude-opus-5-5"
    assert first["body"]["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in first["beta"]
    assert {t["name"] for t in first["body"]["tools"]} >= {"list_stations", "get_station_forecast"}
    assert all(t["eager_input_streaming"] for t in first["body"]["tools"])
    # Second request: the assistant tool_use turn echoed unchanged, then ONE user turn with the real tool result.
    msgs = second["body"]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert any(b["type"] == "tool_use" and b["id"] == "toolu_1" for b in msgs[1]["content"])
    result = msgs[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "toolu_1" and not result["is_error"]
    assert "S_A" in result["content"]  # the API's real station list

    cid = events[0]["id"]
    _, history = bot.conversations.get(cid)
    assert len(history) == 4  # user, assistant(tool_use), user(tool_result), assistant(answer)


def test_copilot_status_and_disabled_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("GROQ_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(k, raising=False)
    assert cp.enabled() is False
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert cp.enabled() is True
