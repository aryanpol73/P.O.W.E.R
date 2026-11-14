"""In-app copilot: Claude answers grid questions using exactly the MCP server's tools.

The copilot never sees the database. Its tool list is the MCP server's own (same names, descriptions and
JSON schemas), and every tool call goes through ``mcp.call_tool``, so the numbers it quotes are the ones the
REST API, the console and any external MCP client see.

Served by the MCP process at ``POST /copilot`` (newline-delimited JSON stream) and reached from the browser
through the web app's server route, which keeps ``GROQ_API_KEY`` and ``POWER_COPILOT_TOKEN`` server-side.
Disabled (``GET /copilot/status`` -> enabled: false) when no Anthropic credentials are configured.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from power_mcp.guards import RateLimiter, RateLimitError

MODEL = os.environ.get("POWER_COPILOT_MODEL", "claude-opus-5-5")
EFFORT = os.environ.get("POWER_COPILOT_EFFORT", "low")  # chat: low is quick and usually enough
MAX_TOOL_ROUNDS = 8
MAX_CONVERSATIONS = 200
CONVERSATION_TTL_S = 2 * 3600
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = (
    "You are the P.O.W.E.R. grid copilot inside an operator console for public EV charging sites. "
    "Answer with the tools; never invent numbers. Every number you state must come from a tool result in this "
    "conversation, and say which tool it came from in a short parenthesis, e.g. (get_station_forecast). "
    "Times from tools are UTC ISO-8601: convert them to the deployment's local time zone "
    "(get_network_status -> deployment.timezone) before showing them. Energy is kWh per hour; risk compares the "
    "q95 forecast with an assumed transformer rating, so say 'assumed' when you mention capacity. If a scenario "
    "returns ood_flags, say the inputs are outside the training range and the effect is indicative. The console may "
    "be a digital twin whose demand is replayed from the training data: say so when asked how accurate it is for "
    "local drivers. Keep answers short and scannable: a direct answer first, then at most a few bullets."
)


class Conversations:
    """Server-side history per conversation, kept unchanged (append-only) so thinking blocks replay intact."""

    def __init__(self) -> None:
        self._items: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()

    def get(self, cid: str | None) -> tuple[str, list[dict[str, Any]]]:
        now = time.monotonic()
        for k in [k for k, (t, _) in self._items.items() if now - t > CONVERSATION_TTL_S]:
            self._items.pop(k, None)
        if cid and cid in self._items:
            self._items.move_to_end(cid)
            return cid, self._items[cid][1]
        cid = uuid.uuid4().hex
        self._items[cid] = (now, [])
        while len(self._items) > MAX_CONVERSATIONS:
            self._items.popitem(last=False)
        return cid, self._items[cid][1]

    def touch(self, cid: str) -> None:
        if cid in self._items:
            self._items[cid] = (time.monotonic(), self._items[cid][1])


def enabled() -> bool:
    return bool(os.environ.get("GROQ_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_PROFILE"))


class Copilot:
    def __init__(self, server: Any, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.server = server  # the MCPServer whose tools the copilot uses
        self.client = client
        self.conversations = Conversations()
        self.limiter = RateLimiter(int(os.environ.get("POWER_COPILOT_RATE_PER_MIN", "10")), 60.0)
        self._tools: list[dict[str, Any]] | None = None

    async def tools(self) -> list[dict[str, Any]]:
        if self._tools is None:
            listed = await self.server.list_tools()
            self._tools = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": t.input_schema,
                    "eager_input_streaming": True,  # inputs stream as generated; validated by the MCP layer
                }
                for t in sorted(listed, key=lambda t: t.name)  # deterministic order keeps the prompt cache warm
            ]
        return self._tools

    async def _run_tool(self, name: str, args: Any) -> tuple[str, bool]:
        if not isinstance(args, dict):
            return json.dumps({"INVALID_JSON": json.dumps(args)}), True
        try:
            res = await self.server.call_tool(name, args)
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
            return f"{type(exc).__name__}: {exc}", True
        text = "\n".join(getattr(c, "text", "") for c in (res.content or []))
        return text[:20_000], bool(res.is_error)

    async def chat(self, conversation_id: str | None, text: str) -> AsyncIterator[dict[str, Any]]:
        """Yield events: conversation, text, tool_call, tool_result, done, error."""
        try:
            self.limiter.check()
        except RateLimitError as exc:
            yield {"type": "error", "message": str(exc)}
            return
        if self.client is None:
            self.client = anthropic.AsyncAnthropic()
        cid, messages = self.conversations.get(conversation_id)
        yield {"type": "conversation", "id": cid}
        messages.append({"role": "user", "content": text})
        tools = await self.tools()
        json_retries = 0
        for _ in range(MAX_TOOL_ROUNDS):
            try:
                async with self.client.beta.messages.stream(
                    model=MODEL,
                    max_tokens=16000,
                    system=SYSTEM,
                    tools=tools,
                    messages=messages,
                    output_config={"effort": EFFORT},
                    cache_control={"type": "ephemeral"},
                    betas=[FALLBACK_BETA],
                    fallbacks="default",  # on a policy decline, re-run on a fallback model in the same call
                ) as stream:
                    async for event in stream:
                        if event.type == "text":
                            yield {"type": "text", "text": event.text}
                    response = await stream.get_final_message()
                json_retries = 0
            except ValueError:
                # Tool-input JSON the SDK could not parse at all: no tool_use id to answer, so re-issue the turn.
                json_retries += 1
                if json_retries > 2:
                    yield {"type": "error", "message": "the model produced unreadable tool input twice; try rephrasing"}
                    return
                continue
            except anthropic.RateLimitError:
                yield {"type": "error", "message": "Claude API rate limit reached; try again in a minute."}
                return
            except anthropic.AuthenticationError:
                yield {"type": "error", "message": "Copilot is not configured: the Anthropic API key was rejected."}
                return
            except anthropic.APIStatusError as exc:
                yield {"type": "error", "message": f"Claude API error {exc.status_code}."}
                return
            except anthropic.APIConnectionError:
                yield {"type": "error", "message": "Cannot reach the Claude API."}
                return

            # Append the assistant turn unchanged (thinking blocks included): history stays append-only.
            messages.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in response.content]})
            self.conversations.touch(cid)
            if response.stop_reason == "pause_turn":
                continue
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason == "refusal":
                yield {"type": "error", "message": "The request was declined."}
                return
            if not tool_uses:
                yield {"type": "done", "model": response.model}
                return
            if response.stop_reason == "max_tokens":
                yield {"type": "error", "message": "The answer was cut off; ask a narrower question."}
                return
            results = []
            for b in tool_uses:
                yield {"type": "tool_call", "name": b.name, "input": b.input}
                out, is_error = await self._run_tool(b.name, b.input)
                yield {"type": "tool_result", "name": b.name, "ok": not is_error}
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": is_error})
            messages.append({"role": "user", "content": results})  # all results in one user turn
        yield {"type": "error", "message": f"stopped after {MAX_TOOL_ROUNDS} tool rounds"}
