"""In-app copilot: an LLM answers grid questions using exactly the MCP server's tools.

The copilot never sees the database. Its tool list is the MCP server's own (same names, descriptions and
JSON schemas), and every tool call goes through ``mcp.call_tool``, so the numbers it quotes are the ones the
REST API, the console and any external MCP client see.

Two interchangeable providers (``POWER_COPILOT_PROVIDER``; default: whichever key is set, Groq first):

* ``groq``      - Groq's OpenAI-compatible chat API (free tier), default model ``openai/gpt-oss-120b``.
* ``anthropic`` - Claude via the Anthropic Messages API, default model ``claude-opus-5-5``.

Served by the MCP process at ``POST /copilot`` (newline-delimited JSON stream) and reached from the browser
through the web app's server route, which keeps the provider key and ``POWER_COPILOT_TOKEN`` server-side.
Disabled (``GET /copilot/status`` -> enabled: false) when the selected provider has no key.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from typing import Any

from power_mcp.guards import RateLimiter, RateLimitError

DEFAULT_MODEL = {"groq": "openai/gpt-oss-120b", "anthropic": "claude-opus-5-5"}
MAX_TOOL_ROUNDS = 8
MAX_CONVERSATIONS = 200
CONVERSATION_TTL_S = 2 * 3600
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = (
    "You are the P.O.W.E.R. grid copilot inside an operator console for public EV charging sites. "
    "Answer with the tools; never invent numbers. Every number you state must come from a tool result in this "
    "conversation, and say which tool it came from in a short parenthesis, e.g. (get_station_forecast). "
    "Tool results give each time in UTC and as a '*_local' field in the deployment's time zone: always show the "
    "'*_local' value (e.g. '21:00 IST'), never the UTC one, and always say which hour you are describing. "
    "Words like 'tonight' or 'this evening' mean hours later today in local time: check get_station_forecast first "
    "to find them, then explain that horizon. Energy is kWh per hour; risk compares the "
    "q95 forecast with an assumed transformer rating, so say 'assumed' when you mention capacity. If a scenario "
    "returns ood_flags, say the inputs are outside the training range and the effect is indicative. The console may "
    "be a digital twin whose demand is replayed from the training data: say so when asked how accurate it is for "
    "local drivers. Keep answers short and scannable: a direct answer first, then at most a few bullets. "
    "Formatting: plain sentences and '-' bullets; **bold** is fine; no tables, no headings."
)


# ------------------------------------------------------------------ configuration (read at call time)
def _has_anthropic() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or os.environ.get("ANTHROPIC_PROFILE"))


def provider() -> str:
    p = os.environ.get("POWER_COPILOT_PROVIDER", "").strip().lower()
    if p in DEFAULT_MODEL:
        return p
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    return "anthropic" if _has_anthropic() else "groq"


def model() -> str:
    """POWER_COPILOT_MODEL, unless it names the other provider's family (e.g. a Claude id with Groq)."""
    p, m = provider(), os.environ.get("POWER_COPILOT_MODEL", "").strip()
    if not m or (p == "groq" and m.startswith("claude")) or (p == "anthropic" and not m.startswith("claude")):
        return DEFAULT_MODEL[p]
    return m


def effort() -> str:
    return os.environ.get("POWER_COPILOT_EFFORT", "low").strip().lower() or "low"


def enabled() -> bool:
    return bool(os.environ.get("GROQ_API_KEY")) if provider() == "groq" else _has_anthropic()


def status() -> dict[str, Any]:
    return {"enabled": enabled(), "provider": provider(), "model": model(), "effort": effort()}


# ------------------------------------------------------------------ conversation store
class Conversations:
    """Server-side history per conversation, append-only (provider-native message format)."""

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


# ------------------------------------------------------------------ copilot
class Copilot:
    def __init__(self, server: Any, client: Any = None) -> None:
        self.server = server  # the MCPServer whose tools the copilot uses
        self.client = client  # AsyncGroq or AsyncAnthropic; created lazily for the selected provider
        self.conversations = Conversations()
        self.limiter = RateLimiter(int(os.environ.get("POWER_COPILOT_RATE_PER_MIN", "10")), 60.0)
        self._listed: list[Any] | None = None

    async def _mcp_tools(self) -> list[Any]:
        if self._listed is None:  # deterministic order keeps prompt prefixes stable (caching)
            self._listed = sorted(await self.server.list_tools(), key=lambda t: t.name)
        return self._listed

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
        cid, messages = self.conversations.get(conversation_id)
        yield {"type": "conversation", "id": cid}
        messages.append({"role": "user", "content": text})
        run = self._chat_groq if provider() == "groq" else self._chat_anthropic
        async for ev in run(messages):
            yield ev
        self.conversations.touch(cid)

    # -------------------------------------------------------------- Groq (OpenAI-compatible tool calling)
    async def _chat_groq(self, messages: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        import groq

        if self.client is None:
            self.client = groq.AsyncGroq()  # GROQ_API_KEY
        mdl = model()
        tools = [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                   "parameters": t.input_schema}} for t in await self._mcp_tools()]
        extra: dict[str, Any] = {}
        if mdl.startswith("openai/gpt-oss"):  # reasoning models: effort control, keep reasoning out of the reply
            extra = {"reasoning_effort": {"xhigh": "high", "max": "high"}.get(effort(), effort()),
                     "include_reasoning": False}
        bad_calls = 0
        for _ in range(MAX_TOOL_ROUNDS):
            text_parts: list[str] = []
            calls: dict[int, dict[str, Any]] = {}
            finish = None
            try:
                stream = await self.client.chat.completions.create(
                    model=mdl,
                    messages=[{"role": "system", "content": SYSTEM}, *messages],
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.2,
                    max_completion_tokens=4096,
                    stream=True,
                    **extra,
                )
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    ch = chunk.choices[0]
                    d = ch.delta
                    if d.content:
                        text_parts.append(d.content)
                        yield {"type": "text", "text": d.content}
                    for tc in d.tool_calls or []:
                        slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
                    if ch.finish_reason:
                        finish = ch.finish_reason
            except groq.BadRequestError as exc:
                # Groq rejects a turn whose generated tool call fails its own parsing ("tool_use_failed"): retry once.
                if "tool_use_failed" in str(exc) and bad_calls < 1:
                    bad_calls += 1
                    continue
                yield {"type": "error", "message": f"Groq rejected the request: {_short(exc)}"}
                return
            except groq.RateLimitError:
                yield {"type": "error", "message": "Groq rate limit reached (free tier); try again in a minute."}
                return
            except groq.AuthenticationError:
                yield {"type": "error", "message": "Copilot is not configured: the Groq API key was rejected."}
                return
            except groq.APIStatusError as exc:
                yield {"type": "error", "message": f"Groq API error {exc.status_code}."}
                return
            except groq.APIConnectionError:
                yield {"type": "error", "message": "Cannot reach the Groq API."}
                return

            ordered = [calls[i] for i in sorted(calls)]
            assistant: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
            if ordered:
                assistant["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                    for c in ordered
                ]
            messages.append(assistant)
            if finish == "length":
                yield {"type": "error", "message": "The answer was cut off; ask a narrower question."}
                return
            if not ordered:
                yield {"type": "done", "model": mdl}
                return
            for c in ordered:
                try:
                    args = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = None  # reported back to the model as INVALID_JSON
                yield {"type": "tool_call", "name": c["name"], "input": args}
                out, is_error = await self._run_tool(c["name"], args)
                yield {"type": "tool_result", "name": c["name"], "ok": not is_error}
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": ("ERROR: " if is_error else "") + out})
        yield {"type": "error", "message": f"stopped after {MAX_TOOL_ROUNDS} tool rounds"}

    # -------------------------------------------------------------- Anthropic (Claude Messages API)
    async def _chat_anthropic(self, messages: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        import anthropic

        if self.client is None:
            self.client = anthropic.AsyncAnthropic()
        tools = [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema,
                  "eager_input_streaming": True} for t in await self._mcp_tools()]
        json_retries = 0
        for _ in range(MAX_TOOL_ROUNDS):
            try:
                async with self.client.beta.messages.stream(
                    model=model(),
                    max_tokens=16000,
                    system=SYSTEM,
                    tools=tools,
                    messages=messages,
                    output_config={"effort": effort()},
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


def _short(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:200]
    return str(exc)[:200]
