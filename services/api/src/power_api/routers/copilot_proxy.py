"""Pass-through to the copilot (MCP service) for single-port hosts such as a Hugging Face Space.

Enabled by ``POWER_COPILOT_UPSTREAM`` (e.g. ``http://127.0.0.1:8765``). The web app's server route then
uses the API's own URL as ``POWER_MCP_URL``. The bearer token is forwarded unchanged and checked by the
MCP service; NDJSON replies are streamed back as they arrive.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

router = APIRouter(tags=["copilot"])
MAX_BODY = 8000


def _upstream(request: Request) -> str:
    return request.app.state.settings.copilot_upstream.rstrip("/")


@router.get("/copilot/status")
async def status(request: Request) -> Response:
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            r = await c.get(f"{_upstream(request)}/copilot/status")
        return Response(r.content, status_code=r.status_code, media_type="application/json")
    except httpx.HTTPError:
        return Response('{"enabled": false, "detail": "copilot starting"}', media_type="application/json")


@router.post("/copilot")
async def chat(request: Request) -> Response:
    body = await request.body()
    if len(body) > MAX_BODY:
        return Response('{"detail": "message too long"}', status_code=413, media_type="application/json")
    headers = {"content-type": "application/json"}
    if auth := request.headers.get("authorization"):
        headers["authorization"] = auth
    client = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=180.0))
    try:
        req = client.build_request("POST", f"{_upstream(request)}/copilot", content=body, headers=headers)
        r = await client.send(req, stream=True)
    except httpx.HTTPError:
        await client.aclose()
        return Response('{"detail": "copilot unreachable"}', status_code=502, media_type="application/json")

    async def relay() -> AsyncIterator[bytes]:
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()

    return StreamingResponse(
        relay(),
        status_code=r.status_code,
        media_type=r.headers.get("content-type", "application/x-ndjson"),
        headers={"cache-control": "no-store", "x-accel-buffering": "no"},
    )
