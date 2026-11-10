"""WebSocket /ws/stream: subscribe to channels, receive pipeline events, 20 s heartbeat."""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
CHANNELS = {"telemetry", "forecast", "alerts", "drift", "sim"}


@router.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    await ws.accept()
    bus = ws.app.state.bus
    q = bus.subscribe()
    wanted: set[str] = set(CHANNELS)

    async def reader() -> None:
        nonlocal wanted
        while True:
            msg = await ws.receive_json()
            if msg.get("op") == "subscribe":
                wanted = {c for c in msg.get("channels", []) if c in CHANNELS} or set(CHANNELS)
                await ws.send_json({"channel": "control", "event": "subscribed", "data": {"channels": sorted(wanted)}})

    task = asyncio.create_task(reader())
    try:
        await ws.send_json({"channel": "control", "event": "hello", "data": {"channels": sorted(wanted)}})
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=20.0)
                if msg["channel"] in wanted:
                    await ws.send_json(msg)
            except TimeoutError:
                await ws.send_json({"channel": "control", "event": "ping", "data": {}})
            if task.done():
                break
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, Exception):
            await task
        bus.unsubscribe(q)
