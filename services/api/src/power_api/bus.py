"""Event bus for WebSocket fan-out.

``InProcessBus`` serves the default deployment, where the worker runs embedded in the API process.
A separate worker process needs a cross-process bus. On PostgreSQL that is ``LISTEN/NOTIFY``
(blueprint section 3.4), not implemented yet because it needs a PostgreSQL server to test against.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any


class InProcessBus:
    def __init__(self, maxsize: int = 256) -> None:
        self._subs: set[asyncio.Queue[dict[str, Any]]] = set()
        self._maxsize = maxsize

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(self._maxsize)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        self._subs.discard(q)

    async def publish(self, channel: str, event: str, data: dict[str, Any]) -> None:
        msg = {"channel": channel, "event": event, "ts": datetime.now(UTC).isoformat(timespec="seconds"), "data": data}
        for q in list(self._subs):
            if q.full():  # slow client: drop its oldest message rather than block the pipeline
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
            q.put_nowait(msg)

    @property
    def subscribers(self) -> int:
        return len(self._subs)
