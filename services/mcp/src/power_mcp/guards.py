"""Guard-rails for the MCP server: rate limit, bounded inputs, response scrubbing."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

# Fields that are never passed to an AI client (none are user-level today; this is defence in depth).
SCRUB = {"session_id", "user_id", "driver_id", "evse_id", "mac_address", "email"}


class RateLimitError(RuntimeError):
    pass


class RateLimiter:
    """Sliding-window limiter: at most ``limit`` calls per ``window_s`` seconds (default 30/min)."""

    def __init__(self, limit: int = 30, window_s: float = 60.0) -> None:
        self.limit, self.window_s = limit, window_s
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def check(self) -> None:
        now = time.monotonic()
        with self._lock:
            while self._calls and now - self._calls[0] > self.window_s:
                self._calls.popleft()
            if len(self._calls) >= self.limit:
                wait = self.window_s - (now - self._calls[0])
                raise RateLimitError(f"rate limit: {self.limit} calls per {self.window_s:.0f} s; retry in {wait:.0f} s")
            self._calls.append(now)


def scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if k not in SCRUB}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def clamp(v: float, lo: float, hi: float, name: str) -> float:
    if not (lo <= v <= hi):
        raise ValueError(f"{name} must be between {lo} and {hi} (got {v})")
    return v
