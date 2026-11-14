"""Guard-rails: sliding-window rate limit, response scrubbing, bounds."""

import pytest

from power_mcp.guards import RateLimiter, RateLimitError, clamp, scrub


def test_rate_limit() -> None:
    lim = RateLimiter(2, 60)
    lim.check()
    lim.check()
    with pytest.raises(RateLimitError):
        lim.check()


def test_scrub_and_clamp() -> None:
    assert scrub({"a": 1, "session_id": "x", "b": [{"email": "e", "c": 2}]}) == {"a": 1, "b": [{"c": 2}]}
    assert clamp(3, 0, 5, "x") == 3
    with pytest.raises(ValueError):
        clamp(9, 0, 5, "x")
