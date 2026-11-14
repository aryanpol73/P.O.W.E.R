"""A1 Telemetry Replay agent: closes one simulated hour by copying held-out history into the live buffer.

* REPLAY:      buffer ts = historical ts (offset 0); accuracy is a genuine out-of-sample measurement.
* LIVE_SHADOW: buffer ts = historical ts + whole weeks, so weekday/hour line up with today.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import and_, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from power_api.db import models as m

COLS = ("energy_kwh", "occ_ratio", "idle_ratio", "active_sessions", "occupied_ports", "arrivals", "is_saturated")
RETENTION = timedelta(days=60)


async def copy_hours(session: AsyncSession, start: datetime, end: datetime, offset_hours: int) -> int:
    """Copy historical rows with source ts in [start - offset, end - offset] into the buffer at ts + offset."""
    off = timedelta(hours=offset_hours)
    rows = (
        (
            await session.execute(
                select(m.HistoricalTelemetry).where(
                    and_(m.HistoricalTelemetry.ts >= start - off, m.HistoricalTelemetry.ts <= end - off)
                )
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        return 0
    await session.execute(delete(m.LiveBuffer).where(and_(m.LiveBuffer.ts >= start, m.LiveBuffer.ts <= end)))
    session.add_all(
        m.LiveBuffer(station_id=r.station_id, ts=r.ts + off, source_ts=r.ts, is_imputed=False, **{c: getattr(r, c) for c in COLS})
        for r in rows
    )
    return len(rows)


async def close_hour(session: AsyncSession, T: datetime, offset_hours: int) -> int:
    n = await copy_hours(session, T, T, offset_hours)
    await session.execute(delete(m.LiveBuffer).where(m.LiveBuffer.ts < T - RETENTION))
    return n
