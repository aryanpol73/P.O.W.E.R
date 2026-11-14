"""A2 Weather Sync agent: live Open-Meteo forecasts for LIVE_SHADOW mode, stored as dated vintages."""

from __future__ import annotations

from datetime import datetime

import httpx
import numpy as np
import pandas as pd
from sqlalchemy import and_, delete
from sqlalchemy.ext.asyncio import AsyncSession

from power_api import repo
from power_api.db import models as m
from power_api.db.session import utcnow

URL = "https://api.open-meteo.com/v1/forecast"


async def sync(
    session: AsyncSession,
    past_days: int = 2,
    forecast_days: int = 2,
    client: httpx.AsyncClient | None = None,
    url: str = URL,
    grid_minute: int = 0,
) -> int:
    """Fetch one batched request for every station; store rows with ``issued_at = now``.

    ``grid_minute``: the minute past each UTC hour on which the panel grid sits (30 for an IST deployment,
    whose local hours start at :30 UTC); hourly forecasts are linearly interpolated onto that grid.
    """
    sts = await repo.stations(session)
    if not sts:
        return 0
    params = {
        "latitude": ",".join(f"{s.latitude:.5f}" for s in sts),
        "longitude": ",".join(f"{s.longitude:.5f}" for s in sts),
        "hourly": "temperature_2m,precipitation",
        "timezone": "GMT",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    own = client is None
    client = client or httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "power-api/0.1"})
    try:
        resp = None
        for attempt in range(3):
            try:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 2:
                    raise
        assert resp is not None
        payload = resp.json()
    finally:
        if own:
            await client.aclose()
    items = payload if isinstance(payload, list) else [payload]
    issued = utcnow().replace(minute=0, second=0, microsecond=0)
    n = 0
    for st, item in zip(sts, items, strict=True):
        h = item["hourly"]
        times_on_hour = pd.to_datetime(h["time"])
        temp = np.asarray(h["temperature_2m"], dtype=float)
        rain = np.asarray(h["precipitation"], dtype=float)
        if grid_minute:
            target = times_on_hour[:-1] + pd.Timedelta(minutes=grid_minute)
            x, tx = times_on_hour.asi8.astype(float), target.asi8.astype(float)
            temp, rain = np.interp(tx, x, temp), np.interp(tx, x, rain)
            times_on_hour = target
        times = times_on_hour.to_pydatetime()
        await session.execute(
            delete(m.WeatherHourly).where(and_(m.WeatherHourly.station_id == st.station_id, m.WeatherHourly.issued_at == issued))
        )
        session.add_all(
            m.WeatherHourly(
                station_id=st.station_id,
                valid_time=t,
                issued_at=issued,
                source="forecast",
                temperature_2m_c=tc,
                precipitation_mm=p,
            )
            for t, tc, p in zip(times, temp.tolist(), rain.tolist(), strict=True)
        )
        n += len(times)
    return n


def week_offset_hours(source_origin: datetime, now: datetime) -> int:
    """Whole weeks that move a historical origin onto (or just before) the current hour."""
    weeks = int((now - source_origin).total_seconds() // (168 * 3600))
    return max(0, weeks) * 168
