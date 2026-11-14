"""Read helpers that turn database rows into the panel format the feature builder expects."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from power_api.db import models as m
from power_api.db.session import from_db, to_db

H = 24
DEMAND_COLS = ("energy_kwh", "occ_ratio", "idle_ratio", "active_sessions", "occupied_ports", "arrivals", "is_saturated")


async def stations(session: AsyncSession, modeled_only: bool = True) -> list[m.Station]:
    q = select(m.Station)
    if modeled_only:
        q = q.where(m.Station.is_modeled.is_(True))
    return list((await session.execute(q.order_by(m.Station.station_id))).scalars())


async def sim_state(session: AsyncSession) -> m.SimState | None:
    return await session.get(m.SimState, 1)


async def _frame(session: AsyncSession, stmt) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    res = await session.execute(stmt)
    return pd.DataFrame(res.mappings().all())


async def weather(session: AsyncSession, start: datetime, end: datetime, origin: datetime) -> pd.DataFrame:
    """One row per (station, valid hour) in [start, end].

    Vintage rule: a *forecast* row counts only if it was issued at or before the origin T, and the
    latest such vintage wins. *Archive* rows (observed weather) are the fallback; in REPLAY mode they
    act as a perfect forecast, the documented assumption of the training data.
    """
    w = await _frame(
        session,
        select(m.WeatherHourly.__table__).where(
            and_(m.WeatherHourly.valid_time >= start, m.WeatherHourly.valid_time <= end)
        ),
    )
    if w.empty:
        return pd.DataFrame(columns=["station_id", "ts", "temperature_2m_c", "precipitation_mm"])
    w = w.loc[(w["source"] == "archive") | (w["issued_at"] <= origin)]
    w["rank"] = np.where(w["source"] == "forecast", 0, 1)  # forecast vintages preferred over archive
    w = w.sort_values(["station_id", "valid_time", "rank", "issued_at"], ascending=[True, True, True, False])
    w = w.drop_duplicates(["station_id", "valid_time"])
    return w.rename(columns={"valid_time": "ts"})[["station_id", "ts", "temperature_2m_c", "precipitation_mm"]]


async def serving_panel(session: AsyncSession, origin: pd.Timestamp, hours: int) -> pd.DataFrame:
    """Gap-free panel for every modeled station: observed buffer up to T, NaN demand for T+1..T+24.

    Missing buffer hours are imputed from the same hour one week earlier (else 0) and flagged
    ``is_imputed``; the forecast agent marks a station degraded after 6 consecutive imputed hours.
    """
    T = to_db(origin)
    start = T - timedelta(hours=hours - 1)
    end = T + timedelta(hours=H)
    sts = [s.station_id for s in await stations(session)]
    buf = await _frame(session, select(m.LiveBuffer.__table__).where(and_(m.LiveBuffer.ts >= start, m.LiveBuffer.ts <= T)))
    wx = await weather(session, start, end, T)
    grid = pd.MultiIndex.from_product([sts, pd.date_range(start, end, freq="h")], names=["station_id", "ts"])
    panel = pd.DataFrame(index=grid).reset_index()
    if not buf.empty:
        panel = panel.merge(buf[["station_id", "ts", *DEMAND_COLS, "is_imputed"]], on=["station_id", "ts"], how="left")
    else:
        for c in (*DEMAND_COLS, "is_imputed"):
            panel[c] = np.nan
    panel = panel.merge(wx, on=["station_id", "ts"], how="left")
    past = panel["ts"] <= T
    missing = past & panel["energy_kwh"].isna()
    if missing.any():
        panel = panel.sort_values(["station_id", "ts"]).reset_index(drop=True)
        week_ago = panel.groupby("station_id")["energy_kwh"].shift(168)
        missing = (panel["ts"] <= T) & panel["energy_kwh"].isna()
        panel.loc[missing, "energy_kwh"] = week_ago[missing].fillna(0.0)
        for c in ("occ_ratio", "active_sessions", "occupied_ports", "arrivals"):
            panel.loc[missing, c] = panel.loc[missing, c].fillna(0)
        panel.loc[missing, "is_saturated"] = False
        panel.loc[missing, "is_imputed"] = True
    panel["is_imputed"] = panel["is_imputed"].astype("boolean").fillna(False).astype(bool)
    for c in ("temperature_2m_c", "precipitation_mm"):
        panel[c] = panel.groupby("station_id")[c].transform(lambda s: s.ffill().bfill())
    panel["is_saturated"] = panel["is_saturated"].astype("boolean").fillna(False).astype(bool) & (panel["ts"] <= T)
    panel["ts"] = pd.to_datetime(panel["ts"]).dt.tz_localize("UTC")
    return panel.sort_values(["station_id", "ts"]).reset_index(drop=True)


async def latest_origin(session: AsyncSession) -> pd.Timestamp | None:
    s = await sim_state(session)
    return from_db(s.sim_now) if s else None
