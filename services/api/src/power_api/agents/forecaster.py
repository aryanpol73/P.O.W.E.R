"""A3 Forecast agent: on each closed hour, run every model for every station and store predictions + SHAP."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pandas as pd
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from power_api import repo
from power_api.db import models as m
from power_api.db.session import from_db, to_db
from power_api.serving.engine import Forecast, ServingEngine


async def run(session: AsyncSession, engine: ServingEngine, T: datetime, history_hours: int) -> Forecast:
    origin = from_db(T)
    panel = await repo.serving_panel(session, origin, history_hours)
    fc = await asyncio.to_thread(engine.forecast, panel, origin)
    cap = {s.station_id: s.transformer_kw for s in await repo.stations(session)}
    await session.execute(delete(m.ModelPrediction).where(m.ModelPrediction.forecast_origin == T))
    await session.execute(delete(m.PredictionExplanation).where(m.PredictionExplanation.forecast_origin == T))
    rows = fc.rows
    session.add_all(
        m.ModelPrediction(
            station_id=r.station_id,
            model_name=r.model,
            forecast_origin=T,
            horizon_h=int(r.h),
            target_time=to_db(r.target_time),
            model_version=engine.version_of(r.model),
            y_mean=float(r.y_mean),
            q05=float(r.q05),
            q50=float(r.q50),
            q95=float(r.q95),
            load_ratio_q95=float(r.q95) / cap[r.station_id] if cap.get(r.station_id) else None,
            is_degraded=fc.degraded.get(r.station_id, False),
            latency_ms=round(fc.latency_ms, 1),
        )
        for r in rows.itertuples(index=False)
    )
    if not fc.explanations.empty:
        session.add_all(
            m.PredictionExplanation(
                station_id=e.station_id,
                forecast_origin=T,
                horizon_h=int(e.h),
                model_name=engine.champion,
                base_value=float(e.base_value),
                contributions=e.contributions,
            )
            for e in fc.explanations.itertuples(index=False)
        )
    return fc


def champion_frame(fc: Forecast, champion: str) -> pd.DataFrame:
    return fc.rows.loc[fc.rows["model"] == champion].copy()
