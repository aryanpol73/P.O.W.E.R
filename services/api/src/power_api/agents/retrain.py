"""A7 Retrain (champion-challenger) agent: operator-triggered, runs in a worker thread, gated promotion.

Data: observed history only (``historical_telemetry`` up to the simulation clock, in the source timeline),
so a challenger can never see an hour the live console has not yet replayed. The heavy lifting is
``power_ml.retrain.fit_challenger``; this module loads the data, stores the run and hot-swaps on a pass.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from power_api import repo
from power_api.agents import base
from power_api.db import models as m
from power_api.serving.engine import ServingEngine
from power_core.config import RetrainCfg
from power_ml.retrain import fit_challenger
from power_ml.training import QUANTILE

COLS = ("energy_kwh", "occ_ratio", "idle_ratio", "active_sessions", "occupied_ports", "arrivals", "is_saturated",
        "temperature_2m_c", "precipitation_mm")


async def _history(session: AsyncSession, start: Any, end: Any) -> pd.DataFrame:
    rows = (await session.execute(
        select(m.HistoricalTelemetry.__table__).where(m.HistoricalTelemetry.ts >= start, m.HistoricalTelemetry.ts <= end)
    )).mappings().all()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize("UTC")
    df["is_saturated"] = df["is_saturated"].astype(bool)
    return df[["station_id", "ts", *COLS]].sort_values(["station_id", "ts"]).reset_index(drop=True)


def _fit(engine: ServingEngine, panel: pd.DataFrame, as_of: pd.Timestamp, rcfg: RetrainCfg) -> Any:
    cfg = engine.manifest["config"]
    q = {k: v for k, v in cfg["lgbm_quantile"].items() if k != "alphas"}
    return fit_challenger(
        engine._to_model(panel),
        engine.stations,
        engine.fcfg,
        rcfg,
        as_of=as_of,
        champion=engine.models[QUANTILE],
        champion_cqr=engine.calibrators[f"{QUANTILE}_cqr"],
        params=q,
        origin_stride=int(cfg["origin_stride"]["lgbm_quantile"]),
        early_stopping_rounds=int(cfg["early_stopping"]["rounds"]),
        holdout_frac=float(cfg["early_stopping"]["holdout_frac"]),
        tz=engine.tz,
        holidays=engine.holidays,
        seed=int(cfg.get("seed", 42)),
        n_jobs=int(cfg.get("n_jobs", -1)),
    )


async def run(
    maker: async_sessionmaker[AsyncSession], engine: ServingEngine, rcfg: RetrainCfg, warmup_hours: int, promote: bool
) -> dict[str, Any]:
    async with maker() as s:
        st = await repo.sim_state(s)
        if st is None:
            raise RuntimeError("simulator not initialised")
        T_src = st.sim_now - timedelta(hours=st.week_offset_hours)  # observed hours, in the stored timeline
        run_id = "rt" + pd.Timestamp(T_src).strftime("%Y%m%dT%H%M")
        if await base.already_done(s, "retrain", run_id):
            return {"run_id": run_id, "skipped": True, "detail": "already retrained at this clock time"}
        await base.record(s, "retrain", run_id, "running", {"promote_requested": promote})
        await s.commit()
        span = rcfg.window_days + rcfg.calib_days + rcfg.shadow_days + 2
        panel = await _history(s, T_src - timedelta(days=span, hours=warmup_hours), T_src)

    as_of = pd.Timestamp(T_src).tz_localize("UTC")
    try:
        res = await asyncio.to_thread(_fit, engine, panel, as_of, rcfg)
    except Exception as exc:  # noqa: BLE001 - recorded and surfaced on the Ops page
        async with maker() as s:
            await base.record(s, "retrain", run_id, "error", {"error": str(exc)})
            await s.commit()
        return {"run_id": run_id, "status": "error", "error": str(exc)}

    promoted = bool(promote and res.passed)
    record = {
        "as_of": as_of.isoformat(),
        "passed": res.passed,
        "promoted": promoted,
        "gate": res.gate,
        "shadow": res.shadow,
        "windows": res.windows,
        "train_rows": res.train_rows,
        "seconds": res.seconds,
        "per_station": {engine._site_id(k): v for k, v in res.per_station.items()},
        "base_version": engine.version,
    }
    await asyncio.to_thread(engine.registry.save_run, run_id, res.model, res.cqr.to_dict(), record)
    if promoted:
        engine.promote(run_id, res.model, res.cqr)
        engine.registry.set_active(run_id)
    async with maker() as s:
        await base.record(s, "retrain", run_id, "ok", record)
        await s.commit()
    return {"run_id": run_id, "status": "ok", **record}


def rollback(engine: ServingEngine) -> dict[str, Any]:
    prev = engine.live_run
    engine.rollback()
    engine.registry.set_active(None)
    return {"rolled_back": prev, "live_champion": engine.version_of(QUANTILE)}


def activate(engine: ServingEngine, run_id: str) -> dict[str, Any]:
    """Promote a stored run manually (e.g. one that failed only the PICP gate, after review)."""
    model, cqr = engine.registry.load(run_id)
    engine.promote(run_id, model, cqr)
    engine.registry.set_active(run_id)
    return {"live_champion": engine.version_of(QUANTILE)}
