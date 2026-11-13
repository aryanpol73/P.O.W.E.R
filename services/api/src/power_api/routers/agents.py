"""Ops endpoints: agent status (A9 watchdog view), drift reports (A5), retrain (A7), discovery (A8)."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from power_api import repo
from power_api.agents import base
from power_api.agents import discovery as discovery_agent
from power_api.agents import retrain as retrain_agent
from power_api.db import models as m
from power_api.db.session import utcnow
from power_api.routers.api import Db, _iso, admin
from power_core.config import threshold_config
from power_ml.training import QUANTILE

router = APIRouter()
log = logging.getLogger("power.agents")
AGENTS = {
    "pipeline": "A1+A3+A6 replay, forecast, alerts (every simulated hour)",
    "drift": "A5 drift detection (each local midnight)",
    "retrain": "A7 champion-challenger retrain (operator-triggered)",
    "discovery": "A8 station discovery from OpenStreetMap",
}


async def _last_runs(s: AsyncSession) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for agent in AGENTS:
        row = (await s.execute(
            select(m.AgentRun)
            .where(m.AgentRun.agent == agent)
            .order_by(m.AgentRun.finished_at.desc(), m.AgentRun.id.desc())
            .limit(1)
        )).scalar()
        n = (await s.execute(select(func.count()).select_from(m.AgentRun).where(m.AgentRun.agent == agent))).scalar_one()
        out[agent] = {
            "description": AGENTS[agent],
            "runs": int(n),
            "last_key": row.idempotency_key if row else None,
            "last_status": row.status if row else None,
            "last_finished_at": _iso(row.finished_at) if row else None,
            "last_details": row.details if row else None,
        }
    return out


@router.get("/agents/status", tags=["ops"])
async def agents_status(request: Request, s: AsyncSession = Db) -> dict[str, Any]:
    eng, sim = request.app.state.engine, request.app.state.sim
    st = await repo.sim_state(s)
    newest = (await s.execute(select(func.max(m.LiveBuffer.ts)))).scalar()
    wx = (await s.execute(select(func.max(m.WeatherHourly.issued_at)).where(m.WeatherHourly.source == "forecast"))).scalar()
    runs = await _last_runs(s)
    task = request.app.state.agent_tasks.get("retrain")
    runs["retrain"]["in_progress"] = bool(task and not task.done())
    lag = (st.sim_now - newest).total_seconds() / 3600 if (st and newest) else None
    return {
        "sim": {"mode": st.mode, "sim_now": _iso(st.sim_now), "paused": st.paused} if st else None,
        "last_error": sim.last_error,
        "buffer": {"newest_hour": _iso(newest), "lag_hours": lag, "healthy": lag is not None and lag <= 1},
        "weather_forecast_issued_at": _iso(wx),
        "live_champion": {"model": QUANTILE, "version": eng.version_of(QUANTILE), "run_id": eng.live_run,
                          "frozen_version": eng.version},
        "ws_subscribers": request.app.state.bus.subscribers,
        "agents": runs,
    }


@router.get("/drift", tags=["ops"])
async def drift_report(days: int = Query(7, ge=1, le=90), s: AsyncSession = Db) -> dict[str, Any]:
    T = (await repo.sim_state(s))
    if T is None:
        raise HTTPException(503, "simulator not initialised")
    rows = (await s.execute(
        select(m.DriftLog).where(m.DriftLog.detected_at > T.sim_now - timedelta(days=days)).order_by(m.DriftLog.detected_at)
    )).scalars().all()
    items = [
        {"detected_at": _iso(r.detected_at), "kind": r.kind, "statistic": r.statistic, "threshold": r.threshold,
         "is_drift": r.is_drift, "details": r.details}
        for r in rows
    ]
    latest_at = items[-1]["detected_at"] if items else None
    latest = list({i["kind"]: i for i in items if i["detected_at"] == latest_at}.values())  # last row per signal
    return {
        "window_days": days,
        "checks": len({i["detected_at"] for i in items}),
        "latest_at": latest_at,
        "latest": latest,
        "recommend_retrain": any((i["details"] or {}).get("recommend_retrain") for i in latest),
        "history": items,
    }


@router.post("/agents/drift/run", tags=["ops"], dependencies=[Depends(admin)])
async def drift_run(request: Request) -> dict[str, Any]:
    return await request.app.state.sim.run_drift()


class RetrainBody(BaseModel):
    promote: bool = True  # promote automatically when every gate passes


@router.post("/agents/retrain", tags=["ops"], status_code=202, dependencies=[Depends(admin)])
async def retrain_start(request: Request, body: RetrainBody | None = None) -> dict[str, Any]:
    tasks = request.app.state.agent_tasks
    if tasks.get("retrain") and not tasks["retrain"].done():
        raise HTTPException(409, "a retrain is already running")
    settings = request.app.state.settings
    cfg = threshold_config(settings.repo_root).retrain
    eng = request.app.state.engine
    promote = body.promote if body else True

    async def job() -> dict[str, Any]:
        res = await retrain_agent.run(request.app.state.maker, eng, cfg, eng.fcfg.warmup_hours, promote)
        await request.app.state.bus.publish("ops", "agent.status", {"agent": "retrain", **{
            k: res.get(k) for k in ("run_id", "status", "passed", "promoted")}})
        return res

    tasks["retrain"] = asyncio.create_task(job(), name="power-retrain")
    return {"started": True, "promote": promote, "window_days": cfg.window_days}


@router.get("/agents/retrain/runs", tags=["ops"])
async def retrain_runs(request: Request) -> dict[str, Any]:
    reg = request.app.state.engine.registry.read()
    reg["runs"] = sorted(reg["runs"], key=lambda r: r.get("saved_at", ""), reverse=True)
    return reg


@router.post("/agents/retrain/rollback", tags=["ops"], dependencies=[Depends(admin)])
async def retrain_rollback(request: Request) -> dict[str, Any]:
    return retrain_agent.rollback(request.app.state.engine)


@router.post("/agents/retrain/{run_id}/activate", tags=["ops"], dependencies=[Depends(admin)])
async def retrain_activate(run_id: str, request: Request) -> dict[str, Any]:
    try:
        return retrain_agent.activate(request.app.state.engine, run_id)
    except (KeyError, FileNotFoundError) as exc:
        raise HTTPException(404, f"unknown run {run_id}") from exc


@router.post("/agents/discovery/run", tags=["ops"], dependencies=[Depends(admin)])
async def discovery_run(request: Request, s: AsyncSession = Db) -> dict[str, Any]:
    key = f"manual-{utcnow().isoformat(timespec='seconds')}"
    try:
        res = await discovery_agent.run(s)
    except Exception as exc:  # noqa: BLE001 - network or upstream failure, reported to the operator
        await base.record(s, "discovery", key, "error", {"error": str(exc)})
        await s.commit()
        raise HTTPException(502, f"discovery failed: {exc}") from exc
    await base.record(s, "discovery", key, "ok", res)
    await s.commit()
    return res
