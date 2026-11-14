"""FastAPI application factory with lifespan: DB, serving engine, event bus, simulator."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select

from power_api.agents import base as agent_base
from power_api.agents import discovery as discovery_agent
from power_api.bus import InProcessBus
from power_api.db import models as m
from power_api.db.session import make_async_engine, sessionmaker_for, utcnow
from power_api.routers import agents, api, push, ws
from power_api.serving.engine import ServingEngine
from power_api.settings import Settings, get_settings
from power_api.simulator import Simulator

log = logging.getLogger("power.api")


async def _discover_once(maker) -> None:  # type: ignore[no-untyped-def]
    """A8 at start-up, skipped when a successful run finished in the last 24 hours (wall clock)."""
    async with maker() as s:
        last = (await s.execute(
            select(func.max(m.AgentRun.finished_at)).where(m.AgentRun.agent == "discovery", m.AgentRun.status == "ok")
        )).scalar()
        if last is not None and (utcnow() - last).total_seconds() < 86_400:
            return
        key = f"startup-{utcnow().isoformat(timespec='seconds')}"
        try:
            res = await discovery_agent.run(s)
            await agent_base.record(s, "discovery", key, "ok", res)
            log.info("discovery: %s", res)
        except Exception as exc:  # noqa: BLE001 - optional enrichment; never blocks the console
            await s.rollback()
            await agent_base.record(s, "discovery", key, "error", {"error": str(exc)})
            log.warning("discovery failed: %s", exc)
        await s.commit()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db = make_async_engine(settings.db_url)
        maker = sessionmaker_for(db)
        async with db.begin() as conn:  # add tables introduced since the database was seeded (idempotent)
            await conn.run_sync(m.Base.metadata.create_all)
        async with maker() as s:
            n = (await s.execute(select(func.count()).select_from(m.Station))).scalar_one()
        if n == 0:
            raise RuntimeError("database is empty: run `uv run power-api seed` first")
        engine = await asyncio.to_thread(
            ServingEngine,
            settings.repo_root,
            settings.source,
            settings.champion,
            settings.model_version or None,
            settings.deployment,
        )
        bus = InProcessBus()
        sim = Simulator(maker, engine, settings, bus)
        await sim.ensure_state()
        if settings.embedded_worker:
            sim.start()
        app.state.settings, app.state.maker, app.state.engine, app.state.bus, app.state.sim = settings, maker, engine, bus, sim
        app.state.agent_tasks = {}
        if settings.discovery_on_start and engine.deployment is not None:
            app.state.agent_tasks["discovery"] = asyncio.create_task(_discover_once(maker), name="power-discovery")
        log.info("P.O.W.E.R. API ready: models %s, champion %s, db %s", engine.version, engine.champion, db.dialect.name)
        try:
            yield
        finally:
            await sim.stop()
            for t in app.state.agent_tasks.values():
                t.cancel()
            await db.dispose()

    app = FastAPI(
        title="P.O.W.E.R. API",
        version="0.1.0",
        lifespan=lifespan,
        description="Predictive Optimization of Wattage for EV Resources: forecasts, alerts, explanations.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api.router, prefix="/api/v1")
    app.include_router(agents.router, prefix="/api/v1")
    app.include_router(push.router, prefix="/api/v1")
    app.include_router(ws.router)
    figures = settings.repo_root / "reports" / "figures"
    if figures.is_dir():  # report figures (SHAP, error analysis, stress test) for the Insights page
        app.mount("/figures", StaticFiles(directory=figures), name="figures")
    return app
