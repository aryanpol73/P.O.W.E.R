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

from power_api.bus import InProcessBus
from power_api.db import models as m
from power_api.db.session import make_async_engine, sessionmaker_for
from power_api.routers import api, ws
from power_api.serving.engine import ServingEngine
from power_api.settings import Settings, get_settings
from power_api.simulator import Simulator

log = logging.getLogger("power.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        db = make_async_engine(settings.db_url)
        maker = sessionmaker_for(db)
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
        log.info("P.O.W.E.R. API ready: models %s, champion %s, db %s", engine.version, engine.champion, db.dialect.name)
        try:
            yield
        finally:
            await sim.stop()
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
    app.include_router(ws.router)
    figures = settings.repo_root / "reports" / "figures"
    if figures.is_dir():  # report figures (SHAP, error analysis, stress test) for the Insights page
        app.mount("/figures", StaticFiles(directory=figures), name="figures")
    return app
