"""Simulation clock and the hourly pipeline: replay -> forecast -> alerts -> publish.

Steps are serialized with a lock, so the background loop and a manual "tick" can never interleave,
and each agent is idempotent per origin.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from power_api import repo
from power_api.agents import alerts as alerts_mod
from power_api.agents import base, forecaster, replay
from power_api.agents import weather as weather_agent
from power_api.bus import InProcessBus
from power_api.db import models as m
from power_api.db.session import from_db, to_db, utcnow
from power_api.serving.engine import ServingEngine
from power_api.settings import Settings
from power_core.config import threshold_config
from power_core.splits import SplitBounds

log = logging.getLogger("power.sim")
HOUR = timedelta(hours=1)


class Simulator:
    def __init__(
        self, maker: async_sessionmaker[AsyncSession], engine: ServingEngine, settings: Settings, bus: InProcessBus
    ) -> None:
        self.maker, self.engine, self.settings, self.bus = maker, engine, settings, bus
        self.alerts = alerts_mod.AlertAgent(threshold_config(settings.repo_root))
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self.last_error: str | None = None

    # ------------------------------------------------------------------ setup
    def _grid_minute(self) -> int:
        return self.engine.deployment.grid_minute if self.engine.deployment else 0

    def default_start(self) -> datetime:
        if self.settings.sim_start:
            return to_db(self.settings.sim_start)
        p = self.settings.repo_root / "data" / "processed" / f"splits_{self.settings.source}.json"
        b = SplitBounds.from_dict(json.loads(p.read_text(encoding="utf-8")))
        dep = self.engine.deployment  # the deployed timeline is the training one moved to local wall-clock time
        return to_db(dep.retime_one(b.test_start) if dep else b.test_start)  # the buffer is pre-filled with the (earlier, observed) history

    async def ensure_state(self) -> m.SimState:
        async with self.maker() as s:
            st = await repo.sim_state(s)
            if st is not None:
                return st
        return await self.reset()

    async def reset(self, start: datetime | None = None, mode: str | None = None) -> m.SimState:
        async with self._lock:
            mode = (mode or self.settings.sim_mode).upper()
            T0 = start or self.default_start()
            offset = weather_agent.week_offset_hours(T0, utcnow()) if mode == "LIVE_SHADOW" else 0
            T = T0 + timedelta(hours=offset)
            async with self.maker() as s:
                for t in (
                    m.PredictionExplanation,
                    m.ModelPrediction,
                    m.RerouteRecommendation,
                    m.Alert,
                    m.LiveBuffer,
                    m.AgentRun,
                    m.SimState,
                ):
                    await s.execute(delete(t))
                if mode == "LIVE_SHADOW":
                    await s.execute(delete(m.WeatherHourly).where(m.WeatherHourly.source == "forecast"))
                await replay.copy_hours(s, T - timedelta(hours=self.settings.history_hours - 1), T, offset)
                st = m.SimState(id=1, mode=mode, sim_now=T, paused=True, speed_ticks_per_min=6.0, week_offset_hours=offset)
                s.add(st)
                await s.commit()
            self.alerts = alerts_mod.AlertAgent(threshold_config(self.settings.repo_root))
            if mode == "LIVE_SHADOW":
                async with self.maker() as s:
                    await weather_agent.sync(
                        s,
                        past_days=min(92, self.settings.history_hours // 24 + 1),
                        grid_minute=self._grid_minute(),
                    )
                    await s.commit()
            await self._process(T)
        await self.bus.publish("sim", "sim.reset", {"sim_now": from_db(T).isoformat(), "mode": mode})
        async with self.maker() as s:
            return await repo.sim_state(s)  # type: ignore[return-value]

    # ------------------------------------------------------------------ pipeline
    async def _process(self, T: datetime) -> dict[str, Any]:
        """Forecast + alerts for origin T (buffer already holds hour T)."""
        key = T.isoformat()
        async with self.maker() as s:
            if await base.already_done(s, "pipeline", key):
                return {"skipped": True}
            fc = await forecaster.run(s, self.engine, T, self.settings.history_hours)
            res = await self.alerts.run(s, T, forecaster.champion_frame(fc, self.engine.champion))
            await base.record(s, "pipeline", key, "ok", {"latency_ms": round(fc.latency_ms, 1), "alerts": len(res["raised"])})
            await s.commit()
        origin = from_db(T).isoformat()
        await self.bus.publish("forecast", "forecast.updated", {"forecast_origin": origin, "latency_ms": round(fc.latency_ms, 1)})
        for a in res["raised"]:
            await self.bus.publish("alerts", "alert.raised", a)
        if res["resolved"]:
            await self.bus.publish("alerts", "alert.resolved", {"alert_ids": res["resolved"]})
        if res["reroutes"]:
            await self.bus.publish("alerts", "reroute.updated", {"count": len(res["reroutes"])})
        return res

    async def step(self) -> dict[str, Any]:
        async with self._lock:
            async with self.maker() as s:
                st = await repo.sim_state(s)
                if st is None:
                    raise RuntimeError("simulator not initialised")
                T = st.sim_now + HOUR
                n = await replay.close_hour(s, T, st.week_offset_hours)
                if n == 0:
                    st.paused = True
                    await s.commit()
                    await self.bus.publish("sim", "sim.end", {"sim_now": from_db(st.sim_now).isoformat()})
                    return {"end_of_data": True}
                if st.mode == "LIVE_SHADOW" and T.hour == 0:
                    await weather_agent.sync(s, grid_minute=self._grid_minute())
                st.sim_now = T
                await s.commit()
            await self.bus.publish("telemetry", "telemetry.tick", {"ts": from_db(T).isoformat(), "rows": n})
            res = await self._process(T)
            await self.bus.publish("sim", "sim.clock", {"sim_now": from_db(T).isoformat()})
            return {"sim_now": from_db(T).isoformat(), **{k: len(v) if isinstance(v, list) else v for k, v in res.items()}}

    # ------------------------------------------------------------------ background loop
    async def run_forever(self) -> None:
        while True:
            try:
                async with self.maker() as s:
                    st = await repo.sim_state(s)
                if st is not None and not st.paused:
                    await self.step()
                    await asyncio.sleep(60.0 / max(st.speed_ticks_per_min, 0.1))
                else:
                    await asyncio.sleep(0.5)
                self.last_error = None
            except asyncio.CancelledError:
                raise
            except Exception as e:  # keep the loop alive; surface the error on /health and pause
                log.exception("simulator step failed")
                self.last_error = f"{type(e).__name__}: {e}"
                async with self.maker() as s:
                    st = await repo.sim_state(s)
                    if st is not None:
                        st.paused = True
                        await s.commit()
                await asyncio.sleep(2.0)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self.run_forever(), name="power-simulator")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
