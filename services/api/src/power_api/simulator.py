"""Simulation clock and the hourly pipeline: replay -> forecast -> alerts -> publish.

Steps are serialized with a lock, so the background loop and a manual "tick" can never interleave,
and each agent is idempotent per origin.

Modes
* REPLAY:      walks the held-out test period at a chosen speed (hours per minute), archived weather.
* LIVE_SHADOW: runs on the real clock. An hour is closed only once it has actually ended, so the console
               always shows the latest finished hour and the forecast for the hours ahead. Demand comes
               from each site's paired training site (its held-out test period, moved by whole weeks so
               weekday and hour match today); weather is the live Open-Meteo forecast, refreshed every
               few hours. When the test period runs out the replay wraps back to its start, still on
               whole weeks, so the live stream never ends.
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

from power_api import push, repo
from power_api.agents import alerts as alerts_mod
from power_api.agents import base, forecaster, replay
from power_api.agents import drift as drift_agent
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
LIVE = "LIVE_SHADOW"
WEATHER_EVERY = timedelta(hours=3)  # live weather refresh
MAX_CATCH_UP = timedelta(hours=48)  # further behind than this after downtime: start afresh at now


class Simulator:
    def __init__(
        self, maker: async_sessionmaker[AsyncSession], engine: ServingEngine, settings: Settings, bus: InProcessBus
    ) -> None:
        self.maker, self.engine, self.settings, self.bus = maker, engine, settings, bus
        self.alerts = alerts_mod.AlertAgent(threshold_config(settings.repo_root))
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self.last_error: str | None = None
        self._push_tasks: set[asyncio.Task[Any]] = set()
        self._vapid: push.VapidKeys | None = None
        self._last_weather: datetime | None = None

    def vapid(self) -> push.VapidKeys:
        if self._vapid is None:
            self._vapid = push.load_keys(self.settings.repo_root)
        return self._vapid

    async def _push(self, raised: list[dict[str, Any]]) -> None:
        try:
            async with self.maker() as s:
                res = await push.notify(s, self.vapid(), raised)
                await s.commit()
            if res["sent"] or res["removed"]:
                log.info("web push: %s", res)
        except Exception:  # noqa: BLE001 - notifications are best-effort
            log.exception("web push failed")

    # ------------------------------------------------------------------ setup
    def _grid_minute(self) -> int:
        return self.engine.deployment.grid_minute if self.engine.deployment else 0

    def test_window(self) -> tuple[datetime, datetime]:
        """The held-out test period on the deployed timeline (training timeline moved to local wall-clock time)."""
        p = self.settings.repo_root / "data" / "processed" / f"splits_{self.settings.source}.json"
        b = SplitBounds.from_dict(json.loads(p.read_text(encoding="utf-8")))
        dep = self.engine.deployment
        if dep is None:
            return to_db(b.test_start), to_db(b.test_end)
        return to_db(dep.retime_one(b.test_start)), to_db(dep.retime_one(b.test_end))

    def default_start(self) -> datetime:
        if self.settings.sim_start:
            return to_db(self.settings.sim_start)
        return self.test_window()[0]  # the buffer is pre-filled with the (earlier, observed) history

    def live_origin(self, now: datetime | None = None) -> datetime:
        """Start of the latest hour that has fully ended, on the deployment's hour grid."""
        now = now or utcnow()
        top = now.replace(minute=self._grid_minute(), second=0, microsecond=0)
        if top > now:
            top -= HOUR
        return top - HOUR

    def live_offset(self, T: datetime, current: int | None = None) -> int:
        """Whole-week shift from origin T back to its source hour in the test period.

        The current shift is kept while its source hour is still inside the test period (a continuous
        stream); otherwise the stream restarts at the first week of the test period, same weekday and hour.
        """
        start, end = self.test_window()
        if current is not None and start <= T - timedelta(hours=current) <= end:
            return current
        return weather_agent.week_offset_hours(start, T)

    async def _sync_weather(self, s: AsyncSession) -> None:
        """One batched Open-Meteo request: the whole history window plus two days ahead."""
        past = min(92, self.settings.history_hours // 24 + 1)
        n = await weather_agent.sync(s, past_days=past, grid_minute=self._grid_minute())
        await weather_agent.prune(s, utcnow() - timedelta(hours=12))  # the newest vintage covers the window
        self._last_weather = utcnow()
        log.info("live weather: %d rows", n)

    async def ensure_state(self) -> m.SimState:
        async with self.maker() as s:
            st = await repo.sim_state(s)
        if st is not None and (st.mode != LIVE or self.live_origin() - st.sim_now <= MAX_CATCH_UP):
            return st  # live: the loop catches up the hours missed while the API was down
        return await self.reset(mode=st.mode if st is not None else None)

    async def reset(self, start: datetime | None = None, mode: str | None = None) -> m.SimState:
        async with self._lock:
            mode = (mode or self.settings.sim_mode).upper()
            if mode == LIVE:  # live always starts at the latest finished hour; `start` is a replay option
                T = self.live_origin()
                offset = self.live_offset(T)
            else:
                T, offset = start or self.default_start(), 0
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
                st = m.SimState(
                    id=1, mode=mode, sim_now=T, paused=mode != LIVE, speed_ticks_per_min=6.0, week_offset_hours=offset
                )
                s.add(st)
                await s.commit()
            self.alerts = alerts_mod.AlertAgent(threshold_config(self.settings.repo_root))
            if mode == LIVE:
                async with self.maker() as s:
                    await self._sync_weather(s)
                    await s.commit()
            await self._process(T, live=mode == LIVE)
        await self.bus.publish("sim", "sim.reset", {"sim_now": from_db(T).isoformat(), "mode": mode})
        async with self.maker() as s:
            return await repo.sim_state(s)  # type: ignore[return-value]

    # ------------------------------------------------------------------ pipeline
    async def _process(self, T: datetime, live: bool = False) -> dict[str, Any]:
        """Forecast + alerts for origin T (buffer already holds hour T).

        Live, the forecast is made when hour T ends (T + 1 h), so a weather forecast issued by then counts.
        """
        key = T.isoformat()
        async with self.maker() as s:
            if await base.already_done(s, "pipeline", key):
                return {"skipped": True}
            decided = T + HOUR if live else None
            fc = await forecaster.run(s, self.engine, T, self.settings.history_hours, decided_at=decided)
            res = await self.alerts.run(s, T, forecaster.champion_frame(fc, self.engine.champion))
            await base.record(s, "pipeline", key, "ok", {"latency_ms": round(fc.latency_ms, 1), "alerts": len(res["raised"])})
            await s.commit()
        if from_db(T).tz_convert(self.engine.tz).hour == 0:  # A5 drift check once per local day
            await self.run_drift(T)
        origin = from_db(T).isoformat()
        await self.bus.publish("forecast", "forecast.updated", {"forecast_origin": origin, "latency_ms": round(fc.latency_ms, 1)})
        for a in res["raised"]:
            await self.bus.publish("alerts", "alert.raised", a)
        if res["raised"] and self.settings.push_enabled:
            # Web Push off the pipeline's critical path; a slow push service never delays the next tick.
            self._push_tasks.add(t := asyncio.create_task(self._push(res["raised"])))
            t.add_done_callback(self._push_tasks.discard)
        if res["resolved"]:
            await self.bus.publish("alerts", "alert.resolved", {"alert_ids": res["resolved"]})
        if res["reroutes"]:
            await self.bus.publish("alerts", "reroute.updated", {"count": len(res["reroutes"])})
        return res

    async def run_drift(self, T: datetime | None = None) -> dict[str, Any]:
        async with self.maker() as s:
            if T is None:
                st = await repo.sim_state(s)
                if st is None:
                    raise RuntimeError("simulator not initialised")
                T = st.sim_now
            key = T.isoformat()
            rep = await drift_agent.run(s, self.engine, T, threshold_config(self.settings.repo_root).drift)
            await base.record(s, "drift", key, "ok", rep)
            await s.commit()
        if rep["n_drift"]:
            await self.bus.publish("drift", "drift.detected", {"as_of": rep["as_of"], "signals": [
                k for k, v in rep["signals"].items() if v["is_drift"]], "recommend_retrain": rep["recommend_retrain"]})
        return rep

    async def step(self) -> dict[str, Any]:
        async with self._lock:
            async with self.maker() as s:
                st = await repo.sim_state(s)
                if st is None:
                    raise RuntimeError("simulator not initialised")
                T = st.sim_now + HOUR
                live = st.mode == LIVE
                if live and utcnow() < T + HOUR:  # hour T has not ended yet
                    return {"sim_now": from_db(st.sim_now).isoformat(), "waiting": True}
                if live:
                    st.week_offset_hours = self.live_offset(T, st.week_offset_hours)  # wraps at the test end
                n = await replay.close_hour(s, T, st.week_offset_hours)
                if n == 0:
                    st.paused = True
                    await s.commit()
                    await self.bus.publish("sim", "sim.end", {"sim_now": from_db(st.sim_now).isoformat()})
                    return {"end_of_data": True}
                if live and (self._last_weather is None or utcnow() - self._last_weather >= WEATHER_EVERY):
                    try:
                        await self._sync_weather(s)
                    except Exception:  # noqa: BLE001 - keep forecasting on the last good weather
                        log.exception("live weather sync failed")
                st.sim_now = T
                await s.commit()
            await self.bus.publish("telemetry", "telemetry.tick", {"ts": from_db(T).isoformat(), "rows": n})
            res = await self._process(T, live=live)
            await self.bus.publish("sim", "sim.clock", {"sim_now": from_db(T).isoformat()})
            return {"sim_now": from_db(T).isoformat(), **{k: len(v) if isinstance(v, list) else v for k, v in res.items()}}

    # ------------------------------------------------------------------ background loop
    async def run_forever(self) -> None:
        while True:
            try:
                async with self.maker() as s:
                    st = await repo.sim_state(s)
                if st is not None and st.mode == LIVE:
                    # real clock: close each hour once it has ended (quickly catching up after downtime)
                    if st.sim_now + 2 * HOUR <= utcnow():
                        await self.step()
                    else:
                        await asyncio.sleep(5.0)
                elif st is not None and not st.paused:
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
