"""A6 Smart-Grid Alert & Reroute agent.

For each station, the worst ``q95 / transformer_kw`` over the next 24 hours drives a hysteresis
state machine (raise at 0.70 / 0.85, clear below 0.65 / 0.80 for two consecutive origins). When a
station is CRITICAL, the headroom-aware planner proposes moving a share of the excess to nearby
stations at the worst hour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from power_api import repo
from power_api.db import models as m
from power_api.db.session import to_db, utcnow
from power_core.config import ThresholdConfig
from power_core.grid import AlertHysteresis, Risk, plan_reroutes


@dataclass
class AlertAgent:
    cfg: ThresholdConfig
    states: dict[str, AlertHysteresis] = field(default_factory=dict)

    async def run(self, session: AsyncSession, T: datetime, champion_rows: pd.DataFrame) -> dict[str, list]:
        sts = {s.station_id: s for s in await repo.stations(session)}
        rows = champion_rows.loc[champion_rows["h"] <= self.cfg.alert.lookahead_hours].copy()
        rows["cap"] = rows["station_id"].map({k: v.transformer_kw for k, v in sts.items()})
        rows["ratio"] = rows["q95"] / rows["cap"]
        raised, resolved = [], []
        for sid, g in rows.groupby("station_id"):
            worst = g.loc[g["ratio"].idxmax()]
            st = self.states.setdefault(sid, AlertHysteresis(self.cfg.alert))
            before = st.level
            level = st.update(float(worst["ratio"]))
            open_alerts = (
                (await session.execute(select(m.Alert).where(m.Alert.station_id == sid, m.Alert.status != "resolved")))
                .scalars()
                .all()
            )
            if level is Risk.NORMAL:
                for a in open_alerts:
                    a.status, a.resolved_at = "resolved", utcnow()
                    resolved.append(a.alert_id)
                continue
            if level is not before or not open_alerts:
                for a in open_alerts:  # superseded by a new level
                    a.status, a.resolved_at = "resolved", utcnow()
                tt = to_db(worst["target_time"])
                key = f"{sid}|{tt.isoformat()}|{level.value}|{T.isoformat()}"
                msg = (
                    f"{level.value}: {sts[sid].name if sid in sts else sid} forecast q95 {worst['q95']:.1f} kWh at h+{int(worst['h'])} is "
                    f"{worst['ratio']:.0%} of the assumed {worst['cap']:.0f} kW transformer rating"
                )
                a = m.Alert(
                    station_id=sid,
                    level=level.value,
                    target_time=tt,
                    forecast_origin=T,
                    load_ratio_q95=float(worst["ratio"]),
                    message=msg,
                    status="open",
                    dedupe_key=key,
                    raised_at=utcnow(),
                )
                session.add(a)
                raised.append({"station_id": sid, "level": level.value, "message": msg})

        # reroutes at the network's worst hour among CRITICAL stations
        await session.execute(delete(m.RerouteRecommendation).where(m.RerouteRecommendation.forecast_origin == T))
        crit = [s for s, h in self.states.items() if h.level is Risk.CRITICAL]
        plans = []
        if crit:
            worst_h = int(rows.loc[rows["station_id"].isin(crit)].sort_values("ratio").iloc[-1]["h"])
            snap = rows.loc[rows["h"] == worst_h].copy()
            snap["lat"] = snap["station_id"].map({k: v.latitude for k, v in sts.items()})
            snap["lon"] = snap["station_id"].map({k: v.longitude for k, v in sts.items()})
            snap["capacity_kw"], snap["is_modeled"] = snap["cap"], True
            for r in plan_reroutes(snap, self.cfg.reroute, self.cfg.alert.critical_raise):
                session.add(
                    m.RerouteRecommendation(
                        forecast_origin=T,
                        target_time=to_db(snap["target_time"].iloc[0]),
                        from_station=r.from_station,
                        to_station=r.to_station,
                        distance_km=r.distance_km,
                        shifted_kwh=r.shifted_kwh,
                        to_headroom_kwh=r.to_headroom_kwh,
                    )
                )
                plans.append(r.__dict__)
        return {"raised": raised, "resolved": resolved, "reroutes": plans}
