"""REST API v1 (blueprint section 3.4). JSON, ISO-8601 UTC timestamps, kWh per hour."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from power_api import repo
from power_api.db import models as m
from power_api.db.session import from_db, to_db
from power_core import metrics
from power_core.conformal import bucket_of
from power_core.grid import Risk, risk_from_ratio

router = APIRouter()


# ------------------------------------------------------------------ dependencies
async def session(request: Request):  # type: ignore[no-untyped-def]
    async with request.app.state.maker() as s:
        yield s


def admin(request: Request, authorization: str | None = Header(default=None)) -> None:
    token = request.app.state.settings.admin_token
    if token and authorization != f"Bearer {token}":
        raise HTTPException(401, "admin token required")


Db = Depends(session)


def _iso(dt: Any) -> str | None:
    t = from_db(dt)
    return t.isoformat() if t is not None else None


async def _origin(s: AsyncSession) -> Any:
    st = await repo.sim_state(s)
    if st is None:
        raise HTTPException(503, "simulator not initialised")
    return st.sim_now


async def _champion_rows(s: AsyncSession, request: Request, T: Any, station_id: str | None = None) -> pd.DataFrame:
    q = select(m.ModelPrediction).where(
        m.ModelPrediction.forecast_origin == T, m.ModelPrediction.model_name == request.app.state.engine.champion
    )
    if station_id:
        q = q.where(m.ModelPrediction.station_id == station_id)
    rows = (await s.execute(q)).scalars().all()
    return pd.DataFrame(
        [
            {
                "station_id": r.station_id,
                "h": r.horizon_h,
                "target_time": r.target_time,
                "q05": r.q05,
                "q50": r.q50,
                "q95": r.q95,
                "ratio": r.load_ratio_q95,
                "degraded": r.is_degraded,
            }
            for r in rows
        ]
    )


def _worst_ratio(ch: pd.DataFrame, request: Request) -> pd.Series:
    """Worst q95 load ratio per station within the alert lookahead (same window as the alert agent)."""
    if ch.empty:
        return pd.Series(dtype=float)
    look = request.app.state.engine.tcfg.alert.lookahead_hours
    return ch.loc[ch["h"] <= look].groupby("station_id")["ratio"].max()


def _risk(ratio_q95: float, request: Request) -> str:
    return risk_from_ratio(ratio_q95, request.app.state.engine.tcfg.alert).value


# ------------------------------------------------------------------ health
@router.get("/health", tags=["system"])
async def health(request: Request, s: AsyncSession = Db) -> dict[str, Any]:
    st = await repo.sim_state(s)
    eng = request.app.state.engine
    return {
        "status": "ok" if not request.app.state.sim.last_error else "degraded",
        "model_version": eng.version,
        "live_champion_version": eng.version_of(eng.champion),
        "champion": eng.champion,
        "models": sorted(eng.models) + [eng.ensemble.name],
        "sim": {"mode": st.mode, "sim_now": _iso(st.sim_now), "paused": st.paused} if st else None,
        "last_error": request.app.state.sim.last_error,
        "ws_subscribers": request.app.state.bus.subscribers,
    }


@router.get("/deployment", tags=["meta"])
async def deployment(request: Request) -> dict[str, Any]:
    """Which city the live console serves, and how that relates to the data the models were trained on."""
    eng = request.app.state.engine
    if eng.deployment is None:
        return {
            "name": eng.source, "display_name": eng.source, "timezone": eng.tz, "is_twin": False, "model_trained_on": eng.source
        }
    return eng.deployment.public()


# ------------------------------------------------------------------ stations
@router.get("/stations", tags=["stations"])
async def list_stations(request: Request, modeled: bool = True, s: AsyncSession = Db) -> list[dict[str, Any]]:
    T = await _origin(s)
    sts = await repo.stations(s, modeled_only=modeled)
    ch = await _champion_rows(s, request, T)
    worst = _worst_ratio(ch, request)
    return [
        {
            "station_id": x.station_id,
            "name": x.name,
            "latitude": x.latitude,
            "longitude": x.longitude,
            "total_ports": x.total_ports,
            "rated_kw_total": x.rated_kw_total,
            "transformer_kw": round(x.transformer_kw, 2),
            "poi_category": x.poi_category,
            "is_modeled": x.is_modeled,
            "risk": _risk(float(worst.get(x.station_id, 0.0)), request) if x.is_modeled else None,
        }
        for x in sts
    ]


async def _station(s: AsyncSession, station_id: str) -> m.Station:
    st = await s.get(m.Station, station_id)
    if st is None:
        raise HTTPException(404, f"unknown station {station_id}")
    return st


@router.get("/stations/{station_id}/telemetry", tags=["stations"])
async def telemetry(station_id: str, hours: int = Query(48, ge=1, le=24 * 30), s: AsyncSession = Db) -> dict[str, Any]:
    await _station(s, station_id)
    T = await _origin(s)
    rows = (
        (
            await s.execute(
                select(m.LiveBuffer)
                .where(m.LiveBuffer.station_id == station_id, m.LiveBuffer.ts > T - timedelta(hours=hours), m.LiveBuffer.ts <= T)
                .order_by(m.LiveBuffer.ts)
            )
        )
        .scalars()
        .all()
    )
    return {
        "station_id": station_id,
        "as_of": _iso(T),
        "points": [
            {
                "ts": _iso(r.ts),
                "energy_kwh": round(r.energy_kwh, 3),
                "occ_ratio": round(r.occ_ratio, 3),
                "idle_ratio": None if r.idle_ratio is None or np.isnan(r.idle_ratio) else round(r.idle_ratio, 3),
                "active_sessions": r.active_sessions,
                "is_imputed": r.is_imputed,
            }
            for r in rows
        ],
    }


@router.get("/stations/{station_id}/forecast", tags=["forecasts"])
async def station_forecast(
    station_id: str,
    request: Request,
    model: str = "champion",
    history_hours: int = Query(48, ge=0, le=24 * 14),
    s: AsyncSession = Db,
) -> dict[str, Any]:
    st = await _station(s, station_id)
    T = await _origin(s)
    eng = request.app.state.engine
    name = eng.champion if model == "champion" else model
    rows = (
        (
            await s.execute(
                select(m.ModelPrediction)
                .where(
                    m.ModelPrediction.forecast_origin == T,
                    m.ModelPrediction.station_id == station_id,
                    m.ModelPrediction.model_name == name,
                )
                .order_by(m.ModelPrediction.horizon_h)
            )
        )
        .scalars()
        .all()
    )
    if not rows:
        raise HTTPException(404, f"no forecast for model {name} at the current origin")
    hist = (await telemetry(station_id, hours=history_hours, s=s))["points"] if history_hours else []
    return {
        "station_id": station_id,
        "model": name,
        "model_version": rows[0].model_version,  # the version that made these forecasts (live challenger tag)
        "forecast_origin": _iso(T),
        "interval_level": eng.tcfg.interval_level,
        "transformer_kw": round(st.transformer_kw, 2),
        "is_degraded": any(r.is_degraded for r in rows),
        "points": [
            {
                "target_time": _iso(r.target_time),
                "h": r.horizon_h,
                "y_mean": round(r.y_mean, 3),
                "q05": round(r.q05, 3),
                "q50": round(r.q50, 3),
                "q95": round(r.q95, 3),
                "load_ratio_q95": round(r.load_ratio_q95 or 0.0, 4),
                "risk": _risk(r.load_ratio_q95 or 0.0, request),
            }
            for r in rows
        ],
        "history": hist,
    }


@router.get("/stations/{station_id}/explain", tags=["explain"])
async def explain(
    station_id: str, request: Request, h: int | None = Query(None, ge=1, le=24), s: AsyncSession = Db
) -> dict[str, Any]:
    await _station(s, station_id)
    T = await _origin(s)
    if h is None:  # default: the forecast peak in the next 24 h
        ch = await _champion_rows(s, request, T, station_id)
        if ch.empty:
            raise HTTPException(404, "no forecast at the current origin")
        h = int(ch.loc[ch["q50"].idxmax(), "h"])
    e = (
        await s.execute(
            select(m.PredictionExplanation).where(
                m.PredictionExplanation.forecast_origin == T,
                m.PredictionExplanation.station_id == station_id,
                m.PredictionExplanation.horizon_h == h,
            )
        )
    ).scalar()
    if e is None:
        raise HTTPException(404, "no explanation stored for this origin/horizon")
    pred = e.base_value + sum(c["shap"] for c in e.contributions)
    parts = " ".join(f"{c['shap']:+.1f} from {c['display']}" for c in e.contributions if c["feature"] != "__other__")
    other = next((c["shap"] for c in e.contributions if c["feature"] == "__other__"), 0.0)
    target = from_db(T) + timedelta(hours=h)
    return {
        "station_id": station_id,
        "model": e.model_name,
        "forecast_origin": _iso(T),
        "h": h,
        "target_time": target.isoformat(),
        "base_value": round(e.base_value, 3),
        "prediction": round(pred, 3),
        "scale": "kWh (identity link: contributions add up to the forecast)",
        "contributions": e.contributions,
        "narrative": f"Base {e.base_value:.1f} kWh {parts} {other:+.1f} other = {pred:.1f} kWh",
    }


# ------------------------------------------------------------------ network
@router.get("/network/kpis", tags=["network"])
async def kpis(request: Request, s: AsyncSession = Db) -> dict[str, Any]:
    T = await _origin(s)
    st = await repo.sim_state(s)
    sts = {x.station_id: x for x in await repo.stations(s)}
    now = (await s.execute(select(m.LiveBuffer).where(m.LiveBuffer.ts == T))).scalars().all()
    ch = await _champion_rows(s, request, T)
    eng = request.app.state.engine
    ports = sum(x.total_ports for x in sts.values())
    occ = sum(r.occ_ratio * sts[r.station_id].total_ports for r in now if r.station_id in sts) / max(ports, 1)
    occ_h = sum(r.occ_ratio * sts[r.station_id].total_ports for r in now if r.station_id in sts)
    idle_h = sum(
        (r.idle_ratio or 0.0) * r.occ_ratio * sts[r.station_id].total_ports
        for r in now
        if r.station_id in sts and r.idle_ratio is not None and not np.isnan(r.idle_ratio)
    )
    h1 = ch.loc[ch["h"] == 1] if not ch.empty else ch
    nxt = float(h1["q50"].sum()) if not h1.empty else None
    half = eng.network_q.get(1)
    worst = _worst_ratio(ch, request)
    by_risk = {r.value: 0 for r in Risk}
    for v in worst:
        by_risk[_risk(float(v), request)] += 1
    status = "CRITICAL" if by_risk["CRITICAL"] else "WARNING" if by_risk["WARNING"] else "NORMAL"
    return {
        "as_of": _iso(T),
        "sim_mode": st.mode if st else None,
        "current_load_kw": round(sum(r.energy_kwh for r in now), 2),
        "next_hour": {
            "q50": round(nxt, 2) if nxt is not None else None,
            "lower": round(max(0.0, nxt - half), 2) if nxt is not None and half else None,
            "upper": round(nxt + half, 2) if nxt is not None and half else None,
            "interval_level": eng.tcfg.interval_level,
            "method": "network-level split conformal (validation)",
        },
        "port_occupancy_pct": round(100 * occ, 1),
        "idle_port_pct": round(100 * idle_h / occ_h, 1) if occ_h > 0 else 0.0,
        "risk_status": status,
        "stations_by_risk": by_risk,
    }


@router.get("/network/snapshot", tags=["network"])
async def snapshot(request: Request, h: int = Query(1, ge=1, le=24), s: AsyncSession = Db) -> dict[str, Any]:
    T = await _origin(s)
    sts = {x.station_id: x for x in await repo.stations(s, modeled_only=False)}
    ch = await _champion_rows(s, request, T)
    at = ch.loc[ch["h"] == h] if not ch.empty else ch
    reroutes = (
        (await s.execute(select(m.RerouteRecommendation).where(m.RerouteRecommendation.forecast_origin == T))).scalars().all()
    )
    out = []
    for r in at.itertuples(index=False):
        x = sts[r.station_id]
        out.append(
            {
                "station_id": r.station_id,
                "name": x.name,
                "lat": x.latitude,
                "lon": x.longitude,
                "q50": round(r.q50, 3),
                "q95": round(r.q95, 3),
                "load_ratio_q50": round(r.q50 / x.transformer_kw, 4),
                "load_ratio_q95": round(r.ratio or 0, 4),
                "risk": _risk(r.ratio or 0, request),
                "is_modeled": True,
                "is_degraded": bool(r.degraded),
            }
        )
    for x in sts.values():
        if not x.is_modeled:
            out.append(
                {
                    "station_id": x.station_id,
                    "name": x.name,
                    "lat": x.latitude,
                    "lon": x.longitude,
                    "q50": None,
                    "q95": None,
                    "load_ratio_q50": None,
                    "load_ratio_q95": None,
                    "risk": None,
                    "is_modeled": False,
                }
            )
    return {
        "forecast_origin": _iso(T),
        "h": h,
        "target_time": (from_db(T) + timedelta(hours=h)).isoformat(),
        "stations": out,
        "reroutes": [
            {
                "from": r.from_station,
                "to": r.to_station,
                "shifted_kwh": round(r.shifted_kwh, 2),
                "distance_km": round(r.distance_km, 3),
                "target_time": _iso(r.target_time),
            }
            for r in reroutes
        ],
    }


# ------------------------------------------------------------------ models
@router.get("/models", tags=["models"])
async def models(request: Request) -> dict[str, Any]:
    man = request.app.state.engine.manifest
    return {
        "version": man["version"],
        "git_sha": man["git_sha"],
        "created_at": man["created_at"],
        "champion": request.app.state.engine.champion,
        "val_wape": man["val_wape"],
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in man["models"].items()},
    }


@router.get("/models/leaderboard", tags=["models"])
async def leaderboard(request: Request) -> dict[str, Any]:
    lb = request.app.state.engine.leaderboard()
    if not lb:
        raise HTTPException(404, "reports/metrics_test.json not found: run `power-ml evaluate --split test`")
    return lb


@router.get("/models/live-arena", tags=["models"])
async def live_arena(hours: int = Query(168, ge=1, le=24 * 60), s: AsyncSession = Db) -> dict[str, Any]:
    """Rolling accuracy of every model on hours that have been observed since they were forecast."""
    T = await _origin(s)
    since = T - timedelta(hours=hours)
    q = (
        select(
            m.ModelPrediction.model_name,
            m.ModelPrediction.horizon_h,
            m.ModelPrediction.q50,
            m.ModelPrediction.q05,
            m.ModelPrediction.q95,
            m.LiveBuffer.energy_kwh,
        )
        .join(
            m.LiveBuffer,
            and_(m.LiveBuffer.station_id == m.ModelPrediction.station_id, m.LiveBuffer.ts == m.ModelPrediction.target_time),
        )
        .where(m.ModelPrediction.target_time > since, m.ModelPrediction.target_time <= T)
    )
    df = pd.DataFrame((await s.execute(q)).all(), columns=["model", "h", "q50", "q05", "q95", "y"])
    if df.empty:
        return {"as_of": _iso(T), "window_hours": hours, "n": 0, "models": {}}
    df["bucket"] = bucket_of(df["h"].to_numpy())
    out = {}
    for name, g in df.groupby("model"):
        out[name] = {
            "n": len(g),
            "wape": round(metrics.wape(g["y"], g["q50"]), 4),
            "mae": round(metrics.mae(g["y"], g["q50"]), 4),
            "picp": round(metrics.picp(g["y"], g["q05"], g["q95"]), 4),
            "wape_by_bucket": {
                ["1-3", "4-12", "13-24"][b]: round(metrics.wape(gb["y"], gb["q50"]), 4) for b, gb in g.groupby("bucket")
            },
        }
    return {
        "as_of": _iso(T),
        "window_hours": hours,
        "n": int(len(df)),
        "models": dict(sorted(out.items(), key=lambda kv: kv[1]["wape"])),
    }


# ------------------------------------------------------------------ alerts
@router.get("/alerts", tags=["alerts"])
async def alerts(
    status: Literal["open", "all"] = "open", limit: int = Query(100, le=1000), s: AsyncSession = Db
) -> list[dict[str, Any]]:
    q = select(m.Alert).order_by(m.Alert.alert_id.desc()).limit(limit)
    if status == "open":
        q = q.where(m.Alert.status != "resolved")
    return [
        {
            "alert_id": a.alert_id,
            "station_id": a.station_id,
            "level": a.level,
            "target_time": _iso(a.target_time),
            "forecast_origin": _iso(a.forecast_origin),
            "load_ratio_q95": round(a.load_ratio_q95, 4),
            "message": a.message,
            "status": a.status,
            "raised_at": _iso(a.raised_at),
            "resolved_at": _iso(a.resolved_at),
        }
        for a in (await s.execute(q)).scalars()
    ]


@router.post("/alerts/{alert_id}/ack", tags=["alerts"], dependencies=[Depends(admin)])
async def ack(alert_id: int, s: AsyncSession = Db) -> dict[str, Any]:
    a = await s.get(m.Alert, alert_id)
    if a is None:
        raise HTTPException(404, "unknown alert")
    if a.status == "open":
        a.status = "acknowledged"
        await s.commit()
    return {"alert_id": alert_id, "status": a.status}


@router.get("/reroutes", tags=["alerts"])
async def reroutes(s: AsyncSession = Db) -> list[dict[str, Any]]:
    T = await _origin(s)
    rows = (await s.execute(select(m.RerouteRecommendation).where(m.RerouteRecommendation.forecast_origin == T))).scalars()
    return [
        {
            "from": r.from_station,
            "to": r.to_station,
            "shifted_kwh": round(r.shifted_kwh, 2),
            "distance_km": round(r.distance_km, 3),
            "to_headroom_kwh": round(r.to_headroom_kwh, 2),
            "target_time": _iso(r.target_time),
        }
        for r in rows
    ]


# ------------------------------------------------------------------ simulation
class SimControl(BaseModel):
    action: Literal["play", "pause", "reset", "set_speed", "step"]
    speed_ticks_per_min: float | None = Field(None, gt=0, le=600)
    mode: Literal["REPLAY", "LIVE_SHADOW"] | None = None
    reset_to: str | None = None


@router.get("/sim/state", tags=["simulation"])
async def sim_state(s: AsyncSession = Db) -> dict[str, Any]:
    st = await repo.sim_state(s)
    if st is None:
        raise HTTPException(503, "simulator not initialised")
    return {
        "mode": st.mode,
        "sim_now": _iso(st.sim_now),
        "paused": st.paused,
        "speed_ticks_per_min": st.speed_ticks_per_min,
        "week_offset_hours": st.week_offset_hours,
    }


@router.post("/sim/control", tags=["simulation"], dependencies=[Depends(admin)])
async def sim_control(body: SimControl, request: Request, s: AsyncSession = Db) -> dict[str, Any]:
    sim = request.app.state.sim
    if body.action == "reset":
        start = to_db(body.reset_to) if body.reset_to else None
        await sim.reset(start=start, mode=body.mode)
    elif body.action == "step":
        return await sim.step()
    else:
        st = await repo.sim_state(s)
        if body.action in ("play", "pause"):
            st.paused = body.action == "pause"
        if body.speed_ticks_per_min:
            st.speed_ticks_per_min = body.speed_ticks_per_min
        await s.commit()
        await request.app.state.bus.publish("sim", "sim.control", {"action": body.action})
    async with request.app.state.maker() as s2:
        return await sim_state(s2)


@router.post("/sim/tick", tags=["simulation"], dependencies=[Depends(admin)])
async def sim_tick(request: Request) -> dict[str, Any]:
    return await request.app.state.sim.step()


# ------------------------------------------------------------------ scenarios
class Scenario(BaseModel):
    station_ids: list[str] = Field(default_factory=lambda: ["*"])
    temperature_delta_c: float = Field(0.0, ge=-20, le=20)
    precipitation_mm: float | None = Field(None, ge=0, le=100)
    force_holiday: bool = False
    demand_multiplier: float = Field(1.0, gt=0, le=3)


@router.post("/scenarios/simulate", tags=["scenarios"])
async def simulate(body: Scenario, request: Request, s: AsyncSession = Db) -> dict[str, Any]:
    """What-if on the current origin. Never changes live state."""
    T = await _origin(s)
    origin = from_db(T)
    eng = request.app.state.engine
    panel = await repo.serving_panel(s, origin, request.app.state.settings.history_hours)
    res = await asyncio.to_thread(
        eng.scenario,
        panel,
        origin,
        temperature_delta_c=body.temperature_delta_c,
        precipitation_mm=body.precipitation_mm,
        force_holiday=body.force_holiday,
        demand_multiplier=body.demand_multiplier,
    )
    keep = None if body.station_ids == ["*"] else set(body.station_ids)
    caps = {x.station_id: x.transformer_kw for x in await repo.stations(s)}

    def pack(df: pd.DataFrame) -> list[dict[str, Any]]:
        if keep is not None:
            df = df.loc[df["station_id"].isin(keep)]
        return [
            {
                "station_id": r.station_id,
                "h": int(r.h),
                "target_time": pd.Timestamp(r.target_time).isoformat(),
                "q05": round(float(r.q05), 3),
                "q50": round(float(r.q50), 3),
                "q95": round(float(r.q95), 3),
            }
            for r in df.itertuples(index=False)
        ]

    b, sc = res["baseline"], res["scenario"]
    new_alerts = []
    for r in sc.itertuples(index=False):
        if (keep is None or r.station_id in keep) and caps.get(r.station_id):
            lvl = _risk(float(r.q95) / caps[r.station_id], request)
            if lvl != "NORMAL":
                new_alerts.append({"station_id": r.station_id, "h": int(r.h), "level": lvl})
    peak = lambda d: float(d.groupby("h")["q50"].sum().max()) if not d.empty else 0.0  # noqa: E731
    return {
        "forecast_origin": origin.isoformat(),
        "model": eng.champion,
        "baseline": pack(b),
        "scenario": pack(sc),
        "delta_network_peak_kwh": round(peak(sc) - peak(b), 3),
        "alerts_in_scenario": new_alerts,
        "ood_flags": res["ood_flags"],
        "assumptions": res["assumptions"],
    }


# ------------------------------------------------------------------ data card
@router.get("/datacard", tags=["system"])
async def datacard(request: Request) -> dict[str, Any]:
    man = request.app.state.engine.manifest
    return {
        "source": man["source"],
        "data_manifest": man.get("data_manifest", {}),
        "assumptions": {
            "transformer_kw": "diversity_factor x sum of port nameplate kW (synthetic; not public data)",
            "replay_weather": "archive (observed) weather is used as a perfect forecast in REPLAY mode",
        },
    }


# ------------------------------------------------------------------ insights (offline analysis results)
def _report(request: Request, name: str) -> Any:
    import json

    p = request.app.state.engine.root / "reports" / name
    if not p.exists():
        raise HTTPException(404, f"reports/{name} not found: run `power-ml analyze`")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/insights/shap", tags=["insights"])
async def insights_shap(request: Request) -> dict[str, Any]:
    """Global SHAP importance, feature groups and local example narratives (test period)."""
    return _report(request, "shap_summary.json")


@router.get("/insights/errors", tags=["insights"])
async def insights_errors(request: Request) -> dict[str, Any]:
    return _report(request, "error_analysis.json")


@router.get("/insights/stress", tags=["insights"])
async def insights_stress(request: Request) -> dict[str, Any]:
    return _report(request, "stress_test.json")


@router.get("/models/horizons", tags=["models"])
async def model_horizons(request: Request, split: Literal["val", "test", "ood"] = "test") -> list[dict[str, Any]]:
    p = request.app.state.engine.root / "reports" / f"metrics_by_horizon_{split}.csv"
    if not p.exists():
        raise HTTPException(404, f"{p.name} not found")
    return pd.read_csv(p).round(4).to_dict(orient="records")
