"""A5 Drift agent: four live signals, checked at each local midnight, logged to ``drift_logs``.

1. ``cusum_bias``   two-sided CUSUM on the daily network forecast bias, standardised by the validation
                    reference (the detector that caught COVID-19 on 2020-03-08 in the stress test).
2. ``seasonal_psi`` PSI of the last week's station demand after removing the hour-of-week profile,
                    against the training period.
3. ``coverage``     rolling champion 90% interval coverage (PICP) below ``coverage_min``.
4. ``wape_ratio``   rolling champion WAPE divided by its test WAPE above ``wape_ratio_max``.

``recommend_retrain`` is set when at least ``min_signals_for_retrain`` signals fire together; the
retrain itself (A7) still needs an operator's go-ahead.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import and_, delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from power_api.db import models as m
from power_api.db.session import from_db
from power_api.serving.engine import ServingEngine
from power_core import metrics
from power_core.config import DriftCfg
from power_core.drift import Cusum, seasonal_psi
from power_core.features.calendar import calendar_features


@lru_cache(maxsize=4)
def _val_bias_reference(path: str, champion: str, tz: str) -> tuple[float, float]:
    p = pd.read_parquet(path, columns=["target_time", "y", champion])
    day = pd.DatetimeIndex(p["target_time"]).tz_convert(tz).normalize()
    g = p.assign(day=day, e=p["y"] - p[champion]).groupby("day").agg(e=("e", "sum"), f=(champion, "sum"))
    b = g["e"] / g["f"].clip(lower=1e-6)
    return float(b.mean()), float(b.std(ddof=0) or 1.0)


def _test_wape(root: Path, champion: str) -> float | None:
    p = root / "reports" / "metrics_test.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["leaderboard"].get(champion, {}).get("wape")


async def _observed(session: AsyncSession, champion: str, start: datetime, end: datetime) -> pd.DataFrame:
    q = (
        select(m.ModelPrediction.target_time, m.ModelPrediction.q05, m.ModelPrediction.q50, m.ModelPrediction.q95,
               m.LiveBuffer.energy_kwh)
        .join(m.LiveBuffer, and_(m.LiveBuffer.station_id == m.ModelPrediction.station_id,
                                 m.LiveBuffer.ts == m.ModelPrediction.target_time))
        .where(m.ModelPrediction.model_name == champion, m.ModelPrediction.target_time > start,
               m.ModelPrediction.target_time <= end)
    )
    return pd.DataFrame((await session.execute(q)).all(), columns=["target_time", "q05", "q50", "q95", "y"])


_REF_CACHE: dict[str, pd.DataFrame] = {}


async def _train_reference(session: AsyncSession, engine: ServingEngine) -> pd.DataFrame:
    """Training-period station demand with hour-of-week (cached per process)."""
    key = f"{engine.root}|{engine.tz}"
    if key not in _REF_CACHE:
        rows = (await session.execute(
            select(m.HistoricalTelemetry.ts, m.HistoricalTelemetry.energy_kwh).where(m.HistoricalTelemetry.split == "train")
        )).all()
        ref = pd.DataFrame(rows, columns=["ts", "y"])
        idx = pd.DatetimeIndex(pd.to_datetime(ref["ts"])).tz_localize("UTC")
        ref["how"] = calendar_features(idx, engine.tz, holidays=engine.holidays)["how"].to_numpy()
        _REF_CACHE[key] = ref
    return _REF_CACHE[key]


async def run(session: AsyncSession, engine: ServingEngine, T: datetime, cfg: DriftCfg) -> dict[str, Any]:
    champion = engine.champion
    T_utc = from_db(T)
    signals: dict[str, dict[str, Any]] = {}

    # 1. CUSUM on daily network bias over every complete local day in the live buffer.
    obs_all = await _observed(session, champion, T - timedelta(days=60), T)
    if not obs_all.empty:
        tt = pd.DatetimeIndex(pd.to_datetime(obs_all["target_time"])).tz_localize("UTC")
        day = tt.tz_convert(engine.tz).normalize()
        last_complete = T_utc.tz_convert(engine.tz).normalize()  # days strictly before today are complete
        d = obs_all.assign(day=day, tt=tt, e=obs_all["y"] - obs_all["q50"]).loc[day < last_complete]
        g = d.groupby("day").agg(e=("e", "sum"), f=("q50", "sum"), hours=("tt", "nunique"))
        g = g.loc[g["hours"] >= 20]  # the replay's first day is partial: too few hours for a stable daily bias
        val = engine.root / "reports" / "predictions_val.parquet"
        if len(g) and val.exists():
            mu, sd = _val_bias_reference(str(val), champion, engine.dcfg.timezone)
            cus = Cusum(k=cfg.cusum_k, h=cfg.cusum_h)
            alarms, stat = [], 0.0
            for dday, row in g.iterrows():
                z = (row["e"] / max(row["f"], 1e-6) - mu) / sd
                direction = cus.update(float(z))
                stat = max(cus.pos, cus.neg)
                if direction:
                    alarms.append({"day": str(pd.Timestamp(dday).date()),
                                   "direction": "under-forecast" if direction == "up" else "over-forecast"})
            latest_day = str(pd.Timestamp(g.index[-1]).date())
            fired = bool(alarms) and alarms[-1]["day"] == latest_day
            signals["cusum_bias"] = {"is_drift": fired, "statistic": round(stat, 3), "threshold": cfg.cusum_h,
                                     "details": {"days": len(g), "alarms": alarms[-5:], "latest_day": latest_day}}

    # 2-4. Rolling window signals.
    win = await _observed(session, champion, T - timedelta(hours=cfg.window_hours), T)
    if len(win) >= 24:
        picp = float(metrics.picp(win["y"], win["q05"], win["q95"]))
        signals["coverage"] = {"is_drift": picp < cfg.coverage_min, "statistic": round(picp, 4),
                               "threshold": cfg.coverage_min, "details": {"rows": len(win)}}
        wape = float(metrics.wape(win["y"], win["q50"]))
        ref = _test_wape(engine.root, champion)
        if ref:
            ratio = wape / ref
            signals["wape_ratio"] = {"is_drift": ratio > cfg.wape_ratio_max, "statistic": round(ratio, 4),
                                     "threshold": cfg.wape_ratio_max,
                                     "details": {"live_wape": round(wape, 4), "test_wape": ref}}

    buf = (await session.execute(
        select(m.LiveBuffer.ts, m.LiveBuffer.energy_kwh).where(
            m.LiveBuffer.ts > T - timedelta(hours=cfg.window_hours), m.LiveBuffer.ts <= T)
    )).all()
    if buf:
        cur = pd.DataFrame(buf, columns=["ts", "y"])
        cur["how"] = calendar_features(pd.DatetimeIndex(pd.to_datetime(cur["ts"])).tz_localize("UTC"), engine.tz,
                                       holidays=engine.holidays)["how"].to_numpy()
        ref_df = await _train_reference(session, engine)
        if len(ref_df):
            v = float(seasonal_psi(ref_df, cur, "y", bins=10))
            if np.isfinite(v):
                signals["seasonal_psi"] = {"is_drift": v > cfg.psi_threshold, "statistic": round(v, 4),
                                           "threshold": cfg.psi_threshold, "details": {"rows": len(cur)}}

    n_drift = sum(s["is_drift"] for s in signals.values())
    recommend = n_drift >= cfg.min_signals_for_retrain
    await session.execute(delete(m.DriftLog).where(m.DriftLog.detected_at == T))  # re-running a check replaces it
    for kind, s in signals.items():
        session.add(m.DriftLog(detected_at=T, station_id=None, kind=kind, statistic=float(s["statistic"]),
                               threshold=float(s["threshold"]), is_drift=bool(s["is_drift"]),
                               details={**s["details"], "recommend_retrain": recommend}))
    return {"as_of": T_utc.isoformat(), "signals": signals, "n_drift": n_drift, "recommend_retrain": recommend}
