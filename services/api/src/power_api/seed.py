"""Load stations, the hourly panel and archive weather into the database (idempotent: replaces)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, text
from sqlalchemy.engine import Engine

from power_api.db import models as m
from power_api.db.session import init_schema, to_db
from power_core.config import threshold_config
from power_core.splits import SplitBounds

CHUNK = 20_000


def _naive_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


def seed(engine: Engine, root: Path, source: str) -> dict[str, int]:
    init_schema(engine)
    stations = pd.read_parquet(root / "data" / "interim" / f"stations_{source}.parquet")
    panel = pd.read_parquet(root / "data" / "processed" / f"panel_{source}.parquet")
    bounds = SplitBounds.from_dict(
        json.loads((root / "data" / "processed" / f"splits_{source}.json").read_text(encoding="utf-8"))
    )
    df_cfg = threshold_config(root).diversity_factor
    modeled = set(panel["station_id"].unique())

    st = pd.DataFrame(
        {
            "station_id": stations["station_id"],
            "name": stations["station_id"].str.replace(r"^[A-Z]+_", "", regex=True).str.replace("_", " ").str.title(),
            "source": source,
            "latitude": stations["latitude"],
            "longitude": stations["longitude"],
            "total_ports": stations["total_ports"].astype(int),
            "rated_kw_total": stations["rated_kw_total"].astype(float),
            "diversity_factor": df_cfg,
            "poi_category": stations["poi_category"],
            "first_seen": _naive_utc(stations["first_seen"]),
            "is_modeled": stations["station_id"].isin(modeled),
        }
    )

    ts = panel["ts"]
    split = pd.Series("gap", index=panel.index)
    split[(ts >= bounds.train_start) & (ts <= bounds.train_end)] = "train"
    split[(ts >= bounds.val_start) & (ts <= bounds.val_end)] = "val"
    split[(ts >= bounds.test_start) & (ts <= bounds.test_end)] = "test"
    split[ts > bounds.test_end] = "ood"
    hist = pd.DataFrame(
        {
            "station_id": panel["station_id"],
            "ts": _naive_utc(ts),
            "energy_kwh": panel["energy_kwh"].astype(float),
            "occ_ratio": panel["occ_ratio"].astype(float),
            "idle_ratio": panel["idle_ratio"].astype(float),
            "active_sessions": panel["active_sessions"].astype(int),
            "occupied_ports": panel["occupied_ports"].astype(int),
            "arrivals": panel["arrivals"].astype(int),
            "is_saturated": panel["is_saturated"].astype(bool),
            "temperature_2m_c": panel["temperature_2m_c"].astype(float),
            "precipitation_mm": panel["precipitation_mm"].astype(float),
            "split": split,
        }
    )
    weather = pd.DataFrame(
        {
            "station_id": hist["station_id"],
            "valid_time": hist["ts"],
            "issued_at": hist["ts"],
            "source": "archive",
            "temperature_2m_c": hist["temperature_2m_c"],
            "precipitation_mm": hist["precipitation_mm"],
        }
    )

    with engine.begin() as conn:
        for table in (
            m.PredictionExplanation,
            m.ModelPrediction,
            m.RerouteRecommendation,
            m.Alert,
            m.LiveBuffer,
            m.WeatherHourly,
            m.HistoricalTelemetry,
            m.SimState,
            m.AgentRun,
            m.DriftLog,
            m.Station,
        ):
            conn.execute(delete(table))
    for name, frame in (("stations", st), ("historical_telemetry", hist), ("weather_hourly", weather)):
        frame.to_sql(
            name,
            engine,
            if_exists="append",
            index=False,
            chunksize=CHUNK,
            method="multi" if engine.dialect.name != "sqlite" else None,
        )
    if engine.dialect.name == "sqlite":
        with engine.begin() as conn:
            conn.execute(text("ANALYZE"))
    return {
        "stations": len(st),
        "historical_rows": len(hist),
        "weather_rows": len(weather),
        "test_start": str(bounds.test_start),
        "first_test_hour_db": str(to_db(bounds.test_start)),
    }
