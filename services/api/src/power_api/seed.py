"""Load stations, the hourly panel and archive weather into the database (idempotent: replaces).

With a deployment profile (e.g. Pune) the database holds the *deployed* sites: donor demand is moved onto
the site ids and the local wall-clock time of the deployment city, and the weather is that city's own
observed weather (see ``power_core.deployment``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, text
from sqlalchemy.engine import Engine

from power_api.db import models as m
from power_api.db.session import init_schema, to_db
from power_core.config import data_config, threshold_config
from power_core.deployment import Deployment, align_to_grid
from power_core.splits import SplitBounds

CHUNK = 20_000
WX = ["temperature_2m_c", "precipitation_mm"]
log = logging.getLogger("power.seed")
# (coords {site: (lat, lon)}, start, end) -> hourly on-the-hour UTC weather:
# station_id, ts, temperature_2m_c, precipitation_mm
WeatherFetcher = Callable[[dict[str, tuple[float, float]], pd.Timestamp, pd.Timestamp], pd.DataFrame]


def _naive_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


def _default_fetcher(root: Path) -> WeatherFetcher:
    from power_ml.weather import fetch_archive

    wcfg = data_config(root).weather

    def fetch(coords: dict[str, tuple[float, float]], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        return fetch_archive(coords, start, end, wcfg.archive_url, list(wcfg.hourly))

    return fetch


def deployment_weather(
    root: Path, dep: Deployment, start: pd.Timestamp, end: pd.Timestamp, fetcher: WeatherFetcher | None = None
) -> pd.DataFrame:
    """Observed hourly weather at every deployed site, cached in ``dep.cfg.weather_cache``."""
    cache = root / dep.cfg.weather_cache
    coords = {s.id: (s.latitude, s.longitude) for s in dep.cfg.sites}
    lo, hi = start.floor("h"), end.floor("h")
    if cache.exists():
        w = pd.read_parquet(cache)
        w["ts"] = pd.to_datetime(w["ts"], utc=True)
        if set(w["station_id"]) >= set(coords) and w["ts"].min() <= lo and w["ts"].max() >= hi:
            return w
    log.info("fetching %s weather for %d sites, %s to %s", dep.cfg.city, len(coords), lo.date(), hi.date())
    w = (fetcher or _default_fetcher(root))(coords, lo - pd.Timedelta(hours=1), hi + pd.Timedelta(hours=2))
    w = w[["station_id", "ts", *WX]].copy()
    w["ts"] = pd.to_datetime(w["ts"], utc=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    w.to_parquet(cache, index=False)
    return w


def _split_labels(ts: pd.Series, bounds: SplitBounds) -> pd.Series:
    split = pd.Series("gap", index=ts.index)
    split[(ts >= bounds.train_start) & (ts <= bounds.train_end)] = "train"
    split[(ts >= bounds.val_start) & (ts <= bounds.val_end)] = "val"
    split[(ts >= bounds.test_start) & (ts <= bounds.test_end)] = "test"
    split[ts > bounds.test_end] = "ood"
    return split


def _deployed_stations(dep: Deployment, stations: pd.DataFrame, modeled: set[str], df_cfg: float) -> pd.DataFrame:
    donors = stations.set_index("station_id")
    missing = [x.donor for x in dep.cfg.sites if x.donor not in donors.index]
    if missing:
        raise ValueError(f"deployment {dep.cfg.name}: donor stations missing from the training data: {missing}")
    sites = dep.cfg.sites
    first_seen = dep.retime(pd.to_datetime(donors.loc[[x.donor for x in sites], "first_seen"], utc=True))
    return pd.DataFrame(
        {
            "station_id": [x.id for x in sites],
            "name": [x.name for x in sites],
            "source": dep.cfg.name,
            "latitude": [x.latitude for x in sites],
            "longitude": [x.longitude for x in sites],
            "total_ports": [int(donors.loc[x.donor, "total_ports"]) for x in sites],
            "rated_kw_total": [float(donors.loc[x.donor, "rated_kw_total"]) for x in sites],
            "diversity_factor": df_cfg,
            "poi_category": [x.poi for x in sites],
            "first_seen": first_seen.tz_localize(None),
            "is_modeled": [x.donor in modeled for x in sites],
        }
    )


def seed(
    engine: Engine,
    root: Path,
    source: str,
    deployment: Deployment | None = None,
    weather_fetcher: WeatherFetcher | None = None,
) -> dict[str, object]:
    init_schema(engine)
    stations = pd.read_parquet(root / "data" / "interim" / f"stations_{source}.parquet")
    panel = pd.read_parquet(root / "data" / "processed" / f"panel_{source}.parquet")
    bounds = SplitBounds.from_dict(
        json.loads((root / "data" / "processed" / f"splits_{source}.json").read_text(encoding="utf-8"))
    )
    df_cfg = threshold_config(root).diversity_factor
    modeled = set(panel["station_id"].unique())
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    # Split labels come from the *training* timeline, before any retiming.
    panel["split"] = _split_labels(panel["ts"], bounds)

    if deployment is None:
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
    else:
        st = _deployed_stations(deployment, stations, modeled, df_cfg)
        value_cols = [c for c in panel.columns if c not in ("station_id", "ts")]
        panel = deployment.retime_panel(panel, value_cols)
        wx = deployment_weather(root, deployment, panel["ts"].min(), panel["ts"].max(), weather_fetcher)
        grid = pd.DatetimeIndex(sorted(panel["ts"].unique()))
        wx = align_to_grid(wx, grid, WX)  # on-the-hour UTC -> the panel's grid (:30 UTC for IST)
        panel = panel.drop(columns=WX).merge(wx, on=["station_id", "ts"], how="left")
        for c in WX:
            panel[c] = panel.groupby("station_id")[c].transform(lambda x: x.interpolate(limit_direction="both"))

    ts = panel["ts"]
    split = panel["split"]
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
        "deployment": deployment.cfg.name if deployment else None,
        "stations": len(st),
        "historical_rows": len(hist),
        "weather_rows": len(weather),
        "test_start": str(bounds.test_start),
        "first_test_hour_db": str(to_db(deployment.retime_one(bounds.test_start) if deployment else bounds.test_start)),
    }
