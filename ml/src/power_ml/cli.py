"""``power-ml`` command line: each stage reads the previous stage's files and writes its own.

    power-ml ingest   --source palo_alto     raw CSV/JSON -> data/interim (anonymised, cleaned)
    power-ml resample --source palo_alto     sessions -> hourly panel
    power-ml weather  --source palo_alto     Open-Meteo archive back-fill (cached)
    power-ml panel    --source palo_alto     merge weather, compute split bounds
    power-ml features --source palo_alto     direct multi-horizon design matrices per split
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from power_core.config import data_config, feature_config, find_repo_root
from power_core.features import build_direct_frame
from power_core.features.weather import augment_frame
from power_core.splits import SplitBounds, assign_split, chronological_bounds
from power_ml import resample as rs
from power_ml import sessions as sx
from power_ml import weather as wx
from power_ml.ingest import acn, palo_alto

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


class Source(StrEnum):
    palo_alto = "palo_alto"
    acn = "acn"


def _root() -> Path:
    return find_repo_root()


def _p(*parts: str) -> Path:
    p = _root().joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _update_manifest(source: str, **entries: object) -> None:
    path = _p("data", "manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest.setdefault(source, {}).update(entries)
    manifest[source]["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


@app.command()
def ingest(source: Source = typer.Option(Source.palo_alto)) -> None:
    """Parse, anonymise and clean raw sessions; build the station table."""
    cfg = data_config(_root())
    if source is Source.palo_alto:
        raw_path = _root() / cfg.palo_alto.raw_path
        if not raw_path.exists():
            raise typer.BadParameter(f"{raw_path} not found. Download the Palo Alto CSV first (see configs/data.yaml).")
        sessions = palo_alto.load(raw_path, cfg.palo_alto, cfg.timezone)
        poi_default, poi_over = cfg.palo_alto.poi_default, cfg.palo_alto.poi_overrides
    else:
        raw_path = _root() / cfg.acn.raw_path
        if not raw_path.exists():
            raise typer.BadParameter(f"{raw_path} not found. Export ACN sessions as JSON first.")
        sessions = acn.load(raw_path, cfg.acn)
        poi_default, poi_over = cfg.acn.poi, {}

    sx.assert_schema(sessions)
    clean, report = sx.clean_sessions(sessions, cfg.cleaning)
    stations = sx.build_station_table(clean, poi_default, poi_over)
    clean, stations, dropped = sx.drop_sparse_stations(clean, stations, cfg.cleaning.min_sessions_per_day_public)
    clean.to_parquet(_p("data", "interim", f"sessions_{source}.parquet"), index=False)
    stations.to_parquet(_p("data", "interim", f"stations_{source}.parquet"), index=False)
    _update_manifest(source.value, raw_file=str(raw_path.relative_to(_root())), raw_sha256=_sha256(raw_path),
                     cleaning=report.to_dict(), sparse_sites_dropped=dropped,
                     sites=len(stations), ports=int(stations["total_ports"].sum()),
                     sessions_after_site_filter=len(clean))
    typer.echo(json.dumps(report.to_dict(), indent=2))
    typer.echo(f"{len(stations)} sites, {int(stations['total_ports'].sum())} ports; dropped sparse sites: {dropped}")


@app.command()
def resample(source: Source = typer.Option(Source.palo_alto)) -> None:
    """Prorate sessions into a gap-free hourly panel over [window.start, ood_end]."""
    cfg = data_config(_root())
    sessions = pd.read_parquet(_p("data", "interim", f"sessions_{source}.parquet"))
    stations = pd.read_parquet(_p("data", "interim", f"stations_{source}.parquet"))
    end = pd.Timestamp(cfg.window.ood_end)
    panel = rs.sessions_to_hourly(sessions, stations, end)
    panel = rs.restrict(panel, pd.Timestamp(cfg.window.start), end)
    panel.to_parquet(_p("data", "processed", f"hourly_{source}.parquet"), index=False)
    main = rs.restrict(panel, pd.Timestamp(cfg.window.start), pd.Timestamp(cfg.window.end))
    stats = {"station_hours_main_window": len(main), "station_hours_incl_ood": len(panel),
             "zero_hour_share_main": round(float((main["energy_kwh"] == 0).mean()), 4),
             "saturated_hour_share_main": round(float(main["is_saturated"].mean()), 4),
             "energy_kwh_main": round(float(main["energy_kwh"].sum()), 1)}
    _update_manifest(source.value, panel=stats)
    typer.echo(json.dumps(stats, indent=2))


@app.command()
def weather(source: Source = typer.Option(Source.palo_alto), refresh: bool = typer.Option(False)) -> None:
    """Back-fill Open-Meteo archive weather for every site (cached to data/external)."""
    cfg = data_config(_root())
    cache = _p("data", "external", f"weather_archive_{source}.parquet")
    if cache.exists() and not refresh:
        typer.echo(f"using cached {cache}")
        return
    stations = pd.read_parquet(_p("data", "interim", f"stations_{source}.parquet"))
    coords = {r.station_id: (float(r.latitude), float(r.longitude)) for r in stations.itertuples()}
    w = wx.fetch_archive(coords, pd.Timestamp(cfg.window.start) - pd.Timedelta(days=1),
                         pd.Timestamp(cfg.window.ood_end) + pd.Timedelta(days=1),
                         cfg.weather.archive_url, cfg.weather.hourly)
    w.to_parquet(cache, index=False)
    typer.echo(f"wrote {len(w)} weather rows for {len(coords)} sites -> {cache}")


@app.command()
def panel(source: Source = typer.Option(Source.palo_alto)) -> None:
    """Merge weather into the hourly panel and freeze chronological split bounds."""
    cfg = data_config(_root())
    hourly = pd.read_parquet(_p("data", "processed", f"hourly_{source}.parquet"))
    bounds = chronological_bounds(pd.Timestamp(cfg.window.start), pd.Timestamp(cfg.window.end),
                                  cfg.split.train_frac, cfg.split.val_frac, cfg.split.embargo_hours)
    wpath = _p("data", "external", f"weather_archive_{source}.parquet")
    if wpath.exists():
        merged, rep = wx.merge_weather(hourly, pd.read_parquet(wpath),
                                       max_gap_hours=cfg.weather.interpolate_max_gap_hours,
                                       climatology_until=bounds.train_end)
    else:
        typer.echo("WARNING: no weather cache; run `power-ml weather` first. Weather columns will be NaN.")
        merged, rep = hourly.assign(temperature_2m_c=np.nan, precipitation_mm=np.nan), {"missing_final": len(hourly)}
    merged.to_parquet(_p("data", "processed", f"panel_{source}.parquet"), index=False)
    _p("data", "processed", f"splits_{source}.json").write_text(json.dumps(bounds.to_dict(), indent=2), encoding="utf-8")
    _update_manifest(source.value, weather_merge=rep, split_bounds=bounds.to_dict())
    typer.echo(json.dumps({"split_bounds": bounds.to_dict(), "weather": rep}, indent=2))


@app.command()
def features(source: Source = typer.Option(Source.palo_alto),
             stride: int = typer.Option(0, help="origin stride in hours (0 = configs/features.yaml)")) -> None:
    """Build direct multi-horizon design matrices and write one parquet per split."""
    dcfg, fcfg = data_config(_root()), feature_config(_root())
    pnl = pd.read_parquet(_p("data", "processed", f"panel_{source}.parquet"))
    stations = pd.read_parquet(_p("data", "interim", f"stations_{source}.parquet"))
    bounds = SplitBounds.from_dict(json.loads(_p("data", "processed", f"splits_{source}.json").read_text(encoding="utf-8")))
    frame = build_direct_frame(pnl, stations, fcfg, tz=dcfg.timezone, pricing_changes=dcfg.pricing_changes,
                               origin_stride_hours=stride or None)
    frame["split"] = assign_split(frame["origin"], frame["target_time"], bounds)
    ood = pd.Timestamp(dcfg.window.ood_start).tz_convert("UTC")
    frame.loc[frame["origin"] >= ood, "split"] = "ood"
    counts = frame["split"].value_counts().to_dict()
    for name in ("train", "val", "test", "ood"):
        part = frame.loc[frame["split"] == name].drop(columns="split")
        # Training rows get forecast-error weather augmentation; evaluation rows keep observed weather.
        if name == "train" and fcfg.weather_augmentation.enabled:
            part = augment_frame(part, fcfg)
        part.to_parquet(_p("data", "processed", f"features_{source}_{name}.parquet"), index=False)
    _update_manifest(source.value, direct_frame_rows={k: int(v) for k, v in counts.items()})
    typer.echo(json.dumps({k: int(v) for k, v in counts.items()}, indent=2))


if __name__ == "__main__":
    app()
