"""Deployment profiles: run the frozen models on a different city's charging sites (digital twin).

The models are trained on one data source (Palo Alto). A deployment profile (``configs/deployments/<name>.yaml``)
places that model on another city's sites, e.g. Pune. Each deployed site is paired with a *donor* training
site whose held-out demand it replays. Three rules keep the twin coherent:

* **Model identity.** The models only know donor station ids (``station_id`` is a categorical feature), so
  the serving engine translates site -> donor before building features and donor -> site afterwards.
* **Wall-clock alignment.** Demand follows the local clock, so a donor hour is moved to the *same local
  wall-clock hour* in the deployment's time zone (08:00 Palo Alto -> 08:00 IST). With a half-hour zone such
  as IST the hourly grid then sits at :30 UTC; every component works on whatever grid the data carries.
* **Local context.** Weather, holidays and time zone are the deployment city's own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel


class Site(BaseModel):
    id: str
    name: str
    area: str = ""
    latitude: float
    longitude: float
    poi: str = "commercial"
    osm: str | None = None
    donor: str


class HolidayCfg(BaseModel):
    country: str = "US"
    subdiv: str | None = "CA"


class DeploymentCfg(BaseModel):
    name: str
    display_name: str
    city: str
    country: str
    timezone: str
    holidays: HolidayCfg
    currency: str = "USD"
    locale: str = "en-US"
    model_source: str
    donor_timezone: str
    weather_cache: str
    sites: list[Site]


@dataclass
class Deployment:
    cfg: DeploymentCfg
    site_to_donor: dict[str, str] = field(init=False)
    donor_to_site: dict[str, str] = field(init=False)

    def __post_init__(self) -> None:
        self.site_to_donor = {s.id: s.donor for s in self.cfg.sites}
        self.donor_to_site = {s.donor: s.id for s in self.cfg.sites}
        if len(self.donor_to_site) != len(self.site_to_donor):
            raise ValueError("each donor station may back only one deployed site")

    @property
    def timezone(self) -> str:
        return self.cfg.timezone

    @property
    def grid_minute(self) -> int:
        """Minute past the UTC hour where local hours start (30 for IST); the deployed hourly grid sits there."""
        off = pd.Timestamp("2020-01-01", tz=self.cfg.timezone).utcoffset()
        return int((-off.total_seconds() // 60) % 60) if off is not None else 0

    @property
    def holidays(self) -> tuple[str, str | None]:
        return self.cfg.holidays.country, self.cfg.holidays.subdiv

    # ------------------------------------------------------------------ time
    def retime(self, ts_utc: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Donor UTC instants -> the deployment instants with the same local wall-clock time."""
        idx = pd.DatetimeIndex(ts_utc)
        if idx.tz is None:
            raise ValueError("retime expects tz-aware timestamps")
        wall = idx.tz_convert(self.cfg.donor_timezone).tz_localize(None)
        return wall.tz_localize(self.cfg.timezone, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")

    def retime_one(self, ts: pd.Timestamp) -> pd.Timestamp:
        out = self.retime(pd.DatetimeIndex([pd.Timestamp(ts)]))[0]
        if pd.isna(out):
            raise ValueError(f"{ts} has no counterpart in {self.timezone}")
        return out

    def retime_panel(self, panel: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
        """Move a donor panel onto deployment site ids and local wall-clock time.

        Donor DST transitions do not exist in the deployment zone (IST has no DST): the repeated autumn
        hour is dropped (first occurrence kept) and the missing spring hour is filled with the previous
        hour's values, so the result is a gap-free hourly grid per site.
        """
        df = panel.loc[panel["station_id"].isin(self.donor_to_site)].copy()
        df["ts"] = self.retime(df["ts"])
        df = df.dropna(subset=["ts"]).drop_duplicates(["station_id", "ts"], keep="first")
        df["station_id"] = df["station_id"].map(self.donor_to_site)
        parts = []
        for sid, g in df.groupby("station_id", sort=True):
            g = g.set_index("ts").sort_index()
            full = pd.date_range(g.index[0], g.index[-1], freq="h")
            g = g.reindex(full)
            with pd.option_context("future.no_silent_downcasting", True):
                g[value_cols] = g[value_cols].ffill().infer_objects()
            g["station_id"] = sid
            parts.append(g.rename_axis("ts").reset_index())
        return pd.concat(parts, ignore_index=True) if parts else df

    # ------------------------------------------------------------------ ids
    def to_donor(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["station_id"] = out["station_id"].astype(str).map(self.site_to_donor).fillna(out["station_id"].astype(str))
        return out

    def to_site(self, s: pd.Series) -> pd.Series:
        return s.astype(str).map(self.donor_to_site).fillna(s.astype(str))

    def public(self) -> dict[str, object]:
        c = self.cfg
        return {
            "name": c.name,
            "display_name": c.display_name,
            "city": c.city,
            "country": c.country,
            "timezone": c.timezone,
            "currency": c.currency,
            "locale": c.locale,
            "holidays": f"{c.holidays.country}" + (f"-{c.holidays.subdiv}" if c.holidays.subdiv else ""),
            "model_trained_on": c.model_source,
            "is_twin": True,
            "note": (
                "Pilot digital twin: sites and weather are Pune's; demand replays a held-out period of the "
                "training data aligned to local time. Not yet validated on Indian charging data."
            ),
        }


def align_to_grid(weather: pd.DataFrame, target: pd.DatetimeIndex, cols: list[str]) -> pd.DataFrame:
    """Linearly interpolate hourly on-the-hour weather (per station) onto ``target`` instants (e.g. :30 UTC)."""
    out = []
    tx = target.asi8.astype(np.float64)
    for sid, g in weather.groupby("station_id", sort=False):
        g = g.sort_values("ts")
        x = pd.DatetimeIndex(g["ts"]).asi8.astype(np.float64)
        rec = {"station_id": sid, "ts": target}
        for c in cols:
            rec[c] = np.interp(tx, x, g[c].to_numpy(float), left=np.nan, right=np.nan)
        out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True) if out else weather.iloc[0:0]


def load_deployment(root: Path, name: str | None) -> Deployment | None:
    """``None`` when no profile is selected (``name`` empty or "none") or the file does not exist."""
    if not name or name.lower() == "none":
        return None
    path = root / "configs" / "deployments" / f"{name}.yaml"
    if not path.exists():
        return None
    return Deployment(DeploymentCfg.model_validate(yaml.safe_load(path.read_text(encoding="utf-8"))))
