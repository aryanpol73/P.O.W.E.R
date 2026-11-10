"""Common session schema, anonymisation, cleaning and station-table construction.

Every source parser returns a DataFrame with ``SESSION_COLUMNS`` (times tz-aware UTC). The
cleaning step returns per-rule counts so the data card can report them exactly.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from power_core.config import CleaningCfg

SESSION_COLUMNS = (
    "session_hash", "station_id", "port_id", "port_kw", "connect_time", "charge_end_time",
    "disconnect_time", "energy_kwh", "latitude", "longitude", "quality_flags",
)
FLAG_CAPPED_POWER = 1
FLAG_IMPUTED_END = 2

# Columns that identify people or their vehicles. They are never written to disk.
PII_COLUMNS = ("mac address", "user id", "userid", "driver postal code", "plug in event id",
               "system s/n", "card", "email", "userinputs")


def salted_hash(values: pd.Series, salt: str | None = None) -> pd.Series:
    """Irreversible session identifier: sha256(salt + value), truncated to 24 hex chars."""
    s = salt if salt is not None else os.environ.get("POWER_HASH_SALT", "power-dev-salt")
    return values.astype(str).map(lambda v: hashlib.sha256((s + v).encode()).hexdigest()[:24])


def drop_pii(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if c.strip().lower() not in PII_COLUMNS]
    return df[keep]


@dataclass
class CleaningReport:
    raw: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    final: int = 0

    def add(self, rule: str, n: int) -> None:
        self.counts[rule] = self.counts.get(rule, 0) + int(n)

    def to_dict(self) -> dict[str, object]:
        return {"raw_sessions": self.raw, "rules": self.counts, "clean_sessions": self.final,
                "removed_pct": round(100 * (1 - self.final / self.raw), 3) if self.raw else 0.0}


def clean_sessions(df: pd.DataFrame, cfg: CleaningCfg) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the cleaning rules of report Section 4.1 in a fixed order."""
    rep = CleaningReport(raw=len(df))
    df = df.copy()
    df["quality_flags"] = df.get("quality_flags", 0)
    df["quality_flags"] = df["quality_flags"].fillna(0).astype(int)

    dup = df.duplicated(["port_id", "connect_time", "disconnect_time", "energy_kwh"])
    rep.add("duplicate", dup.sum())
    df = df.loc[~dup]

    missing = df["connect_time"].isna() | df["disconnect_time"].isna() | df["energy_kwh"].isna()
    rep.add("missing_core_field", missing.sum())
    df = df.loc[~missing]

    dur_h = (df["disconnect_time"] - df["connect_time"]) / pd.Timedelta(hours=1)
    invalid = (dur_h <= 0) | (dur_h > cfg.max_duration_hours) | (df["energy_kwh"] < 0)
    rep.add("invalid_duration_or_energy", invalid.sum())
    df, dur_h = df.loc[~invalid], dur_h.loc[~invalid]

    failed = (df["energy_kwh"] == 0) & (dur_h * 60 < cfg.min_failed_duration_minutes)
    rep.add("failed_plug_in", failed.sum())
    df = df.loc[~failed].copy()

    # Missing charge end: assume constant nameplate power from connection.
    no_end = df["charge_end_time"].isna()
    est = df["connect_time"] + pd.to_timedelta(df["energy_kwh"] / df["port_kw"], unit="h")
    df.loc[no_end, "charge_end_time"] = est[no_end]
    df.loc[no_end, "quality_flags"] |= FLAG_IMPUTED_END
    rep.add("imputed_charge_end", no_end.sum())

    # Clamp the charging window into the connection window.
    df["charge_end_time"] = df["charge_end_time"].clip(lower=df["connect_time"], upper=df["disconnect_time"])

    # Physics: implied power may not exceed tolerance x nameplate. Cap energy if it does.
    ch_h = (df["charge_end_time"] - df["connect_time"]) / pd.Timedelta(hours=1)
    max_kwh = cfg.power_tolerance * df["port_kw"] * ch_h
    over = df["energy_kwh"] > max_kwh + 1e-9
    df.loc[over, "energy_kwh"] = max_kwh[over]
    df.loc[over, "quality_flags"] |= FLAG_CAPPED_POWER
    rep.add("capped_power", over.sum())

    rep.final = len(df)
    return df.loc[:, list(SESSION_COLUMNS)].reset_index(drop=True), rep


def build_station_table(sessions: pd.DataFrame, poi_default: str, poi_overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """One row per site. ``total_ports`` counts distinct ports ever seen (an upper bound early on)."""
    ports = sessions.drop_duplicates(["station_id", "port_id"])[["station_id", "port_id", "port_kw", "latitude", "longitude"]]
    agg = ports.groupby("station_id").agg(
        total_ports=("port_id", "nunique"),
        rated_kw_total=("port_kw", "sum"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    )
    agg["first_seen"] = sessions.groupby("station_id")["connect_time"].min()
    agg["last_seen"] = sessions.groupby("station_id")["disconnect_time"].max()
    agg["n_sessions"] = sessions.groupby("station_id").size()
    overrides = poi_overrides or {}
    agg["poi_category"] = [overrides.get(s, poi_default) for s in agg.index]
    out = agg.reset_index()
    out["total_ports"] = out["total_ports"].astype(int)
    return out[["station_id", "latitude", "longitude", "total_ports", "rated_kw_total", "poi_category",
                "first_seen", "last_seen", "n_sessions"]]


def drop_sparse_stations(
    sessions: pd.DataFrame, stations: pd.DataFrame, min_per_day: float
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Remove sites with fewer than ``min_per_day`` sessions per active day (privacy + signal)."""
    days = ((stations["last_seen"] - stations["first_seen"]) / pd.Timedelta(days=1)).clip(lower=1)
    keep = stations["n_sessions"] / days >= min_per_day
    dropped = stations.loc[~keep, "station_id"].tolist()
    st = stations.loc[keep].reset_index(drop=True)
    return sessions.loc[sessions["station_id"].isin(st["station_id"])].reset_index(drop=True), st, dropped


def assert_schema(df: pd.DataFrame) -> None:
    missing = [c for c in SESSION_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"session frame missing {missing}")
    for c in ("connect_time", "charge_end_time", "disconnect_time"):
        if getattr(df[c].dt, "tz", None) is None:
            raise ValueError(f"{c} must be tz-aware UTC")
    if not np.issubdtype(df["energy_kwh"].dtype, np.number):
        raise ValueError("energy_kwh must be numeric")
