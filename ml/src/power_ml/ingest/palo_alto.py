"""Parser for the City of Palo Alto "Electric Vehicle Charging Station Usage" CSV.

Column names are matched case-insensitively with parenthetical units removed, so
"Charging Time (hh:mm:ss)" and "charging time" are the same column. Personal identifiers
(MAC address, user id, driver postal code) are dropped before anything is stored.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from power_core.config import PaloAltoCfg
from power_ml.sessions import SESSION_COLUMNS, drop_pii, salted_hash

TZ_OFFSETS_H = {"PDT": -7, "PST": -8, "UTC": 0, "GMT": 0}
REQUIRED = ("station name", "start date", "energy", "charging time", "total duration")


def _norm(col: str) -> str:
    return re.sub(r"\s*\(.*?\)", "", col).strip().lower()


def site_of(station_name: str, regex: str, prefix: str) -> str:
    m = re.match(regex, str(station_name).strip())
    raw = m.group("site") if m else str(station_name)
    return prefix + re.sub(r"[^A-Z0-9]+", "_", raw.strip().upper()).strip("_")


def _local_to_utc(naive: pd.Series, tz_abbrev: pd.Series | None, tz: str) -> pd.Series:
    if tz_abbrev is not None:
        off = tz_abbrev.astype(str).str.strip().str.upper().map(TZ_OFFSETS_H)
        known = off.notna()
        out = pd.Series(pd.NaT, index=naive.index, dtype="datetime64[ns, UTC]")
        out[known] = (naive[known] - pd.to_timedelta(off[known], unit="h")).dt.tz_localize("UTC")
        if (~known).any():
            out[~known] = naive[~known].dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
        return out
    return naive.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")


def parse(raw: pd.DataFrame, cfg: PaloAltoCfg, tz: str) -> pd.DataFrame:
    df = raw.rename(columns={c: _norm(c) for c in raw.columns})
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(f"Palo Alto CSV missing columns {missing}; found {list(df.columns)}")

    # Stable identifier for hashing, taken before PII columns are removed.
    key_cols = [c for c in ("plug in event id", "evse id", "port number", "start date", "station name") if c in df]
    key = df[key_cols].astype(str).agg("|".join, axis=1)
    df = drop_pii(df)

    start_naive = pd.to_datetime(df["start date"], format="mixed", errors="coerce")
    connect = _local_to_utc(start_naive, df.get("start time zone"), tz)
    total = pd.to_timedelta(df["total duration"].astype(str).str.strip(), errors="coerce")
    charging = pd.to_timedelta(df["charging time"].astype(str).str.strip(), errors="coerce")

    port_type = df["port type"].astype(str).str.strip() if "port type" in df else pd.Series("Level 2", index=df.index)
    port_kw = port_type.map(cfg.port_kw).fillna(cfg.default_port_kw).astype(float)
    # A physical port is (station name, port number). EVSE ID is blank in older rows and filled in
    # newer ones (and changes when hardware is swapped), so including it would double-count ports.
    station_name = df["station name"].astype(str).str.strip()
    port_no = df["port number"].fillna("1").astype(str) if "port number" in df else pd.Series("1", index=df.index)

    out = pd.DataFrame(
        {
            "session_hash": salted_hash(key),
            "station_id": df["station name"].map(lambda s: site_of(s, cfg.site_regex, cfg.site_prefix)),
            "port_id": station_name + ":" + port_no.str.strip(),
            "port_kw": port_kw,
            "connect_time": connect,
            "charge_end_time": connect + charging,
            "disconnect_time": connect + total,
            "energy_kwh": pd.to_numeric(df["energy"], errors="coerce"),
            "latitude": pd.to_numeric(df.get("latitude"), errors="coerce") if "latitude" in df else np.nan,
            "longitude": pd.to_numeric(df.get("longitude"), errors="coerce") if "longitude" in df else np.nan,
            "quality_flags": 0,
        }
    )
    return out.loc[:, list(SESSION_COLUMNS)]


def load(path: str | Path, cfg: PaloAltoCfg, tz: str) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str, keep_default_na=True)
    return parse(raw, cfg, tz)
