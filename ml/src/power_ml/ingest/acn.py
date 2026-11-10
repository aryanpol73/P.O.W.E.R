"""Parser for ACN-Data session exports (Lee et al., 2019).

Export sessions from the ACN-Data portal as JSON (a list of sessions, or an object with an
``_items`` list). Fields used: connectionTime, disconnectTime, doneChargingTime, kWhDelivered,
sessionID, stationID, siteID. ``userID`` and ``userInputs`` are dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from power_core.config import AcnCfg
from power_ml.sessions import SESSION_COLUMNS, salted_hash


def parse(records: list[dict], cfg: AcnCfg) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    for c in ("connectionTime", "disconnectTime", "kWhDelivered", "sessionID", "stationID"):
        if c not in df:
            raise ValueError(f"ACN export missing field {c}")
    df = df.drop(columns=[c for c in ("userID", "userInputs") if c in df])

    def ts(col: str) -> pd.Series:
        if col not in df:
            return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
        return pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")

    # Each export is for one site; the siteID field is numeric (e.g. "0002"), so name it from config.
    out = pd.DataFrame(
        {
            "session_hash": salted_hash(df["sessionID"]),
            "station_id": "ACN_" + cfg.site_id.upper(),
            "port_id": df["stationID"].astype(str),
            "port_kw": float(cfg.port_kw),
            "connect_time": ts("connectionTime"),
            "charge_end_time": ts("doneChargingTime"),
            "disconnect_time": ts("disconnectTime"),
            "energy_kwh": pd.to_numeric(df["kWhDelivered"], errors="coerce"),
            "latitude": cfg.latitude,
            "longitude": cfg.longitude,
            "quality_flags": 0,
        }
    )
    return out.loc[:, list(SESSION_COLUMNS)]


def raw_files(root: Path, raw_path: str) -> list[Path]:
    """The configured file plus any chunks named like it, e.g. acn_caltech_sessions_2019q1.json."""
    p = root / raw_path
    return sorted({*p.parent.glob(f"{p.stem}*{p.suffix}")})


def load(paths: str | Path | list[Path], cfg: AcnCfg) -> pd.DataFrame:
    """Load one or more ACN JSON exports and de-duplicate sessions across overlapping chunks."""
    files = [Path(paths)] if isinstance(paths, str | Path) else list(paths)
    records: list[dict] = []
    for f in files:
        try:
            with f.open(encoding="utf-8") as fh:
                payload = json.load(fh)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"{f.name} is not complete JSON ({e.msg} at line {e.lineno}): the download was probably cut off. "
                "Re-download it, or split the date range into smaller files."
            ) from e
        records.extend(payload["_items"] if isinstance(payload, dict) and "_items" in payload else payload)
    out = parse(records, cfg)
    return out.drop_duplicates("session_hash").reset_index(drop=True)
