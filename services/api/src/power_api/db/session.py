"""Engines and sessions. Async for the API/agents, sync for bulk seeding."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from power_api.db.models import Base

HYPERTABLES = {
    "historical_telemetry": "ts",
    "live_rolling_buffer": "ts",
    "weather_hourly": "valid_time",
    "model_predictions": "forecast_origin",
}


def sync_url(url: str) -> str:
    return url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg")


def _sqlite_pragmas(engine: Engine | AsyncEngine) -> None:
    target = engine.sync_engine if isinstance(engine, AsyncEngine) else engine

    @event.listens_for(target, "connect")
    def _on_connect(dbapi_conn, _record) -> None:  # type: ignore[no-untyped-def]
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")  # readers do not block the single writer
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


def make_async_engine(url: str) -> AsyncEngine:
    eng = create_async_engine(url, future=True, pool_pre_ping=True)
    if url.startswith("sqlite"):
        _sqlite_pragmas(eng)
    return eng


def make_sync_engine(url: str) -> Engine:
    eng = create_engine(sync_url(url), future=True)
    if url.startswith("sqlite"):
        _sqlite_pragmas(eng)
    return eng


def init_schema(engine: Engine) -> None:
    """Create tables; on PostgreSQL with TimescaleDB available, convert time-series tables to hypertables."""
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            has_ts = conn.execute(text("SELECT 1 FROM pg_available_extensions WHERE name='timescaledb'")).first()
            if has_ts:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
                for table, col in HYPERTABLES.items():
                    conn.execute(
                        text(f"SELECT create_hypertable('{table}', '{col}', if_not_exists => TRUE, migrate_data => TRUE)")
                    )


def sessionmaker_for(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def session_scope(maker: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with maker() as s:
        yield s


# ------------------------------------------------------------------ timestamp convention
def to_db(ts: pd.Timestamp | datetime) -> datetime:
    """Any timestamp -> naive UTC datetime for storage."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError(f"naive timestamp {ts!r}: attach a timezone before storing")
    return t.tz_convert("UTC").tz_localize(None).to_pydatetime()


def from_db(dt: datetime | pd.Timestamp | None) -> pd.Timestamp | None:
    """Stored naive UTC -> tz-aware UTC Timestamp."""
    if dt is None:
        return None
    t = pd.Timestamp(dt)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
