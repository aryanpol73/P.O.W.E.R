"""ORM schema (portable SQLite / PostgreSQL). Mirrors docs/02_system_blueprint.md section 3.3.

Convention: every timestamp column holds **naive UTC**. Convert with ``power_api.db.timeconv``.
On PostgreSQL the time-series tables can be turned into TimescaleDB hypertables (see session.py).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Index, Integer, SmallInteger, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Station(Base):
    __tablename__ = "stations"
    station_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(32))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    total_ports: Mapped[int] = mapped_column(Integer)
    rated_kw_total: Mapped[float] = mapped_column(Float)
    diversity_factor: Mapped[float] = mapped_column(Float, default=0.70)
    poi_category: Mapped[str] = mapped_column(String(64), default="unknown")
    first_seen: Mapped[datetime | None] = mapped_column(DateTime)
    is_modeled: Mapped[bool] = mapped_column(Boolean, default=True)

    @property
    def transformer_kw(self) -> float:
        """SYNTHETIC planning assumption: DF x sum of port nameplate kW."""
        return self.rated_kw_total * self.diversity_factor


class HistoricalTelemetry(Base):
    __tablename__ = "historical_telemetry"
    station_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    energy_kwh: Mapped[float] = mapped_column(Float)
    occ_ratio: Mapped[float] = mapped_column(Float)
    idle_ratio: Mapped[float | None] = mapped_column(Float)
    active_sessions: Mapped[int] = mapped_column(SmallInteger)
    occupied_ports: Mapped[int] = mapped_column(SmallInteger)
    arrivals: Mapped[int] = mapped_column(SmallInteger)
    is_saturated: Mapped[bool] = mapped_column(Boolean)
    temperature_2m_c: Mapped[float | None] = mapped_column(Float)
    precipitation_mm: Mapped[float | None] = mapped_column(Float)
    split: Mapped[str] = mapped_column(String(8))
    __table_args__ = (Index("ix_hist_ts", "ts"),)


class LiveBuffer(Base):
    """Raw hourly actuals in sim/live time. Features are computed on read, never stored."""

    __tablename__ = "live_rolling_buffer"
    station_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    energy_kwh: Mapped[float] = mapped_column(Float)
    occ_ratio: Mapped[float] = mapped_column(Float)
    idle_ratio: Mapped[float | None] = mapped_column(Float)
    active_sessions: Mapped[int] = mapped_column(SmallInteger)
    occupied_ports: Mapped[int] = mapped_column(SmallInteger)
    arrivals: Mapped[int] = mapped_column(SmallInteger)
    is_saturated: Mapped[bool] = mapped_column(Boolean)
    is_imputed: Mapped[bool] = mapped_column(Boolean, default=False)
    source_ts: Mapped[datetime | None] = mapped_column(DateTime)
    ingested_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    __table_args__ = (Index("ix_buffer_ts", "ts"),)


class WeatherHourly(Base):
    """Weather with vintages: serving reads the latest ``issued_at <= T`` for each valid hour."""

    __tablename__ = "weather_hourly"
    station_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    valid_time: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    source: Mapped[str] = mapped_column(String(16))  # archive | forecast | climatology
    temperature_2m_c: Mapped[float | None] = mapped_column(Float)
    precipitation_mm: Mapped[float | None] = mapped_column(Float)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    station_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    forecast_origin: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    horizon_h: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    target_time: Mapped[datetime] = mapped_column(DateTime)
    model_version: Mapped[str] = mapped_column(String(32))
    y_mean: Mapped[float] = mapped_column(Float)
    q05: Mapped[float | None] = mapped_column(Float)
    q50: Mapped[float | None] = mapped_column(Float)
    q95: Mapped[float | None] = mapped_column(Float)
    load_ratio_q95: Mapped[float | None] = mapped_column(Float)
    is_degraded: Mapped[bool] = mapped_column(Boolean, default=False)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    __table_args__ = (Index("ix_pred_target", "station_id", "target_time"), Index("ix_pred_origin", "forecast_origin"))


class PredictionExplanation(Base):
    __tablename__ = "prediction_explanations"
    station_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    forecast_origin: Mapped[datetime] = mapped_column(DateTime, primary_key=True)
    horizon_h: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(64))
    base_value: Mapped[float] = mapped_column(Float)
    contributions: Mapped[list] = mapped_column(JSON)


class Alert(Base):
    __tablename__ = "alerts"
    alert_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    station_id: Mapped[str] = mapped_column(String(64))
    level: Mapped[str] = mapped_column(String(16))  # WARNING | CRITICAL
    target_time: Mapped[datetime] = mapped_column(DateTime)
    forecast_origin: Mapped[datetime] = mapped_column(DateTime)
    load_ratio_q95: Mapped[float] = mapped_column(Float)
    message: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(16), default="open")  # open | acknowledged | resolved
    dedupe_key: Mapped[str] = mapped_column(String(160))
    raised_at: Mapped[datetime] = mapped_column(DateTime)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)
    __table_args__ = (UniqueConstraint("dedupe_key"),)


class RerouteRecommendation(Base):
    __tablename__ = "reroute_recommendations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    forecast_origin: Mapped[datetime] = mapped_column(DateTime)
    target_time: Mapped[datetime] = mapped_column(DateTime)
    from_station: Mapped[str] = mapped_column(String(64))
    to_station: Mapped[str] = mapped_column(String(64))
    distance_km: Mapped[float] = mapped_column(Float)
    shifted_kwh: Mapped[float] = mapped_column(Float)
    to_headroom_kwh: Mapped[float] = mapped_column(Float)


class SimState(Base):
    __tablename__ = "sim_state"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    mode: Mapped[str] = mapped_column(String(16))
    sim_now: Mapped[datetime] = mapped_column(DateTime)  # last *closed* hour = current forecast origin T
    paused: Mapped[bool] = mapped_column(Boolean, default=True)
    speed_ticks_per_min: Mapped[float] = mapped_column(Float, default=6.0)
    week_offset_hours: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class AgentRun(Base):
    __tablename__ = "agent_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    agent: Mapped[str] = mapped_column(String(32))
    idempotency_key: Mapped[str] = mapped_column(String(128))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(16))  # running | ok | error | skipped
    details: Mapped[dict | None] = mapped_column(JSON)
    __table_args__ = (UniqueConstraint("agent", "idempotency_key"),)


class DriftLog(Base):
    __tablename__ = "drift_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime)
    station_id: Mapped[str | None] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(32))
    statistic: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    is_drift: Mapped[bool] = mapped_column(Boolean)
    details: Mapped[dict | None] = mapped_column(JSON)


class PushSubscription(Base):
    """A browser's Web Push subscription (endpoint + encryption keys) and the lowest alert level it wants."""

    __tablename__ = "push_subscriptions"
    endpoint: Mapped[str] = mapped_column(String(1024), primary_key=True)
    p256dh: Mapped[str] = mapped_column(String(256))
    auth: Mapped[str] = mapped_column(String(64))
    min_level: Mapped[str] = mapped_column(String(16), default="CRITICAL")  # CRITICAL | WARNING
    created_at: Mapped[datetime] = mapped_column(DateTime)
    last_sent_at: Mapped[datetime | None] = mapped_column(DateTime)
    failures: Mapped[int] = mapped_column(Integer, default=0)
