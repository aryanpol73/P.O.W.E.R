"""Agent run bookkeeping: one ``agent_runs`` row per (agent, idempotency key)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from power_api.db import models as m
from power_api.db.session import utcnow


async def already_done(session: AsyncSession, agent: str, key: str) -> bool:
    row = (await session.execute(select(m.AgentRun).where(m.AgentRun.agent == agent, m.AgentRun.idempotency_key == key))).scalar()
    return row is not None and row.status == "ok"


async def record(session: AsyncSession, agent: str, key: str, status: str, details: dict[str, Any] | None = None) -> None:
    row = (await session.execute(select(m.AgentRun).where(m.AgentRun.agent == agent, m.AgentRun.idempotency_key == key))).scalar()
    now = utcnow()
    if row is None:
        session.add(m.AgentRun(agent=agent, idempotency_key=key, started_at=now, finished_at=now, status=status, details=details))
    else:
        row.status, row.finished_at, row.details = status, now, details
