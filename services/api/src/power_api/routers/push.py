"""Web Push subscription endpoints."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from power_api import push
from power_api.db import models as m
from power_api.db.session import utcnow
from power_api.routers.api import Db, admin

router = APIRouter()
MAX_SUBSCRIPTIONS = 5000


class Keys(BaseModel):
    p256dh: str = Field(min_length=16, max_length=256)
    auth: str = Field(min_length=8, max_length=64)


class Subscription(BaseModel):
    endpoint: str = Field(max_length=1024)
    keys: Keys

    @field_validator("endpoint")
    @classmethod
    def https_only(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError("push endpoints must be https URLs")
        return v


class SubscribeBody(BaseModel):
    subscription: Subscription
    min_level: Literal["CRITICAL", "WARNING"] = "CRITICAL"


class UnsubscribeBody(BaseModel):
    endpoint: str = Field(max_length=1024)


@router.get("/push/vapid-public-key", tags=["push"])
async def vapid_public_key(request: Request) -> dict[str, Any]:
    return {"public_key": request.app.state.sim.vapid().public, "enabled": request.app.state.settings.push_enabled}


@router.post("/push/subscribe", tags=["push"], status_code=201)
async def subscribe(body: SubscribeBody, s: AsyncSession = Db) -> dict[str, Any]:
    sub = body.subscription
    row = await s.get(m.PushSubscription, sub.endpoint)
    if row is None:
        n = (await s.execute(select(func.count()).select_from(m.PushSubscription))).scalar_one()
        if n >= MAX_SUBSCRIPTIONS:
            raise HTTPException(429, "subscription limit reached")
        s.add(m.PushSubscription(endpoint=sub.endpoint, p256dh=sub.keys.p256dh, auth=sub.keys.auth,
                                 min_level=body.min_level, created_at=utcnow(), failures=0))
    else:
        row.p256dh, row.auth, row.min_level, row.failures = sub.keys.p256dh, sub.keys.auth, body.min_level, 0
    await s.commit()
    return {"subscribed": True, "min_level": body.min_level}


@router.post("/push/unsubscribe", tags=["push"])
async def unsubscribe(body: UnsubscribeBody, s: AsyncSession = Db) -> dict[str, Any]:
    res = await s.execute(delete(m.PushSubscription).where(m.PushSubscription.endpoint == body.endpoint))
    await s.commit()
    return {"removed": res.rowcount or 0}


@router.post("/push/test", tags=["push"], dependencies=[Depends(admin)])
async def test_push(request: Request, s: AsyncSession = Db) -> dict[str, Any]:
    """Send a test notification to every subscription (operator check that delivery works)."""
    keys = request.app.state.sim.vapid()
    subs = list((await s.execute(select(m.PushSubscription))).scalars())
    payload = json.dumps({"title": "P.O.W.E.R. test alert", "body": "Notifications are working.", "url": "/ops",
                          "tag": "power-test", "level": "TEST"})
    statuses = []
    for sub in subs:
        info = {"endpoint": sub.endpoint, "keys": {"p256dh": sub.p256dh, "auth": sub.auth}}
        statuses.append(await asyncio.to_thread(push._send_one, keys, info, payload))
    return {"subscriptions": len(subs), "accepted": sum(200 <= c < 300 for c in statuses), "statuses": statuses}
