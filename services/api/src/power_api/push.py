"""Web Push (VAPID) for grid alerts: one notification per pipeline tick that raises alerts.

Keys come from ``VAPID_PUBLIC_KEY`` / ``VAPID_PRIVATE_KEY`` (base64url: uncompressed P-256 point / raw 32-byte
scalar). Without them a key pair is generated once and kept in ``data/vapid.json`` (git-ignored), so a dev
console works out of the box. Subscriptions that the push service reports gone (404/410) are deleted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import py_vapid
from cryptography.hazmat.primitives import serialization
from pywebpush import WebPushException, webpush
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from power_api.db import models as m
from power_api.db.session import utcnow

log = logging.getLogger("power.push")
LEVELS = {"WARNING": 1, "CRITICAL": 2}


@dataclass(frozen=True)
class VapidKeys:
    public: str  # applicationServerKey for the browser
    private: str
    subject: str


def load_keys(root: Path) -> VapidKeys:
    subject = os.environ.get("VAPID_SUBJECT", "mailto:admin@localhost")
    pub, priv = os.environ.get("VAPID_PUBLIC_KEY", ""), os.environ.get("VAPID_PRIVATE_KEY", "")
    if pub and priv:
        return VapidKeys(pub, priv, subject)
    path = root / "data" / "vapid.json"
    if path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        return VapidKeys(d["public"], d["private"], subject)
    v = py_vapid.Vapid02()
    v.generate_keys()
    raw_pub = v.public_key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    raw_priv = v.private_key.private_numbers().private_value.to_bytes(32, "big")
    keys = VapidKeys(py_vapid.b64urlencode(raw_pub), py_vapid.b64urlencode(raw_priv), subject)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"public": keys.public, "private": keys.private}), encoding="utf-8")
    log.info("generated a VAPID key pair in %s (set VAPID_PUBLIC_KEY/VAPID_PRIVATE_KEY in production)", path)
    return keys


def _send_one(keys: VapidKeys, sub: dict[str, Any], payload: str) -> int:
    """Blocking send; returns the push service's HTTP status (201 = accepted)."""
    try:
        resp = webpush(
            sub,
            data=payload,
            vapid_private_key=py_vapid.Vapid02.from_raw(keys.private.encode()),
            vapid_claims={"sub": keys.subject},
            ttl=3600,
            timeout=10,
        )
        return int(getattr(resp, "status_code", 201))
    except WebPushException as exc:
        return int(exc.response.status_code) if exc.response is not None else 0


def build_payload(raised: list[dict[str, Any]], names: dict[str, str]) -> dict[str, Any] | None:
    if not raised:
        return None
    crit = [a for a in raised if a["level"] == "CRITICAL"]
    worst = crit or raised
    sites = sorted({names.get(a["station_id"], a["station_id"]) for a in worst})
    level = "CRITICAL" if crit else "WARNING"
    title = f"{level}: {sites[0]}" + (f" +{len(sites) - 1} more" if len(sites) > 1 else "")
    first = worst[0]
    return {
        "title": title,
        "body": first["message"],
        "level": level,
        "url": f"/stations/{first['station_id']}" if len(sites) == 1 else "/",
        "tag": f"power-{level.lower()}",
    }


async def notify(session: AsyncSession, keys: VapidKeys, raised: list[dict[str, Any]]) -> dict[str, int]:
    """Send one coalesced notification to every subscription whose threshold the new alerts reach."""
    if not raised:
        return {"sent": 0, "removed": 0}
    names = {s.station_id: s.name for s in (await session.execute(select(m.Station))).scalars()}
    top = max(LEVELS.get(a["level"], 0) for a in raised)
    subs = [s for s in (await session.execute(select(m.PushSubscription))).scalars() if LEVELS.get(s.min_level, 2) <= top]
    if not subs:
        return {"sent": 0, "removed": 0}
    sent = removed = 0
    for s in subs:
        eligible = [a for a in raised if LEVELS.get(a["level"], 0) >= LEVELS.get(s.min_level, 2)]
        payload = build_payload(eligible, names)
        if payload is None:
            continue
        info = {"endpoint": s.endpoint, "keys": {"p256dh": s.p256dh, "auth": s.auth}}
        status = await asyncio.to_thread(_send_one, keys, info, json.dumps(payload))
        if status in (404, 410):
            await session.execute(delete(m.PushSubscription).where(m.PushSubscription.endpoint == s.endpoint))
            removed += 1
        elif 200 <= status < 300:
            s.last_sent_at = utcnow()
            s.failures = 0
            sent += 1
        else:
            s.failures = (s.failures or 0) + 1
            log.warning("push to %s... failed with %s", s.endpoint[:40], status)
    return {"sent": sent, "removed": removed}
