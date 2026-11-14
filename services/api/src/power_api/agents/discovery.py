"""A8 Station Discovery agent: other public chargers in the deployment city, from OpenStreetMap.

Queries the Overpass API for ``amenity=charging_station`` inside a padded bounding box around the
deployed sites and upserts them as *unmodelled* stations (grey pins on the map). They never get
forecasts and never enter accuracy metrics; they show where demand could be rerouted next and where the
network could grow. OpenStreetMap data (c) OpenStreetMap contributors, ODbL.
"""

from __future__ import annotations

from typing import Any

import httpx
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from power_api.db import models as m
from power_api.db.session import utcnow
from power_core.geo import haversine_km

URLS = ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter")
PAD_DEG = 0.06
MIN_SEPARATION_KM = 0.08  # closer than this to a known station = the same place


def _query(bbox: tuple[float, float, float, float]) -> str:
    s, w, n, e = bbox
    return f'[out:json][timeout:60];nwr["amenity"="charging_station"]({s:.5f},{w:.5f},{n:.5f},{e:.5f});out center tags;'


def _name(tags: dict[str, str]) -> str:
    base = tags.get("name") or tags.get("operator") or tags.get("brand") or "Public charger"
    where = tags.get("addr:suburb") or tags.get("addr:street")
    return f"{base}, {where}" if where and where not in base else base


def _int(v: str | None) -> int | None:
    try:
        return int(str(v).split()[0])
    except (TypeError, ValueError):
        return None


async def run(
    session: AsyncSession, client: httpx.AsyncClient | None = None, urls: tuple[str, ...] = URLS
) -> dict[str, Any]:
    known = list((await session.execute(select(m.Station))).scalars())
    sites = [k for k in known if k.source != "osm"]
    if not sites:
        return {"added": 0, "updated": 0, "found": 0}
    lats = [k.latitude for k in sites]
    lons = [k.longitude for k in sites]
    bbox = (min(lats) - PAD_DEG, min(lons) - PAD_DEG, max(lats) + PAD_DEG, max(lons) + PAD_DEG)

    own = client is None
    client = client or httpx.AsyncClient(timeout=90.0, headers={"User-Agent": "power-ev-console/0.1 (student project)"})
    try:
        errors: list[str] = []
        for url in urls:  # public Overpass instances are often busy (429/502/504): try the next mirror
            try:
                resp = await client.post(url, data={"data": _query(bbox)})
                resp.raise_for_status()
                elements = resp.json().get("elements", [])
                break
            except httpx.HTTPStatusError as exc:
                errors.append(f"{httpx.URL(url).host}: HTTP {exc.response.status_code}")
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(f"{httpx.URL(url).host}: {type(exc).__name__}")
        else:
            raise RuntimeError("Overpass unavailable (" + "; ".join(errors) + "); public instances are often busy, retry later")
    finally:
        if own:
            await client.aclose()

    site_lat = np.array(lats)
    site_lon = np.array(lons)
    by_id = {k.station_id: k for k in known}
    added = updated = 0
    for el in elements:
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        if lat is None or lon is None:
            continue
        if (haversine_km(lat, lon, site_lat, site_lon) < MIN_SEPARATION_KM).any():
            continue  # already a deployed site
        tags = el.get("tags", {})
        sid = f"OSM_{el['type'][0].upper()}{el['id']}"
        ports = _int(tags.get("capacity")) or 0
        row = by_id.get(sid)
        if row is None:
            session.add(m.Station(
                station_id=sid, name=_name(tags), source="osm", latitude=float(lat), longitude=float(lon),
                total_ports=ports, rated_kw_total=0.0, diversity_factor=1.0,
                poi_category=tags.get("access", "public"), first_seen=utcnow(), is_modeled=False,
            ))
            added += 1
        else:
            row.name, row.latitude, row.longitude, row.total_ports = _name(tags), float(lat), float(lon), ports
            updated += 1
    return {"found": len(elements), "added": added, "updated": updated, "bbox": [round(x, 4) for x in bbox]}
