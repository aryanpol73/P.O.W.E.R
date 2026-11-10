"use client";

// City map: MapLibre GL with free OpenFreeMap vector tiles (no token). Circle colour = q95 risk,
// radius = forecast median load, pulsing halo for CRITICAL, curved arcs for reroute suggestions.

import maplibregl, { type GeoJSONSource, type Map as MLMap } from "maplibre-gl";
import { useEffect, useRef, useState } from "react";

import type { Snapshot } from "@/lib/api";
import { fmtNum, fmtPct } from "@/lib/format";

const STYLE = { light: "https://tiles.openfreemap.org/styles/positron", dark: "https://tiles.openfreemap.org/styles/dark" };
const prefersDark = () => typeof window !== "undefined" && window.matchMedia("(prefers-color-scheme: dark)").matches;
const COLORS = { NORMAL: "#1f9d55", WARNING: "#d98e04", CRITICAL: "#d23b3b", CONTEXT: "#8a96a8", ARC: "#1f4fd1" };

function arc(a: [number, number], b: [number, number], steps = 24): [number, number][] {
  // quadratic Bezier with the control point offset perpendicular to the chord
  const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const c: [number, number] = [mx - dy * 0.35, my + dx * 0.35];
  return Array.from({ length: steps + 1 }, (_, i) => {
    const t = i / steps;
    return [(1 - t) ** 2 * a[0] + 2 * (1 - t) * t * c[0] + t ** 2 * b[0], (1 - t) ** 2 * a[1] + 2 * (1 - t) * t * c[1] + t ** 2 * b[1]];
  });
}

function toGeoJSON(s: Snapshot) {
  const pos = new Map(s.stations.map((x) => [x.station_id, [x.lon, x.lat] as [number, number]]));
  return {
    stations: {
      type: "FeatureCollection" as const,
      features: s.stations.map((x) => ({
        type: "Feature" as const,
        geometry: { type: "Point" as const, coordinates: [x.lon, x.lat] },
        properties: {
          id: x.station_id, name: x.name, risk: x.is_modeled ? x.risk ?? "NORMAL" : "CONTEXT", q50: x.q50 ?? 0,
          q95: x.q95 ?? 0, ratio: x.load_ratio_q95 ?? 0, modeled: x.is_modeled,
        },
      })),
    },
    arcs: {
      type: "FeatureCollection" as const,
      features: s.reroutes
        .filter((r) => pos.has(r.from) && pos.has(r.to))
        .map((r) => ({
          type: "Feature" as const,
          geometry: { type: "LineString" as const, coordinates: arc(pos.get(r.from)!, pos.get(r.to)!) },
          properties: { kwh: r.shifted_kwh },
        })),
    },
  };
}

export function StationMap({ snapshot, onSelect }: { snapshot: Snapshot | undefined; onSelect?: (id: string) => void }) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<MLMap | null>(null);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const fitted = useRef(false);
  const selectRef = useRef(onSelect);
  selectRef.current = onSelect;

  useEffect(() => {
    if (!el.current || map.current) return;
    const dark = prefersDark();
    const m = new maplibregl.Map({ container: el.current, style: dark ? STYLE.dark : STYLE.light, center: [-122.14, 37.43], zoom: 13, attributionControl: { compact: true } });
    map.current = m;
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    m.on("error", (e) => {
      if (!m.isStyleLoaded() && String(e.error?.message ?? "").length) setFailed(true);
    });
    m.on("load", () => {
      const empty = { type: "FeatureCollection" as const, features: [] };
      m.addSource("stations", { type: "geojson", data: empty });
      m.addSource("arcs", { type: "geojson", data: empty });
      m.addLayer({
        id: "arcs", type: "line", source: "arcs",
        paint: { "line-color": COLORS.ARC, "line-width": ["interpolate", ["linear"], ["get", "kwh"], 0, 2, 10, 6], "line-dasharray": [2, 1.5], "line-opacity": 0.85 },
      });
      m.addLayer({
        id: "halo", type: "circle", source: "stations", filter: ["==", ["get", "risk"], "CRITICAL"],
        paint: { "circle-color": COLORS.CRITICAL, "circle-radius": 22, "circle-opacity": 0.25, "circle-blur": 0.4 },
      });
      m.addLayer({
        id: "stations", type: "circle", source: "stations",
        paint: {
          "circle-color": ["match", ["get", "risk"], "CRITICAL", COLORS.CRITICAL, "WARNING", COLORS.WARNING, "NORMAL", COLORS.NORMAL, COLORS.CONTEXT],
          "circle-radius": ["interpolate", ["linear"], ["get", "q50"], 0, 7, 10, 12, 40, 22],
          "circle-stroke-color": "#ffffff", "circle-stroke-width": 2, "circle-opacity": 0.92,
        },
      });
      m.addLayer({
        id: "labels", type: "symbol", source: "stations",
        layout: { "text-field": ["get", "name"], "text-size": 11, "text-offset": [0, 1.9], "text-font": ["Noto Sans Regular"] },
        paint: { "text-color": dark ? "#e6ecf5" : "#1b2a3d", "text-halo-color": dark ? "#0b1220" : "#ffffff", "text-halo-width": 1.4 },
      });
      m.on("mouseenter", "stations", () => (m.getCanvas().style.cursor = "pointer"));
      m.on("mouseleave", "stations", () => (m.getCanvas().style.cursor = ""));
      m.on("click", "stations", (e) => {
        const f = e.features?.[0];
        if (!f) return;
        const p = f.properties as Record<string, string | number | boolean>;
        new maplibregl.Popup({ offset: 14 })
          .setLngLat((f.geometry as { coordinates: [number, number] }).coordinates)
          .setHTML(
            p.modeled === true || p.modeled === "true"
              ? `<strong>${p.name}</strong><br/>median ${fmtNum(Number(p.q50))} kWh · q95 ${fmtNum(Number(p.q95))} kWh<br/>${fmtPct(Number(p.ratio))} of assumed capacity`
              : `<strong>${p.name}</strong><br/>not modelled (no history in the window)`,
          )
          .addTo(m);
        if (p.modeled === true || p.modeled === "true") selectRef.current?.(String(p.id));
      });
      setReady(true);
    });
    let raf = 0;
    const pulse = (t: number) => {
      if (m.getLayer("halo")) {
        const k = (t % 1600) / 1600;
        m.setPaintProperty("halo", "circle-radius", 16 + 18 * k);
        m.setPaintProperty("halo", "circle-opacity", 0.35 * (1 - k));
      }
      raf = requestAnimationFrame(pulse);
    };
    raf = requestAnimationFrame(pulse);
    return () => {
      cancelAnimationFrame(raf);
      m.remove();
      map.current = null;
    };
  }, []);

  useEffect(() => {
    const m = map.current;
    if (!ready || !m || !snapshot) return;
    const g = toGeoJSON(snapshot);
    (m.getSource("stations") as GeoJSONSource).setData(g.stations);
    (m.getSource("arcs") as GeoJSONSource).setData(g.arcs);
    if (!fitted.current && snapshot.stations.length) {
      // frame every station once; afterwards keep the user's pan/zoom
      const b = new maplibregl.LngLatBounds();
      snapshot.stations.forEach((x) => b.extend([x.lon, x.lat]));
      m.fitBounds(b, { padding: 60, maxZoom: 15, duration: 0 });
      fitted.current = true;
    }
  }, [ready, snapshot]);

  return (
    <div className="relative h-full min-h-[360px] overflow-hidden rounded-[14px]">
      {/* sized directly: maplibre-gl.css sets .maplibregl-map { position: relative }, which would cancel absolute positioning */}
      <div ref={el} className="h-full min-h-[360px] w-full" />
      {failed && (
        <div className="absolute inset-0 grid place-items-center bg-surface-2 p-6 text-center text-sm text-muted">
          Map tiles could not be loaded (offline?). Station data is still listed alongside.
        </div>
      )}
      <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap gap-3 rounded-lg bg-surface/90 px-3 py-2 text-[11px] text-muted shadow">
        {(["NORMAL", "WARNING", "CRITICAL"] as const).map((r) => (
          <span key={r} className="flex items-center gap-1.5">
            <span className="size-2.5 rounded-full" style={{ background: COLORS[r] }} />
            {r.toLowerCase()}
          </span>
        ))}
        <span className="flex items-center gap-1.5">
          <span className="h-0.5 w-4" style={{ background: COLORS.ARC }} /> reroute
        </span>
        <span>size = forecast load</span>
      </div>
    </div>
  );
}
