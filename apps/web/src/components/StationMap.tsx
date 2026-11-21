"use client";

// City map: MapLibre GL with free OpenFreeMap vector tiles (no token). A realistic day map (parks, water,
// road hierarchy) while the sun is up in the city and a dark night map after sunset, switched on the
// simulation clock. Circle colour = q95 risk, radius = forecast median, pulsing halo for CRITICAL, a ring and
// a station card for the selected site, and flowing dashed arcs for reroute suggestions.

import maplibregl, { type GeoJSONSource, type Map as MLMap, type Popup } from "maplibre-gl";
import { useEffect, useRef, useState } from "react";

import type { Snapshot, SnapshotStation } from "@/lib/api";
import { fmtNum, fmtPct, LOCALE, prettyStation, SITE_TZ } from "@/lib/format";
import { isNight, sunTimes } from "@/lib/sun";

type Mode = "day" | "night";
const STYLES: Record<Mode, string> = {
  day: "https://tiles.openfreemap.org/styles/liberty", // colourful, OSM-like: green parks, blue water, yellow roads
  night: "https://tiles.openfreemap.org/styles/fiord", // dark blue with readable roads and labels
};
// Map-specific ink so pins and labels read on either base map (the panels keep the app theme).
const INK: Record<Mode, { STROKE: string; LABEL: string; HALO: string; CONTEXT: string; SELECT: string }> = {
  day: { STROKE: "#ffffff", LABEL: "#1f2937", HALO: "#ffffff", CONTEXT: "#64748b", SELECT: "#111827" },
  night: { STROKE: "#0b1020", LABEL: "#e5e7eb", HALO: "#0b1020", CONTEXT: "#94a3b8", SELECT: "#ffffff" },
};
const DEFAULT_CENTRE: [number, number] = [73.83, 18.53]; // lon, lat (Pune) until stations load
const HHMM = new Intl.DateTimeFormat(LOCALE, { timeZone: SITE_TZ, hour: "2-digit", minute: "2-digit", hour12: false });
const hhmm = (d: Date) => HHMM.format(d);
const reducedMotion = () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// MapLibre paints on a canvas, so it cannot use var(); resolve the design tokens once the map loads.
const TOKENS = {
  NORMAL: "--ok", WARNING: "--warn", CRITICAL: "--crit", CONTEXT: "--context", ARC: "--accent",
  STROKE: "--pin-stroke", LABEL: "--map-label", HALO: "--map-label-halo", BG: "--map-bg", WATER: "--map-water", SELECT: "--text",
} as const;
type Palette = Record<keyof typeof TOKENS, string>;
function readPalette(): Palette {
  const cs = getComputedStyle(document.documentElement);
  return Object.fromEntries(Object.entries(TOKENS).map(([k, v]) => [k, cs.getPropertyValue(v).trim()])) as Palette;
}

function paletteFor(mode: Mode): Palette {
  return { ...readPalette(), ...INK[mode] };
}

// Our overlay layers. Called on every style load, because swapping the base style drops custom layers.
function addOverlays(m: MLMap, C: Palette) {
  const empty = { type: "FeatureCollection" as const, features: [] };
  m.addSource("stations", { type: "geojson", data: empty });
  m.addSource("arcs", { type: "geojson", data: empty });
  m.addLayer({
    id: "arcs-glow", type: "line", source: "arcs", layout: { "line-cap": "round" },
    paint: { "line-color": C.ARC, "line-width": ["interpolate", ["linear"], ["get", "kwh"], 0, 6, 10, 12], "line-opacity": 0.18, "line-blur": 3 },
  });
  m.addLayer({
    id: "arcs", type: "line", source: "arcs", layout: { "line-cap": "round" },
    paint: { "line-color": C.ARC, "line-width": ["interpolate", ["linear"], ["get", "kwh"], 0, 2, 10, 4], "line-dasharray": [0, 4, 3] },
  });
  m.addLayer({
    id: "halo", type: "circle", source: "stations", filter: ["==", ["get", "risk"], "CRITICAL"],
    paint: { "circle-color": C.CRITICAL, "circle-radius": 24, "circle-opacity": 0.28, "circle-blur": 0.45 },
  });
  m.addLayer({
    id: "glow", type: "circle", source: "stations", filter: ["!=", ["get", "risk"], "CONTEXT"],
    paint: {
      "circle-color": ["match", ["get", "risk"], "CRITICAL", C.CRITICAL, "WARNING", C.WARNING, C.NORMAL],
      "circle-radius": ["interpolate", ["linear"], ["get", "q50"], 0, 14, 10, 20, 40, 32],
      "circle-opacity": 0.18, "circle-blur": 1,
    },
  });
  m.addLayer({
    id: "selected", type: "circle", source: "stations", filter: ["==", ["get", "id"], ""],
    paint: {
      "circle-color": "rgba(0,0,0,0)",
      "circle-radius": ["+", ["interpolate", ["linear"], ["get", "q50"], 0, 7, 10, 12, 40, 22], 6],
      "circle-stroke-color": C.SELECT, "circle-stroke-width": 2,
    },
  });
  m.addLayer({
    id: "stations", type: "circle", source: "stations",
    paint: {
      "circle-color": ["match", ["get", "risk"], "CRITICAL", C.CRITICAL, "WARNING", C.WARNING, "NORMAL", C.NORMAL, C.CONTEXT],
      "circle-radius": ["match", ["get", "risk"], "CONTEXT", 4.5, ["interpolate", ["linear"], ["get", "q50"], 0, 7, 10, 12, 40, 22]],
      "circle-stroke-color": C.STROKE,
      "circle-stroke-width": ["match", ["get", "risk"], "CONTEXT", 1.2, 2.5],
      "circle-opacity": ["match", ["get", "risk"], "CONTEXT", 0.85, 1],
    },
  });
  m.addLayer({
    id: "labels", type: "symbol", source: "stations", filter: ["==", ["get", "modeled"], true],
    layout: { "text-field": ["get", "name"], "text-size": 12, "text-offset": [0, 2], "text-font": ["Noto Sans Bold"], "text-optional": true },
    paint: { "text-color": C.LABEL, "text-halo-color": C.HALO, "text-halo-width": 1.8 },
  });
}

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
        properties: { id: x.station_id, name: x.name, risk: x.is_modeled ? x.risk ?? "NORMAL" : "CONTEXT", q50: x.q50 ?? 0, modeled: x.is_modeled },
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

// Station card shown in the popup. Built with DOM nodes (names come from the API, never as HTML).
function stationCard(x: SnapshotStation, s: Snapshot, colour: string, onOpen?: (id: string) => void): HTMLElement {
  const el = (tag: string, cls: string, text?: string) => {
    const n = document.createElement(tag);
    n.className = cls;
    if (text) n.textContent = text;
    return n;
  };
  const root = el("div", "min-w-52");
  const head = el("div", "flex items-center gap-2 pr-5");
  if (x.is_modeled) {
    const pct = el("span", "tabular rounded-full px-2 py-0.5 font-mono text-[11px] font-semibold text-[#0b0f20]", fmtPct(x.load_ratio_q95));
    pct.style.background = colour;
    head.append(pct);
  }
  head.append(el("strong", "text-[13px]", x.name));
  root.append(head);
  if (!x.is_modeled) {
    root.append(el("div", "mt-1 text-muted", "Not modelled (no usage history in the window)."));
    return root;
  }
  root.append(el("div", "tabular mt-1.5 text-muted", `median ${fmtNum(x.q50)} kWh · q95 ${fmtNum(x.q95)} kWh`));
  const r = s.reroutes.find((y) => y.from === x.station_id);
  if (r) root.append(el("div", "tabular mt-0.5 text-[#c7d2fe]", `Reroute ${fmtNum(r.shifted_kwh)} kWh → ${prettyStation(r.to)} (${fmtNum(r.distance_km, 1)} km)`));
  const open = el("a", "btn btn-primary btn-sm mt-3 w-full", "Open station  →") as HTMLAnchorElement;
  open.href = `/stations/${x.station_id}`;
  open.addEventListener("click", (e) => {
    if (!onOpen) return;
    e.preventDefault();
    onOpen(x.station_id);
  });
  root.append(open);
  return root;
}

export type MapPadding = { top: number; right: number; bottom: number; left: number };

export function StationMap({
  snapshot,
  selectedId,
  onSelect,
  onOpen,
  padding = { top: 60, right: 60, bottom: 60, left: 60 },
  focusCity = null,
}: {
  snapshot: Snapshot | undefined;
  selectedId?: string | null;
  onSelect?: (id: string | null) => void;
  onOpen?: (id: string) => void;
  padding?: MapPadding;
  /** fly to this city's stations (null = all of them); every station stays on the map */
  focusCity?: string | null;
}) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<MLMap | null>(null);
  const popup = useRef<Popup | null>(null);
  const palette = useRef<Palette | null>(null);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(false);
  const fitted = useRef(false);
  const cb = useRef({ onSelect, onOpen });
  cb.current = { onSelect, onOpen };
  const pad = useRef(padding);
  pad.current = padding;
  // day or night where the map is looking (the focused city, else all stations), at the console clock
  const clock = snapshot ? new Date(snapshot.forecast_origin) : new Date();
  const inView = snapshot?.stations.filter((x) => !focusCity || x.city === focusCity) ?? [];
  const centre: [number, number] = inView.length
    ? [inView.reduce((a, x) => a + x.lon, 0) / inView.length, inView.reduce((a, x) => a + x.lat, 0) / inView.length]
    : DEFAULT_CENTRE;
  const mode: Mode = isNight(clock, centre[1], centre[0]) ? "night" : "day";
  const sun = sunTimes(clock, centre[1], centre[0]);
  const modeRef = useRef(mode);

  useEffect(() => {
    if (!el.current || map.current) return;
    const m = new maplibregl.Map({
      container: el.current, style: STYLES[modeRef.current], center: [73.86, 18.52], zoom: 11.5, attributionControl: { compact: true },
    });
    map.current = m;
    m.addControl(new maplibregl.NavigationControl({ showCompass: false }), "bottom-right");
    m.on("error", (e) => {
      if (!m.isStyleLoaded() && String(e.error?.message ?? "").length) setFailed(true);
    });
    // the base style names a few pattern images it does not ship; give them a transparent pixel
    m.on("styleimagemissing", (e) => {
      if (!m.hasImage(e.id)) m.addImage(e.id, { width: 1, height: 1, data: new Uint8Array(4) });
    });
    // fires for the first style and after every day/night swap: (re)build our layers on top of the base map
    m.on("style.load", () => {
      const C = paletteFor(modeRef.current);
      palette.current = C;
      addOverlays(m, C);
      setReady(true);
    });
    // layer-scoped handlers are kept by the map across style swaps, so register them once
    m.on("mouseenter", "stations", () => (m.getCanvas().style.cursor = "pointer"));
    m.on("mouseleave", "stations", () => (m.getCanvas().style.cursor = ""));
    // one click handler: a station selects it, empty map clears the selection
    m.on("click", (e) => {
      if (!m.getLayer("stations")) return;
      const f = m.queryRenderedFeatures(e.point, { layers: ["stations"] })[0];
      cb.current.onSelect?.(f ? String((f.properties as { id: string }).id) : null);
    });

    let raf = 0;
    // flowing dashes: step through phase-shifted dash arrays (MapLibre cannot animate a dash offset)
    const DASHES = [[0, 4, 3], [0.5, 4, 2.5], [1, 4, 2], [1.5, 4, 1.5], [2, 4, 1], [2.5, 4, 0.5], [3, 4, 0], [0, 0.5, 3, 3.5], [0, 1, 3, 3], [0, 1.5, 3, 2.5], [0, 2, 3, 2], [0, 2.5, 3, 1.5], [0, 3, 3, 1], [0, 3.5, 3, 0.5]];
    let dashStep = -1;
    const tick = (t: number) => {
      if (m.getLayer("halo")) {
        const k = (t % 2000) / 2000;
        const e = 1 - (1 - k) ** 3; // ease-out: fast bloom, slow fade
        m.setPaintProperty("halo", "circle-radius", 14 + 22 * e);
        m.setPaintProperty("halo", "circle-opacity", 0.42 * (1 - e));
      }
      const step = Math.floor(t / 70) % DASHES.length;
      if (step !== dashStep && m.getLayer("arcs")) {
        m.setPaintProperty("arcs", "line-dasharray", DASHES[step]);
        dashStep = step;
      }
      raf = requestAnimationFrame(tick);
    };
    // with reduced motion the halo stays a static ring and the arcs stay dashed
    if (!reducedMotion()) raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      const p = popup.current;
      popup.current = null;
      p?.remove();
      m.remove();
      map.current = null;
    };
  }, []);

  // swap the base map at sunrise and sunset; style.load re-adds the overlays and the data effect refills them
  useEffect(() => {
    const m = map.current;
    if (!m || modeRef.current === mode) return;
    modeRef.current = mode;
    setReady(false);
    m.setStyle(STYLES[mode]);
  }, [mode]);

  // data
  useEffect(() => {
    const m = map.current;
    if (!ready || !m || !snapshot) return;
    const g = toGeoJSON(snapshot);
    (m.getSource("stations") as GeoJSONSource).setData(g.stations);
    (m.getSource("arcs") as GeoJSONSource).setData(g.arcs);
    if (!fitted.current && snapshot.stations.length) {
      // frame every station once, clear of the floating panels; afterwards keep the user's pan/zoom
      const b = new maplibregl.LngLatBounds();
      snapshot.stations.forEach((x) => b.extend([x.lon, x.lat]));
      m.fitBounds(b, { padding: pad.current, maxZoom: 14.5, duration: 0 });
      fitted.current = true;
    }
  }, [ready, snapshot]);

  // fly to the chosen city (or back out to every station)
  const lastFocus = useRef<string | null | undefined>(undefined);
  useEffect(() => {
    const m = map.current;
    if (!ready || !m || !snapshot || lastFocus.current === focusCity) return;
    const pts = snapshot.stations.filter((x) => !focusCity || x.city === focusCity);
    if (!pts.length) return;
    const first = lastFocus.current === undefined;
    lastFocus.current = focusCity;
    if (first && !focusCity) return; // the initial framing above already shows everything
    const b = new maplibregl.LngLatBounds();
    pts.forEach((x) => b.extend([x.lon, x.lat]));
    m.fitBounds(b, { padding: pad.current, maxZoom: 13.5, duration: reducedMotion() ? 0 : 1100 });
  }, [ready, snapshot, focusCity]);

  // selection: ring, card, and a smooth glide so the station sits in the clear part of the map
  useEffect(() => {
    const m = map.current;
    if (!ready || !m) return;
    m.setFilter("selected", ["==", ["get", "id"], selectedId ?? ""]);
    const x = selectedId ? snapshot?.stations.find((s) => s.station_id === selectedId) : undefined;
    if (!x || !snapshot) {
      const p = popup.current;
      popup.current = null;
      p?.remove();
      return;
    }
    const C = palette.current;
    const colour = C ? (x.risk === "CRITICAL" ? C.CRITICAL : x.risk === "WARNING" ? C.WARNING : C.NORMAL) : "#fff";
    const card = stationCard(x, snapshot, colour, cb.current.onOpen);
    if (!popup.current) {
      const p = new maplibregl.Popup({ offset: 18, maxWidth: "280px", focusAfterOpen: false, closeOnClick: false });
      p.on("close", () => {
        // the card's own close button; removals made here clear popup.current first
        if (popup.current !== p) return;
        popup.current = null;
        cb.current.onSelect?.(null);
      });
      popup.current = p;
    }
    popup.current.setLngLat([x.lon, x.lat]).setDOMContent(card);
    if (!popup.current.isOpen()) popup.current.addTo(m);
  }, [ready, selectedId, snapshot]);

  // glide only when the selection itself changes (not on every data refresh)
  useEffect(() => {
    const m = map.current;
    if (!ready || !m || !selectedId) return;
    const x = snapshot?.stations.find((s) => s.station_id === selectedId);
    if (x) m.easeTo({ center: [x.lon, x.lat], padding: pad.current, duration: reducedMotion() ? 0 : 900, easing: (t) => 1 - (1 - t) ** 3 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, selectedId]);

  return (
    <div className="absolute inset-0">
      {/* sized directly: maplibre-gl.css sets .maplibregl-map { position: relative }, which would cancel absolute positioning */}
      <div ref={el} className="h-full w-full" />
      <div
        className="pointer-events-none absolute right-14 bottom-3 z-[1] rounded-full border border-border bg-surface/85 px-2.5 py-1 text-[11px] text-muted shadow-sm backdrop-blur"
        title="The map follows day and night in the city on the console clock"
      >
        {mode === "day" ? "☀ Day map" : "☾ Night map"} · sunrise {hhmm(sun.sunrise)} · sunset {hhmm(sun.sunset)}
      </div>
      {failed && (
        <div className="absolute inset-0 grid place-items-center bg-bg p-6 text-center text-sm text-muted">
          Map tiles could not be loaded (offline?). Station data is still listed alongside.
        </div>
      )}
    </div>
  );
}
