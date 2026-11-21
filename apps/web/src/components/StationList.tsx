"use client";

import clsx from "clsx";
import { ArrowUpRight, MapPin, Search } from "lucide-react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import type { Risk, SnapshotStation } from "@/lib/api";
import { fmtNum, fmtPct } from "@/lib/format";

const TONE: Record<Risk, { badge: string; bar: string }> = {
  NORMAL: { badge: "bg-ok/[0.12] text-ok shadow-[inset_0_0_0_1px_rgba(45,212,191,0.24)]", bar: "from-cyan to-ok" },
  WARNING: { badge: "bg-warn/[0.13] text-warn shadow-[inset_0_0_0_1px_rgba(251,146,60,0.28)]", bar: "from-amber to-warn" },
  CRITICAL: { badge: "bg-crit/[0.14] text-crit shadow-[inset_0_0_0_1px_rgba(251,113,133,0.32)]", bar: "from-[#f472b6] to-crit" },
};

// Forecast stations ranked by q95 load. A row selects the station on the map; the arrow opens its page.
export function StationList({
  stations,
  h,
  selectedId,
  onSelect,
  className,
  cities = [],
  city = null,
  onCity,
}: {
  stations: SnapshotStation[];
  h: number;
  selectedId: string | null;
  onSelect: (id: string) => void;
  className?: string;
  cities?: { name: string; sites: number }[];
  city?: string | null;
  onCity?: (city: string | null) => void;
}) {
  const [q, setQ] = useState("");
  const input = useRef<HTMLInputElement>(null);
  const inCity = city ? stations.filter((s) => s.city === city) : stations;
  const shown = q ? inCity.filter((s) => s.name.toLowerCase().includes(q.toLowerCase())) : inCity;

  // "/" focuses the search, like most consoles
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.key === "/" && !/input|textarea|select/i.test(t.tagName) && !t.isContentEditable) {
        e.preventDefault();
        input.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <aside aria-labelledby="stations-title" className={clsx("hud flex min-h-0 flex-col overflow-hidden", className)}>
      <div className="px-4 pt-4 pb-2.5">
        <h2 id="stations-title" className="display text-lg leading-tight">
          Stations at <span className="text-grad">+{h} h</span>
        </h2>
        <p className="text-xs text-muted">q95 load as a share of the transformer rating</p>
        {cities.length > 1 && onCity && (
          <label className="mt-3 flex h-10 items-center gap-2 rounded-xl bg-black/25 px-3 shadow-[inset_0_0_0_1px_rgba(165,180,252,0.1)] transition-shadow focus-within:shadow-[inset_0_0_0_1px_var(--accent)]">
            <MapPin size={15} className="shrink-0 text-muted" aria-hidden="true" />
            <span className="sr-only">City</span>
            <select
              value={city ?? ""}
              onChange={(e) => onCity(e.target.value || null)}
              className="min-w-0 flex-1 cursor-pointer bg-transparent text-[13px] text-text outline-none [&>option]:bg-surface"
            >
              <option value="">All cities ({cities.reduce((a, c) => a + c.sites, 0)} sites)</option>
              {cities.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name} ({c.sites})
                </option>
              ))}
            </select>
          </label>
        )}
        <label className="mt-3 flex h-10 items-center gap-2 rounded-xl bg-black/25 px-3 shadow-[inset_0_0_0_1px_rgba(165,180,252,0.1)] transition-shadow focus-within:shadow-[inset_0_0_0_1px_var(--accent)]">
          <Search size={15} className="shrink-0 text-muted" aria-hidden="true" />
          <span className="sr-only">Search stations</span>
          <input
            ref={input}
            type="search"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder={`Search ${inCity.length} sites`}
            className="min-w-0 flex-1 bg-transparent text-[13px] text-text outline-none placeholder:text-muted"
          />
          <kbd className="rounded-[5px] px-1.5 font-mono text-[10px] text-muted shadow-[inset_0_0_0_1px_rgba(165,180,252,0.2)]">/</kbd>
        </label>
      </div>
      <ul className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        {shown.map((s) => {
          const risk = s.risk ?? "NORMAL";
          const on = s.station_id === selectedId;
          return (
            <li key={s.station_id} className="group relative">
              <button
                onClick={() => onSelect(s.station_id)}
                aria-pressed={on}
                className={clsx("row-hover flex w-full items-center gap-3 rounded-xl py-2 pr-10 pl-2 text-left", on && "bg-[rgba(99,102,241,0.14)] hover:bg-[rgba(99,102,241,0.18)]")}
              >
                <span className={clsx("tabular grid h-10 w-11 shrink-0 place-items-center rounded-[11px] font-mono text-xs font-semibold transition-colors duration-500", TONE[risk].badge)}>
                  {fmtPct(s.load_ratio_q95)}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[13px] font-bold">{s.name}</span>
                  {!city && cities.length > 1 && s.city && <span className="block truncate text-[11px] text-muted">{s.city}</span>}
                  <span className="mt-1.5 block h-1 overflow-hidden rounded-full bg-white/[0.06]">
                    <span
                      className={clsx("block h-full rounded-full bg-gradient-to-r transition-[width] duration-500 ease-[var(--ease)]", TONE[risk].bar)}
                      style={{ width: `${Math.min(100, Math.round((s.load_ratio_q95 ?? 0) * 100))}%` }}
                    />
                  </span>
                  <span className="tabular mt-1 block text-[11px] text-muted">{fmtNum(s.q50)} kWh median</span>
                </span>
              </button>
              <Link
                href={`/stations/${s.station_id}`}
                aria-label={`Open ${s.name}`}
                title="Open station"
                className="absolute top-1/2 right-2 grid size-8 -translate-y-1/2 place-items-center rounded-lg text-muted opacity-60 transition-[opacity,color,background-color] group-hover:opacity-100 hover:bg-white/[0.08] hover:text-text focus-visible:opacity-100"
              >
                <ArrowUpRight size={15} />
              </Link>
            </li>
          );
        })}
        {inCity.length > 0 && shown.length === 0 && <li className="px-3 py-6 text-center text-sm text-muted">No station matches “{q}”.</li>}
        {stations.length === 0 && <li className="px-3 py-6 text-center text-sm text-muted">No forecast stations in this snapshot.</li>}
      </ul>
    </aside>
  );
}
