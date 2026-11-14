"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";

import { api, type Risk } from "@/lib/api";
import { fmtNum } from "@/lib/format";

function Tile({ label, value, unit, sub, tone, wide }: { label: string; value: string; unit?: string; sub?: string; tone?: Risk; wide?: boolean }) {
  return (
    <div className={clsx("card min-w-0 px-4 py-3", wide && "col-span-2 lg:col-span-1")}>
      <div className="text-[11px] uppercase tracking-wider text-muted">{label}</div>
      <div
        className={clsx(
          "tabular mt-1 text-2xl font-semibold",
          tone === "CRITICAL" && "text-crit",
          tone === "WARNING" && "text-warn",
          tone === "NORMAL" && "text-ok",
        )}
      >
        {value}
        {unit && <span className="ml-1 text-sm font-normal text-muted">{unit}</span>}
      </div>
      {sub && <div className="mt-0.5 truncate text-xs text-muted">{sub}</div>}
    </div>
  );
}

export function KpiRibbon() {
  const { data: k, isError } = useQuery({ queryKey: ["kpis"], queryFn: api.kpis });
  if (isError) return <div className="card p-4 text-sm text-crit">Cannot reach the API. Is `uv run power-api serve` running?</div>;
  const n = k?.next_hour;
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
      <Tile label="Network load now" value={fmtNum(k?.current_load_kw)} unit="kW" sub="mean over the last closed hour" />
      <Tile
        label="Next hour (median)"
        value={fmtNum(n?.q50)}
        unit="kWh"
        sub={n?.lower != null ? `${Math.round((n.interval_level ?? 0.9) * 100)}% interval ${fmtNum(n.lower)} – ${fmtNum(n.upper)}` : undefined}
      />
      <Tile label="Port occupancy" value={fmtNum(k?.port_occupancy_pct, 0)} unit="%" sub={`${fmtNum(k?.idle_port_pct, 0)}% of occupied ports idle`} />
      <Tile
        label="Grid risk (next 12 h)"
        value={k?.risk_status ?? "…"}
        tone={k?.risk_status}
        sub={k ? `${k.stations_by_risk.CRITICAL} critical · ${k.stations_by_risk.WARNING} warning` : undefined}
      />
      <Tile
        wide
        label="Mode"
        value={k?.sim_mode === "LIVE_SHADOW" ? "Live-shadow" : "Replay"}
        sub="Pune pilot twin · replays held-out demand, never seen in training"
      />
    </div>
  );
}
