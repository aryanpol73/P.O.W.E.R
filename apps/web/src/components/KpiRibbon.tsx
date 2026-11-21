"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { PlugZap, TrendingUp, TriangleAlert, Zap, type LucideIcon } from "lucide-react";

import { api, type Risk } from "@/lib/api";
import { fmtNum } from "@/lib/format";

// Each figure has its own colour (amber load, violet forecast, cyan ports); grid risk takes the risk colour.
const TONE = {
  amber: { chip: "bg-amber/15 text-amber", tile: "bg-amber/[0.05]" },
  violet: { chip: "bg-violet/15 text-violet", tile: "bg-violet/[0.06]" },
  cyan: { chip: "bg-cyan/15 text-cyan", tile: "bg-cyan/[0.05]" },
  NORMAL: { chip: "bg-ok/15 text-ok", tile: "bg-ok/[0.06]" },
  WARNING: { chip: "bg-warn/15 text-warn", tile: "bg-warn/[0.07]" },
  CRITICAL: { chip: "bg-crit/15 text-crit", tile: "bg-crit/[0.08]" },
} as const;

function Tile({ icon: Icon, tone, label, value, unit, sub, valueClass }: { icon: LucideIcon; tone: keyof typeof TONE; label: string; value: string; unit?: string; sub?: string; valueClass?: string }) {
  return (
    <div className={clsx("flex min-w-0 items-center gap-3 rounded-[14px] py-2 pr-4 pl-2 transition-colors duration-500", TONE[tone].tile)}>
      <span className={clsx("grid size-8 shrink-0 place-items-center rounded-[10px] transition-colors duration-500", TONE[tone].chip)} aria-hidden="true">
        <Icon size={16} strokeWidth={2.2} />
      </span>
      <div className="min-w-0">
        <div className="truncate text-[11px] font-semibold text-muted">{label}</div>
        <div className={clsx("tabular truncate font-mono text-[17px] leading-tight font-medium", valueClass)}>
          {value}
          {unit && <span className="ml-1 text-[11px] text-muted">{unit}</span>}
        </div>
        {sub && <div className="tabular truncate text-[11px] text-muted">{sub}</div>}
      </div>
    </div>
  );
}

export function KpiRibbon({ layout = "strip", className }: { layout?: "strip" | "grid"; className?: string }) {
  const { data: k, isError } = useQuery({ queryKey: ["kpis"], queryFn: api.kpis });
  if (isError)
    return <div className={clsx("hud p-4 text-sm text-crit", className)}>Cannot reach the API. Is `uv run power-api serve` running?</div>;
  const n = k?.next_hour;
  const risk: Risk = k?.risk_status ?? "NORMAL";
  const riskText = !k ? "…" : risk === "NORMAL" ? "All normal" : `${k.stations_by_risk[risk]} ${risk.toLowerCase()}`;
  return (
    <div role="group" aria-label="Key figures" className={clsx("hud p-1.5", layout === "strip" ? "flex gap-1.5" : "grid grid-cols-2 gap-1.5", className)}>
      <Tile icon={Zap} tone="amber" label="Load now" value={fmtNum(k?.current_load_kw, 0)} unit="kW" sub="last closed hour" />
      <Tile
        icon={TrendingUp}
        tone="violet"
        label="Next hour"
        value={fmtNum(n?.q50, 0)}
        unit="kWh"
        sub={n?.lower != null ? `${Math.round((n.interval_level ?? 0.9) * 100)}% range ${fmtNum(n.lower, 0)}–${fmtNum(n.upper, 0)}` : undefined}
      />
      <Tile icon={PlugZap} tone="cyan" label="Ports occupied" value={fmtNum(k?.port_occupancy_pct, 0)} unit="%" sub={`${fmtNum(k?.idle_port_pct, 0)}% of them idle`} />
      <Tile
        icon={TriangleAlert}
        tone={risk}
        label="Grid risk, 12 h"
        value={riskText}
        sub={k ? `${k.stations_by_risk.CRITICAL} crit · ${k.stations_by_risk.WARNING} warn` : undefined}
        valueClass={clsx("font-sans text-[15px] font-extrabold capitalize", risk === "CRITICAL" ? "text-crit" : risk === "WARNING" ? "text-warn" : "text-ok")}
      />
    </div>
  );
}
