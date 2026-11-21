"use client";

import clsx from "clsx";
import { ChevronLeft, ChevronRight } from "lucide-react";

import type { NetworkHorizon, Risk } from "@/lib/api";
import { fmtHour, fmtNum } from "@/lib/format";

const BAR: Record<Risk, string> = {
  NORMAL: "from-ok to-ok/25",
  WARNING: "from-warn to-warn/25",
  CRITICAL: "from-crit to-crit/25",
};
const GLOW: Record<Risk, string> = {
  NORMAL: "shadow-[0_0_0_2px_var(--bg),0_0_0_4px_var(--accent),0_0_20px_rgba(45,212,191,0.45)]",
  WARNING: "shadow-[0_0_0_2px_var(--bg),0_0_0_4px_var(--accent),0_0_20px_rgba(251,146,60,0.5)]",
  CRITICAL: "shadow-[0_0_0_2px_var(--bg),0_0_0_4px_var(--accent),0_0_20px_rgba(251,113,133,0.55)]",
};

// Forecast horizon picker: one bar per hour ahead, height = network median, colour = that hour's worst site.
export function HorizonDock({
  h,
  onChange,
  data,
  showOthers,
  onToggleOthers,
  className,
}: {
  h: number;
  onChange: (h: number) => void;
  data: NetworkHorizon | undefined;
  showOthers: boolean;
  onToggleOthers: (v: boolean) => void;
  className?: string;
}) {
  const hours = data?.hours ?? [];
  const max = Math.max(1, ...hours.map((x) => x.q50));
  const cur = hours.find((x) => x.h === h);

  return (
    <section aria-label="Forecast horizon" className={clsx("hud px-4 pt-3.5 pb-3 md:px-5", className)}>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <h2 className="text-[13px] font-extrabold">Forecast horizon</h2>
        <div className="hidden items-center gap-3 text-[11px] text-muted md:flex">
          <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-ok shadow-[0_0_8px_var(--ok)]" />Normal</span>
          <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-warn shadow-[0_0_8px_var(--warn)]" />≥70%</span>
          <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-crit shadow-[0_0_8px_var(--crit)]" />≥85%</span>
          <span className="flex items-center gap-1.5"><span className="w-3 border-t-2 border-dashed border-accent" />Reroute</span>
        </div>
        <div className="flex-1" />
        <label className="flex cursor-pointer items-center gap-2 text-[11px] font-semibold text-muted select-none hover:text-text">
          <span className="relative inline-flex">
            <input type="checkbox" checked={showOthers} onChange={(e) => onToggleOthers(e.target.checked)} className="peer sr-only" />
            <span className="h-5 w-9 rounded-full bg-white/10 shadow-[inset_0_0_0_1px_rgba(165,180,252,0.2)] transition-colors duration-300 peer-checked:bg-gradient-to-r peer-checked:from-[var(--grad-a)] peer-checked:to-[var(--grad-b)] peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-[var(--ring)]" />
            <span className="absolute top-0.5 left-0.5 size-4 rounded-full bg-white shadow transition-transform duration-300 ease-[var(--ease)] peer-checked:translate-x-4" />
          </span>
          <span className="hidden sm:inline">Other chargers</span>
          <span className="sr-only sm:hidden">Other chargers</span>
        </label>
        <div className="flex items-center gap-1.5">
          <button aria-label="Previous hour" onClick={() => onChange(Math.max(1, h - 1))} disabled={h <= 1} className="btn btn-secondary btn-sm btn-icon">
            <ChevronLeft size={15} strokeWidth={2.4} />
          </button>
          <span className="tabular min-w-[6.5rem] sm:min-w-[7.5rem] rounded-[10px] bg-[rgba(99,102,241,0.18)] px-2.5 py-1.5 text-center font-mono text-[13px] font-semibold text-[#e0e7ff] shadow-[inset_0_0_0_1px_rgba(165,180,252,0.28)]" aria-live="polite">
            +{h} h · {cur ? fmtHour(cur.target_time) : "…"}
          </span>
          <button aria-label="Next hour" onClick={() => onChange(Math.min(24, h + 1))} disabled={h >= 24} className="btn btn-secondary btn-sm btn-icon">
            <ChevronRight size={15} strokeWidth={2.4} />
          </button>
        </div>
      </div>

      <div className="mt-3 flex h-12 items-end gap-[3px] md:gap-[5px]" role="group" aria-label="Hours ahead">
        {hours.length === 0
          ? Array.from({ length: 24 }, (_, i) => <span key={i} className="flex-1 animate-pulse rounded-[5px] bg-white/[0.06]" style={{ height: `${30 + ((i * 37) % 50)}%` }} />)
          : hours.map((x) => {
              const on = x.h === h;
              return (
                <button
                  key={x.h}
                  onClick={() => onChange(x.h)}
                  aria-pressed={on}
                  aria-label={`+${x.h} hours, ${fmtHour(x.target_time)}: network median ${fmtNum(x.q50, 0)} kWh, ${x.risk.toLowerCase()}`}
                  title={`+${x.h} h · ${fmtHour(x.target_time)} · ${fmtNum(x.q50, 0)} kWh`}
                  className={clsx(
                    "flex-1 rounded-[5px] bg-gradient-to-b transition-[height,opacity,box-shadow,translate] duration-500 ease-[var(--ease)] hover:-translate-y-0.5 hover:opacity-100",
                    BAR[x.risk],
                    on ? clsx("opacity-100", GLOW[x.risk]) : "opacity-50",
                  )}
                  style={{ height: `${Math.max(10, (x.q50 / max) * 100)}%` }}
                />
              );
            })}
      </div>
      <div className="tabular mt-2 flex justify-between font-mono text-[11px] text-muted">
        <span>+1 h</span>
        <span>+6 h</span>
        <span>+12 h</span>
        <span>+18 h</span>
        <span>+24 h</span>
      </div>
    </section>
  );
}
