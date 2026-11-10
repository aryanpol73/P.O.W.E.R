"use client";

import type { Explanation } from "@/lib/api";
import { fmtNum } from "@/lib/format";

// Additive waterfall: base value + each contribution = forecast (identity link, so the bars sum in kWh).
export function ShapWaterfall({ e }: { e: Explanation }) {
  const steps = [{ label: "base value (average forecast)", value: e.base_value, base: true }, ...e.contributions.map((c) => ({
    label: c.feature === "__other__" ? "all other features" : `${c.display}${c.value !== null && typeof c.value === "number" ? ` = ${fmtNum(c.value, 1)}` : ""}`,
    value: c.shap,
    base: false,
  }))];
  let cum = 0;
  const bars = steps.map((s) => {
    const start = s.base ? 0 : cum;
    cum = start + s.value;
    return { ...s, start, end: cum };
  });
  const lo = Math.min(0, ...bars.map((b) => Math.min(b.start, b.end)));
  const hi = Math.max(...bars.map((b) => Math.max(b.start, b.end)), e.prediction);
  const x = (v: number) => ((v - lo) / (hi - lo || 1)) * 100;

  return (
    <div className="flex flex-col gap-1.5">
      {bars.map((b, i) => (
        <div key={i} className="grid grid-cols-[minmax(0,44%)_1fr_56px] items-center gap-2 text-xs">
          <span className="truncate text-muted" title={b.label}>
            {b.label}
          </span>
          <div className="relative h-4 rounded bg-surface-2">
            <div
              className="absolute top-0 h-4 rounded"
              style={{
                left: `${x(Math.min(b.start, b.end))}%`,
                width: `${Math.max(0.6, Math.abs(x(b.end) - x(b.start)))}%`,
                background: b.base ? "var(--muted)" : b.value >= 0 ? "var(--crit)" : "var(--accent)",
                opacity: b.base ? 0.6 : 0.9,
              }}
            />
          </div>
          <span className="tabular text-right">{b.base ? fmtNum(b.value) : `${b.value >= 0 ? "+" : ""}${fmtNum(b.value)}`}</span>
        </div>
      ))}
      <div className="mt-1 grid grid-cols-[minmax(0,44%)_1fr_56px] items-center gap-2 border-t border-border pt-2 text-xs font-semibold">
        <span>forecast</span>
        <span />
        <span className="tabular text-right">{fmtNum(e.prediction)}</span>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-muted">
        Red bars push the forecast up, blue bars pull it down. Values are TreeSHAP contributions of the champion model in kWh,
        so they add up exactly to the forecast.
      </p>
    </div>
  );
}
