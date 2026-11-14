"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { api, type LeaderRow } from "@/lib/api";
import { fmtNum, modelLabel } from "@/lib/format";

// "target": best is closest to the nominal 90% coverage, not the highest (100% would just be too wide).
const COLS: { key: keyof LeaderRow | "picp"; label: string; better: "low" | "high" | "target"; digits: number }[] = [
  { key: "wape", label: "WAPE", better: "low", digits: 3 },
  { key: "mase", label: "MASE", better: "low", digits: 3 },
  { key: "mae", label: "MAE kWh", better: "low", digits: 2 },
  { key: "rmse", label: "RMSE kWh", better: "low", digits: 2 },
  { key: "r2", label: "R²", better: "high", digits: 3 },
  { key: "peak_mae", label: "Peak MAE", better: "low", digits: 2 },
  { key: "picp", label: "90% cover", better: "target", digits: 3 },
  { key: "train_time_s", label: "Train s", better: "low", digits: 0 },
  { key: "latency_ms_one_origin", label: "ms / origin", better: "low", digits: 1 },
];
const CURVES = ["B1_seasonal_naive_24h", "B3_climatology", "M1_ridge", "M3_xgboost", "M4_lightgbm", "M4_lightgbm_quantile", "M5_lstm"];
const PALETTE = ["#8a96a8", "#b07cc6", "#e08a3c", "#2a9d8f", "#1f4fd1", "#d23b3b", "#16a34a"];

const value = (r: LeaderRow, k: string) => (k === "picp" ? (r.interval_cqr ?? r.interval_split_conformal).picp : (r as unknown as Record<string, number | null>)[k]);

export default function ArenaPage() {
  const lb = useQuery({ queryKey: ["leaderboard"], queryFn: api.leaderboard, staleTime: Infinity });
  const hz = useQuery({ queryKey: ["horizons"], queryFn: api.horizons, staleTime: Infinity });
  const live = useQuery({ queryKey: ["arena"], queryFn: () => api.liveArena(168) });
  const [sortKey, setSortKey] = useState("wape");

  const rows = useMemo(() => {
    const entries = Object.entries(lb.data?.leaderboard ?? {});
    const col = COLS.find((c) => c.key === sortKey)!;
    return entries.sort((a, b) => {
      const va = value(a[1], sortKey) ?? Infinity, vb = value(b[1], sortKey) ?? Infinity;
      if (col.better === "target") return Math.abs(va - 0.9) - Math.abs(vb - 0.9);
      return col.better === "low" ? va - vb : vb - va;
    });
  }, [lb.data, sortKey]);
  const best = useMemo(() => {
    const out: Record<string, number> = {};
    for (const c of COLS) {
      const vals = rows.map(([, r]) => value(r, c.key)).filter((v): v is number => typeof v === "number");
      out[c.key] =
        c.better === "target"
          ? vals.reduce((a, b) => (Math.abs(b - 0.9) < Math.abs(a - 0.9) ? b : a), vals[0])
          : c.better === "low"
            ? Math.min(...vals)
            : Math.max(...vals);
    }
    return out;
  }, [rows]);
  const curve = useMemo(() => {
    const byH = new Map<number, Record<string, number>>();
    for (const r of hz.data ?? []) {
      if (!CURVES.includes(r.model)) continue;
      byH.set(r.h, { ...(byH.get(r.h) ?? { h: r.h }), [r.model]: r.wape });
    }
    return [...byH.values()].sort((a, b) => a.h - b.h);
  }, [hz.data]);

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h1 className="text-lg font-semibold">Model Arena</h1>
        <p className="text-sm text-muted">
          Held-out test period {lb.data ? `(${lb.data.rows.toLocaleString()} forecasts, evaluated once)` : ""}. Click a column to sort. WAPE is
          the primary metric: MAPE is undefined on the many zero-demand hours.
        </p>
      </div>

      <section className="card overflow-x-auto">
        <table className="w-full min-w-[860px] text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs text-muted">
              <th className="px-4 py-3 font-medium">Model</th>
              {COLS.map((c) => (
                <th key={c.key} className="px-3 py-3 text-right font-medium">
                  <button onClick={() => setSortKey(c.key)} className={clsx("hover:text-text", sortKey === c.key && "text-accent")}>
                    {c.label}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(([name, r]) => (
              <tr key={name} className={clsx("border-b border-border last:border-0", name === "M4_lightgbm_quantile" && "bg-accent-soft/50")}>
                <td className="px-4 py-2.5 font-medium">{modelLabel(name)}</td>
                {COLS.map((c) => {
                  const v = value(r, c.key);
                  return (
                    <td key={c.key} className={clsx("tabular px-3 py-2.5 text-right", v === best[c.key] && "font-semibold text-ok")}>
                      {typeof v === "number" ? fmtNum(v, c.digits) : "-"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        {lb.isError && <p className="p-4 text-sm text-muted">{(lb.error as Error).message}</p>}
      </section>

      <div className="grid gap-4 xl:grid-cols-[1fr_380px]">
        <section className="card p-4">
          <h2 className="text-sm font-semibold">Error by forecast horizon (test)</h2>
          <p className="mb-2 text-xs text-muted">The LSTM wins the next hour; gradient boosting wins from about 6 hours ahead.</p>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={curve} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="h" stroke="var(--muted)" tick={{ fontSize: 11 }} tickFormatter={(v) => `+${v}h`} />
              <YAxis stroke="var(--muted)" tick={{ fontSize: 11 }} width={40} domain={["auto", "auto"]} />
              <Tooltip
                contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }}
                formatter={(v, n) => [fmtNum(Number(v), 3), modelLabel(String(n))]}
                labelFormatter={(v) => `horizon +${v} h`}
              />
              <Legend formatter={(v) => <span className="text-xs">{modelLabel(String(v))}</span>} />
              {CURVES.map((m, i) => (
                <Line key={m} dataKey={m} stroke={PALETTE[i]} dot={false} strokeWidth={m === "M4_lightgbm_quantile" ? 2.6 : 1.6} isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </section>

        <section className="card p-4">
          <h2 className="text-sm font-semibold">Live arena</h2>
          <p className="mb-3 text-xs text-muted">
            Rolling accuracy on hours replayed in this session (last 7 days of simulated time, {live.data?.n.toLocaleString() ?? 0} scored forecasts).
          </p>
          {live.data && live.data.n === 0 && <p className="text-sm text-muted">Press play: scores appear once forecast hours have been observed.</p>}
          <ul className="flex flex-col gap-2">
            {Object.entries(live.data?.models ?? {}).map(([name, m]) => (
              <li key={name} className="grid grid-cols-[1fr_auto] items-center gap-2 text-xs">
                <span className="truncate">{modelLabel(name)}</span>
                <span className="tabular font-medium">WAPE {fmtNum(m.wape, 3)}</span>
                <div className="col-span-2 h-1.5 overflow-hidden rounded bg-surface-2">
                  <div className="h-full rounded bg-accent" style={{ width: `${Math.min(100, m.wape * 70)}%` }} />
                </div>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
