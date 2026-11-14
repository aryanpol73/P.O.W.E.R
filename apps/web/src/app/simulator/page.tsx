"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { AlertTriangle, FlaskConical } from "lucide-react";
import { useMemo, useState } from "react";
import { Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { api, type ScenarioResult } from "@/lib/api";
import { fmtHour, fmtNum, fmtTime, prettyStation } from "@/lib/format";

function aggregate(res: ScenarioResult) {
  const by = new Map<number, { h: number; t: string; base: number; scen: number; band: [number, number] }>();
  for (const p of res.baseline) {
    const r = by.get(p.h) ?? { h: p.h, t: p.target_time, base: 0, scen: 0, band: [0, 0] as [number, number] };
    r.base += p.q50;
    by.set(p.h, r);
  }
  for (const p of res.scenario) {
    const r = by.get(p.h)!;
    r.scen += p.q50;
    r.band = [r.band[0] + p.q05, r.band[1] + p.q95];
  }
  return [...by.values()].sort((a, b) => a.h - b.h);
}

function worstPerStation(alerts: ScenarioResult["alerts_in_scenario"]): string {
  const worst = new Map<string, string>();
  for (const a of alerts) if (worst.get(a.station_id) !== "CRITICAL") worst.set(a.station_id, a.level);
  return [...worst].map(([s, l]) => `${prettyStation(s)} (${l.toLowerCase()})`).join(", ");
}

export default function SimulatorPage() {
  const stations = useQuery({ queryKey: ["stations"], queryFn: api.stations });
  const [temp, setTemp] = useState(0);
  const [rain, setRain] = useState(0);
  const [holiday, setHoliday] = useState(false);
  const [mult, setMult] = useState(1);
  const [station, setStation] = useState("*");
  const run = useMutation({ mutationFn: api.simulate });
  const data = useMemo(() => (run.data ? aggregate(run.data) : []), [run.data]);

  const submit = () =>
    run.mutate({ temperature_delta_c: temp, precipitation_mm: rain > 0 ? rain : null, force_holiday: holiday, demand_multiplier: mult, station_ids: [station] });

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h1 className="text-lg font-semibold">What-If Simulator</h1>
        <p className="text-sm text-muted">
          Shock the next 24 hours of weather or calendar and see how the champion model responds. Runs on a copy: live forecasts and
          alerts are never changed.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <section className="card flex flex-col gap-4 p-4">
          <label className="flex flex-col gap-1 text-sm">
            <span className="flex justify-between"><span>Temperature change</span><span className="tabular text-muted">{temp > 0 ? "+" : ""}{temp} °C</span></span>
            <input type="range" min={-15} max={15} step={1} value={temp} onChange={(e) => setTemp(Number(e.target.value))} className="accent-[var(--accent)]" />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="flex justify-between"><span>Rain every hour</span><span className="tabular text-muted">{rain} mm</span></span>
            <input type="range" min={0} max={10} step={0.5} value={rain} onChange={(e) => setRain(Number(e.target.value))} className="accent-[var(--accent)]" />
          </label>
          <label className="flex flex-col gap-1 text-sm">
            <span className="flex justify-between"><span>Demand multiplier (assumption)</span><span className="tabular text-muted">×{mult.toFixed(2)}</span></span>
            <input type="range" min={0.5} max={2} step={0.05} value={mult} onChange={(e) => setMult(Number(e.target.value))} className="accent-[var(--accent)]" />
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" checked={holiday} onChange={(e) => setHoliday(e.target.checked)} className="size-4 accent-[var(--accent)]" />
            Treat the whole day as a public holiday
          </label>
          <label className="flex flex-col gap-1 text-sm">
            Stations
            <select value={station} onChange={(e) => setStation(e.target.value)} className="h-9 rounded-lg border border-border bg-surface px-2">
              <option value="*">Whole network</option>
              {stations.data?.filter((s) => s.is_modeled).map((s) => (
                <option key={s.station_id} value={s.station_id}>{s.name}</option>
              ))}
            </select>
          </label>
          <button onClick={submit} disabled={run.isPending} className="inline-flex items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-medium text-white disabled:opacity-60">
            <FlaskConical size={16} /> {run.isPending ? "Running…" : "Run scenario"}
          </button>
          {run.isError && <p className="text-xs text-crit">{(run.error as Error).message}</p>}
        </section>

        <section className="card p-4">
          {!run.data && <div className="grid h-full min-h-[300px] place-items-center text-sm text-muted">Set a scenario and press Run.</div>}
          {run.data && (
            <>
              <div className="mb-3 flex flex-wrap items-baseline gap-x-4 gap-y-1">
                <h2 className="text-sm font-semibold">{station === "*" ? "Network" : prettyStation(station)}: baseline vs scenario</h2>
                <span className="text-xs text-muted">origin {fmtTime(run.data.forecast_origin)}</span>
                <span className="ml-auto text-sm">
                  network peak change <strong className={run.data.delta_network_peak_kwh >= 0 ? "text-crit" : "text-ok"}>
                    {run.data.delta_network_peak_kwh >= 0 ? "+" : ""}{fmtNum(run.data.delta_network_peak_kwh)} kWh
                  </strong>
                </span>
              </div>
              {run.data.ood_flags.map((f) => (
                <div key={f.feature} className="mb-3 flex gap-2 rounded-lg border border-warn/40 bg-warn/10 px-3 py-2 text-xs">
                  <AlertTriangle size={14} className="mt-0.5 shrink-0 text-warn" />
                  <span>
                    <strong>{f.feature === "temperature_2m_c" ? "Temperature" : "Rain"} outside the training range</strong> ({fmtNum(f.min)} to {fmtNum(f.max)} vs
                    the range seen in training, {fmtNum(f.train_min)} to {fmtNum(f.train_max)}). {f.message}
                  </span>
                </div>
              ))}
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="t" stroke="var(--muted)" tick={{ fontSize: 11 }} tickFormatter={(v: string) => fmtHour(v)} minTickGap={30} />
                  <YAxis stroke="var(--muted)" tick={{ fontSize: 11 }} width={44} />
                  <Tooltip
                    contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }}
                    labelFormatter={(v) => fmtTime(String(v))}
                    formatter={(v, n) => (Array.isArray(v) ? [`${fmtNum(v[0])} – ${fmtNum(v[1])} kWh`, "scenario band (sum of sites)"] : [`${fmtNum(Number(v))} kWh`, n === "base" ? "baseline" : "scenario"])}
                  />
                  <Legend formatter={(v) => <span className="text-xs">{v === "base" ? "baseline (median)" : v === "scen" ? "scenario (median)" : "scenario band"}</span>} />
                  <Area dataKey="band" stroke="none" fill="var(--band)" isAnimationActive={false} />
                  <Line dataKey="base" stroke="var(--muted)" strokeDasharray="5 4" dot={false} strokeWidth={1.8} isAnimationActive={false} />
                  <Line dataKey="scen" stroke="var(--accent)" dot={false} strokeWidth={2.4} isAnimationActive={false} />
                </ComposedChart>
              </ResponsiveContainer>
              <div className="mt-3 grid gap-3 text-xs md:grid-cols-2">
                <div>
                  <div className="mb-1 font-semibold">Stations at risk in the scenario</div>
                  {run.data.alerts_in_scenario.length === 0 ? (
                    <span className="text-muted">None above 70% of assumed capacity.</span>
                  ) : (
                    <span>{worstPerStation(run.data.alerts_in_scenario)}</span>
                  )}
                </div>
                <div>
                  <div className="mb-1 font-semibold">Assumptions</div>
                  <span className="text-muted">{run.data.assumptions.length ? run.data.assumptions.join("; ") : "Model response only, no overlays."}</span>
                </div>
              </div>
              {station === "*" && <p className="mt-2 text-[11px] text-muted">The band adds site intervals, so it overstates network uncertainty; the KPI ribbon uses a network-calibrated interval.</p>}
            </>
          )}
        </section>
      </div>
    </div>
  );
}
