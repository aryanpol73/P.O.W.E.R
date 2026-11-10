"use client";

import { useQuery } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { api } from "@/lib/api";
import { fmtNum, modelLabel } from "@/lib/format";

/* eslint-disable @typescript-eslint/no-explicit-any */
const FEATURE_LABELS: Record<string, string> = {
  same_how_4w_mean: "4-week same hour-of-week avg",
  same_hour_7d_avg: "7-day same-hour avg",
  hour_cos: "hour of day (cos)",
  hour_sin: "hour of day (sin)",
  dow_sin: "day of week (sin)",
  dow_cos: "day of week (cos)",
  level_ratio: "today vs month level",
  station_id: "station",
  doy_cos: "season (cos)",
  rolling_mean_24h: "24 h average load",
  lag_24h: "same hour yesterday",
  rolling_std_24h: "24 h volatility",
  active_sessions_T: "cars charging now",
  rolling_max_24h: "24 h peak",
  level_28d: "28-day level",
};

function Figure({ name, caption }: { name: string; caption: string }) {
  return (
    <figure className="card overflow-hidden">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={api.figure(name)} alt={caption} loading="lazy" className="w-full bg-white" />
      <figcaption className="px-4 py-2 text-xs text-muted">{caption}</figcaption>
    </figure>
  );
}

export default function InsightsPage() {
  const shap = useQuery({ queryKey: ["insights", "shap"], queryFn: api.shap, staleTime: Infinity });
  const errs = useQuery({ queryKey: ["insights", "errors"], queryFn: api.errors, staleTime: Infinity });
  const stress = useQuery({ queryKey: ["insights", "stress"], queryFn: api.stress, staleTime: Infinity });
  const top = Object.entries((shap.data?.top_features ?? {}) as Record<string, number>).map(([f, v]) => ({ f: FEATURE_LABELS[f] ?? f, v }));
  const groups = Object.entries((shap.data?.group_importance ?? {}) as Record<string, number>).map(([g, v]) => ({ g, v }));
  const hl = (errs.data?.highlights ?? {}) as Record<string, Record<string, number>>;
  const st = stress.data as any;
  const alarms = (st?.cusum?.alarms ?? []) as { date: string; split: string; direction: string }[];
  const testAlarms = alarms.filter((a) => a.split === "test");

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h1 className="text-lg font-semibold">Insights</h1>
        <p className="text-sm text-muted">What drives the forecasts, where they fail, and how the system behaved when COVID-19 hit. Test period unless stated.</p>
      </div>
      {(shap.isError || errs.isError) && <p className="card p-4 text-sm text-muted">Run `uv run power-ml analyze` to produce these results.</p>}

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="card p-4">
          <h2 className="text-sm font-semibold">Top drivers (mean |SHAP|, kWh)</h2>
          <p className="mb-2 text-xs text-muted">Recurring weekly and daily patterns dominate; weather barely matters in mild Palo Alto.</p>
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={top} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" stroke="var(--muted)" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="f" width={170} stroke="var(--muted)" tick={{ fontSize: 11 }} interval={0} />
              <Tooltip formatter={(v) => [`${fmtNum(Number(v), 3)} kWh`, "mean |SHAP|"]} contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }} />
              <Bar dataKey="v" fill="var(--accent)" radius={[0, 4, 4, 0]} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </section>

        <section className="card flex flex-col gap-4 p-4">
          <div>
            <h2 className="text-sm font-semibold">By feature group</h2>
            <ul className="mt-2 flex flex-col gap-1.5">
              {groups.map(({ g, v }) => (
                <li key={g} className="grid grid-cols-[110px_1fr_60px] items-center gap-2 text-xs">
                  <span className="capitalize">{g}</span>
                  <div className="h-2 rounded bg-surface-2">
                    <div className="h-2 rounded bg-accent" style={{ width: `${(v / (groups[0]?.v || 1)) * 100}%` }} />
                  </div>
                  <span className="tabular text-right">{fmtNum(v, 2)}</span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h2 className="text-sm font-semibold">Where errors are largest (WAPE, champion)</h2>
            <dl className="mt-2 grid grid-cols-2 gap-2 text-xs">
              {hl.is_holiday && (<><dt className="text-muted">Holidays vs normal days</dt><dd className="tabular">{fmtNum(hl.is_holiday.true, 2)} vs {fmtNum(hl.is_holiday.false, 2)}</dd></>)}
              {hl.temp_tercile && (<><dt className="text-muted">Cool vs warm hours</dt><dd className="tabular">{fmtNum(hl.temp_tercile.cool, 2)} vs {fmtNum(hl.temp_tercile.warm, 2)}</dd></>)}
              {hl.rain && (<><dt className="text-muted">Rain vs dry</dt><dd className="tabular">{fmtNum(hl.rain.rain, 2)} vs {fmtNum(hl.rain.dry, 2)}</dd></>)}
              {hl.saturated && (<><dt className="text-muted">Full vs not-full stations</dt><dd className="tabular">{fmtNum(hl.saturated.saturated, 2)} vs {fmtNum(hl.saturated["not saturated"], 2)}</dd></>)}
              {hl.h_bucket && (<><dt className="text-muted">1–3 h vs 13–24 h ahead</dt><dd className="tabular">{fmtNum(hl.h_bucket["1-3"], 2)} vs {fmtNum(hl.h_bucket["13-24"], 2)}</dd></>)}
            </dl>
          </div>
          <div>
            <h2 className="text-sm font-semibold">Example explanations</h2>
            <ul className="mt-2 flex flex-col gap-2 text-xs leading-relaxed text-muted">
              {((shap.data?.local_examples ?? []) as any[]).map((ex, i) => (<li key={i}>{ex.narrative}</li>))}
            </ul>
          </div>
        </section>
      </div>

      {st && (
        <section className="card p-4">
          <h2 className="text-sm font-semibold">COVID-19 stress test (frozen models, Mar–Jun 2020)</h2>
          <div className="mt-3 grid gap-4 md:grid-cols-3">
            <div className="text-sm">
              <div className="text-xs text-muted">First drift alarm in the COVID period</div>
              <div className="text-lg font-semibold">{st.cusum?.first_alarm_in_covid_period ?? "-"}</div>
              <div className="text-xs text-muted">CUSUM on daily forecast bias · {testAlarms.length} alarms in the 172 normal test days</div>
            </div>
            <div className="text-sm">
              <div className="text-xs text-muted">Demand-distribution alarm (seasonal PSI)</div>
              <div className="text-lg font-semibold">week of {st.seasonal_psi?.first_alarm_week ?? "-"}</div>
              <div className="text-xs text-muted">Santa Clara County shelter-in-place: 17 Mar 2020</div>
            </div>
            <div className="text-xs">
              <div className="mb-1 text-muted">WAPE, normal test → COVID</div>
              {["B3_climatology", "M4_lightgbm", "M4_lightgbm_quantile", "M5_lstm"].map((m) =>
                st.models?.[m] ? (
                  <div key={m} className="flex justify-between">
                    <span>{modelLabel(m)}</span>
                    <span className="tabular">{fmtNum(st.models[m].wape_test, 2)} → {fmtNum(st.models[m].wape_ood, 2)}</span>
                  </div>
                ) : null,
              )}
            </div>
          </div>
        </section>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <Figure name="covid_stress_test.png" caption="Daily network energy vs forecast with drift alarms (orange = over-forecast, green = under-forecast)." />
        <Figure name="forecast_example.png" caption="Two weeks of 1-hour-ahead forecasts with the calibrated 90% interval." />
        <Figure name="error_heatmap_weekday_hour.png" caption="WAPE by weekday and local hour of the target (test period, champion model)." />
        <Figure name="shap_beeswarm.png" caption="SHAP beeswarm: each dot is one forecast; colour is the feature value." />
      </div>
    </div>
  );
}
