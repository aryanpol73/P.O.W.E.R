"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { use, useState } from "react";

import { ForecastChart } from "@/components/ForecastChart";
import { ShapWaterfall } from "@/components/ShapWaterfall";
import { api } from "@/lib/api";
import { fmtNum, fmtPct, fmtTime, modelLabel, prettyStation, RISK_COLOR } from "@/lib/format";

export default function StationPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [model, setModel] = useState("champion");
  const [h, setH] = useState<number | undefined>(undefined);
  const health = useQuery({ queryKey: ["health"], queryFn: api.health });
  const fc = useQuery({ queryKey: ["forecast", id, model], queryFn: () => api.forecast(id, model) });
  const ex = useQuery({ queryKey: ["explain", id, h], queryFn: () => api.explain(id, h) });
  const stations = useQuery({ queryKey: ["stations"], queryFn: api.stations });
  const st = stations.data?.find((s) => s.station_id === id);
  const peak = fc.data?.points.reduce((a, b) => (b.q50 > a.q50 ? b : a), fc.data.points[0]);
  const worst = fc.data?.points.reduce((a, b) => (b.load_ratio_q95 > a.load_ratio_q95 ? b : a), fc.data.points[0]);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/" className="grid size-8 place-items-center rounded-lg border border-border bg-surface" aria-label="Back">
          <ArrowLeft size={16} />
        </Link>
        <div>
          <h1 className="text-lg font-semibold">{st?.name ?? prettyStation(id)}</h1>
          <p className="text-xs text-muted">
            {st ? `${st.total_ports} ports · ${fmtNum(st.rated_kw_total)} kW nameplate · assumed transformer ${fmtNum(st.transformer_kw)} kW` : id}
          </p>
        </div>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="ml-auto h-9 rounded-lg border border-border bg-surface px-2 text-sm"
          aria-label="Model"
        >
          <option value="champion">Champion (LightGBM q50)</option>
          {health.data?.models.map((m) => (
            <option key={m} value={m}>
              {modelLabel(m)}
            </option>
          ))}
        </select>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <div className="card px-4 py-3">
          <div className="text-[11px] uppercase tracking-wider text-muted">Next-24 h peak</div>
          <div className="tabular text-xl font-semibold">{fmtNum(peak?.q50)} kWh</div>
          <div className="text-xs text-muted">{peak ? fmtTime(peak.target_time) : "-"}</div>
        </div>
        <div className="card px-4 py-3">
          <div className="text-[11px] uppercase tracking-wider text-muted">Worst q95 vs capacity (24 h)</div>
          <div className="tabular text-xl font-semibold" style={{ color: RISK_COLOR[worst?.risk ?? "NORMAL"] }}>
            {fmtPct(worst?.load_ratio_q95)}
          </div>
          <div className="text-xs text-muted">{worst ? `${worst.risk.toLowerCase()} at +${worst.h} h` : "-"}</div>
        </div>
        <div className="card px-4 py-3">
          <div className="text-[11px] uppercase tracking-wider text-muted">Next hour</div>
          <div className="tabular text-xl font-semibold">{fmtNum(fc.data?.points[0]?.q50)} kWh</div>
          <div className="text-xs text-muted">
            90% interval {fmtNum(fc.data?.points[0]?.q05)} – {fmtNum(fc.data?.points[0]?.q95)}
          </div>
        </div>
        <div className="card px-4 py-3">
          <div className="text-[11px] uppercase tracking-wider text-muted">Data quality</div>
          <div className={clsx("text-xl font-semibold", fc.data?.is_degraded ? "text-warn" : "text-ok")}>
            {fc.data ? (fc.data.is_degraded ? "Degraded" : "Good") : "-"}
          </div>
          <div className="text-xs text-muted">{fc.data?.is_degraded ? "recent hours were imputed" : "all recent hours observed"}</div>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_420px]">
        <section className="card p-4">
          <div className="mb-2 flex flex-wrap items-baseline gap-2">
            <h2 className="text-sm font-semibold">Last 48 h and next 24 h</h2>
            <span className="text-xs text-muted">
              {fc.data ? `${modelLabel(fc.data.model)} · origin ${fmtTime(fc.data.forecast_origin)} · click a forecast hour to explain it` : ""}
            </span>
          </div>
          {fc.isError && <p className="text-sm text-crit">{(fc.error as Error).message}</p>}
          {fc.data && <ForecastChart data={fc.data} onPickHour={setH} />}
          <div className="mt-2 flex flex-wrap gap-4 text-[11px] text-muted">
            <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 bg-[var(--text)]" /> actual</span>
            <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 bg-accent" /> forecast (median)</span>
            <span className="flex items-center gap-1.5"><span className="h-3 w-4 rounded-sm" style={{ background: "var(--band)" }} /> 90% interval (calibrated)</span>
          </div>
        </section>

        <section className="card p-4">
          <div className="mb-3 flex items-baseline justify-between gap-2">
            <h2 className="text-sm font-semibold">Why this forecast?</h2>
            <select
              value={h ?? ""}
              onChange={(e) => setH(e.target.value ? Number(e.target.value) : undefined)}
              className="h-8 rounded-lg border border-border bg-surface px-2 text-xs"
              aria-label="Hour to explain"
            >
              <option value="">peak hour</option>
              {Array.from({ length: 24 }, (_, i) => i + 1).map((v) => (
                <option key={v} value={v}>
                  +{v} h
                </option>
              ))}
            </select>
          </div>
          {ex.data && (
            <>
              <p className="mb-3 text-xs text-muted">
                {fmtTime(ex.data.target_time)} (+{ex.data.h} h)
              </p>
              <ShapWaterfall e={ex.data} />
            </>
          )}
          {ex.isError && <p className="text-sm text-muted">{(ex.error as Error).message}</p>}
        </section>
      </div>
    </div>
  );
}
