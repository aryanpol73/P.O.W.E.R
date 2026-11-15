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
        <Link href="/" className="btn btn-secondary btn-icon" aria-label="Back">
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
          className="field ml-auto"
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
          <div className="eyebrow">Next-24 h peak</div>
          <div className="kpi mt-1.5 text-xl">{fmtNum(peak?.q50)} kWh</div>
          <div className="tabular mt-0.5 text-xs text-muted">{peak ? fmtTime(peak.target_time) : "-"}</div>
        </div>
        <div className="card px-4 py-3">
          <div className="eyebrow">Worst q95 vs capacity (24 h)</div>
          <div className="kpi mt-1.5 text-xl" style={{ color: RISK_COLOR[worst?.risk ?? "NORMAL"] }}>
            {fmtPct(worst?.load_ratio_q95)}
          </div>
          <div className="tabular mt-0.5 text-xs text-muted">{worst ? `${worst.risk.toLowerCase()} at +${worst.h} h` : "-"}</div>
        </div>
        <div className="card px-4 py-3">
          <div className="eyebrow">Next hour</div>
          <div className="kpi mt-1.5 text-xl">{fmtNum(fc.data?.points[0]?.q50)} kWh</div>
          <div className="tabular mt-0.5 text-xs text-muted">
            90% interval {fmtNum(fc.data?.points[0]?.q05)} – {fmtNum(fc.data?.points[0]?.q95)}
          </div>
        </div>
        <div className="card px-4 py-3">
          <div className="eyebrow">Data quality</div>
          <div className={clsx("kpi mt-1.5 text-xl", fc.data?.is_degraded ? "text-warn" : "text-ok")}>
            {fc.data ? (fc.data.is_degraded ? "Degraded" : "Good") : "-"}
          </div>
          <div className="mt-0.5 text-xs text-muted">{fc.data?.is_degraded ? "recent hours were imputed" : "all recent hours observed"}</div>
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
          <div className="mt-2 flex flex-wrap gap-4 text-[11px] font-medium text-muted">
            <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 rounded-full bg-text/85" /> actual</span>
            <span className="flex items-center gap-1.5"><span className="h-[3px] w-4 rounded-full bg-accent" /> forecast (median)</span>
            <span className="flex items-center gap-1.5"><span className="h-3 w-4 rounded-sm border border-accent/30" style={{ background: "var(--band)" }} /> 90% interval (calibrated)</span>
          </div>
        </section>

        <section className="card p-4">
          <div className="mb-3 flex items-baseline justify-between gap-2">
            <h2 className="text-sm font-semibold">Why this forecast?</h2>
            <select
              value={h ?? ""}
              onChange={(e) => setH(e.target.value ? Number(e.target.value) : undefined)}
              className="field field-sm tabular"
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
              <p className="tabular mb-3 text-xs text-muted">
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
