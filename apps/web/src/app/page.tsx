"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { ArrowRight } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { AlertFeed } from "@/components/AlertFeed";
import { KpiRibbon } from "@/components/KpiRibbon";
import { StationMap } from "@/components/StationMap";
import { api } from "@/lib/api";
import { fmtNum, fmtPct, fmtTime, RISK_COLOR } from "@/lib/format";

export default function CommandCenter() {
  const router = useRouter();
  const [h, setH] = useState(1);
  const snap = useQuery({ queryKey: ["snapshot", h], queryFn: () => api.snapshot(h), placeholderData: keepPreviousData });
  const modeled = snap.data?.stations.filter((s) => s.is_modeled).sort((a, b) => (b.load_ratio_q95 ?? 0) - (a.load_ratio_q95 ?? 0)) ?? [];

  return (
    <div className="flex flex-col gap-4">
      <KpiRibbon />

      <div className="grid gap-4 xl:grid-cols-[1fr_360px]">
        <section className="card flex flex-col p-3">
          <div className="flex flex-wrap items-center gap-3 px-1 pb-3">
            <div>
              <h1 className="text-base font-semibold">City map · forecast in {h} h</h1>
              <p className="text-xs text-muted">{snap.data ? fmtTime(snap.data.target_time) : "…"} · colour = risk at q95, size = median load</p>
            </div>
            <label className="ml-auto flex w-full items-center gap-3 text-xs text-muted sm:w-auto">
              +1 h
              <input
                type="range"
                min={1}
                max={24}
                value={h}
                onChange={(e) => setH(Number(e.target.value))}
                className="w-full accent-[var(--accent)] sm:w-56"
                aria-label="Forecast horizon in hours"
              />
              +24 h
            </label>
          </div>
          <div className="h-[420px] md:h-[520px]">
            <StationMap snapshot={snap.data} onSelect={(id) => router.push(`/stations/${id}`)} />
          </div>
          {snap.data && snap.data.reroutes.length > 0 && (
            <div className="mt-3 rounded-lg bg-accent-soft px-3 py-2 text-xs">
              <strong>Reroute suggestions</strong> (for the network&apos;s worst hour):{" "}
              {snap.data.reroutes.map((r, i) => (
                <span key={i}>
                  {i > 0 && " · "}
                  {r.from.replace("PA_", "")} → {r.to.replace("PA_", "")} {fmtNum(r.shifted_kwh)} kWh ({fmtNum(r.distance_km, 2)} km)
                </span>
              ))}
            </div>
          )}
        </section>

        <div className="flex flex-col gap-4">
          <AlertFeed />
          <div className="card">
            <div className="border-b border-border px-4 py-3 text-sm font-semibold">Stations at +{h} h</div>
            <ul className="divide-y divide-border">
              {modeled.map((s) => (
                <li key={s.station_id}>
                  <Link href={`/stations/${s.station_id}`} className="flex items-center gap-3 px-4 py-2.5 text-sm hover:bg-surface-2">
                    <span className="size-2.5 rounded-full" style={{ background: RISK_COLOR[s.risk ?? "NORMAL"] }} />
                    <span className="flex-1 truncate">{s.name}</span>
                    <span className="tabular text-xs text-muted">{fmtNum(s.q50)} kWh</span>
                    <span className={clsx("tabular w-12 text-right text-xs", (s.load_ratio_q95 ?? 0) >= 0.85 ? "text-crit" : "text-muted")}>
                      {fmtPct(s.load_ratio_q95)}
                    </span>
                    <ArrowRight size={14} className="text-muted" />
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
