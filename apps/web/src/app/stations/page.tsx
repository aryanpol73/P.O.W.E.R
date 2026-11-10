"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";

import { api } from "@/lib/api";
import { fmtNum, RISK_COLOR } from "@/lib/format";

export default function StationsPage() {
  const { data, isError } = useQuery({ queryKey: ["stations"], queryFn: api.stations });
  return (
    <div className="flex flex-col gap-4">
      <div>
        <h1 className="text-lg font-semibold">Stations</h1>
        <p className="text-sm text-muted">Charging sites grouped from the City of Palo Alto ChargePoint data. Risk uses the next 12 hours.</p>
      </div>
      {isError && <p className="text-sm text-crit">Cannot reach the API.</p>}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {data?.map((s) => {
          const body = (
            <>
              <div className="flex items-center gap-2">
                <span className="size-2.5 rounded-full" style={{ background: s.is_modeled ? RISK_COLOR[s.risk ?? "NORMAL"] : "var(--context)" }} />
                <span className="font-medium">{s.name}</span>
                <span className="ml-auto text-xs text-muted">{s.is_modeled ? s.risk?.toLowerCase() : "not modelled"}</span>
              </div>
              <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-muted">
                <div>
                  <div className="tabular text-base text-text">{s.total_ports}</div>ports
                </div>
                <div>
                  <div className="tabular text-base text-text">{fmtNum(s.rated_kw_total)}</div>kW nameplate
                </div>
                <div>
                  <div className="tabular text-base text-text">{s.poi_category.replace(/_/g, " ")}</div>site type
                </div>
              </div>
              {!s.is_modeled && <p className="mt-2 text-xs text-muted">Opened after the modelling window, so it appears on the map as context only.</p>}
            </>
          );
          return s.is_modeled ? (
            <Link key={s.station_id} href={`/stations/${s.station_id}`} className="card p-4 transition-colors hover:bg-surface-2">
              {body}
            </Link>
          ) : (
            <div key={s.station_id} className="card p-4 opacity-80">
              {body}
            </div>
          );
        })}
      </div>
    </div>
  );
}
