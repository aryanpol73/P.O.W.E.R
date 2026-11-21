"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";

import { api, type Station } from "@/lib/api";
import { fmtNum, RISK_COLOR } from "@/lib/format";

const RISK_ORDER = { CRITICAL: 0, WARNING: 1, NORMAL: 2 } as const;

function accessLabel(poi: string): string {
  // OSM "access" tag values -> plain words ("yes" means open to everyone)
  const v = poi.toLowerCase();
  if (v === "yes" || v === "public") return "Public";
  if (v === "customers") return "Customers only";
  if (v === "private" || v === "no") return "Private";
  return poi.replace(/_/g, " ");
}

function PilotCard({ s }: { s: Station }) {
  const body = (
    <>
      <div className="flex items-center gap-2">
        <span className="size-2.5 shrink-0 rounded-full" style={{ background: s.kind === "pilot" ? RISK_COLOR[s.risk ?? "NORMAL"] : "var(--context)" }} />
        <span className="min-w-0 truncate font-medium">{s.name}</span>
        <span className="ml-auto shrink-0 text-xs text-muted">{s.kind === "pilot" ? s.risk?.toLowerCase() : "no forecast"}</span>
      </div>
      <div className="mt-2 grid grid-cols-3 gap-2 text-xs text-muted">
        <div>
          <div className="kpi text-base text-text">{s.total_ports}</div>ports
        </div>
        <div>
          <div className="kpi text-base text-text">{fmtNum(s.rated_kw_total)}</div>kW nameplate
        </div>
        <div>
          <div className="kpi text-base text-text">{fmtNum(s.transformer_kw)}</div>kW assumed capacity
        </div>
      </div>
      {s.kind === "pilot_no_history" && (
        <p className="mt-2 text-xs text-muted">Pilot site without usage history to replay yet, so it is shown on the map but not forecast.</p>
      )}
    </>
  );
  return s.kind === "pilot" ? (
    <Link href={`/stations/${s.station_id}`} className="card card-link p-4">
      {body}
    </Link>
  ) : (
    <div className="card p-4 opacity-80">{body}</div>
  );
}

function DiscoveredCard({ s }: { s: Station }) {
  return (
    <div className="card p-4">
      <div className="flex items-center gap-2">
        <span className="size-2.5 shrink-0 rounded-full" style={{ background: "var(--context)" }} />
        <span className="min-w-0 truncate font-medium">{s.name}</span>
        <span className="ml-auto shrink-0 text-xs text-muted">{accessLabel(s.poi_category)}</span>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 text-xs text-muted">
        <div>
          <div className="kpi text-base text-text">{s.total_ports > 0 ? s.total_ports : "–"}</div>
          {s.total_ports > 0 ? "ports (OpenStreetMap)" : "ports not recorded"}
        </div>
        <div>
          <div className="kpi text-base text-text">{s.estimate_next_hour_kwh != null ? `≈ ${fmtNum(s.estimate_next_hour_kwh)}` : "–"}</div>
          {s.estimate_next_hour_kwh != null ? "kWh next hour (estimate)" : "no estimate without a port count"}
        </div>
      </div>
    </div>
  );
}

export default function StationsPage() {
  const { data, isError } = useQuery({ queryKey: ["stations"], queryFn: api.stations });
  const pilot = (data ?? [])
    .filter((s) => s.kind !== "discovered")
    .sort((a, b) =>
      a.kind !== b.kind
        ? a.kind === "pilot" ? -1 : 1
        : (RISK_ORDER[a.risk ?? "NORMAL"] - RISK_ORDER[b.risk ?? "NORMAL"]) || a.name.localeCompare(b.name),
    );
  const discovered = (data ?? []).filter((s) => s.kind === "discovered").sort((a, b) => b.total_ports - a.total_ports || a.name.localeCompare(b.name));
  const dep = useQuery({ queryKey: ["deployment"], queryFn: api.deployment, staleTime: Infinity });
  const region = dep.data?.city ?? "Pilot";
  // cities in profile order (largest first would hide the smaller ones under the fold)
  const order = (dep.data?.cities ?? []).map((c) => c.name);
  const groups = (list: Station[]) => {
    const by = new Map<string, Station[]>();
    for (const s of list) {
      const k = s.city ?? region;
      by.set(k, [...(by.get(k) ?? []), s]);
    }
    return [...by.entries()].sort((a, b) => (order.indexOf(a[0]) + 1 || 99) - (order.indexOf(b[0]) + 1 || 99));
  };
  const multi = order.length > 1;

  return (
    <div className="flex flex-col gap-6">
      <section className="flex flex-col gap-3">
        <div>
          <h1 className="text-lg font-semibold">{region} pilot sites</h1>
          <p className="text-sm text-muted">
            The {pilot.filter((s) => s.kind === "pilot").length} forecast sites{multi ? ` in ${order.length} cities` : ""}, highest risk first. Risk
            compares the next 12 hours with the assumed transformer capacity. Open a site for its forecast and explanation.
          </p>
        </div>
        {isError && <p className="text-sm text-crit">Cannot reach the API.</p>}
        {groups(pilot).map(([c, list]) => (
          <div key={c} className="flex flex-col gap-2">
            {multi && (
              <h2 className="text-sm font-semibold text-muted">
                {c} <span className="font-normal">· {list.length} {list.length === 1 ? "site" : "sites"}</span>
              </h2>
            )}
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {list.map((s) => (
                <PilotCard key={s.station_id} s={s} />
              ))}
            </div>
          </div>
        ))}
      </section>

      {discovered.length > 0 && (
        <section className="flex flex-col gap-3">
          <div>
            <h2 className="text-base font-semibold">Other chargers nearby ({discovered.length})</h2>
            <p className="text-sm text-muted">
              Public chargers in {region} found on OpenStreetMap. Their operators publish no usage data, so they are not forecast and never raise alerts.
              The estimate is a rough guide only: the pilot sites&apos; typical next-hour energy per port, times this charger&apos;s port count.
            </p>
          </div>
          {groups(discovered).map(([c, list]) => (
            <div key={c} className="flex flex-col gap-2">
              {multi && <h3 className="text-sm font-semibold text-muted">{c} <span className="font-normal">· {list.length}</span></h3>}
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {list.map((s) => (
                  <DiscoveredCard key={s.station_id} s={s} />
                ))}
              </div>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}
