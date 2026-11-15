"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { BellRing, Check } from "lucide-react";
import Link from "next/link";

import { api } from "@/lib/api";
import { fmtTime, prettyStation } from "@/lib/format";

import { PushToggle } from "./PushToggle";

export function AlertFeed() {
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({ queryKey: ["alerts"], queryFn: () => api.alerts("open") });
  useQuery({ queryKey: ["stations"], queryFn: api.stations }); // re-render once display names are registered
  const ack = useMutation({ mutationFn: api.ack, onSuccess: () => qc.invalidateQueries({ queryKey: ["alerts"] }) });
  return (
    <div className="card flex min-h-0 flex-col">
      <div className="flex items-center gap-2 border-b border-border px-4 py-3 text-sm font-semibold">
        <BellRing size={15} className="shrink-0" /> <span className="whitespace-nowrap">Open alerts</span>
        <span className="tabular rounded-full border border-border bg-surface-2 px-2 text-xs font-medium text-muted">{data?.length ?? 0}</span>
        <PushToggle className="ml-auto font-normal" />
      </div>
      <ul className="max-h-[320px] divide-y divide-divider overflow-y-auto">
        {isLoading && <li className="px-4 py-3 text-sm text-muted">Loading…</li>}
        {data?.length === 0 && <li className="px-4 py-6 text-center text-sm text-muted">No open alerts. All stations are below 70% of assumed capacity.</li>}
        {data?.map((a) => (
          <li key={a.alert_id} className="flex gap-3 px-4 py-3">
            <span
              className={clsx(
                "mt-1.5 size-2.5 shrink-0",
                a.level === "CRITICAL" ? "pulse-crit rounded-full bg-crit" : "rotate-45 rounded-[2px] bg-warn",
              )}
            />
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-2 text-sm">
                <Link href={`/stations/${a.station_id}`} className="font-medium underline-offset-2 hover:text-accent hover:underline">
                  {prettyStation(a.station_id)}
                </Link>
                <span
                  className={clsx(
                    "rounded-md border px-1.5 py-px text-[10px] font-semibold tracking-[0.06em]",
                    a.level === "CRITICAL" ? "border-crit/40 bg-crit/10 text-crit" : "border-warn/40 bg-warn/10 text-warn",
                  )}
                >
                  {a.level}
                </span>
              </div>
              <div className="tabular text-xs text-muted">
                peak {fmtTime(a.target_time)} · q95 at {Math.round(a.load_ratio_q95 * 100)}% of capacity
              </div>
              {a.status === "open" ? (
                <button
                  onClick={() => ack.mutate(a.alert_id)}
                  className="btn btn-ghost btn-sm mt-1 -ml-2"
                >
                  <Check size={12} /> Acknowledge
                </button>
              ) : (
                <span className="mt-1.5 inline-block text-xs text-muted">acknowledged</span>
              )}
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
