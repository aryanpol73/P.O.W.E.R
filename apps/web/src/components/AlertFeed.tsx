"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { BellRing, Check } from "lucide-react";
import Link from "next/link";

import { api } from "@/lib/api";
import { fmtTime, prettyStation } from "@/lib/format";

export function AlertFeed() {
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({ queryKey: ["alerts"], queryFn: () => api.alerts("open") });
  const ack = useMutation({ mutationFn: api.ack, onSuccess: () => qc.invalidateQueries({ queryKey: ["alerts"] }) });
  return (
    <div className="card flex min-h-0 flex-col">
      <div className="flex items-center gap-2 border-b border-border px-4 py-3 text-sm font-semibold">
        <BellRing size={15} /> Open alerts
        <span className="ml-auto rounded-full bg-surface-2 px-2 text-xs font-normal text-muted">{data?.length ?? 0}</span>
      </div>
      <ul className="max-h-[320px] divide-y divide-border overflow-y-auto">
        {isLoading && <li className="px-4 py-3 text-sm text-muted">Loading…</li>}
        {data?.length === 0 && <li className="px-4 py-6 text-center text-sm text-muted">No open alerts. All stations are below 70% of assumed capacity.</li>}
        {data?.map((a) => (
          <li key={a.alert_id} className="flex gap-3 px-4 py-3">
            <span className={clsx("mt-1 size-2.5 shrink-0 rounded-full", a.level === "CRITICAL" ? "bg-crit" : "bg-warn")} />
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline justify-between gap-2 text-sm">
                <Link href={`/stations/${a.station_id}`} className="font-medium hover:underline">
                  {prettyStation(a.station_id)}
                </Link>
                <span className={clsx("text-xs font-semibold", a.level === "CRITICAL" ? "text-crit" : "text-warn")}>{a.level}</span>
              </div>
              <div className="text-xs text-muted">
                peak {fmtTime(a.target_time)} · q95 at {Math.round(a.load_ratio_q95 * 100)}% of capacity
              </div>
              {a.status === "open" ? (
                <button
                  onClick={() => ack.mutate(a.alert_id)}
                  className="mt-1.5 inline-flex items-center gap-1 rounded-md border border-border px-2 py-0.5 text-xs hover:bg-surface-2"
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
