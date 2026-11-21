"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Check, Sparkles, TriangleAlert } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { api, type Alert } from "@/lib/api";
import { fmtHour, prettyStation } from "@/lib/format";

import { PushToggle } from "./PushToggle";

const LEVEL = {
  CRITICAL: { text: "text-crit", chip: "bg-crit/15 text-crit", edge: "shadow-[inset_0_0_0_1px_rgba(251,113,133,0.3)]" },
  WARNING: { text: "text-warn", chip: "bg-warn/15 text-warn", edge: "shadow-[inset_0_0_0_1px_rgba(251,146,60,0.28)]" },
  NORMAL: { text: "text-ok", chip: "bg-ok/15 text-ok", edge: "shadow-[inset_0_0_0_1px_rgba(45,212,191,0.24)]" },
} as const;

// Open alerts as stacked cards. Acknowledging slides the card out and closes the gap before the list refetches.
export function AlertFeed({ className, onSelect }: { className?: string; onSelect?: (stationId: string) => void }) {
  const qc = useQueryClient();
  const { data, isLoading, isError } = useQuery({ queryKey: ["alerts"], queryFn: () => api.alerts("open") });
  useQuery({ queryKey: ["stations"], queryFn: api.stations }); // re-render once display names are registered
  const [leaving, setLeaving] = useState<Set<number>>(new Set());
  const ack = useMutation({
    mutationFn: api.ack,
    onSettled: () => setTimeout(() => qc.invalidateQueries({ queryKey: ["alerts"] }), 480),
    onError: (_e, id) => setLeaving((s) => new Set([...s].filter((x) => x !== id))),
  });
  const open = (data ?? []).filter((a) => a.status === "open");
  const visible = open.filter((a) => !leaving.has(a.alert_id)).length;

  return (
    <section aria-labelledby="alerts-title" className={clsx("flex min-h-0 flex-col", className)}>
      <div className="flex items-center gap-2 px-1 pb-2.5">
        <h2 id="alerts-title" className="flex items-center gap-2 text-[13px] font-extrabold">
          Open alerts
          <span className="tabular rounded-full bg-crit/20 px-2 py-px font-mono text-[11px] font-semibold text-[#fecdd3]">{visible}</span>
        </h2>
        <PushToggle compact className="ml-auto" />
      </div>
      <div className="-mx-1 min-h-0 flex-1 overflow-y-auto px-1 pb-1">
        {isLoading && <div className="hud p-4 text-sm text-muted">Loading…</div>}
        {isError && <div className="hud p-4 text-sm text-crit">Alerts unavailable.</div>}
        {data && visible === 0 && (
          <div className="rounded-2xl bg-ok/[0.08] p-4 text-[13px] text-[#99f6e4] shadow-[inset_0_0_0_1px_rgba(45,212,191,0.25)]">
            No open alerts. Every station is below 70% of its assumed rating.
          </div>
        )}
        {open.map((a) => (
          <AlertCard
            key={a.alert_id}
            a={a}
            leaving={leaving.has(a.alert_id)}
            onSelect={onSelect}
            onAck={() => {
              setLeaving((s) => new Set(s).add(a.alert_id));
              ack.mutate(a.alert_id);
            }}
          />
        ))}
      </div>
    </section>
  );
}

function AlertCard({ a, leaving, onAck, onSelect }: { a: Alert; leaving: boolean; onAck: () => void; onSelect?: (id: string) => void }) {
  const L = LEVEL[a.level];
  return (
    <div className="slide-collapse" data-open={!leaving}>
      <div>
        <article className={clsx("hud mb-2.5 p-3.5", L.edge)}>
          <div className="flex items-center gap-2.5">
            <span className={clsx("grid size-7 place-items-center rounded-[9px]", L.chip, a.level === "CRITICAL" && "pulse-crit")} aria-hidden="true">
              <TriangleAlert size={14} strokeWidth={2.4} className="relative z-10" />
            </span>
            <span className={clsx("text-[10px] font-extrabold tracking-[0.12em]", L.text)}>{a.level}</span>
            <span className="tabular ml-auto text-xs text-muted">peaks {fmtHour(a.target_time)}</span>
          </div>
          <p className="mt-2 text-[13px] font-bold">
            {onSelect ? (
              <button onClick={() => onSelect(a.station_id)} className="text-left hover:text-accent">
                {prettyStation(a.station_id)}
              </button>
            ) : (
              prettyStation(a.station_id)
            )}{" "}
            <span className="font-semibold text-muted">may reach</span> <span className={clsx("tabular font-mono", L.text)}>{Math.round(a.load_ratio_q95 * 100)}%</span>{" "}
            <span className="font-semibold text-muted">of rating</span>
          </p>
          <div className="mt-3 flex gap-2">
            <Link href={`/stations/${a.station_id}`} className="btn btn-primary btn-sm">
              <Sparkles size={13} /> Explain
            </Link>
            <button onClick={onAck} disabled={leaving} className="btn btn-secondary btn-sm">
              <Check size={13} strokeWidth={2.6} /> Acknowledge
            </button>
          </div>
        </article>
      </div>
    </div>
  );
}
