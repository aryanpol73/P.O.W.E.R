"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import { useState } from "react";

import { api } from "@/lib/api";
import { fmtTime, TZ_LABEL } from "@/lib/format";
import { useLive } from "@/lib/live";

const SPEEDS = [6, 12, 30, 60];

export function SimControls() {
  const qc = useQueryClient();
  const { status } = useLive();
  const sim = useQuery({ queryKey: ["sim"], queryFn: api.sim, refetchInterval: status === "live" ? false : 10_000 });
  const [error, setError] = useState<string | null>(null);
  const control = useMutation({
    mutationFn: api.simControl,
    onSuccess: () => {
      setError(null);
      qc.invalidateQueries({ queryKey: ["sim"] });
    },
    onError: (e: Error) => setError(e.message),
  });
  const s = sim.data;
  const busy = control.isPending;

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 md:px-6">
      <div className="min-w-0">
        <div className="text-[11px] uppercase tracking-wider text-muted">
          {s?.mode === "LIVE_SHADOW" ? "Live-shadow" : "Replay"} clock · forecast origin ({TZ_LABEL})
        </div>
        <div className="tabular truncate text-base font-semibold">{s ? fmtTime(s.sim_now) : sim.isError ? "API unreachable" : "…"}</div>
      </div>

      <div className="flex items-center gap-1.5">
        <button
          aria-label={s?.paused ? "Play" : "Pause"}
          disabled={!s || busy}
          onClick={() => control.mutate({ action: s?.paused ? "play" : "pause" })}
          className="grid size-9 place-items-center rounded-lg bg-accent text-white disabled:opacity-50"
        >
          {s?.paused ? <Play size={16} /> : <Pause size={16} />}
        </button>
        <button
          aria-label="Advance one hour"
          title="Advance one hour"
          disabled={!s || busy || !s.paused}
          onClick={() => control.mutate({ action: "step" })}
          className="grid size-9 place-items-center rounded-lg border border-border bg-surface disabled:opacity-40"
        >
          <SkipForward size={16} />
        </button>
        <button
          aria-label="Reset to the start of the test period"
          title="Reset to the start of the test period"
          disabled={!s || busy}
          onClick={() => control.mutate({ action: "reset" })}
          className="grid size-9 place-items-center rounded-lg border border-border bg-surface disabled:opacity-40"
        >
          <RotateCcw size={16} />
        </button>
        <select
          aria-label="Simulated hours per minute"
          value={s?.speed_ticks_per_min ?? 6}
          disabled={!s || busy}
          onChange={(e) => control.mutate({ action: "set_speed", speed_ticks_per_min: Number(e.target.value) })}
          className="h-9 rounded-lg border border-border bg-surface px-2 text-sm"
        >
          {SPEEDS.map((v) => (
            <option key={v} value={v}>
              {v} h/min
            </option>
          ))}
        </select>
      </div>

      <div className="ml-auto flex items-center gap-3 text-xs">
        {error && <span className="text-crit">{error}</span>}
        <span className="flex items-center gap-1.5 text-muted">
          <span
            className={clsx("size-2 rounded-full", status === "live" ? "bg-ok" : status === "connecting" ? "bg-warn" : "bg-crit")}
          />
          {status === "live" ? "Live" : status === "connecting" ? "Connecting" : "Offline"}
        </span>
      </div>
    </div>
  );
}
