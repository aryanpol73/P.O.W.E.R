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

  const dot = status === "live" ? "bg-ok text-ok breathe" : status === "connecting" ? "bg-warn text-warn" : "bg-crit text-crit";
  const statusText = status === "live" ? "Live" : status === "connecting" ? "Connecting" : "Offline";

  return (
    <div className="flex items-center gap-2">
      <span className={clsx("size-2 shrink-0 rounded-full", dot)} title={statusText} aria-hidden="true" />
      <div className="mr-1 min-w-0 leading-tight">
        <div className="tabular truncate font-mono text-[13px] font-medium">{s ? fmtTime(s.sim_now) : sim.isError ? "API unreachable" : "…"}</div>
        <div className="hidden truncate text-[11px] text-muted sm:block">
          <span className="sr-only">Connection {statusText}. </span>
          {s?.mode === "LIVE_SHADOW" ? "Live-shadow" : "Replay"} · {TZ_LABEL}
          {s ? (s.paused ? " · paused" : ` · ${s.speed_ticks_per_min} h/min`) : ""}
        </div>
      </div>
      {error && <span className="hidden max-w-40 truncate text-xs text-crit lg:inline" title={error}>{error}</span>}
      <select
        aria-label="Simulated hours per minute"
        value={s?.speed_ticks_per_min ?? 6}
        disabled={!s || busy}
        onChange={(e) => control.mutate({ action: "set_speed", speed_ticks_per_min: Number(e.target.value) })}
        className="field tabular hidden lg:block"
      >
        {SPEEDS.map((v) => (
          <option key={v} value={v}>
            {v} h/min
          </option>
        ))}
      </select>
      <button
        aria-label="Reset to the start of the test period"
        title="Reset to the start of the test period"
        disabled={!s || busy}
        onClick={() => control.mutate({ action: "reset" })}
        className="btn btn-secondary btn-icon hidden sm:inline-flex"
      >
        <RotateCcw size={15} />
      </button>
      <button
        aria-label="Advance one hour"
        title="Advance one hour"
        disabled={!s || busy || !s.paused}
        onClick={() => control.mutate({ action: "step" })}
        className="btn btn-secondary btn-icon hidden sm:inline-flex"
      >
        <SkipForward size={15} />
      </button>
      <button
        aria-label={s?.paused ? "Play" : "Pause"}
        aria-busy={busy}
        disabled={!s || busy}
        onClick={() => control.mutate({ action: s?.paused ? "play" : "pause" })}
        className="btn btn-primary btn-icon"
      >
        {s?.paused ? <Play size={15} fill="currentColor" /> : <Pause size={15} fill="currentColor" />}
      </button>
    </div>
  );
}
