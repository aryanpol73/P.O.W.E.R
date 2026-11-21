"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { Pause, Play, RotateCcw, SkipForward } from "lucide-react";
import { useEffect, useState } from "react";

import { api, type SimState } from "@/lib/api";
import { fmtTime, TZ_LABEL } from "@/lib/format";
import { useLive } from "@/lib/live";

const SPEEDS = [6, 12, 30, 60];
const HOUR = 3_600_000;

/** Wall-clock time, ticking every second (live mode shows real time, not the replay clock). */
function useNow(enabled: boolean): Date {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    if (!enabled) return;
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, [enabled]);
  return now;
}

const clock = (d: Date) => fmtTime(d.toISOString(), { second: "2-digit" });
const hhmm = (iso: string) => fmtTime(iso, { weekday: undefined, day: undefined, month: undefined });

export function SimControls() {
  const qc = useQueryClient();
  const { status } = useLive();
  const sim = useQuery({ queryKey: ["sim"], queryFn: api.sim, refetchInterval: status === "live" ? false : 10_000 });
  const [error, setError] = useState<string | null>(null);
  const control = useMutation({
    mutationFn: api.simControl,
    onSuccess: () => {
      setError(null);
      qc.invalidateQueries();
    },
    onError: (e: Error) => setError(e.message),
  });
  const s = sim.data;
  const busy = control.isPending;
  const live = s?.mode === "LIVE_SHADOW";
  const now = useNow(live);

  const dot = status === "live" ? "bg-ok text-ok breathe" : status === "connecting" ? "bg-warn text-warn" : "bg-crit text-crit";
  const statusText = status === "live" ? "Connected" : status === "connecting" ? "Connecting" : "Offline";
  // hour T closes at T + 1 h; the forecast shown was made then
  const updated = s ? new Date(new Date(s.sim_now).getTime() + HOUR).toISOString() : null;

  return (
    <div className="flex items-center gap-2">
      <span className={clsx("size-2 shrink-0 rounded-full", dot)} title={statusText} aria-hidden="true" />
      <div className="mr-1 min-w-0 leading-tight">
        <div className="tabular truncate font-mono text-[13px] font-medium">
          {!s ? (sim.isError ? "API unreachable" : "…") : live ? clock(now) : fmtTime(s.sim_now)}
        </div>
        <div className="hidden truncate text-[11px] text-muted sm:block">
          <span className="sr-only">Connection {statusText}. </span>
          {!s ? "" : live ? (
            <span title="Real clock and live weather. Each site's demand stream replays its paired training site's held-out data on today's weekday and hour.">
              Live · {TZ_LABEL} · updated {hhmm(updated!)}
              {s.next_update ? ` · next ${hhmm(s.next_update)}` : ""}
            </span>
          ) : (
            <>
              Replay · {TZ_LABEL}
              {s.paused ? " · paused" : ` · ${s.speed_ticks_per_min} h/min`}
            </>
          )}
        </div>
      </div>
      {error && <span className="hidden max-w-40 truncate text-xs text-crit lg:inline" title={error}>{error}</span>}
      <ModeSwitch s={s} busy={busy} onMode={(mode) => control.mutate({ action: "reset", mode })} />
      {s && !live && (
        <>
          <select
            aria-label="Simulated hours per minute"
            value={s.speed_ticks_per_min}
            disabled={busy}
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
            disabled={busy}
            onClick={() => control.mutate({ action: "reset", mode: "REPLAY" })}
            className="btn btn-secondary btn-icon hidden sm:inline-flex"
          >
            <RotateCcw size={15} />
          </button>
          <button
            aria-label="Advance one hour"
            title="Advance one hour"
            disabled={busy || !s.paused}
            onClick={() => control.mutate({ action: "step" })}
            className="btn btn-secondary btn-icon hidden sm:inline-flex"
          >
            <SkipForward size={15} />
          </button>
          <button
            aria-label={s.paused ? "Play" : "Pause"}
            aria-busy={busy}
            disabled={busy}
            onClick={() => control.mutate({ action: s.paused ? "play" : "pause" })}
            className="btn btn-primary btn-icon"
          >
            {s.paused ? <Play size={15} fill="currentColor" /> : <Pause size={15} fill="currentColor" />}
          </button>
        </>
      )}
    </div>
  );
}

function ModeSwitch({ s, busy, onMode }: { s?: SimState; busy: boolean; onMode: (m: SimState["mode"]) => void }) {
  const opts: [SimState["mode"], string, string][] = [
    ["LIVE_SHADOW", "Live", "Run on the real clock with live weather"],
    ["REPLAY", "Replay", "Walk the held-out test period at a chosen speed"],
  ];
  return (
    <div role="radiogroup" aria-label="Clock mode" className="hidden rounded-xl bg-black/25 p-0.5 shadow-[inset_0_0_0_1px_rgba(165,180,252,0.1)] md:flex">
      {opts.map(([m, label, title]) => {
        const on = s?.mode === m;
        return (
          <button
            key={m}
            role="radio"
            aria-checked={on}
            title={title}
            disabled={!s || busy || on}
            onClick={() => onMode(m)}
            className={clsx(
              "h-8 rounded-[10px] px-3 text-[12px] font-semibold transition-colors",
              on ? "bg-accent text-white" : "text-muted hover:text-text",
            )}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
