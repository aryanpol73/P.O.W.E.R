"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { CheckCircle2, CircleAlert, Loader2, MapPinned, RefreshCw, RotateCcw, Search, XCircle } from "lucide-react";
import { useState } from "react";

import { api, type DriftItem, type RetrainRun } from "@/lib/api";
import { fmtNum, fmtTime } from "@/lib/format";

const SIGNAL_LABEL: Record<string, string> = {
  cusum_bias: "Forecast bias (CUSUM)",
  seasonal_psi: "Demand shift (seasonal PSI)",
  coverage: "90% interval coverage",
  wape_ratio: "Live WAPE / test WAPE",
};

function Pill({ ok, label }: { ok: boolean | null; label: string }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium",
        ok === null ? "bg-surface-2 text-muted" : ok ? "bg-ok/15 text-ok" : "bg-crit/15 text-crit",
      )}
    >
      {ok === null ? null : ok ? <CheckCircle2 size={12} /> : <XCircle size={12} />}
      {label}
    </span>
  );
}

function Signal({ d }: { d: DriftItem }) {
  const pct = d.kind === "coverage" ? `${fmtNum(d.statistic * 100, 1)}%` : fmtNum(d.statistic, 3);
  const rule = d.kind === "coverage" ? `alarm below ${fmtNum(d.threshold * 100, 0)}%` : `alarm above ${fmtNum(d.threshold, 2)}`;
  return (
    <li className="flex items-center justify-between gap-3 rounded-lg border border-border px-3 py-2 text-sm">
      <div className="min-w-0">
        <div className="font-medium">{SIGNAL_LABEL[d.kind] ?? d.kind}</div>
        <div className="text-xs text-muted">{rule}</div>
      </div>
      <div className="flex items-center gap-2">
        <span className="tabular">{pct}</span>
        <Pill ok={!d.is_drift} label={d.is_drift ? "drift" : "ok"} />
      </div>
    </li>
  );
}

function RunCard({ r, active, onActivate, busy }: { r: RetrainRun; active: boolean; onActivate: () => void; busy: boolean }) {
  return (
    <div className={clsx("rounded-lg border px-3 py-3 text-sm", active ? "border-accent" : "border-border")}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="font-medium">
          {r.run_id} {active && <span className="ml-1 text-xs text-accent">(live champion)</span>}
        </div>
        <Pill ok={r.passed} label={r.passed ? "gate passed" : "gate failed"} />
      </div>
      <div className="mt-1 text-xs text-muted">
        data up to {fmtTime(r.as_of)} · {r.train_rows.toLocaleString()} training rows · {fmtNum(r.seconds, 0)} s
      </div>
      <table className="mt-2 w-full text-xs">
        <thead className="text-muted">
          <tr>
            <th className="text-left font-normal">shadow week ({r.shadow.rows.toLocaleString()} forecasts)</th>
            <th className="text-right font-normal">WAPE</th>
            <th className="text-right font-normal">PICP</th>
          </tr>
        </thead>
        <tbody className="tabular">
          <tr>
            <td>challenger</td>
            <td className="text-right">{fmtNum(r.shadow.challenger.wape, 3)}</td>
            <td className="text-right">{fmtNum(r.shadow.challenger.picp, 3)}</td>
          </tr>
          <tr>
            <td>current champion</td>
            <td className="text-right">{fmtNum(r.shadow.champion.wape, 3)}</td>
            <td className="text-right">{fmtNum(r.shadow.champion.picp, 3)}</td>
          </tr>
        </tbody>
      </table>
      <ul className="mt-2 flex flex-wrap gap-1.5">
        {Object.entries(r.gate).map(([k, g]) => (
          <li key={k} title={g.rule}>
            <Pill ok={g.ok} label={`${k}: ${g.rule}`} />
          </li>
        ))}
      </ul>
      {!active && (
        <button onClick={onActivate} disabled={busy} className="mt-2 text-xs text-accent underline disabled:opacity-50">
          make this the live champion
        </button>
      )}
    </div>
  );
}

export default function OpsPage() {
  const qc = useQueryClient();
  const [msg, setMsg] = useState<string | null>(null);
  const status = useQuery({ queryKey: ["agents"], queryFn: api.agents, refetchInterval: 5000 });
  const drift = useQuery({ queryKey: ["drift"], queryFn: () => api.drift(14) });
  const runs = useQuery({ queryKey: ["retrain-runs"], queryFn: api.retrainRuns, refetchInterval: status.data?.agents.retrain.in_progress ? 4000 : false });
  const refresh = () => ["agents", "drift", "retrain-runs", "forecast", "snapshot", "stations"].forEach((k) => qc.invalidateQueries({ queryKey: [k] }));
  const act = useMutation({
    mutationFn: async (fn: () => Promise<unknown>) => fn(),
    onSuccess: (r) => {
      setMsg(r && typeof r === "object" ? JSON.stringify(r).slice(0, 220) : "done");
      refresh();
    },
    onError: (e: Error) => setMsg(e.message),
  });
  const st = status.data;
  const retraining = st?.agents.retrain.in_progress ?? false;
  const checks = drift.data?.history ?? [];
  const days = [...new Set(checks.map((c) => c.detected_at))].slice(-14);

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h1 className="text-lg font-semibold">Operations</h1>
        <p className="text-sm text-muted">
          Background agents, drift monitoring and champion-challenger retraining. Retrains never touch the frozen test-scored
          models; a promoted challenger only changes live forecasts and can be rolled back.
        </p>
      </div>
      {msg && <p className="card px-4 py-2 text-xs text-muted">{msg}</p>}

      <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {st &&
          Object.entries(st.agents).map(([name, a]) => (
            <div key={name} className="card px-4 py-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold capitalize">{name}</span>
                {a.in_progress ? (
                  <Loader2 size={16} className="animate-spin text-accent" />
                ) : (
                  <Pill ok={a.last_status === null ? null : a.last_status === "ok"} label={a.last_status ?? "not run"} />
                )}
              </div>
              <div className="mt-1 text-xs text-muted">{a.description}</div>
              <div className="mt-2 text-xs">
                {a.runs.toLocaleString()} runs{a.last_finished_at ? ` · last ${new Date(a.last_finished_at).toLocaleTimeString()}` : ""}
              </div>
            </div>
          ))}
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <section className="card flex flex-col gap-3 p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold">Drift checks</h2>
            <button
              onClick={() => act.mutate(api.runDrift)}
              disabled={act.isPending}
              className="inline-flex items-center gap-1 rounded-lg border border-border px-2.5 py-1 text-xs hover:bg-surface-2 disabled:opacity-50"
            >
              <Search size={13} /> Check now
            </button>
          </div>
          <p className="text-xs text-muted">
            Runs automatically at each local midnight. {drift.data?.latest_at ? `Latest: ${fmtTime(drift.data.latest_at)}.` : "No check yet."}
          </p>
          <ul className="flex flex-col gap-2">{drift.data?.latest.map((d) => <Signal key={d.kind} d={d} />)}</ul>
          {drift.data?.recommend_retrain && (
            <div className="flex gap-2 rounded-lg border border-warn/40 bg-warn/10 px-3 py-2 text-xs">
              <CircleAlert size={14} className="mt-0.5 shrink-0 text-warn" />
              Two or more signals fired together: a retrain is recommended.
            </div>
          )}
          {days.length > 1 && (
            <div>
              <div className="mb-1 text-xs text-muted">Last {days.length} checks (red = signal fired)</div>
              <div className="grid gap-1" style={{ gridTemplateColumns: `90px repeat(${days.length}, minmax(0, 1fr))` }}>
                {Object.keys(SIGNAL_LABEL).map((kind) => (
                  <div key={kind} className="contents">
                    <span className="truncate text-[10px] text-muted">{SIGNAL_LABEL[kind].split(" (")[0]}</span>
                    {days.map((day) => {
                      const c = checks.find((x) => x.detected_at === day && x.kind === kind);
                      return (
                        <span
                          key={day}
                          title={`${fmtTime(day)}: ${c ? fmtNum(c.statistic, 3) : "n/a"}`}
                          className={clsx("h-3 rounded-sm", !c ? "bg-surface-2" : c.is_drift ? "bg-crit" : "bg-ok/60")}
                        />
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        <section className="card flex flex-col gap-3 p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-sm font-semibold">Champion-challenger retraining</h2>
            <div className="flex gap-1.5">
              <button
                onClick={() => act.mutate(() => api.startRetrain(true))}
                disabled={act.isPending || retraining}
                className="inline-flex items-center gap-1 rounded-lg bg-accent px-2.5 py-1 text-xs font-medium text-white disabled:opacity-50"
              >
                {retraining ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
                {retraining ? "Retraining…" : "Retrain now"}
              </button>
              {st?.live_champion.run_id && (
                <button
                  onClick={() => act.mutate(api.rollback)}
                  disabled={act.isPending}
                  className="inline-flex items-center gap-1 rounded-lg border border-border px-2.5 py-1 text-xs hover:bg-surface-2 disabled:opacity-50"
                >
                  <RotateCcw size={13} /> Roll back
                </button>
              )}
            </div>
          </div>
          <p className="text-xs text-muted">
            Trains LightGBM quantile models on the last 365 days observed so far, recalibrates on the next 14 days and
            scores both models on the final 7 days, which neither has seen. Promoted only if WAPE improves by 2%,
            coverage stays within 87–93% and no site gets more than 10% worse. Takes about a minute.
          </p>
          <div className="text-xs">
            Live champion: <span className="font-medium">{st?.live_champion.version ?? "…"}</span>
            {st && !st.live_champion.run_id && <span className="text-muted"> (frozen, test-scored artefacts)</span>}
          </div>
          <div className="flex flex-col gap-2">
            {runs.data?.runs.length === 0 && <p className="text-xs text-muted">No retrains yet.</p>}
            {runs.data?.runs.slice(0, 4).map((r) => (
              <RunCard key={r.run_id} r={r} active={runs.data?.active === r.run_id} busy={act.isPending} onActivate={() => act.mutate(() => api.activate(r.run_id))} />
            ))}
          </div>
        </section>
      </div>

      <section className="card flex flex-wrap items-center justify-between gap-3 p-4 text-sm">
        <div>
          <h2 className="font-semibold">Station discovery</h2>
          <p className="text-xs text-muted">
            Adds other public chargers in the city from OpenStreetMap as grey, unforecast pins (runs once a day at start-up).
            {st?.agents.discovery.last_details && " Last run: " + JSON.stringify(st.agents.discovery.last_details).slice(0, 120)}
          </p>
        </div>
        <button
          onClick={() => act.mutate(api.runDiscovery)}
          disabled={act.isPending}
          className="inline-flex items-center gap-1 rounded-lg border border-border px-2.5 py-1 text-xs hover:bg-surface-2 disabled:opacity-50"
        >
          <MapPinned size={13} /> Discover now
        </button>
      </section>

      {st && (
        <section className="card grid gap-2 p-4 text-xs sm:grid-cols-3">
          <div>
            <div className="text-muted">Telemetry buffer</div>
            <Pill ok={st.buffer.healthy} label={st.buffer.healthy ? "fresh" : `${fmtNum(st.buffer.lag_hours, 0)} h behind`} />
          </div>
          <div>
            <div className="text-muted">Live WebSocket clients</div>
            <div className="tabular text-sm">{st.ws_subscribers}</div>
          </div>
          <div>
            <div className="text-muted">Simulator</div>
            <div className="text-sm">{st.last_error ? <span className="text-crit">{st.last_error}</span> : st.sim?.paused ? "paused" : "running"}</div>
          </div>
        </section>
      )}
    </div>
  );
}
