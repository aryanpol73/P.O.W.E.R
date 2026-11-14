"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Loader2, MessageSquareText, Send, Sparkles, Wrench, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

type Step = { name: string; ok: boolean | null };
type Turn = { role: "user" | "assistant"; text: string; steps?: Step[]; error?: string };
type Status = { enabled: boolean; model?: string; detail?: string };

const TOOL_LABEL: Record<string, string> = {
  list_stations: "listing sites",
  get_network_status: "reading network status",
  get_station_forecast: "reading a forecast",
  explain_peak_shap: "explaining the peak",
  simulate_weather_shock: "running a what-if",
  compare_models: "comparing models",
  get_alerts: "reading alerts",
  get_drift_report: "reading drift checks",
  get_agent_status: "checking agents",
};

const SUGGESTIONS = [
  "Which sites are at risk in the next 12 hours, and why?",
  "What drives tonight's peak at Westend Mall?",
  "Simulate a +6 °C heatwave across the network.",
  "Which model is most accurate, on the test set and live?",
];

export function CopilotPanel() {
  const [open, setOpen] = useState(false);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const conv = useRef<string | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const status = useQuery<Status>({
    queryKey: ["copilot-status"],
    queryFn: () => fetch("/api/copilot").then((r) => r.json()),
    staleTime: 60_000,
    enabled: open,
  });

  useEffect(() => bottom.current?.scrollIntoView({ block: "end" }), [turns]);

  const patchLast = (f: (t: Turn) => Turn) => setTurns((ts) => [...ts.slice(0, -1), f(ts[ts.length - 1])]);

  async function ask(text: string) {
    const q = text.trim();
    if (!q || busy) return;
    setInput("");
    setBusy(true);
    setTurns((ts) => [...ts, { role: "user", text: q }, { role: "assistant", text: "", steps: [] }]);
    try {
      const res = await fetch("/api/copilot", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ message: q, conversation_id: conv.current }),
      });
      if (!res.ok || !res.body) {
        const detail = await res.json().then((j) => j.detail).catch(() => res.statusText);
        patchLast((t) => ({ ...t, error: String(detail) }));
        return;
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let nl: number;
        while ((nl = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          const ev = JSON.parse(line);
          if (ev.type === "conversation") conv.current = ev.id;
          else if (ev.type === "text") patchLast((t) => ({ ...t, text: t.text + ev.text }));
          else if (ev.type === "tool_call") patchLast((t) => ({ ...t, steps: [...(t.steps ?? []), { name: ev.name, ok: null }] }));
          else if (ev.type === "tool_result")
            patchLast((t) => {
              const steps = [...(t.steps ?? [])];
              const i = steps.findIndex((s) => s.name === ev.name && s.ok === null);
              if (i >= 0) steps[i] = { name: ev.name, ok: ev.ok };
              return { ...t, steps };
            });
          else if (ev.type === "error") patchLast((t) => ({ ...t, error: ev.message }));
        }
      }
    } catch (e) {
      patchLast((t) => ({ ...t, error: (e as Error).message }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <button
        onClick={() => setOpen((o) => !o)}
        aria-label={open ? "Close copilot" : "Open copilot"}
        className="fixed bottom-20 right-4 z-40 grid size-12 place-items-center rounded-full bg-accent text-white shadow-lg md:bottom-6"
      >
        {open ? <X size={20} /> : <MessageSquareText size={20} />}
      </button>
      {open && (
        <section className="card fixed bottom-36 right-4 z-40 flex h-[min(70dvh,560px)] w-[min(calc(100vw-2rem),400px)] flex-col overflow-hidden shadow-2xl md:bottom-20">
          <header className="flex items-center gap-2 border-b border-border px-4 py-3">
            <Sparkles size={16} className="text-accent" />
            <div className="min-w-0">
              <div className="text-sm font-semibold">Grid copilot</div>
              <div className="truncate text-[11px] text-muted">
                Answers only from the console&apos;s own tools{status.data?.model ? ` · ${status.data.model}` : ""}
              </div>
            </div>
          </header>
          <div className="flex-1 space-y-3 overflow-y-auto px-4 py-3 text-sm">
            {status.data && !status.data.enabled && (
              <p className="rounded-lg bg-surface-2 px-3 py-2 text-xs text-muted">
                The copilot is off. Set <code>GROQ_API_KEY</code> on the MCP service to turn it on.
                {status.data.detail ? ` (${status.data.detail})` : ""}
              </p>
            )}
            {turns.length === 0 && status.data?.enabled !== false && (
              <div className="flex flex-col gap-2">
                <p className="text-xs text-muted">Ask about risk, forecasts, drivers, scenarios or model accuracy.</p>
                {SUGGESTIONS.map((s) => (
                  <button key={s} onClick={() => ask(s)} className="rounded-lg border border-border px-3 py-2 text-left text-xs hover:bg-surface-2">
                    {s}
                  </button>
                ))}
              </div>
            )}
            {turns.map((t, i) => (
              <div key={i} className={clsx("flex", t.role === "user" ? "justify-end" : "justify-start")}>
                <div className={clsx("max-w-[90%] rounded-xl px-3 py-2", t.role === "user" ? "bg-accent text-white" : "bg-surface-2")}>
                  {t.steps && t.steps.length > 0 && (
                    <ul className="mb-1.5 flex flex-wrap gap-1">
                      {t.steps.map((s, j) => (
                        <li key={j} className={clsx("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px]", s.ok === false ? "bg-crit/15 text-crit" : "bg-bg text-muted")}>
                          {s.ok === null ? <Loader2 size={10} className="animate-spin" /> : <Wrench size={10} />}
                          {TOOL_LABEL[s.name] ?? s.name}
                        </li>
                      ))}
                    </ul>
                  )}
                  {t.text && <div className="whitespace-pre-wrap leading-relaxed">{t.text}</div>}
                  {!t.text && !t.error && t.role === "assistant" && busy && i === turns.length - 1 && (
                    <Loader2 size={14} className="animate-spin text-muted" />
                  )}
                  {t.error && <div className="text-xs text-crit">{t.error}</div>}
                </div>
              </div>
            ))}
            <div ref={bottom} />
          </div>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              ask(input);
            }}
            className="flex gap-2 border-t border-border p-3"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask the copilot…"
              disabled={busy || status.data?.enabled === false}
              className="h-9 min-w-0 flex-1 rounded-lg border border-border bg-surface px-3 text-sm"
            />
            <button type="submit" disabled={busy || !input.trim()} aria-label="Send" className="grid size-9 place-items-center rounded-lg bg-accent text-white disabled:opacity-50">
              {busy ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
            </button>
          </form>
        </section>
      )}
    </>
  );
}
