"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";

/**
 * The hosted API runs on a free plan that can sleep when idle and needs up to a minute to wake.
 * While it does not answer, say so plainly instead of leaving empty panels, keep checking every
 * 5 seconds, and reload every query once it is back.
 */
export function ServerWaking() {
  const qc = useQueryClient();
  const health = useQuery({
    queryKey: ["health-probe"],
    queryFn: api.health,
    retry: false,
    refetchInterval: (q) => (q.state.status === "success" ? false : 5_000),
    refetchOnWindowFocus: false,
  });
  const ok = health.isSuccess;
  const [slow, setSlow] = useState(false);
  const wasDown = useRef(false);
  const [since] = useState(() => Date.now());
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (ok) return;
    const t = setTimeout(() => setSlow(true), 2_500); // a fast answer never shows the banner
    const id = setInterval(() => setElapsed(Math.round((Date.now() - since) / 1000)), 1_000);
    return () => {
      clearTimeout(t);
      clearInterval(id);
    };
  }, [ok, since]);

  useEffect(() => {
    if (!ok && slow) wasDown.current = true;
    if (ok && wasDown.current) {
      wasDown.current = false;
      qc.invalidateQueries(); // panels that gave up while the server slept load now
    }
  }, [ok, slow, qc]);

  if (ok || !slow) return null;
  return (
    <div
      role="status"
      className="hud pointer-events-none fixed inset-x-0 top-24 z-40 mx-auto flex w-fit max-w-[calc(100%-2rem)] items-center gap-3 px-4 py-3 text-sm md:top-28"
    >
      <Loader2 className="size-4 shrink-0 animate-spin text-accent" aria-hidden="true" />
      <span>
        <span className="font-semibold">Starting the forecasting server…</span>{" "}
        <span className="text-muted">
          It runs on a free plan and can take up to a minute to wake. This page fills in by itself
          {elapsed > 0 ? ` (${elapsed} s)` : ""}.
        </span>
      </span>
    </div>
  );
}
