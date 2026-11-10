"use client";

// Live layer: one WebSocket to /ws/stream. Pipeline events invalidate the matching queries so every
// view refreshes when a simulated hour closes. Reconnects with exponential backoff.

import { useQueryClient } from "@tanstack/react-query";
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";

import { WS_URL } from "./api";

export type LiveStatus = "connecting" | "live" | "offline";

interface LiveEvent {
  channel: string;
  event: string;
  ts: string;
  data: Record<string, unknown>;
}

const LiveContext = createContext<{ status: LiveStatus; last: LiveEvent | null }>({ status: "connecting", last: null });

export function useLive() {
  return useContext(LiveContext);
}

const INVALIDATE: Record<string, string[][]> = {
  "forecast.updated": [["kpis"], ["snapshot"], ["stations"], ["forecast"], ["explain"], ["arena"], ["sim"]],
  "alert.raised": [["alerts"]],
  "alert.resolved": [["alerts"]],
  "reroute.updated": [["snapshot"]],
  "sim.clock": [["sim"], ["health"]],
  "sim.control": [["sim"]],
  "sim.reset": [["sim"], ["kpis"], ["snapshot"], ["stations"], ["forecast"], ["alerts"], ["arena"], ["explain"]],
  "sim.end": [["sim"]],
};

export function LiveProvider({ children }: { children: ReactNode }) {
  const qc = useQueryClient();
  const [status, setStatus] = useState<LiveStatus>("connecting");
  const [last, setLast] = useState<LiveEvent | null>(null);
  const retry = useRef(0);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let closed = false;

    const connect = () => {
      setStatus("connecting");
      ws = new WebSocket(WS_URL);
      ws.onopen = () => {
        retry.current = 0;
        setStatus("live");
      };
      ws.onmessage = (msg) => {
        const ev = JSON.parse(msg.data) as LiveEvent;
        if (ev.channel === "control") return;
        setLast(ev);
        for (const key of INVALIDATE[ev.event] ?? []) qc.invalidateQueries({ queryKey: key });
      };
      ws.onclose = () => {
        setStatus("offline");
        if (closed) return;
        const delay = Math.min(15000, 1000 * 2 ** retry.current++);
        timer = setTimeout(connect, delay);
      };
      ws.onerror = () => ws?.close();
    };
    connect();
    return () => {
      closed = true;
      if (timer) clearTimeout(timer);
      ws?.close();
    };
  }, [qc]);

  return <LiveContext.Provider value={{ status, last }}>{children}</LiveContext.Provider>;
}
