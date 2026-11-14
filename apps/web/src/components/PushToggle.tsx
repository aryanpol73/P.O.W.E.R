"use client";

import clsx from "clsx";
import { Bell, BellOff, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";

import { api } from "@/lib/api";

type State = "unsupported" | "no-sw" | "denied" | "off" | "on" | "busy";

function keyBytes(b64url: string): Uint8Array<ArrayBuffer> {
  const pad = "=".repeat((4 - (b64url.length % 4)) % 4);
  const raw = atob((b64url + pad).replace(/-/g, "+").replace(/_/g, "/"));
  const out = new Uint8Array(new ArrayBuffer(raw.length));
  for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
  return out;
}

async function registration(): Promise<ServiceWorkerRegistration | null> {
  if (!("serviceWorker" in navigator)) return null;
  return (await navigator.serviceWorker.getRegistration()) ?? null;
}

/** Browser notifications for new grid alerts (Web Push through the service worker). */
export function PushToggle({ level = "CRITICAL", className }: { level?: "CRITICAL" | "WARNING"; className?: string }) {
  const [state, setState] = useState<State>("busy");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      if (!("PushManager" in window) || !("Notification" in window)) return setState("unsupported");
      if (Notification.permission === "denied") return setState("denied");
      const reg = await registration();
      if (!reg) return setState("no-sw");
      setState((await reg.pushManager.getSubscription()) ? "on" : "off");
    })().catch(() => setState("unsupported"));
  }, []);

  async function toggle() {
    setError(null);
    const reg = await registration();
    if (!reg) return setState("no-sw");
    setState("busy");
    try {
      const existing = await reg.pushManager.getSubscription();
      if (existing) {
        await api.unsubscribePush(existing.endpoint).catch(() => undefined);
        await existing.unsubscribe();
        return setState("off");
      }
      if ((await Notification.requestPermission()) !== "granted") return setState("denied");
      const { public_key } = await api.vapidKey();
      const sub = await reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: keyBytes(public_key) });
      await api.subscribePush(sub.toJSON(), level);
      setState("on");
    } catch (e) {
      setError((e as Error).message);
      setState("off");
    }
  }

  const label: Record<State, string> = {
    unsupported: "Notifications not supported in this browser",
    "no-sw": "Notifications need the installed app (production build)",
    denied: "Notifications blocked in browser settings",
    off: level === "CRITICAL" ? "Notify me of critical alerts" : "Notify me of alerts",
    on: "Alert notifications on",
    busy: "…",
  };
  const usable = state === "off" || state === "on";
  return (
    <div className={clsx("flex flex-col items-end", className)}>
      <button
        onClick={toggle}
        disabled={!usable}
        title={label[state]}
        className={clsx(
          "inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs disabled:cursor-not-allowed disabled:opacity-60",
          state === "on" ? "border-accent bg-accent-soft text-accent" : "border-border hover:bg-surface-2",
        )}
      >
        {state === "busy" ? <Loader2 size={13} className="animate-spin" /> : state === "on" ? <Bell size={13} /> : <BellOff size={13} />}
        <span className="hidden sm:inline">{label[state]}</span>
      </button>
      {error && <span className="mt-1 text-[11px] text-crit">{error}</span>}
    </div>
  );
}
