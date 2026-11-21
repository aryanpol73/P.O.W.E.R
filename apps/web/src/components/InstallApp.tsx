"use client";

import { Download, Share, X } from "lucide-react";
import { useEffect, useState } from "react";

// Chromium (Android, Windows, macOS, ChromeOS) fires `beforeinstallprompt`; iOS/iPadOS Safari never does,
// so there we explain "Share -> Add to Home Screen" instead.
type InstallPrompt = Event & { prompt: () => Promise<void>; userChoice: Promise<{ outcome: "accepted" | "dismissed" }> };

function isStandalone(): boolean {
  return (
    window.matchMedia("(display-mode: standalone)").matches ||
    (navigator as Navigator & { standalone?: boolean }).standalone === true
  );
}

function isIos(): boolean {
  const ua = navigator.userAgent;
  return /iPad|iPhone|iPod/.test(ua) || (ua.includes("Macintosh") && navigator.maxTouchPoints > 1);
}

const DISMISS_KEY = "power-install-dismissed";

export function InstallApp({ variant }: { variant: "header" | "banner" }) {
  const [prompt, setPrompt] = useState<InstallPrompt | null>(null);
  const [ios, setIos] = useState(false);
  const [installed, setInstalled] = useState(true);
  const [dismissed, setDismissed] = useState(true);
  const [showIosHelp, setShowIosHelp] = useState(false);

  useEffect(() => {
    setInstalled(isStandalone());
    setIos(isIos());
    try {
      setDismissed(localStorage.getItem(DISMISS_KEY) === "1");
    } catch {
      setDismissed(false);
    }
    const onPrompt = (e: Event) => {
      e.preventDefault(); // keep it for our own button
      setPrompt(e as InstallPrompt);
    };
    const onInstalled = () => {
      setInstalled(true);
      setPrompt(null);
    };
    window.addEventListener("beforeinstallprompt", onPrompt);
    window.addEventListener("appinstalled", onInstalled);
    return () => {
      window.removeEventListener("beforeinstallprompt", onPrompt);
      window.removeEventListener("appinstalled", onInstalled);
    };
  }, []);

  if (installed || (!prompt && !ios)) return null;

  async function install() {
    if (prompt) {
      await prompt.prompt();
      const { outcome } = await prompt.userChoice;
      if (outcome === "accepted") setInstalled(true);
      setPrompt(null);
    } else {
      setShowIosHelp((v) => !v);
    }
  }

  function dismiss() {
    setDismissed(true);
    try {
      localStorage.setItem(DISMISS_KEY, "1");
    } catch {
      /* private mode: dismissal lasts for this visit */
    }
  }

  const help = showIosHelp && (
    <p className="flex items-start gap-1.5 text-[11px] leading-snug text-muted">
      <Share size={13} className="mt-px shrink-0" />
      In Safari, tap Share, then <strong className="font-semibold text-text">Add to Home Screen</strong>.
    </p>
  );

  if (variant === "header") {
    return (
      <div className="relative">
        <button onClick={install} title="Install app" aria-label="Install app" className="btn btn-secondary btn-icon">
          <Download size={15} />
        </button>
        {help && <div className="glass-panel panel-in absolute top-12 right-0 z-40 w-60 p-3 [transform-origin:top_right]">{help}</div>}
      </div>
    );
  }

  if (dismissed) return null;
  return (
    <div className="card fixed inset-x-3 bottom-20 z-40 flex items-center gap-3 p-3 shadow-xl md:hidden">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/icons/icon-192.png" alt="" className="size-10 shrink-0 rounded-xl" />
      <div className="min-w-0 flex-1">
        <div className="text-sm font-semibold">Install P.O.W.E.R.</div>
        <div className="text-[11px] text-muted">{ios ? "Add it to your Home Screen for full-screen alerts." : "Full-screen app with alert notifications."}</div>
        {help && <div className="mt-2">{help}</div>}
      </div>
      <button onClick={install} className="btn btn-primary btn-sm shrink-0">
        {ios ? "How" : "Install"}
      </button>
      <button onClick={dismiss} aria-label="Dismiss" className="grid size-8 shrink-0 place-items-center rounded-lg text-muted hover:bg-surface-2">
        <X size={16} />
      </button>
    </div>
  );
}
