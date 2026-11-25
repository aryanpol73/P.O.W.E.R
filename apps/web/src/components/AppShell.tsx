"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Activity, BarChart3, Bot, FlaskConical, Info, Lightbulb, Map as MapIcon } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useLayoutEffect, useRef, useState, type ReactNode } from "react";

import { api } from "@/lib/api";

import { CopilotPanel } from "./CopilotPanel";
import { InstallApp } from "./InstallApp";
import { ServerWaking } from "./ServerWaking";
import { SimControls } from "./SimControls";

const NAV = [
  { href: "/", label: "Command", icon: MapIcon },
  { href: "/stations", label: "Stations", icon: Activity },
  { href: "/arena", label: "Arena", icon: BarChart3 },
  { href: "/simulator", label: "What-If", icon: FlaskConical },
  { href: "/insights", label: "Insights", icon: Lightbulb },
  { href: "/ops", label: "Ops", icon: Bot },
];

// Atlas Night shell: floating header pills over the content. The Command Center fills the viewport
// with its map; every other page scrolls underneath the header.
export function AppShell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const isActive = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));
  const activeIndex = NAV.findIndex((n) => isActive(n.href));
  const dep = useQuery({ queryKey: ["deployment"], queryFn: api.deployment, staleTime: Infinity });
  useQuery({ queryKey: ["stations"], queryFn: api.stations }); // registers display names app-wide
  const isMap = path === "/";

  return (
    <div className="min-h-dvh">
      <header className="pointer-events-none fixed inset-x-0 top-0 z-30 flex items-start gap-3 p-3 md:p-5">
        <div className="hud rise pointer-events-auto flex items-center gap-1.5 p-1.5">
          <Link href="/" className="flex items-center gap-2.5 rounded-xl py-0.5 pr-2 pl-1 md:pr-3">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/icons/logo.png" alt="" width={32} height={32} className="size-8 rounded-[10px] shadow-[0_6px_16px_-6px_rgba(34,197,94,0.7)]" />
            <span className="hidden text-sm font-extrabold tracking-[0.06em] sm:inline">P.O.W.E.R.</span>
          </Link>
          <TwinBadge city={dep.data?.city} isTwin={dep.data?.is_twin} cities={dep.data?.cities?.length} />
          <NavTabs activeIndex={activeIndex} isActive={isActive} />
        </div>
        <div className="flex-1" />
        <div className="hud rise pointer-events-auto flex items-center gap-2 p-1.5 pl-3 [--d:80ms]">
          <SimControls />
          <InstallApp variant="header" />
        </div>
      </header>

      <main className={clsx(isMap ? "relative h-dvh overflow-hidden" : "mx-auto w-full max-w-[1400px] px-4 pt-24 pb-28 md:px-6 md:pt-28 md:pb-12")}>
        {children}
      </main>
      <ServerWaking />
      <CopilotPanel />
      <InstallApp variant="banner" />

      <nav aria-label="Main" className="glass fixed inset-x-0 bottom-0 z-30 grid grid-cols-6 border-t border-divider pb-[env(safe-area-inset-bottom)] md:hidden">
        {NAV.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            aria-current={isActive(href) ? "page" : undefined}
            className={clsx(
              "relative flex min-h-14 flex-col items-center justify-center gap-0.5 text-[10px] font-semibold",
              isActive(href) ? "text-accent before:absolute before:inset-x-5 before:top-0 before:h-0.5 before:rounded-b-full before:bg-gradient-to-r before:from-[var(--grad-a)] before:to-[var(--grad-b)]" : "text-muted",
            )}
          >
            <Icon size={18} strokeWidth={isActive(href) ? 2.4 : 2} />
            {label}
          </Link>
        ))}
      </nav>
    </div>
  );
}

// Tab row with one gradient pill that slides to the active tab (measured, so label widths can vary).
function NavTabs({ activeIndex, isActive }: { activeIndex: number; isActive: (href: string) => boolean }) {
  const refs = useRef<(HTMLAnchorElement | null)[]>([]);
  const [pill, setPill] = useState<{ x: number; w: number } | null>(null);

  useLayoutEffect(() => {
    const measure = () => {
      const el = refs.current[activeIndex];
      setPill(el ? { x: el.offsetLeft, w: el.offsetWidth } : null);
    };
    measure();
    window.addEventListener("resize", measure);
    document.fonts?.ready.then(measure).catch(() => {});
    return () => window.removeEventListener("resize", measure);
  }, [activeIndex]);

  return (
    <nav aria-label="Main" className="relative hidden rounded-xl bg-black/25 p-[3px] md:flex">
      {pill && (
        <span
          aria-hidden="true"
          className="absolute top-[3px] bottom-[3px] left-0 rounded-[10px] bg-gradient-to-br from-[var(--grad-a)] to-[var(--grad-b)] shadow-[inset_0_1px_0_rgba(255,255,255,0.22),0_8px_18px_-8px_rgba(124,58,237,0.8)] transition-[transform,width] duration-[460ms] ease-[var(--ease)]"
          style={{ transform: `translateX(${pill.x}px)`, width: pill.w }}
        />
      )}
      {NAV.map(({ href, label, icon: Icon }, i) => (
        <Link
          key={href}
          href={href}
          ref={(el) => {
            refs.current[i] = el;
          }}
          aria-current={isActive(href) ? "page" : undefined}
          title={label}
          className={clsx(
            "relative flex h-9 items-center gap-2 rounded-[10px] px-3 text-[13px] font-semibold xl:px-3.5",
            isActive(href) ? "text-white" : "text-muted hover:text-text",
          )}
        >
          <Icon size={15} strokeWidth={2.2} className="xl:hidden" />
          <span className="hidden xl:inline">{label}</span>
          <span className="sr-only xl:hidden">{label}</span>
        </Link>
      ))}
    </nav>
  );
}

// The pilot-twin disclaimer used to live in the sidebar footer; it now opens from a badge.
function TwinBadge({ city, isTwin, cities }: { city?: string; isTwin?: boolean; cities?: number }) {
  return (
    <details className="group relative hidden sm:block">
      <summary className="flex h-8 cursor-pointer list-none items-center gap-1.5 rounded-lg px-2.5 text-[11px] font-bold tracking-[0.04em] text-amber ring-1 ring-amber/30 ring-inset hover:bg-amber/10 [&::-webkit-details-marker]:hidden">
        <Info size={12} strokeWidth={2.5} />
        {isTwin ? `${city ?? "Pilot"} twin` : city ?? "Pilot"}
      </summary>
      <div className="glass-panel panel-in absolute top-11 left-0 z-40 w-72 p-4 text-xs leading-relaxed text-muted [transform-origin:top_left]">
        {isTwin ? (
          <>
            <strong className="mb-1 block text-sm text-text">Pilot digital twin</strong>
            Real {city ?? "local"} charging sites{cities ? ` in ${cities} cities` : ""} with their own weather, Indian holidays and IST. Models
            trained on US open charging data (2017–2020) and not yet validated on Indian demand. Transformer ratings are a planning
            assumption.
          </>
        ) : (
          <>Project #41. Transformer ratings are a planning assumption.</>
        )}
      </div>
    </details>
  );
}
