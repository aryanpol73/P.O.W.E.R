"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Activity, BarChart3, Bot, FlaskConical, LayoutDashboard, Lightbulb } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { api } from "@/lib/api";

import { CopilotPanel } from "./CopilotPanel";
import { InstallApp } from "./InstallApp";
import { SimControls } from "./SimControls";

const NAV = [
  { href: "/", label: "Command Center", icon: LayoutDashboard },
  { href: "/stations", label: "Stations", icon: Activity },
  { href: "/arena", label: "Model Arena", icon: BarChart3 },
  { href: "/simulator", label: "What-If", icon: FlaskConical },
  { href: "/insights", label: "Insights", icon: Lightbulb },
  { href: "/ops", label: "Ops", icon: Bot },
];

export function AppShell({ children }: { children: ReactNode }) {
  const path = usePathname();
  const active = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));
  const dep = useQuery({ queryKey: ["deployment"], queryFn: api.deployment, staleTime: Infinity });
  useQuery({ queryKey: ["stations"], queryFn: api.stations }); // registers display names app-wide
  return (
    <div className="min-h-dvh md:grid md:grid-cols-[220px_1fr]">
      <aside className="hidden border-r border-border bg-surface md:flex md:flex-col">
        <div className="flex items-center gap-2.5 px-5 py-5">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/icons/logo.png" alt="P.O.W.E.R. logo" width={36} height={36} className="size-9 rounded-[10px] shadow-[0_6px_16px_-6px_rgba(34,197,94,0.55)]" />
          <div>
            <div className="text-sm font-semibold tracking-[0.06em]">P.O.W.E.R.</div>
            <div className="text-[11px] text-muted">{dep.data?.city ? `${dep.data.city} EV grid console` : "EV grid console"}</div>
          </div>
        </div>
        <nav className="flex flex-col gap-0.5 px-3">
          {NAV.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              aria-current={active(href) ? "page" : undefined}
              className={clsx(
                "relative flex min-h-10 items-center gap-3 rounded-lg px-3 text-sm",
                active(href)
                  ? "bg-accent-soft font-medium text-accent before:absolute before:inset-y-2 before:-left-3 before:w-[3px] before:rounded-r-full before:bg-accent"
                  : "text-muted hover:bg-surface-2 hover:text-text",
              )}
            >
              <Icon size={16} strokeWidth={active(href) ? 2.25 : 2} />
              {label}
            </Link>
          ))}
        </nav>
        <div className="mt-auto">
          <InstallApp variant="sidebar" />
        </div>
        <div className="border-t border-divider px-5 py-4 text-[11px] leading-relaxed text-muted">
          {dep.data?.is_twin ? (
            <>
              <span className="mb-1.5 inline-flex items-center rounded-md border border-warn/40 bg-warn/12 px-1.5 py-0.5 font-semibold text-warn">Pilot digital twin</span>
              <br />
              Pune sites and weather, Indian holidays, IST. Models trained on US open charging data (2017–2020) and not yet
              validated on Indian demand. Transformer ratings are a planning assumption.
            </>
          ) : (
            <>Project #41. Transformer ratings are a planning assumption.</>
          )}
        </div>
      </aside>

      <div className="flex min-w-0 flex-col pb-16 md:pb-0">
        <header className="glass sticky top-0 z-20 border-b border-border">
          <SimControls />
        </header>
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-5 md:px-6">{children}</main>
        <CopilotPanel />
        <InstallApp variant="banner" />
      </div>

      <nav className="glass fixed inset-x-0 bottom-0 z-30 grid grid-cols-6 border-t border-border pb-[env(safe-area-inset-bottom)] md:hidden">
        {NAV.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            aria-current={active(href) ? "page" : undefined}
            className={clsx(
              "relative flex min-h-12 flex-col items-center justify-center gap-0.5 text-[10px]",
              active(href)
                ? "font-semibold text-accent before:absolute before:inset-x-4 before:top-0 before:h-0.5 before:rounded-b-full before:bg-accent"
                : "text-muted",
            )}
          >
            <Icon size={18} strokeWidth={active(href) ? 2.25 : 2} />
            {label.split(" ")[0]}
          </Link>
        ))}
      </nav>
    </div>
  );
}
