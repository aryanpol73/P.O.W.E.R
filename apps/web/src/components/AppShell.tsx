"use client";

import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { Activity, BarChart3, Bot, FlaskConical, LayoutDashboard, Lightbulb, Zap } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { api } from "@/lib/api";

import { CopilotPanel } from "./CopilotPanel";
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
        <div className="flex items-center gap-2 px-5 py-5">
          <span className="grid size-8 place-items-center rounded-lg bg-accent text-white">
            <Zap size={18} />
          </span>
          <div>
            <div className="text-sm font-semibold tracking-wide">P.O.W.E.R.</div>
            <div className="text-[11px] text-muted">{dep.data?.city ? `${dep.data.city} EV grid console` : "EV grid console"}</div>
          </div>
        </div>
        <nav className="flex flex-col gap-1 px-3">
          {NAV.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                active(href) ? "bg-accent-soft font-medium text-accent" : "text-muted hover:bg-surface-2 hover:text-text",
              )}
            >
              <Icon size={16} />
              {label}
            </Link>
          ))}
        </nav>
        <div className="mt-auto px-5 py-4 text-[11px] leading-relaxed text-muted">
          {dep.data?.is_twin ? (
            <>
              <span className="mb-1 inline-block rounded bg-warn/15 px-1.5 py-0.5 font-medium text-warn">Pilot digital twin</span>
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
        <header className="sticky top-0 z-20 border-b border-border bg-bg/90 backdrop-blur">
          <SimControls />
        </header>
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-5 md:px-6">{children}</main>
        <CopilotPanel />
      </div>

      <nav className="fixed inset-x-0 bottom-0 z-30 grid grid-cols-6 border-t border-border bg-surface md:hidden">
        {NAV.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={clsx("flex flex-col items-center gap-0.5 py-2 text-[10px]", active(href) ? "text-accent" : "text-muted")}
          >
            <Icon size={18} />
            {label.split(" ")[0]}
          </Link>
        ))}
      </nav>
    </div>
  );
}
