"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { List, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

import { AlertFeed } from "@/components/AlertFeed";
import { HorizonDock } from "@/components/HorizonDock";
import { KpiRibbon } from "@/components/KpiRibbon";
import { StationList } from "@/components/StationList";
import { StationMap, type MapPadding } from "@/components/StationMap";
import { api } from "@/lib/api";

// Atlas Night Command Center: the city map fills the screen and everything else floats over it.
// lg+: stations on the left, figures and alerts on the right, the horizon dock at the bottom.
// Smaller screens: figures scroll across the top, the dock sits above the tab bar, lists open in a sheet.
export default function CommandCenter() {
  const router = useRouter();
  const [h, setH] = useState(1);
  const [showOthers, setShowOthers] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [city, setCity] = useState<string | null>(null); // null = the whole deployment
  const [sheet, setSheet] = useState(false);
  const wide = useWide();

  const snap = useQuery({ queryKey: ["snapshot", h], queryFn: () => api.snapshot(h), placeholderData: keepPreviousData });
  const horizon = useQuery({ queryKey: ["horizon"], queryFn: api.horizon, placeholderData: keepPreviousData });
  useQuery({ queryKey: ["stations"], queryFn: api.stations }); // display names for reroute text
  const dep = useQuery({ queryKey: ["deployment"], queryFn: api.deployment, staleTime: Infinity });
  const cities = dep.data?.cities ?? [];
  const pickCity = (c: string | null) => {
    setCity(c);
    setSelected(null);
  };

  const modeled = useMemo(
    () => snap.data?.stations.filter((s) => s.is_modeled).sort((a, b) => (b.load_ratio_q95 ?? 0) - (a.load_ratio_q95 ?? 0)) ?? [],
    [snap.data],
  );
  const shownSnap = useMemo(
    () => (snap.data && !showOthers ? { ...snap.data, stations: snap.data.stations.filter((s) => s.is_modeled) } : snap.data),
    [snap.data, showOthers],
  );
  // keep the map's framing clear of the floating panels
  const padding: MapPadding = wide ? { top: 110, right: 360, bottom: 190, left: 350 } : { top: 170, right: 40, bottom: 240, left: 40 };

  const select = (id: string | null) => {
    setSelected(id);
    if (id) setSheet(false);
  };

  return (
    <div className="absolute inset-0">
      <StationMap
        snapshot={shownSnap}
        selectedId={selected}
        onSelect={select}
        onOpen={(id) => router.push(`/stations/${id}`)}
        padding={padding}
        focusCity={city}
      />

      {/* figures: a centred strip on very wide screens, otherwise at the top of the right column (lg) or scrolling (small) */}
      <KpiRibbon className="rise absolute top-[92px] left-1/2 hidden -translate-x-1/2 [--d:120ms] 2xl:flex" />
      <div className="absolute inset-x-3 top-[76px] overflow-x-auto [scrollbar-width:none] lg:hidden">
        <KpiRibbon className="rise w-max [--d:120ms]" />
      </div>

      <StationList
        stations={modeled}
        h={h}
        selectedId={selected}
        onSelect={select}
        cities={cities}
        city={city}
        onCity={pickCity}
        className="rise absolute top-[92px] bottom-[176px] left-5 hidden w-[310px] [--d:160ms] lg:flex"
      />

      <div className="absolute top-[92px] right-5 bottom-[176px] hidden w-[320px] flex-col gap-3 lg:flex">
        <KpiRibbon layout="grid" className="rise [--d:120ms] 2xl:hidden" />
        <AlertFeed onSelect={select} className="rise min-h-0 flex-1 [--d:220ms]" />
      </div>

      <HorizonDock
        h={h}
        onChange={setH}
        data={horizon.data}
        showOthers={showOthers}
        onToggleOthers={setShowOthers}
        className="rise absolute right-[72px] bottom-[76px] left-3 [--d:260ms] lg:right-auto lg:bottom-5 lg:left-1/2 lg:w-[min(820px,calc(100%-12rem))] lg:-translate-x-1/2"
      />

      {/* small screens: stations and alerts in a sheet */}
      <button onClick={() => setSheet((v) => !v)} aria-expanded={sheet} className="btn btn-secondary hud absolute top-[164px] left-3 lg:hidden">
        {sheet ? <X size={15} /> : <List size={15} />} Stations &amp; alerts
      </button>
      <div
        className={clsx(
          "hud absolute inset-x-3 top-[214px] bottom-[220px] flex flex-col gap-4 overflow-y-auto p-3 transition-[opacity,transform] duration-[480ms] ease-[var(--ease)] lg:hidden",
          sheet ? "opacity-100" : "pointer-events-none translate-y-4 opacity-0",
        )}
        aria-hidden={!sheet}
        inert={!sheet}
      >
        <StationList stations={modeled} h={h} selectedId={selected} onSelect={select} cities={cities} city={city} onCity={pickCity} className="max-h-[60vh] shrink-0 !bg-transparent !shadow-none !backdrop-blur-none" />
        <AlertFeed onSelect={select} className="shrink-0" />
      </div>
    </div>
  );
}

function useWide() {
  const [wide, setWide] = useState(true);
  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)");
    const on = () => setWide(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return wide;
}
