"use client";

import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { Forecast } from "@/lib/api";
import { AXIS, GRID, TOOLTIP_CURSOR, TOOLTIP_LABEL_STYLE, TOOLTIP_STYLE } from "@/lib/chart";
import { fmtHour, fmtNum, fmtTime } from "@/lib/format";

interface Row {
  t: number;
  actual?: number;
  q50?: number;
  band?: [number, number];
  h?: number;
}

export function ForecastChart({ data, onPickHour }: { data: Forecast; onPickHour?: (h: number) => void }) {
  const rows: Row[] = [
    ...data.history.map((p) => ({ t: Date.parse(p.ts), actual: p.energy_kwh })),
    ...data.points.map((p) => ({ t: Date.parse(p.target_time), q50: p.q50, band: [p.q05, p.q95] as [number, number], h: p.h })),
  ];
  const origin = Date.parse(data.forecast_origin);
  const last = data.history.at(-1);
  if (last) {
    // join the forecast line to the last observation so the two read as one series
    const joint = rows.find((r) => r.t === Date.parse(last.ts));
    if (joint) joint.q50 = last.energy_kwh;
  }
  const ymax = Math.max(data.transformer_kw * 1.05, ...data.points.map((p) => p.q95), ...data.history.map((p) => p.energy_kwh));

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ComposedChart
        data={rows}
        margin={{ top: 10, right: 16, bottom: 0, left: 0 }}
        onClick={(e) => {
          const r = (e as unknown as { activePayload?: { payload: Row }[] })?.activePayload?.[0]?.payload;
          if (r?.h && onPickHour) onPickHour(r.h);
        }}
      >
        <CartesianGrid {...GRID} />
        <XAxis
          dataKey="t"
          type="number"
          scale="time"
          domain={["dataMin", "dataMax"]}
          tickFormatter={(v: number) => fmtHour(new Date(v).toISOString())}
          {...AXIS}
          minTickGap={40}
        />
        <YAxis {...AXIS} axisLine={false} width={44} domain={[0, Math.ceil(ymax)]} unit="" />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          labelStyle={TOOLTIP_LABEL_STYLE}
          cursor={TOOLTIP_CURSOR}
          labelFormatter={(v) => fmtTime(new Date(Number(v)).toISOString())}
          formatter={(value, name) => {
            if (Array.isArray(value)) return [`${fmtNum(value[0])} – ${fmtNum(value[1])} kWh`, "90% interval"];
            return [`${fmtNum(Number(value))} kWh`, name === "actual" ? "actual" : "forecast (median)"];
          }}
        />
        <Area dataKey="band" stroke="none" fill="var(--band)" isAnimationActive={false} connectNulls />
        <Line dataKey="actual" stroke="var(--text)" strokeOpacity={0.85} strokeWidth={1.6} dot={false} isAnimationActive={false} />
        <Line
          dataKey="q50"
          stroke="var(--accent)"
          strokeWidth={2.4}
          strokeLinecap="round"
          dot={false}
          activeDot={{ r: 4, fill: "var(--accent)", stroke: "var(--surface)", strokeWidth: 2 }}
          isAnimationActive={false}
          connectNulls
        />
        <ReferenceLine
          y={data.transformer_kw}
          stroke="var(--crit)"
          strokeDasharray="6 4"
          strokeWidth={1.5}
          label={{ value: `assumed capacity ${fmtNum(data.transformer_kw)} kW`, position: "insideTopRight", fill: "var(--crit)", fontSize: 11, fontWeight: 600 }}
        />
        <ReferenceLine
          x={origin}
          stroke="var(--muted)"
          strokeDasharray="2 3"
          label={{ value: "now", position: "insideTopLeft", fill: "var(--muted)", fontSize: 11, fontWeight: 600 }}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
