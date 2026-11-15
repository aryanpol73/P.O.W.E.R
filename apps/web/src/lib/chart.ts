// Shared Recharts styling. Everything reads the CSS design tokens so charts follow the theme.
import type { CSSProperties } from "react";

/** Faint horizontal gridlines that stay behind the data. */
export const GRID = { stroke: "var(--divider)", vertical: false } as const;

/** Axis line in --border, tick labels in --muted, no tick marks. */
export const AXIS = {
  stroke: "var(--border)",
  tick: { fontSize: 11, fill: "var(--muted)" },
  tickLine: false,
} as const;

export const TOOLTIP_STYLE: CSSProperties = {
  background: "var(--glass-strong)",
  backdropFilter: "blur(12px)",
  border: "1px solid var(--border)",
  borderRadius: 12,
  boxShadow: "var(--shadow-lg)",
  fontSize: 12,
  padding: "8px 12px",
  color: "var(--text)",
  fontVariantNumeric: "tabular-nums",
};

export const TOOLTIP_LABEL_STYLE: CSSProperties = { color: "var(--muted)", marginBottom: 4, fontWeight: 500 };

export const TOOLTIP_CURSOR = { stroke: "var(--muted)", strokeOpacity: 0.4, strokeDasharray: "3 3" };
