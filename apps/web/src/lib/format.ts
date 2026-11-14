// Display helpers. The live console serves the Pune deployment, so times are shown in IST and numbers
// use the Indian locale. Override with NEXT_PUBLIC_SITE_TZ / NEXT_PUBLIC_LOCALE for another deployment.

export const SITE_TZ = process.env.NEXT_PUBLIC_SITE_TZ ?? "Asia/Kolkata";
export const LOCALE = process.env.NEXT_PUBLIC_LOCALE ?? "en-IN";
export const TZ_LABEL = SITE_TZ === "Asia/Kolkata" ? "IST" : SITE_TZ;

// Station display names, registered when the station list loads (api.stations).
const NAMES = new Map<string, string>();
export function registerStationNames(list: { station_id: string; name: string }[]): void {
  for (const s of list) NAMES.set(s.station_id, s.name);
}

export function fmtTime(iso: string | null | undefined, opts: Intl.DateTimeFormatOptions = {}): string {
  if (!iso) return "-";
  return new Intl.DateTimeFormat(LOCALE, {
    timeZone: SITE_TZ,
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
    ...opts,
  }).format(new Date(iso));
}

export function fmtHour(iso: string): string {
  return new Intl.DateTimeFormat(LOCALE, { timeZone: SITE_TZ, hour: "2-digit", hour12: false }).format(new Date(iso)) + ":00";
}

export function fmtNum(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  return v.toLocaleString(LOCALE, { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtPct(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  return `${(v * 100).toFixed(digits)}%`;
}

export function prettyStation(id: string): string {
  const name = NAMES.get(id);
  if (name) return name;
  return id.replace(/^[A-Z]+_/, "").replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
}

export const MODEL_LABELS: Record<string, string> = {
  B0_persistence: "B0 Persistence",
  B1_seasonal_naive_24h: "B1 Seasonal naive (24 h)",
  B2_seasonal_naive_168h: "B2 Seasonal naive (168 h)",
  B3_climatology: "B3 Hour-of-week climatology",
  M1_ridge: "M1 Ridge",
  M2_random_forest: "M2 Random Forest",
  M3_xgboost: "M3 XGBoost",
  M4_lightgbm: "M4 LightGBM",
  M4_lightgbm_quantile: "M4 LightGBM q50 (champion)",
  M5_lstm: "M5 LSTM (Seq2Seq)",
  E1_ensemble: "E1 Ensemble",
};

export const modelLabel = (m: string) => MODEL_LABELS[m] ?? m;

export const RISK_COLOR: Record<string, string> = {
  NORMAL: "var(--ok)",
  WARNING: "var(--warn)",
  CRITICAL: "var(--crit)",
};
