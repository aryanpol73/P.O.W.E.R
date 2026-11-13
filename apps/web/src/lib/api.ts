// Typed client for the P.O.W.E.R. API (services/api). Reads go straight to the API; state-changing
// admin calls go through /api/admin/* so the admin token stays on the server.

import { registerStationNames } from "./format";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
export const WS_URL = API_URL.replace(/^http/, "ws") + "/ws/stream";

export type Risk = "NORMAL" | "WARNING" | "CRITICAL";

export interface Deployment {
  name: string;
  display_name: string;
  city?: string;
  country?: string;
  timezone: string;
  currency?: string;
  holidays?: string;
  model_trained_on: string;
  is_twin: boolean;
  note?: string;
}

export interface Health {
  status: string;
  model_version: string;
  champion: string;
  models: string[];
  sim: { mode: string; sim_now: string; paused: boolean } | null;
  last_error: string | null;
}

export interface SimState {
  mode: "REPLAY" | "LIVE_SHADOW";
  sim_now: string;
  paused: boolean;
  speed_ticks_per_min: number;
  week_offset_hours: number;
}

export interface Kpis {
  as_of: string;
  sim_mode: string;
  current_load_kw: number;
  next_hour: { q50: number | null; lower: number | null; upper: number | null; interval_level: number; method: string };
  port_occupancy_pct: number;
  idle_port_pct: number;
  risk_status: Risk;
  stations_by_risk: Record<Risk, number>;
}

export interface Station {
  station_id: string;
  name: string;
  latitude: number;
  longitude: number;
  total_ports: number;
  rated_kw_total: number;
  transformer_kw: number;
  poi_category: string;
  is_modeled: boolean;
  risk: Risk | null;
}

export interface SnapshotStation {
  station_id: string;
  name: string;
  lat: number;
  lon: number;
  q50: number | null;
  q95: number | null;
  load_ratio_q50: number | null;
  load_ratio_q95: number | null;
  risk: Risk | null;
  is_modeled: boolean;
  is_degraded?: boolean;
}

export interface Reroute {
  from: string;
  to: string;
  shifted_kwh: number;
  distance_km: number;
  target_time: string;
}

export interface Snapshot {
  forecast_origin: string;
  h: number;
  target_time: string;
  stations: SnapshotStation[];
  reroutes: Reroute[];
}

export interface ForecastPoint {
  target_time: string;
  h: number;
  y_mean: number;
  q05: number;
  q50: number;
  q95: number;
  load_ratio_q95: number;
  risk: Risk;
}

export interface Forecast {
  station_id: string;
  model: string;
  model_version: string;
  forecast_origin: string;
  interval_level: number;
  transformer_kw: number;
  is_degraded: boolean;
  points: ForecastPoint[];
  history: { ts: string; energy_kwh: number; occ_ratio: number; idle_ratio: number | null; is_imputed: boolean }[];
}

export interface Contribution {
  feature: string;
  display: string;
  value: number | string | null;
  shap: number;
}

export interface Explanation {
  station_id: string;
  model: string;
  forecast_origin: string;
  h: number;
  target_time: string;
  base_value: number;
  prediction: number;
  contributions: Contribution[];
  narrative: string;
}

export interface Alert {
  alert_id: number;
  station_id: string;
  level: Risk;
  target_time: string;
  forecast_origin: string;
  load_ratio_q95: number;
  message: string;
  status: "open" | "acknowledged" | "resolved";
  raised_at: string;
  resolved_at: string | null;
}

export interface LeaderRow {
  wape: number;
  mase?: number;
  mae: number;
  rmse: number;
  r2: number;
  peak_mae?: number;
  wape_day_ahead_midnight?: number;
  interval_split_conformal: { picp: number; mpiw: number; winkler: number };
  interval_cqr?: { picp: number; mpiw: number; winkler: number };
  train_time_s: number | null;
  latency_ms_one_origin: number;
}

export interface Leaderboard {
  split: string;
  model_version: string;
  rows: number;
  origin_range: [string, string];
  ensemble_weights: Record<string, number>;
  leaderboard: Record<string, LeaderRow>;
}

export interface LiveArena {
  as_of: string;
  window_hours: number;
  n: number;
  models: Record<string, { n: number; wape: number; mae: number; picp: number; wape_by_bucket: Record<string, number> }>;
}

export interface HorizonRow {
  model: string;
  h: number;
  wape: number;
  mae: number;
  rmse: number;
}

export interface ScenarioPoint {
  station_id: string;
  h: number;
  target_time: string;
  q05: number;
  q50: number;
  q95: number;
}

export interface ScenarioResult {
  forecast_origin: string;
  model: string;
  baseline: ScenarioPoint[];
  scenario: ScenarioPoint[];
  delta_network_peak_kwh: number;
  alerts_in_scenario: { station_id: string; h: number; level: Risk }[];
  ood_flags: { feature: string; min: number; max: number; train_min: number; train_max: number; message: string }[];
  assumptions: string[];
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, { ...init, headers: { "content-type": "application/json", ...(init?.headers ?? {}) } });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* keep status text */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

const get = <T,>(path: string) => request<T>(`${API_URL}/api/v1${path}`);
const admin = <T,>(path: string, body?: unknown) =>
  request<T>(`/api/admin/${path}`, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  health: () => get<Health>("/health"),
  sim: () => get<SimState>("/sim/state"),
  kpis: () => get<Kpis>("/network/kpis"),
  deployment: () => get<Deployment>("/deployment"),
  stations: () =>
    get<Station[]>("/stations?modeled=false").then((list) => {
      registerStationNames(list);
      return list;
    }),
  snapshot: (h: number) => get<Snapshot>(`/network/snapshot?h=${h}`),
  forecast: (id: string, model = "champion") => get<Forecast>(`/stations/${id}/forecast?model=${encodeURIComponent(model)}&history_hours=48`),
  explain: (id: string, h?: number) => get<Explanation>(`/stations/${id}/explain${h ? `?h=${h}` : ""}`),
  alerts: (status: "open" | "all" = "open") => get<Alert[]>(`/alerts?status=${status}&limit=50`),
  leaderboard: () => get<Leaderboard>("/models/leaderboard"),
  liveArena: (hours = 168) => get<LiveArena>(`/models/live-arena?hours=${hours}`),
  horizons: () => get<HorizonRow[]>("/models/horizons?split=test"),
  shap: () => get<Record<string, unknown>>("/insights/shap"),
  errors: () => get<Record<string, unknown>>("/insights/errors"),
  stress: () => get<Record<string, unknown>>("/insights/stress"),
  simulate: (body: { temperature_delta_c: number; precipitation_mm: number | null; force_holiday: boolean; demand_multiplier: number; station_ids: string[] }) =>
    request<ScenarioResult>(`${API_URL}/api/v1/scenarios/simulate`, { method: "POST", body: JSON.stringify(body) }),
  simControl: (body: { action: "play" | "pause" | "reset" | "set_speed" | "step"; speed_ticks_per_min?: number; mode?: string }) =>
    admin<SimState | Record<string, unknown>>("sim/control", body),
  ack: (id: number) => admin<{ alert_id: number; status: string }>(`alerts/${id}/ack`),
  figure: (name: string) => `${API_URL}/figures/${name}`,
};
