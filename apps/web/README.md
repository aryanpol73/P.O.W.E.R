# P.O.W.E.R. web console (PWA)

Next.js 15 (App Router, TypeScript) operator console for the P.O.W.E.R. API.

| Page | What it shows |
|---|---|
| `/` Command Center | KPI ribbon, MapLibre city map (colour = q95 risk, size = median load, pulsing halo when critical, reroute arcs), 24 h time scrubber, open alerts, station ranking |
| `/stations/[id]` | 48 h actuals + 24 h forecast with calibrated 90% band and assumed capacity line; SHAP waterfall for any forecast hour; model switcher |
| `/arena` | Test-set leaderboard (sortable), error-vs-horizon curves, live rolling accuracy |
| `/simulator` | What-if: temperature, rain, holiday, demand multiplier; out-of-training-range warnings |
| `/insights` | SHAP drivers, error analysis, COVID-19 stress test with drift alarms, report figures |

Live updates: one WebSocket (`/ws/stream`) invalidates TanStack Query caches when a simulated hour closes.
Admin actions (play/pause/reset, alert ack) go through `src/app/api/admin/[...path]/route.ts`, which adds
`ADMIN_TOKEN` server-side so the token never reaches the browser.

## Run

```bash
npm install
npm run dev          # http://localhost:3000 (needs the API on :8000: `uv run power-api serve`)
npm run build && npm start   # production build with the Serwist service worker
```

Environment: `NEXT_PUBLIC_API_URL` (default `http://127.0.0.1:8000`), `ADMIN_TOKEN` (server-only, must match the API).

Map tiles: OpenFreeMap (free, no key). The service worker is only built in production; it precaches the app
shell, serves `/offline` for navigations without a connection, and keeps recent API reads (network-first).
