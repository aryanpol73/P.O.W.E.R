# P.O.W.E.R.
### Predictive Optimization of Wattage for EV Resources

An end-to-end, real-time machine-learning engine and installable PWA for **electric-vehicle charging demand forecasting and grid load management**.
B.Tech Credit-Based ML Course · Project Bank **#41 — Electric Vehicle Charging Demand Forecasting**.

> It forecasts hourly station energy demand (kWh) 1–24 h ahead with calibrated 90% prediction intervals and port-occupancy/idle risk. It raises transformer-overload alerts and recommends rerouting overflow demand to nearby stations. Every forecast is explained with SHAP.

---

## Highlights

- **Leakage-proof direct multi-horizon forecasting.** Features are typed by anchor (target / origin / static) and checked by a future-poisoning unit test.
- **Model arena:** seasonal-naive baselines · Ridge · Random Forest · XGBoost · LightGBM · LSTM Seq2Seq · weighted ensemble, compared with WAPE, MASE, RMSE, MAE, R², pinball, PICP, MPIW, Winkler and Diebold–Mariano tests.
- **Calibrated uncertainty:** conformalized quantile regression (CQR), plus adaptive conformal inference online.
- **Real-time pipeline:** replay / live-shadow simulation clock, Open-Meteo live weather, nine autonomous workers (forecast, arena, drift, alerts, retraining, and others).
- **PWA:** Next.js + MapLibre/deck.gl city map with a 24-hour time scrubber, reroute arcs, what-if simulator, SHAP waterfalls and Web Push alerts.
- **MCP server:** query the grid in natural language from Claude Desktop / Claude Code or the in-app copilot.

## Architecture

```
Palo Alto / ACN sessions ─┐
Open-Meteo archive ───────┼─► power_ml (offline) ─► artifacts/models/<version>
                          │                                   │
Open-Meteo forecast ──► worker agents ◄── Postgres+Timescale ─┤
OpenChargeMap ─────────┘   │  (replay, weather, forecast,     │
                           │   arena, drift, alerts, retrain) │
                           ▼                                  ▼
                    LISTEN/NOTIFY ──► FastAPI (REST + WS + serving + SHAP)
                                         │                 │
                                   Next.js PWA         MCP server ──► Claude
```
The full design is in [docs/02_system_blueprint.md](docs/02_system_blueprint.md). The methodology audit is in [docs/01_audit_and_refinement.md](docs/01_audit_and_refinement.md).

## Pune pilot deployment (digital twin)

The models are trained and scored on the Palo Alto open charging data, and every reported accuracy number
comes from that data. The live console serves Pune instead (`configs/deployments/pune.yaml`):

* **Sites:** 9 real charging-station locations in Pune from OpenStreetMap (Aundh, Baner, Deccan Gymkhana,
  Erandwane, Kothrud, Karve Nagar, Sahakar Nagar). Each site keeps the port count and rating of the training
  site it is paired with, so the capacity assumption and the alert thresholds mean the same thing.
* **Demand:** each site replays the held-out demand of its paired training site, aligned on local
  wall-clock time (08:00 Palo Alto becomes 08:00 IST). There is no public per-session Indian dataset to
  replay instead.
* **Context:** Pune's own observed weather (Open-Meteo archive), Indian and Maharashtra holidays, and IST.
  Because IST is UTC+5:30, the hourly grid sits at :30 UTC.

The serving engine maps site ids to the training ids before building features and maps them back
afterwards, so no training-site id reaches the API or the UI. Live accuracy on the console therefore
measures the models on a transferred demand profile, not on Indian drivers. Validating on Indian data is
future work.

## Repository layout

| Path | What |
|---|---|
| `packages/power_core` | Shared feature builder, splits, metrics, conformal, drift, grid logic (used by training **and** serving) |
| `ml/` | Offline pipeline: ingest → resample → tune → train → calibrate → evaluate → explain → export |
| `services/api` | FastAPI REST + WebSocket + model serving |
| `services/worker` | Autonomous agents (APScheduler + Postgres LISTEN/NOTIFY) |
| `services/mcp` | Model Context Protocol server |
| `apps/web` | Next.js 15 PWA |
| `configs/` | Data window, features, model spaces, thresholds, agent schedules |
| `notebooks/` | Narrative EDA / results notebooks (logic lives in packages) |
| `reports/` | Metrics, figures, data card, model card, final report |

## Implementation status

| Phase | Scope | State |
|---|---|---|
| P0 | Monorepo scaffold, configs, CI | ✅ done |
| P1 | Ingest (Palo Alto, ACN), anonymisation, cleaning, hourly resampling, Open-Meteo back-fill | ✅ done (run on real data pending) |
| P2 | `power_core`: anchor-typed features, panel splits, metrics, DM test, conformal, drift, grid logic + leakage tests | ✅ done |
| P3–P4 | 4 baselines + Ridge, RF, XGBoost, LightGBM (+ q05/q50/q95), LSTM, ensemble; Optuna; CQR; evaluation; SHAP; error analysis; COVID stress test | ✅ done |
| P5 | Backend: database, FastAPI REST + WebSocket, serving engine, simulator + replay / forecast / alert / weather agents | ✅ done (SQLite; PostgreSQL supported but not yet run) |
| P6–P7 | Next.js PWA, MCP server, retrain / drift / discovery agents | ⏳ next |

What works today (Python 3.12 + `uv`):

```bash
uv sync                                   # install the locked environment
uv run pytest -q                          # 38 tests, incl. leakage proofs (-m leakage)
uv run power-ml ingest   --source palo_alto   # needs data/raw/palo_alto_sessions.csv
uv run power-ml resample --source palo_alto
uv run power-ml weather  --source palo_alto   # Open-Meteo archive, cached
uv run power-ml panel    --source palo_alto
uv run power-ml features --source palo_alto   # data/processed/features_palo_alto_{train,val,test,ood}.parquet
uv run power-ml tune     --source palo_alto   # Optuna -> reports/tuning.json
uv run power-ml train    --source palo_alto   # artifacts/models/<version>/
uv run power-ml train-lstm --source palo_alto # needs: uv sync --extra dl
uv run power-ml evaluate --source palo_alto --split test
uv run power-ml analyze  --source palo_alto   # SHAP, error analysis, COVID stress test
```

Backend (after the steps above):

```bash
uv run power-api seed                     # load stations, hourly history, weather into data/power.db (SQLite)
uv run power-api serve                    # http://127.0.0.1:8000/docs ; simulator runs inside the API
```
Then `POST /api/v1/sim/control {"action": "play"}` starts the replay clock; `WS /ws/stream` streams events.
Set `DATABASE_URL=postgresql+asyncpg://...` and `uv sync --extra postgres` for PostgreSQL.

## Prerequisites

- Docker 24+ and Docker Compose v2
- Python 3.12 with [`uv`](https://docs.astral.sh/uv/) (only for running training outside Docker)
- Node.js 20+ with `pnpm` 9 (only for frontend development outside Docker)
- Free API keys: **ACN-Data** token (if using ACN) and **OpenChargeMap** key (optional, context stations). Open-Meteo needs no key (non-commercial use).

## Quick start (one command after data)

```bash
git clone https://github.com/aryanpol73/P.O.W.E.R.git
cd P.O.W.E.R
cp .env.example .env            # fill keys if you have them
make data                       # download datasets + weather archive, verify sha256
make train                      # full offline pipeline (≈ 20–60 min on a laptop CPU)
make up                         # db + api + worker + web + mcp
```
Open **http://localhost:3000** (PWA) and **http://localhost:8000/docs** (API).

## Environment variables (`.env.example`)

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://power:power@db:5432/power` | Postgres + TimescaleDB |
| `MODEL_VERSION` | `latest` | Registry version served |
| `SIM_MODE` | `REPLAY` | `REPLAY` · `LIVE_SHADOW` · `LIVE` |
| `POWER_DEPLOYMENT` | `pune` | live-console city profile in `configs/deployments/`; `none` = training sites |
| `SIM_START` | first test-split hour | Replay start point |
| `ADMIN_TOKEN` | — | Required for sim control / ack / retrain |
| `MCP_READONLY_TOKEN` | — | Token the MCP server uses against the API |
| `ACN_API_TOKEN` | — | ACN-Data (optional secondary dataset) |
| `OCM_API_KEY` | — | OpenChargeMap (optional context stations) |
| `OPEN_METEO_BASE` | `https://api.open-meteo.com/v1` | Forecast API |
| `OPEN_METEO_ARCHIVE` | `https://archive-api.open-meteo.com/v1` | Archive API |
| `VAPID_PUBLIC_KEY` / `VAPID_PRIVATE_KEY` | — | Web Push (`make vapid` generates) |
| `ANTHROPIC_API_KEY` | — | Optional in-app copilot; feature disabled when empty |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Frontend → API |
| `SEED` | `42` | Global determinism |

## Reproducible training

Every stage is deterministic (`SEED=42`, pinned `uv.lock` / `pnpm-lock.yaml`, sha256-verified data, MLflow-logged git SHA + data hash).

```bash
make data         # power-ml ingest  -> data/raw, data/manifest.json
make features     # power-ml resample + features -> data/processed/*.parquet
make leakage-test # pytest -m leakage (future-poisoning, split purity, vintage checks)
make tune         # Optuna (PanelTimeSeriesSplit inside train only)
make train        # fit all models, early stopping on val
make calibrate    # CQR per horizon bucket on val
make evaluate     # ONE pass on test -> reports/metrics_test.json, dm_tests.csv
make ablations    # recursive vs direct, Tweedie vs L2, no-weather, DF sensitivity
make explain      # SHAP global + error analysis figures
make stress-test  # COVID-period out-of-distribution evaluation
make export       # artifacts/models/<version>/manifest.json
```

The test split is evaluated **once** by `make evaluate`. Re-running it with the same artefacts gives identical numbers.

## Datasets

| Dataset | Use | Notes |
|---|---|---|
| City of Palo Alto — EV Charging Station Usage | Primary (multi-site, coordinates, charging vs connected time) | Modelling window pre-COVID; exact counts in `data/manifest.json` and the data card |
| Caltech ACN-Data (Lee et al., 2019) | Secondary: out-of-domain workplace test, idle-occupancy study | Requires free token |
| Open-Meteo Archive / Forecast | Weather back-fill / live | Archive = observed (training); forecast = live serving |
| OpenChargeMap | Context stations on map (cold-start priors only) | Not used in accuracy metrics |

Transformer capacities are a **synthetic planning assumption** (`C = DF × Σ port kW`, DF = 1.0 by default and configurable). No public dataset publishes them.

## Tests

```bash
make test         # pytest (unit + leakage + API with testcontainers) + vitest
make e2e          # Playwright: viva demo flow against the compose stack
make lint         # ruff, mypy, eslint, tsc
```

## Using the MCP server

Add to Claude Desktop / Claude Code MCP config:
```json
{ "mcpServers": { "power": {
  "command": "uv",
  "args": ["run", "--directory", "services/mcp", "power-mcp", "--transport", "stdio"],
  "env": { "POWER_API_URL": "http://localhost:8000", "POWER_API_TOKEN": "<MCP_READONLY_TOKEN>" } } } }
```
Try: *"Which stations are at risk in the next 24 hours, and what drives the peak at the worst one?"* or *"Simulate a +8 °C heatwave across the network."*

## Viva demo flow (≈ 8 minutes)

1. **Data & Methodology page:** sources, exact record counts, pre-COVID window, split diagram, green leakage tests.
2. **Model Arena:** test leaderboard vs seasonal-naive baseline (MASE < 1), error-vs-horizon curve showing recursive vs direct, DM significance, ensemble vs components with training time.
3. **Command Center:** press **Play** (6 ticks/min). Watch the KPI ribbon, map colours and the live arena update over WebSocket.
4. **Scrub the time slider** +1 → +24 h. Halos move from workplace to downtown sites; a CRITICAL halo pulses and reroute arcs appear.
5. **Click the critical station:** 48 h + 24 h chart with 90% band and transformer line, SHAP waterfall with plain-language narrative.
6. **What-If:** +10 °C and holiday. The curve recalculates and the OOD badge shows where tree models stop extrapolating.
7. **Ops:** switch to `LIVE_SHADOW` (live Open-Meteo weather). Show drift panel and champion–challenger gate.
8. **Copilot / Claude Desktop via MCP:** ask for a grid briefing.

## Responsible AI

Session and user identifiers are dropped or hashed at ingestion; no MAC addresses, user IDs or zip codes are stored. Forecasts are **decision support**, not autonomous grid control. Reroute suggestions are advisory and bounded by assumed driver willingness. The model learns from where chargers already exist, so planning use must be paired with equity analysis (see the report, Chapter 10).

## Team & contributions

| Member | Role | Owned modules |
|---|---|---|
| _Name 1_ | Data & feature engineering | `ml/ingest`, `power_core.features`, data card |
| _Name 2_ | Modelling & evaluation | `ml/models`, `tune`, `evaluate`, `ablations` |
| _Name 3_ | Backend & real-time | `services/api`, `services/worker`, DB |
| _Name 4_ | Frontend, XAI & MCP | `apps/web`, `ml/explain`, `services/mcp` |

## Citation / references

Lee, Li & Low (2019) ACN-Data; Amezquita et al. (2024) *Energies* 17:3396; Zhu et al. (2019) *Energies* 12:2692; Skala et al. (2023) *Energies* 16:4093; Shern et al. (2024) *Energies* 17:5772; Adnane et al. (2023) *Energies* 16:4897; Romano et al. (2019) CQR; Gibbs & Candès (2021) ACI.

## License

Code: MIT. Datasets remain under their providers' terms.
