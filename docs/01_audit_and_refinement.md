# P.O.W.E.R. — Deliverables 1 & 2
## Audit of the Gemini 4 Argon plan, and the refined Feature Schema & Model Suite

**Project #41 · Electric Vehicle Charging Demand Forecasting**
*Predictive Optimization of Wattage for EV Resources*

Sources audited: *50 Real-World Machine Learning Projects* (course bank: §8 rubric, §9 minimum requirements, §10 project-type guidance, Final Recommendation), *EV_Charging_Demand_Forecasting_Project_Spec.docx* (Gemini spec), *ev station forecast research papers.pdf* (literature summary), and the prompt's Sections 1–8.

Notation used throughout:

| Symbol | Meaning |
|---|---|
| `s` | station (site) index, `s = 1..S` |
| `T` | **forecast origin**: the last fully observed hour when a forecast is issued |
| `h` | horizon, `h ∈ {1,…,24}` hours |
| `t = T + h` | target hour being forecast |
| `y_{s,τ}` | energy dispensed at station `s` during hour `τ` (kWh). Numerically equal to mean power in kW over that hour |
| `N_s` | number of ports at station `s` |
| `P_s^port` | total nameplate power of all ports at `s` (kW) |
| `C_s` | assumed transformer capacity for `s` (kW). **Synthetic assumption**, see F12 |
| `q̂_α` | predicted α-quantile |

---

# DELIVERABLE 1 — Audit, Verification & Rating

## 1.1 What Gemini 4 Argon got right

These parts are correct and should stay:

1. **Problem framing.** Treating EV load as stochastic, peaky and constrained by hardware matches ACN-Data (Lee et al., 2019) and is the right way to argue the problem matters for the 10-mark criterion.
2. **Physical anomalies.** Idle occupancy (`doneChargingTime < disconnectTime`) and right-censoring at full occupancy are real and well documented. Most student projects ignore them.
3. **Algorithm coverage.** Baseline + Ridge + RF + Gradient Boosting + a sequence model covers every algorithm family the course lists for #41 (Regression, Random Forest, Gradient Boosting, time-series forecasting) and exceeds the "≥ 3 algorithms + baseline" minimum.
4. **Leakage basics.** Chronological split, no shuffling, fit-on-train transformers, `shift(1)` before rolling, and a gap in CV are all correct *as principles*.
5. **Session → hourly proration.** Spreading `kWhDelivered` across the active charging window instead of dumping it in the start hour is correct and often missed.
6. **Probabilistic output.** Prediction intervals are the right product for transformer protection: an operator sizes against p95, not the mean.
7. **Spatial framing + XAI.** A map, a time scrubber and local SHAP explanations are exactly what makes a viva memorable.

## 1.2 Verification of specific technical claims

| # | Claim (source) | Verdict | Correction |
|---|---|---|---|
| V1 | "Caltech ACN-Data: 50,000+ sessions" (prompt §4.1) | **Unverified / inconsistent** | The PDF itself quotes Lee et al. as "over 30,000 sessions" (as of 2019). The live ACN API has more today across `caltech`, `jpl`, `office001`, but **you must report the exact count you download**. ACN-Data also requires a free API token (registration). |
| V2 | "UrbanEV / City of Palo Alto Open EV Charging Dataset" (prompt §4.1) | **Wrong — these are two different datasets** | *City of Palo Alto EV Charging Station Usage* is a US session-level log (≈2011–2020, dozens of named stations with lat/lon, `Total Duration` **and** `Charging Time`). *UrbanEV* is a Shenzhen benchmark of zone-level hourly occupancy/volume with prices and weather. Treat them separately. |
| V3 | "50k rows ≈ 5.7 years hourly for one site" (PDF) | Arithmetic correct (50,000 / 8,760 = 5.7) | It exposes an ambiguity the report must resolve: **is "50,000 records" sessions or hourly rows?** The course asks for "number of records". Report both: raw sessions *and* modelled station-hours. |
| V4 | "XGBoost safer on 50k rows because of explicit L1/L2 Hessian regularization" (PDF/spec) | **Overstated** | LightGBM has the same knobs (`lambda_l1`, `lambda_l2`, `min_sum_hessian_in_leaf`, `min_data_in_leaf`, `max_depth`). Level-wise vs leaf-wise matters mainly *before* tuning. After Optuna tunes both, the difference is an empirical question, which is also what Amezquita et al. (2024) report: near-identical accuracy. |
| V5 | "LightGBM natively partitions categoricals; XGBoost needs one-hot" (PDF) | **Outdated** | XGBoost ≥ 1.6 supports native categoricals (`enable_categorical=True`, `tree_method="hist"`), and `hist` is the default since 2.0. Both handle `station_id` natively. |
| V6 | "LightGBM trains 3–5× faster → better for real-time retraining" (PDF) | True in general, **irrelevant at this scale** | Both train on ≤ 5M rows in seconds to minutes on a laptop. Retraining speed should not decide the champion; accuracy and calibration should. |
| V7 | "Amezquita et al.: GBDTs beat RF, MLR, LSTM" | Consistent with the PDF summary | Cite it as one study's finding, not a law. Your own test-set table decides. |
| V8 | "Skala et al. (2023) — interval forecasting" | Correct citation, **indirect relevance** | It is **household** load with EVs, not station load. Use it to motivate intervals, not as a station benchmark. |
| V9 | "90% prediction intervals" (§2.2) vs "shaded 10th–90th percentile bands" (§7.1) | **Internal contradiction** | The 10th–90th band is an **80%** interval. A 90% interval needs the 5th and 95th percentiles. |
| V10 | "Compute lags in < 50 ms" | Feasible | Reading 168–672 rows per station from an indexed hypertable and computing ~30 features takes a few ms. Not a risk. |

## 1.3 Scorecard (out of 10)

| Dimension | Score | Why it is not higher |
|---|---|---|
| **1. Rubric alignment (100-mark optimisation)** | **7.5 / 10** | Covers all 7 criteria and the §9 minimums. Misses the course's §10 ensemble rule ("compare the ensemble against its individual component models… training time and interpretability"), the Final Recommendation's *proposal* step, statistical significance between models, and an individual-contribution plan. It also over-invests in UI: the course says assessment should reward "sound methodology… not just a polished user interface". |
| **2. ML & statistical rigour** | **5.5 / 10** | Three critical defects: horizon leakage in the 24-h forecast (F1), a panel-data CV bug (F2) and a COVID regime break inside the test split (F3). Also MAPE on zero-heavy data (F5), an 80%-vs-90% interval mix-up with no coverage guarantee (F4), and oracle weather (F6). |
| **3. Real-time systems feasibility** | **6.0 / 10** | The components are right (scheduler, buffer, serving, drift), but it never says who runs inference after each tick. Replay is paired with live weather from a different year (F8), cold-start and missing ticks are unhandled (F14), the drift test fires on the daily cycle (F11), and APScheduler is placed inside multi-worker Uvicorn (F13). |
| **4. UI / geospatial / XAI impact** | **7.5 / 10** | Strong concepts. But the what-if sliders extrapolate outside the training data (F9), network intervals are summed incorrectly (F10), transformer ratings do not exist in any dataset (F12), ACN has only 3 geographic sites (F7), and Mapbox needs a billed token. |
| **Overall (weighted 30/30/20/20)** | **6.6 / 10** | A good research brief with a solid literature base. It is not yet an executable, leakage-proof specification. |

## 1.4 Flaws, edge cases & upgraded solutions

Severity: **C** = would cost marks or invalidate results · **H** = high · **M** = medium · **L** = low.

### F1 (C) — Horizon leakage in the 24-hour forecast
**Problem.** The schema defines `lag_1h … lag_3h`, `rolling_*` and `ramp_rate` relative to the **target** hour `t`. That is valid only for `h = 1`. To forecast `t = T + 24` at origin `T`, `y_{t−1} = y_{T+23}` is in the future. Training with target-anchored short lags and then serving 24 steps ahead is train/serve skew. Run as "recursive" instead, it becomes error accumulation. Section 5.5 ("evaluate direct vs. recursive") leaves this undecided. That is the single most important design decision in the project.

**Upgrade.** Split every feature into exactly one anchor class and enforce it in code:
- **Target-anchored, known in advance:** calendar of `t`, weather forecast for `t`, and **seasonal lags with lag ≥ h**: `y_{t−24}`, `y_{t−48}`, `y_{t−168}`. These are always observed when `h ≤ 24`.
- **Origin-anchored state:** `y_T, y_{T−1}, y_{T−2}`, rolling statistics up to `T`, and occupancy at `T`.
- **Horizon** `h` as an explicit feature.

This is the **Direct multi-horizon, horizon-as-feature** strategy (§2.3). A unit test enforces it (the "future-poisoning test", §2.6).

### F2 (C) — `TimeSeriesSplit(gap=24)` is wrong on panel data
**Problem.** scikit-learn's `TimeSeriesSplit` splits by **row position**. With `S` stations stacked per timestamp, `gap=24` removes 24 *rows*: with 10 stations that is 2.4 hours, not 24. Fold boundaries can also cut through a timestamp, so station A at 14:00 is in train while station B at 14:00 is in validation.

**Upgrade.** Write a custom `PanelTimeSeriesSplit` that splits on **unique timestamps** and applies the gap in **hours**. Set the gap to `max(H) = 24 h` plus the longest origin-window feature footprint:

```
gap_hours = H_max + 0  (origin features end at T, so a 24 h gap suffices for horizon purity)
embargo   = 168 h       (recommended: removes weekly autocorrelation bleed between folds)
```

Use `n_splits = 5`, expanding window, splitting only *inside* the 70% training block.

### F3 (C) — COVID-19 regime break inside a chronological split
**Problem.** Both candidate US datasets run into 2020/2021. Workplace and downtown charging collapsed from March 2020. A plain 70/15/15 cut over the full range puts validation and test inside lockdown: the model is tuned on one world and tested on another. Results become uninterpretable, and in the viva it looks like you did not look at the data.

**Upgrade.**
- **Primary experiment window: pre-COVID.** For example 2017-01-01 → 2020-02-29 for Palo Alto, or 2018-05-01 → 2020-02-29 for ACN Caltech. Split 70/15/15 inside this window.
- **Secondary "stress test":** evaluate frozen models on Mar–Jun 2020 as an explicit out-of-distribution case study. This feeds the Novelty & Insights criterion and lets the drift agent show it detects a real regime change.
- Also document known **pricing regime changes**. Caltech moved from free to paid charging (~$0.12/kWh) in 2018 per Lee et al. Verify the date and encode it as a step dummy.

### F4 (H) — Interval definition, crossing and coverage
**Problem.** The spec mixes 80% and 90% intervals (V9). Independently trained quantile models can **cross** (`q̂_0.05 > q̂_0.5`). Raw quantile GBDTs are typically **under-covered** on held-out data, and nothing guarantees the advertised 90%.

**Upgrade.** **Conformalized Quantile Regression (CQR)** calibrated on the validation block, one calibration per horizon bucket (Mondrian CQR). Add **Adaptive Conformal Inference (ACI)** online in the real-time loop. Sort the quantiles, then clip to `[0, P_s^port]`. Report PICP (coverage) and MPIW (width). Formulas in §2.4.

### F5 (H) — MAPE is undefined on zero-demand hours
**Problem.** Night-time and weekend station-hours are frequently exactly 0 kWh. MAPE divides by `y`, so it explodes or is undefined, and the course asks for "metrics appropriate to the task".

**Upgrade.** Make **WAPE** (= ΣAbsErr / Σy) and **MASE** (scaled by the seasonal-naive error) primary. Keep MAE, RMSE, R², and add **peak-hour MAE** and probabilistic metrics. Report MAPE only on hours with `y ≥ 1 kWh`, labelled as such. Formulas in §2.5.

### F6 (H) — Oracle weather (train/serve skew)
**Problem.** Training uses **observed** historical weather (Open-Meteo archive/reanalysis). Live serving uses **forecast** weather, which is noisier and gets worse with lead time. The trained model over-trusts weather.

**Upgrade.**
1. Name the assumption explicitly in the report ("perfect-weather-forecast upper bound").
2. **Forecast-error augmentation:** during training, add `ε ~ N(0, σ_T(h)²)` to temperature, with `σ_T(h) = 0.8 + 0.05·h` °C (a conservative 1–24 h error band; document it as an assumption). Do the same for precipitation as a multiplicative log-normal.
3. **Weather ablation:** train with and without weather. If weather adds < 1% WAPE (likely in mild Palo Alto), say so honestly. That is a real, gradeable insight.
4. The Open-Meteo *Historical Forecast API* only archives forecasts from about 2022 onward, so it cannot back-fill 2017–2020. Note this as the reason for (2).

### F7 (H) — Dataset choice and the meaning of "station"
**Problem.** "Station" is never defined. In ACN-Data a `stationID` is one EVSE port (~50+ at Caltech). Per-port hourly demand is mostly zeros: very sparse, hard to forecast and useless for a city map. ACN also has only 3 geographic sites (Pasadena, La Cañada, Silicon Valley), which makes a weak map.

**Upgrade (recommended data strategy):**
- **Primary: City of Palo Alto EV Charging Station Usage.** It has many named stations with coordinates within a few km of each other (ideal for map + 3 km rerouting), `Charging Time` vs `Total Duration` (idle occupancy is computable directly), and plenty of station-hours.
- **Modelling unit = site.** Group ports by street location/garage (e.g., all `"PALO ALTO CA / HAMILTON #n"` → `HAMILTON`). Each site is one forecast series.
- **Secondary: ACN-Data (Caltech)** for an out-of-domain workplace test and the idle-occupancy deep dive (`doneChargingTime`).
- Record exact counts at ingestion: raw sessions, sessions after cleaning, sites, ports, modelled station-hours.

### F8 (H) — Replay vs live data from different years
**Problem.** Replaying 2019 demand while fetching **today's** 2026 weather and calendar means the model sees weather and holidays that never co-occurred with that demand. The live accuracy shown in the arena is then meaningless.

**Upgrade.** Two explicit sim modes in one `sim_state` table:
- **`REPLAY` (default for viva accuracy):** a simulated clock walks through the held-out test period; weather comes from the **archive** for that sim timestamp. Live arena metrics are genuine out-of-sample errors.
- **`LIVE_SHADOW` (default for the real-time story):** historical demand is **re-stamped** by a whole number of weeks, `Δ = ⌈(now − t_hist)/168 h⌉·168 h`, so weekday and hour line up with today. Weather and forecasts are live from Open-Meteo. The UI labels it "historical demand pattern, live weather".

### F9 (M) — What-if simulator extrapolation
**Problem.** Tree models **cannot extrapolate**: a +10 °C shock beyond the observed maximum gives the same output as the edge value. A ToU-tariff slider does nothing if the training data never contained tariff variation, or produces artefacts. An examiner who drags the slider and sees nothing happen will ask why.

**Upgrade.**
- Each scenario returns `ood_flags` when any shocked feature leaves the training **[p1, p99]** range. The UI shows "outside training data" badges.
- Physically motivated hinge features (`cold_stress`, `heat_stress`, §2.2) plus **monotone constraints** in LightGBM/XGBoost (non-decreasing in `heat_stress` and `cold_stress`) give a directionally sensible response.
- The tariff slider is implemented **only** if a documented pricing step exists in the data (F3). Otherwise it becomes a rule-based **elasticity overlay** (`ŷ' = ŷ·(1 + ε·Δp/p)`, with ε taken from published ACN free→paid behaviour). It is clearly labelled as a scenario assumption, not a model output.

### F10 (M) — Summing intervals across stations
**Problem.** "Network predicted peak + 90% CI" cannot be built by summing station quantiles: `Σ_s q̂_0.95,s ≠ q̂_0.95(Σ_s y_s)` because of diversification.

**Upgrade.** Network point forecast = `Σ_s ŷ_s` (coherent bottom-up). Network interval = **separately conformal-calibrated** on validation residuals of the aggregate.

### F11 (M) — KS-test drift on raw load
**Problem.** A KS-test comparing "recent load" with "training load" fires every few hours, because the recent window samples a different part of the daily and weekly cycle.

**Upgrade.**
- **Feature drift:** Population Stability Index (PSI) per feature on **hour-of-week-standardised values**: each value becomes `z = (x − μ_how)/σ_how` using the training reference's mean and std for its hour-of-week, then one PSI compares the last 7 days with the reference. (A separate PSI per bucket would be noise, because a 7-day window holds only one sample per hour-of-week.)
- **Concept drift:** **Page-Hinkley** (or ADWIN) on the standardised residual stream `z_t = (y_t − ŷ_t)/σ̂_h`, plus rolling-WAPE ratio vs the validation WAPE.
- Thresholds: PSI > 0.2 = drift; Page-Hinkley λ tuned so that the false-alarm rate on validation is < 1/month.

### F12 (M) — Transformer capacity is not in any dataset
**Problem.** Neither Palo Alto nor ACN publishes distribution-transformer ratings, so ">85% of transformer rating" has no ground truth.

**Upgrade.** Define `C_s = DF · P_s^port` with a documented **diversity factor** `DF` (default 0.7, configurable per station), state it as a **synthetic planning assumption** in the data card, and run a sensitivity table of alert counts for DF ∈ {0.5, 0.7, 0.9}. Also note that hourly kWh is **mean** power: intra-hour peaks are higher, so alerts are evaluated on the **upper quantile** `q̂_0.95`.

### F13 (M) — Serving-stack pitfalls
- **APScheduler inside Uvicorn with `--workers > 1`** starts every job once per worker. → Run a **separate `worker` process** (single instance) and keep the API stateless.
- **Redis is unnecessary.** Fan-out from the worker to API WebSockets uses **PostgreSQL `LISTEN/NOTIFY`**, and the database is already there.
- **Mapbox GL** needs an access token and billing. → **MapLibre GL JS** (open-source fork) with free vector tiles + **deck.gl** for arcs and halos.
- **PyTorch in the serving image** adds ~1 GB. → Export the LSTM to **ONNX**, serve with `onnxruntime`.
- **SHAP latency:** TreeSHAP is fast for GBDTs (ms per row) but slow for a deep 500-tree RF. → Live local SHAP only for the champion GBDT; RF/LSTM explanations precomputed offline.

### F14 (M) — Missing ticks and cold-start stations
**Missing ticks.** If an hour's meter value is missing, `y_T` is NaN and lag features break.
→ Policy: impute with the seasonal profile (`ŷ = same_how_4w_mean`) and set `is_imputed = 1`, which is stored and visible. If more than 6 consecutive hours are imputed for a station, its forecast **falls back to seasonal-naive** and it is flagged `DEGRADED`. Weather gaps: carry the last forecast vintage forward up to 6 h, then use the climatology for that hour-of-year.

**Cold-start stations** (new OpenChargeMap pins or new ports): there are no lags.
→ A **prior model** (LightGBM without autoregressive features: calendar + weather + `poi_category` + `N_s` + `P_s^port`). It is blended with the station model as history accumulates:
```
ŷ = w·ŷ_station + (1 − w)·ŷ_prior,   w = n / (n + k),   n = hours of history, k = 336 (2 weeks)
```
Stations discovered via OpenChargeMap but absent from training are shown as **grey "context" pins** with prior estimates only, never mixed into accuracy metrics.

### F15 (L) — Energy proration detail
Spreading kWh uniformly over `[connectionTime, doneChargingTime]` is reasonable for ACN. For Palo Alto, use `[start, start + Charging Time]`. Both are better than `[start, end]`, which smears energy into idle time. Add a check that charging power stays physical: `kWh / charging_hours ≤ 1.1 × port_kW`. Rows that violate it are flagged and capped. Always handle **DST**: store UTC, derive calendar features in `America/Los_Angeles`.

### F16 (L) — Censoring acknowledged but never used
→ Make it measurable: `is_saturated_{s,τ} = 1[occupied_ports_{s,τ} ≥ N_s]` (from reconstructed occupancy). (a) Report all metrics separately on saturated vs unsaturated hours. (b) Use `saturation_hours_24h` as a feature. (c) State in the limitations that saturated hours give a **lower bound** on latent demand. Tobit/censored-likelihood regression is listed as future work.

### F17 (L) — Rubric hygiene gaps
Add: the faculty **proposal** (Final Recommendation), **Diebold–Mariano** significance tests between the top models, **ensemble vs component** comparison with **training time** (§10), an **individual contribution matrix**, a **data card** and a **model card**.

## 1.5 Summary of the upgrade

| Area | Gemini 4 Argon | P.O.W.E.R. upgrade |
|---|---|---|
| Multi-step | Undecided (direct vs recursive) | Direct multi-horizon, horizon-as-feature, strict anchor classes + leakage unit test |
| CV | `TimeSeriesSplit(gap=24)` | `PanelTimeSeriesSplit` on timestamps, 24 h gap + 168 h embargo |
| Window | Full dataset span | Pre-COVID main experiment + COVID OOD stress test |
| Intervals | Quantile LightGBM, 80/90 ambiguous | CQR (Mondrian by horizon) + online ACI; PICP/MPIW/Winkler reported |
| Metrics | RMSE/MAE/MAPE/R²/Pinball | + WAPE, MASE, peak-hour MAE, PICP, MPIW, Winkler, Diebold–Mariano |
| Loss | Squared error | Tweedie objective (zero-inflated, non-negative) vs L2 ablation |
| Weather | Observed = oracle | Forecast-error augmentation + with/without ablation |
| Data | ACN or "UrbanEV/Palo Alto" | Palo Alto site-level (primary) + ACN (OOD, idle study) |
| Real-time | Replay + live weather mixed | `REPLAY` vs `LIVE_SHADOW` modes with week-aligned restamping |
| Drift | KS on raw load | PSI on hour-of-week-standardised features + Page-Hinkley on residuals |
| Alerts | > 85% of rating | q̂_0.95 vs `C_s = DF·P_port` (assumption) + hysteresis |
| Infra | APScheduler in API, Redis, Mapbox | Separate worker, Postgres LISTEN/NOTIFY, MapLibre + deck.gl, ONNX LSTM |

---

# DELIVERABLE 2 — Feature Schema & Model Suite Refinement

## 2.1 Targets

| Target | Definition | Role |
|---|---|---|
| `y = energy_kwh` | `y_{s,τ} = Σ_{sessions i} E_i · \|[a_i, b_i) ∩ [τ, τ+1h)\| / (b_i − a_i)` where `E_i` is kWh delivered, `a_i` charging start, `b_i` charging end (`doneChargingTime` or start + `Charging Time`) | **Primary.** All arena models |
| `occ_ratio` | `occ_{s,τ} = (1/N_s)·Σ_i \|[c_i, d_i) ∩ [τ, τ+1h)\| / 1h`, where `[c_i, d_i)` is connect → disconnect | Secondary (LightGBM only). Port utilisation % |
| `idle_ratio` | `(Σ_i \|[b_i, d_i) ∩ [τ,τ+1h)\|) / (Σ_i \|[c_i, d_i) ∩ [τ,τ+1h)\| + ε)` | Secondary. Overstay / idle risk for pricing nudges |
| `is_saturated` | `1[occ_{s,τ}·N_s ≥ N_s − 0.5]` | **Evaluation stratifier only, never a feature at time t** |

## 2.2 Audited feature table

Legend: **K** = keep, **M** = modify, **R** = remove, **A** = add. Anchor: **Tgt** = known for target hour `t` in advance, **Org** = state at origin `T`, **Sta** = static. Models: R = Ridge, F = RF, G = XGB/LGBM, L = LSTM, P = cold-start prior.

### Group 0 — Horizon
| Feature | Dec. | Formula | Anchor | Models | Rationale |
|---|---|---|---|---|---|
| `h` | **A** | `h = t − T ∈ {1..24}` | — | R F G | Lets one direct model learn horizon-dependent reliance on recent state |

### Group A — Calendar (all computed in local time `America/Los_Angeles`)
| Feature | Dec. | Formula | Anchor | Models | Rationale |
|---|---|---|---|---|---|
| `hour_sin`, `hour_cos` | K | `sin/cos(2π·hour(t)/24)` | Tgt | R F G L P | Continuity across midnight |
| `dow_sin`, `dow_cos` | K | `sin/cos(2π·dow(t)/7)` | Tgt | R F G L P | Weekly cycle |
| `how_f{k}_sin/cos`, k=1..3 | **A** | `sin/cos(2πk·how(t)/168)`, `how = 24·dow + hour` | Tgt | R | Ridge cannot form hour×weekday interactions on its own |
| `how_onehot` (167 dummies) | **A** | `1[how(t) = j]` | Tgt | R | Exact weekly profile for the linear model (ridge-penalised) |
| `month_sin`, `month_cos` | **M → `doy_sin/cos`** | `sin/cos(2π·doy(t)/365.25)`, K=1 only | Tgt | R F G | With ~3 years of data, monthly dummies overfit. One smooth annual harmonic is enough |
| `is_weekend` | K | `1[dow(t) ∈ {Sat,Sun}]` | Tgt | R F G L P | Cheap, very strong for Ridge |
| `is_holiday` | K | `holidays.US(subdiv="CA")` on date(t) | Tgt | R F G L P | Holiday collapse of workplace demand |
| `is_bridge_day` | **A** | weekday between a holiday and a weekend | Tgt | F G | Captures "long weekend" effects |
| `peak_tariff_regime` | **M → `pricing_era`** | step dummy `1[t ≥ t_price_change]` **only if** a documented price change exists in the window | Tgt | R F G | A fixed ToU schedule is a deterministic function of hour + weekday: zero new information. Only a regime change adds signal |

### Group B — Autoregressive lags
| Feature | Dec. | Formula | Anchor | Models | Rationale |
|---|---|---|---|---|---|
| `lag_1h`, `lag_2h`, `lag_3h` | **M → `y_org_0/1/2`** | `y_T, y_{T−1}, y_{T−2}` | Org | R F G | Re-anchored to origin (F1). With `h` as a feature, trees learn how quickly recent state decays |
| `lag_24h` | **M (re-anchored)** | `y_{t−24}` (valid since 24 ≥ h) | Tgt | R F G | Diurnal persistence, the strongest single feature in practice |
| `lag_48h` | K (re-anchored) | `y_{t−48}` | Tgt | F G | Robustness when yesterday was anomalous |
| `lag_168h` | K (re-anchored) | `y_{t−168}` | Tgt | R F G | Same hour last week |
| `same_hour_7d_avg` | **M** | `(1/7)·Σ_{k=1..7} y_{t−24k}` | Tgt | R F G | Smoothed daily profile; valid for h ≤ 24 |
| `same_how_4w_mean` | **A** | `(1/4)·Σ_{k=1..4} y_{t−168k}` | Tgt | R F G P* | Robust weekly profile; also the imputation value for missing ticks |
| `hours_since_nonzero` | **A** | `T − max{τ ≤ T : y_τ > 0}` | Org | F G | Sparse sites: "how long has it been dead" |

### Group C — Rolling & volatility (computed on `y_{≤T}`, i.e. `shift(0)` at origin = `shift(1)` relative to `T+1`)
| Feature | Dec. | Formula | Anchor | Models | Rationale |
|---|---|---|---|---|---|
| `rolling_mean_3h` | **R** | — | — | — | Exactly `(y_org_0 + y_org_1 + y_org_2)/3`: perfect collinearity with Group B |
| `rolling_mean_6h` | **R** | — | — | — | Near-duplicate of `ewma` (ρ > 0.95 expected) |
| `rolling_mean_24h` | K | `(1/24)·Σ_{k=0..23} y_{T−k}` | Org | R F G | Daily level |
| `rolling_std_6h` | **R** | — | — | — | n = 6 with many zeros is a very noisy estimator |
| `rolling_std_24h` | K | `std(y_{T−23..T})` | Org | F G | Volatility regime; also feeds interval width |
| `rolling_max_24h` | K | `max(y_{T−23..T})` | Org | F G | Peak envelope |
| `ewma_6h` | **M** | `e_T = α·y_T + (1−α)·e_{T−1}`, α = 0.3 (span ≈ 5.7 h), seeded with 24 h mean | Org | R F G | Replaces the 3 h/6 h means |
| `load_ramp_rate_1h` | **R** | — | — | — | `y_T − y_{T−1}` is a linear combination of existing lags: singular for Ridge, redundant for trees |
| `level_28d` | **A** | `(1/672)·Σ_{k=0..671} y_{T−k}` | Org | R F G | Slow adoption / level shift. Trees cannot extrapolate trend, so give them the level |
| `level_ratio` | **A** | `rolling_mean_24h / (level_28d + 0.1)` | Org | F G | "Today vs this month": an anomaly signal |

### Group O — Occupancy state (new; operationalises idle occupancy and censoring)
| Feature | Dec. | Formula | Anchor | Models | Rationale |
|---|---|---|---|---|---|
| `active_sessions_T` | **A** | # sessions with `a_i ≤ T+1h < b_i` (charging at end of hour T) | Org | F G | Ongoing sessions keep drawing power in T+1, T+2: the strongest short-horizon signal |
| `occupied_ports_T` | **A** | # sessions with `c_i ≤ T+1h < d_i` | Org | F G | Physical occupancy |
| `idle_ports_T` | **A** | `occupied_ports_T − active_sessions_T` | Org | F G | Blocked plugs reduce capacity for new arrivals |
| `arrivals_T` | **A** | # sessions with `c_i ∈ [T, T+1h)` | Org | F G | Arrival momentum (morning ramp) |
| `headroom_ports_T` | **A** | `N_s − occupied_ports_T` | Org | F G | Physical ceiling on new demand |
| `saturation_hours_24h` | **A** | `Σ_{k=0..23} is_saturated_{T−k}` | Org | F G | Censoring awareness |

### Group D — Weather (Open-Meteo; archive for training, forecast for serving; value **at target hour t**)
| Feature | Dec. | Formula | Anchor | Models | Rationale |
|---|---|---|---|---|---|
| `temperature_2m_c` | K | Open-Meteo `temperature_2m` at t | Tgt | R F G L P | Main exogenous driver |
| `apparent_temperature_c` | **R** | — | — | — | Derived from temperature + humidity + wind; ρ > 0.95 with temperature |
| `precipitation_mm` | K | `precipitation` at t | Tgt | R F G L P | Travel/dwell effect |
| `is_rain` | **A** | `1[precipitation ≥ 0.2 mm]` | Tgt | R F G | Rain is a threshold effect; mm is heavy-tailed |
| `relative_humidity_pct` | **R** | — | — | — | Weak signal, collinear with temperature and precipitation. Keep only in the weather ablation |
| `cloud_cover_pct`, `shortwave_radiation_wm2` | **R** | — | — | — | No co-located PV in either dataset. Re-add only if PV generation is modelled |
| `temp_deviation_from_comfort` | **M → hinge pair** | `cold_stress = max(0, 10 − T_c)`, `heat_stress = max(0, T_c − 28)` | Tgt | R F G P | Separates battery-preconditioning (cold) from HVAC (heat) effects. Enables monotone constraints (F9) |

### Group E — Static & spatial
| Feature | Dec. | Formula | Anchor | Models | Rationale |
|---|---|---|---|---|---|
| `station_id` | **M** | Native categorical (LGBM/XGB); one-hot (Ridge); 8-dim embedding (LSTM). **No target encoding** | Sta | R F G L | Target encoding adds leakage risk and is unnecessary with native categoricals |
| `latitude`, `longitude` | **R from models** (keep in DB/map) | — | Sta | P only | With ~10 sites, coordinates are just a station-ID proxy and invite overfitting |
| `total_ports` (`N_s`) | K | count of ports | Sta | R F G P | Scale and utilisation denominator |
| `max_station_capacity_kw` | **M → `rated_kw_total`** | `P_s^port = Σ port kW` | Sta | P | Constant per station, so redundant with `station_id` except for cold-start |
| `poi_category` | K | {workplace, downtown_garage, library/community, retail, residential} | Sta | G P | Mostly for the cold-start prior |
| `station_age_days` | **A** | `t − first_session_date_s` | Tgt | F G P | Utilisation ramps for months after a site opens |

**Final count:** ≈ 34 features for GBDT/RF, ≈ 210 columns for Ridge (with 167 hour-of-week dummies and station one-hots), and a 168-step × 12-channel sequence for the LSTM.

## 2.3 Multi-step strategy — decision

**Chosen: Direct multi-horizon with horizon-as-feature (a pooled direct, or "stacked origins", model).**

```
ŷ_{s,T+h} = f( x^org_{s,T},  x^tgt_{s,T+h},  x^sta_s,  h ),     h = 1..24
```

Training rows are `(s, T, h)` triples. Size ≈ `S × |T_train| × 24`, for example 10 sites × 18,000 origins × 24 ≈ 4.3 M rows. GBDTs with `hist` handle that easily. RF and Ridge are trained on a sub-sample of origins (every 3rd origin, ≈ 1.4 M rows; `max_samples=0.25` for RF).

| Strategy | Verdict | Reason |
|---|---|---|
| **Recursive** (feed `ŷ` back as lag) | Ablation only | Error compounds with h; quantiles can't be propagated without simulation; train/serve mismatch. We implement it for LightGBM **to show the error-vs-horizon curve** in the report |
| **Direct, 24 separate models** | Rejected | 24 × 5 models × 3 quantiles = 360 artefacts; no horizon sharing |
| **Direct pooled (horizon as feature)** | **Chosen** | One artefact per algorithm, causally clean, every arena model produces all 24 steps |
| **Seq2Seq / MIMO** | Used for the LSTM | Encoder over 168 h, dense decoder emits 24 outputs at once (no recursion) |

**Evaluation reporting.** Every metric is reported (a) per horizon `h = 1, 6, 12, 24`, (b) averaged over h = 1..24, and (c) for the **operational day-ahead** issue time (origin 14:00 local, target the next day 00–23).

## 2.4 Uncertainty: quantiles + conformal calibration

1. **Quantile LightGBM** (`objective="quantile"`) for α ∈ {0.05, 0.50, 0.95}, minimising the **pinball loss**:
   `L_α(y, q) = max(α·(y − q), (α − 1)·(y − q))`
2. **Fix crossing:** `(q_lo, q_med, q_hi) ← sort(q̂_0.05, q̂_0.5, q̂_0.95)`.
3. **CQR calibration** on the validation block, separately per horizon bucket `B ∈ {1–3, 4–12, 13–24}`:
   `E_i = max(q̂_lo(x_i) − y_i, y_i − q̂_hi(x_i))`,
   `Q_B = Quantile_{(1−α)(1 + 1/n_B)}({E_i}_{i∈B})` with α = 0.10,
   `PI(x) = [ max(0, q̂_lo(x) − Q_B),  min(P_s^port, q̂_hi(x) + Q_B) ]`.
   Under exchangeability this gives ≥ 90% marginal coverage. Time series are not exactly exchangeable, so we add (4).
4. **Adaptive Conformal Inference (online, in the Arena agent):**
   `α_{t+1} = α_t + γ·(α − err_t)`, `err_t = 1[y_t ∉ PI_t]`, γ = 0.005. This keeps realised coverage near 90% under drift.
5. **Every arena model gets an interval.** Non-quantile models (Ridge, RF, XGB, LSTM) get split-conformal intervals `ŷ ± Q_B(|y − ŷ|)`. That makes the comparison on PICP, MPIW and Winkler fair.
6. **Network interval:** conformal on aggregate residuals (F10).

## 2.5 Metrics

With `e = y − ŷ` over the evaluation set `𝒯`:

| Metric | Formula | Notes |
|---|---|---|
| MAE | `(1/n)·Σ\|e\|` | Robust; units kWh |
| RMSE | `√((1/n)·Σe²)` | Penalises spikes (transformer risk) |
| **WAPE** (primary) | `Σ\|e\| / Σy` | Well defined with zeros; comparable across stations |
| **MASE** | `MAE / MAE_{seasonal-naive-24, train}` | < 1 means it beats the baseline |
| R² | `1 − Σe² / Σ(y − ȳ)²` | Requested by the course |
| MAPE (filtered) | `(100/\|𝒯⁺\|)·Σ_{𝒯⁺} \|e/y\|`, `𝒯⁺ = {y ≥ 1 kWh}` | Only reported with this filter |
| Peak-hour MAE | MAE on hours where `y ≥ p90(y_train,s)` | What matters for overload |
| Pinball loss | `(1/n)·Σ L_α(y, q̂_α)` | Per quantile |
| PICP | `(1/n)·Σ 1[ℓ ≤ y ≤ u]` | Target 0.90 |
| MPIW | `(1/n)·Σ (u − ℓ)` | Sharpness |
| Winkler (α = 0.1) | `(u − ℓ) + (2/α)·(ℓ − y)·1[y < ℓ] + (2/α)·(y − u)·1[y > u]` | Joint coverage + width |
| Diebold–Mariano | `DM = d̄ / √(2π·f̂_d(0)/n)`, `d_t = \|e_{1,t}\| − \|e_{2,t}\|`, HAC variance with h−1 lags | Is model 1 *significantly* better than model 2? |

Stratified reporting: by station, by hour-of-day, weekday vs weekend, saturated vs unsaturated, and temperature tercile.

## 2.6 Leakage guard-rails (codified as tests)

1. **Future-poisoning test:** for random `(s, T)`, overwrite all `y_{s,τ>T}` with `NaN` or `1e9` and recompute `x^org`. The features must be identical. For target-anchored lags with lag ≥ h, they must also be identical.
2. **Split-purity test:** `max(train.timestamp) + gap ≤ min(val.timestamp)`, and the same for val/test. No timestamp appears in two splits.
3. **Transformer-fit test:** scalers, imputers and the CQR calibrator are fitted only on their designated block (asserted by stored fit-range metadata).
4. **Weather-vintage test:** during serving, weather rows used for `t` must have `issued_at ≤ T`.

## 2.7 Final model suite

| ID | Model | Implementation | Key settings / Optuna search space | Purpose |
|---|---|---|---|---|
| B0 | Persistence | `ŷ_{T+h} = y_T` | — | Short-horizon floor |
| **B1** | **Seasonal naive (daily)** | `ŷ_t = y_{t−24}` | — | **Mandatory baseline**, MASE denominator |
| B2 | Seasonal naive (weekly) | `ŷ_t = y_{t−168}` | — | Weekly floor |
| B3 | Hour-of-week climatology | `ŷ_t = mean_train(y \| s, how(t))` | — | Strong "dumb" benchmark |
| **M1** | **Ridge regression** | `sklearn.linear_model.Ridge` on `log1p(y)` + Duan smearing back-transform | `alpha ∈ [1e−3, 1e3]` log | Regression family; interpretable coefficients |
| **M2** | **Random Forest** | `RandomForestRegressor` | `n_estimators ∈ [200, 600]`, `max_depth ∈ [8, 24]`, `min_samples_leaf ∈ [5, 100]`, `max_features ∈ {0.3, 0.5, sqrt}`, `max_samples=0.25` | Bagging family |
| **M3** | **XGBoost** | `XGBRegressor(tree_method="hist", enable_categorical=True, objective="reg:tweedie")` | `max_depth ∈ [3, 10]`, `eta ∈ [0.01, 0.2]`, `min_child_weight ∈ [1, 50]`, `subsample, colsample ∈ [0.5, 1]`, `lambda, alpha ∈ [1e−3, 10]` log, `tweedie_variance_power ∈ [1.1, 1.7]` | Boosting family (depth-wise) |
| **M4** | **LightGBM** (mean + q05/q50/q95) | `LGBMRegressor(objective="tweedie")` + 3 quantile heads | `num_leaves ∈ [15, 255]`, `min_data_in_leaf ∈ [20, 500]`, `learning_rate ∈ [0.01, 0.2]`, `feature_fraction, bagging_fraction ∈ [0.5, 1]`, `lambda_l1, lambda_l2 ∈ [1e−3, 10]`, monotone +1 on `heat_stress`, `cold_stress` | Boosting family (leaf-wise); **expected champion**; intervals |
| **M5** | **LSTM Seq2Seq (MIMO)** | PyTorch: 2-layer LSTM (hidden 64), station embedding, dense head → 24 outputs; Tweedie or Huber loss; early stopping on val; export ONNX | `hidden ∈ {32, 64, 128}`, `dropout ∈ [0, 0.3]`, `lr ∈ [1e−4, 3e−3]`, lookback ∈ {72, 168} | "Time-series forecasting" family (Zhu et al., 2019) |
| **E1** | **Weighted ensemble** M3 + M4 (+ M5) | `ŷ = Σ w_k·ŷ_k`, `w ≥ 0`, `Σw = 1`, fitted by NNLS on the validation block | — | Course §10: compare against components incl. training time |
| A1 | Recursive LightGBM | Same as M4 but h = 1 only, fed back 24× | — | **Ablation:** error accumulation |
| A2 | M4 with L2 loss | `objective="regression"` | — | **Ablation:** Tweedie vs L2 |
| A3 | M4 without weather | drop Group D | — | **Ablation:** value of weather |
| S1 | Occupancy model | LightGBM on `occ_ratio` (+ `idle_ratio`) | as M4 | Secondary operational target |
| P0 | Cold-start prior | LightGBM without Groups B, C, O | as M4 | New stations (F14) |

**Prophet: dropped.** Hourly, zero-inflated, multi-station data with strong lag dependence suits it poorly: it has no lag terms, one model per station, and is slow to fit. It would add a weak model without satisfying anything the LSTM doesn't already cover. You can mention it in the literature review.

**Optuna protocol.** TPE sampler (`seed=42`), MedianPruner, 60 trials for GBDTs, 30 for RF, 20 for LSTM. Objective = mean WAPE across `PanelTimeSeriesSplit` folds **inside the training block**. The validation block is used only for early stopping, ensemble weights and CQR calibration. The **test block is evaluated exactly once**, by `make evaluate`, which writes `reports/metrics_test.json`.

**Expected outcome (hypothesis, not a result).** M4 ≈ M3 > M2 > M5 ≈ M1 > B3 > B1 on WAPE. The ensemble E1 gives a small gain (≤ 3%). Weather adds little in mild Palo Alto. Recursive error grows roughly with √h. Whatever the numbers turn out to be, report them; a hypothesis that turns out wrong is still a valid finding.
