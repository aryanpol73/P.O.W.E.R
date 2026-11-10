[[TITLE]]

# Abstract

Uncoordinated electric-vehicle (EV) charging concentrates demand into sharp morning and evening peaks that stress local distribution transformers and widen the grid's net-load "duck curve". This project, **P.O.W.E.R. (Predictive Optimization of Wattage for EV Resources)**, builds an end-to-end, real-time machine-learning system that forecasts hourly charging-station energy demand (kWh) 1 to 24 hours ahead, with calibrated 90% prediction intervals and port-occupancy and idle-time risk. Session-level charging logs from the City of Palo Alto (primary) and Caltech ACN-Data (secondary) are converted into a gap-free hourly panel by prorating delivered energy over each session's active charging window. Historical weather from Open-Meteo is merged in. Following the course guidance for Project #41, we compare four seasonal-naive and climatology baselines against Ridge Regression, Random Forest, XGBoost, LightGBM, a Seq2Seq LSTM and a weighted ensemble, under a strictly chronological, leakage-tested direct multi-horizon protocol. Uncertainty is quantified with conformalized quantile regression and maintained online with adaptive conformal inference. Models are evaluated with WAPE, MASE, MAE, RMSE, R², pinball loss, interval coverage and width, and Diebold–Mariano significance tests, and interpreted with SHAP. A production-style progressive web application (Next.js, FastAPI, PostgreSQL/TimescaleDB) streams simulated live telemetry through nine autonomous workers. It renders a city map with a 24-hour forecast scrubber, transformer-overload alerts, reroute suggestions, a what-if simulator and per-forecast SHAP explanations. A Model Context Protocol server exposes the system to natural-language querying. ⟨Insert one sentence with the headline result, e.g. "LightGBM reduced WAPE by X% relative to seasonal-naive at a 24-hour horizon while achieving Y% interval coverage."⟩

**Keywords:** EV charging demand, time-series forecasting, gradient boosting, conformal prediction, SHAP, real-time ML systems, smart grid.

# Rubric Compliance Matrix

| Rubric criterion (marks) | Where addressed | Evidence produced |
|---|---|---|
| Problem definition & real-world relevance (10) | Chapter 1 | Formal problem statement, user/decision table, measurable objectives |
| Dataset selection & preprocessing (15) | Chapters 3–4 | Data card with exact counts, cleaning rules, imputation policy, encodings, leakage tests |
| Correct implementation of ML algorithms (20) | Chapter 5 | Mathematical formulation of every model; modular code in `ml/` and `power_core` |
| Model comparison & evaluation (20) | Chapters 6–7 | Chronological split, panel time-series CV, baseline comparison, multi-metric leaderboard, DM tests, ablations |
| Novelty, interpretation & practical insights (10) | Chapter 8 | SHAP global/local, residual analysis, censoring analysis, COVID stress test, actionable findings, limitations |
| Working prototype / demonstration (15) | Chapter 9 | PWA + API + real-time workers + MCP; demo script |
| Report, presentation & viva (10) | Chapters 10–11, Appendices | Ethics, reproducibility, contribution matrix, viva question bank |
| §9 minimum requirements | Ch 1, 3, 5–7, 10, 11 | Problem, dataset documentation, ≥ 3 algorithms + baseline, error analysis, prototype, reproducibility, ethics |
| §10 ensemble guidance | §7.4 | Ensemble vs components incl. training time and interpretability |

[[TOC]]

# List of Abbreviations

| Abbreviation | Meaning |
|---|---|
| ACI | Adaptive Conformal Inference |
| CPO | Charge Point Operator |
| CQR | Conformalized Quantile Regression |
| DM | Diebold–Mariano test |
| EVSE | Electric Vehicle Supply Equipment |
| GBDT | Gradient-Boosted Decision Trees |
| MASE | Mean Absolute Scaled Error |
| MCP | Model Context Protocol |
| MPIW | Mean Prediction Interval Width |
| OCPP | Open Charge Point Protocol |
| OOD | Out-of-Distribution |
| PICP | Prediction Interval Coverage Probability |
| PSI | Population Stability Index |
| PWA | Progressive Web App |
| SHAP | SHapley Additive exPlanations |
| ToU | Time-of-Use tariff |
| WAPE | Weighted Absolute Percentage Error |

# 1. Problem Definition and Real-World Relevance

## 1.1 Background

EV adoption shifts transport energy demand onto the electricity distribution network. Unlike residential load, charging-station load is highly stochastic and has a high peak-to-average ratio. Arrivals cluster around workplace mornings (08:00–10:00) and evening returns (17:00–20:00). When many Level-2 (7–22 kW) or DC fast chargers (50–350 kW) draw power at once, the local distribution transformer can be pushed beyond its rating, and the evening ramp adds to the system-level duck curve (Lee et al., 2019; Shern et al., 2024).

Three physical properties make the problem harder than ordinary regression:

- **Idle occupancy.** Vehicles stay plugged in long after charging finishes (`doneChargingTime` precedes `disconnectTime` by hours in ACN-Data), so a station can be 100% occupied while drawing 0 kW.
- **Right-censored demand.** When all ports are occupied, further arrivals leave unserved. Meter data records only the served demand, which is therefore a lower bound on true demand.
- **Exogenous sensitivity.** Ambient temperature affects battery efficiency and cabin HVAC load; holidays and tariffs shift behaviour.

## 1.2 Formal problem statement

For each charging site s and each forecast origin T (the last fully observed hour), predict the energy demand for each of the next 24 hours together with a 90% prediction interval:

$$ ŷ_{s,T+h} ,  [ℓ_{s,T+h} , u_{s,T+h}]  for  h = 1,…,24,   with   P(ℓ ≤ y_{s,T+h} ≤ u) ≈ 0.90 $$

Secondary outputs: predicted port utilisation, idle-port share, a transformer load ratio, a three-level risk status (NORMAL / WARNING / CRITICAL), and reroute recommendations to nearby stations with headroom.

## 1.3 Target users and decisions supported

| User | Decision supported | Output used |
|---|---|---|
| Distribution grid operator | Pre-emptive load management, transformer protection | Network & station q95 forecasts, CRITICAL alerts |
| Charge Point Operator (CPO) | Dynamic pricing, idle fees, staffing, maintenance windows | Occupancy and idle-share forecasts, peak hours |
| Urban / infrastructure planner | Where to add ports or upgrade transformers | Saturation statistics, persistent-risk stations, SHAP drivers |
| EV driver (indirect) | Where to charge now | Reroute suggestions to stations with headroom |

## 1.4 Objectives and expected benefits

1. Beat the seasonal-naive baseline at every horizon (MASE < 1) and report the improvement with statistical significance.
2. Deliver 90% intervals whose empirical coverage on unseen data lies within 87–93%.
3. Detect predicted transformer overloads at least 6 hours in advance with a documented precision/recall on the test period.
4. Explain every forecast in plain language (SHAP).
5. Demonstrate the full loop in real time: ingest → forecast → alert → explain.

**Expected benefits:** fewer unplanned overload events, better-informed port placement, reduced idle blocking through targeted pricing, and an auditable decision-support tool.

## 1.5 Scope and assumptions

In scope: hourly site-level forecasting for public Level-2 sites in one city, simulated real-time replay, and live weather. Out of scope: actuating chargers (OCPP `SetChargingProfile`), sub-hourly power control and retail tariff design. **Key assumption:** transformer ratings are not public. Capacity is modelled as C_s = DF × Σ port kW with DF = 1.0 (EV chargers are continuous loads whose supply is sized at ≥ 125% of charger rating), and alert frequency is reported for DF ∈ {0.85, 1.0} and lookaheads of 6, 12 and 24 hours.

# 2. Literature Review

## 2.1 Evolution of approaches

Early work modelled aggregate charging load with Multiple Linear Regression, ARIMA and SARIMA. These capture smooth diurnal cycles but miss station-level non-linear spikes. Tree ensembles (Random Forest, Gradient Boosting, XGBoost, LightGBM) recast forecasting as supervised tabular learning over lags, calendar and weather features. Amezquita et al. (2024) report that Optuna-tuned LightGBM and XGBoost outperform MLR, decision trees, RF and LSTM on station data. Recurrent networks (LSTM, Bi-LSTM, GRU) model sequential dependence across multi-step horizons (Zhu et al., 2019), and interval methods such as LSTM–Bayesian networks quantify uncertainty (Skala et al., 2023). Forecasts feed smart-charging controllers that shift flexible load (Lee et al., 2019; Shern et al., 2024).

## 2.2 Key studies

| Study | Problem | Approach | Limitation | How this project uses it |
|---|---|---|---|---|
| Lee et al. (2019), ACN-Data | No open fine-grained workplace charging data | Open dataset + API; behavioural analysis incl. free→paid pricing | California workplace garages only | Secondary dataset; idle-occupancy evidence; pricing-regime dummy |
| Amezquita et al. (2024) | Predict plug-in status and hourly energy | MLR, DT, RF, LSTM, XGBoost, LightGBM + Optuna | Lag-heavy; no traffic data | Model suite and tuning protocol; hypothesis that GBDTs lead |
| Zhu et al. (2019) | Multi-step station load forecasting | RNN, LSTM, Bi-LSTM, GRU across resolutions | Slow training; lags regime shifts | Seq2Seq LSTM design; horizon evaluation |
| Skala et al. (2023) | Point forecasts fail on EV spikes (households) | LSTM + Bayesian NN intervals; weather merge | Household (not station) load; heavy inference | Motivation for intervals; weather-merge practice |
| Shern et al. (2024) | Peak congestion; smart-charging allocation | RF, SVR, GBR, XGBoost formulations | Little deployment focus | Mathematical formulations; ensemble comparison |
| Adnane et al. (2023) | No unified taxonomy for EV-energy ML | Systematic mapping study | Survey only | Justifies metric and validation choices |

## 2.3 Research gap and contributions

Most studies report point accuracy on random or single-horizon splits and stop at offline evaluation. This project contributes:

1. A **leakage-tested direct multi-horizon** formulation that separates target-anchored and origin-anchored features.
2. **Distribution-free calibrated intervals** (CQR + ACI) compared fairly across all models.
3. **Explicit treatment of idle occupancy and censoring** as features, as evaluation strata and in the discussion of limitations.
4. A **real-time production system** with drift detection, champion–challenger retraining and an explainable, map-based operator interface.

# 3. Dataset Selection and Documentation

## 3.1 Sources

| Dataset | Provider / licence | Granularity | Period used | Role |
|---|---|---|---|---|
| EV Charging Station Usage, City of Palo Alto | City of Palo Alto Open Data | Session (start, end, total duration, charging time, kWh, station, port, coordinates) | ⟨2017-01-01 → 2020-02-29⟩ main; Mar–Jun 2020 OOD | Primary training/evaluation |
| ACN-Data, Caltech site | Caltech / PowerFlex, research use (token) | Session (connectionTime, doneChargingTime, disconnectTime, kWhDelivered, spaceID) | ⟨2018-05-01 → 2020-02-29⟩ | Out-of-domain test; idle-time study |
| Open-Meteo Historical Weather | Open-Meteo, CC-BY 4.0 | Hourly reanalysis at station coordinates | Same as above | Training weather |
| Open-Meteo Forecast | Open-Meteo | Hourly forecast, 48 h | Live | Serving weather |
| OpenChargeMap | OpenChargeMap, CC-BY-SA | Station POIs | Live | Map context, cold-start priors |

## 3.2 Record counts (fill from `data/manifest.json`)

| Quantity | Palo Alto | ACN Caltech |
|---|---|---|
| Raw sessions downloaded | ⟨n⟩ | ⟨n⟩ |
| Sessions after cleaning | ⟨n⟩ (⟨%⟩ removed) | ⟨n⟩ |
| Sessions in modelling window | ⟨n⟩ | ⟨n⟩ |
| Sites (modelling units) / ports | ⟨S⟩ / ⟨N⟩ | 1 / ⟨N⟩ |
| Modelled station-hours (rows) | ⟨S × hours⟩ (≥ 50,000) | ⟨n⟩ |
| Direct multi-horizon training rows | ⟨rows × 24⟩ | — |
| Share of zero-demand hours | ⟨%⟩ | ⟨%⟩ |
| Share of saturated hours | ⟨%⟩ | ⟨%⟩ |

Both counts are reported because "50,000+ records" is ambiguous between sessions and modelled hours.

## 3.3 Target variable

The primary target is **hourly energy dispensed per site, y (kWh)**, which is numerically equal to the mean power in kW over that hour. Secondary targets are port-utilisation ratio and idle ratio (definitions in §4.3).

## 3.4 Modelling window: justification

Charging demand collapsed from March 2020 (COVID-19 restrictions). A chronological split across that break would tune models in one regime and test them in another. The main experiment therefore uses a pre-COVID window. The March–June 2020 period is kept as an explicit out-of-distribution stress test (§8.5) that also exercises the drift detector.

## 3.5 Data limitations

- **Regional bias:** a mild-climate, affluent Californian city and a university campus; findings may not transfer to highway corridors, cold climates or Indian cities.
- **Censoring:** demand is unobserved when all ports are full.
- **No transformer ratings:** capacity is a synthetic assumption.
- **No sub-hourly power for Palo Alto:** intra-hour peaks exceed the hourly mean.
- **Weather is observed, not forecast,** during training (mitigated in §4.6).
- **Station additions over time** cause cold-start periods.
- **Reporting artefacts:** duplicate sessions, zero-energy sessions, negative durations (handled in §4.1).

# 4. Preprocessing and Feature Engineering

## 4.1 Cleaning rules

| Rule | Action | Rationale |
|---|---|---|
| Duplicate session (same port, start, end, kWh) | Drop | Export artefact |
| `disconnect ≤ connect` or duration > 48 h | Drop | Invalid or abandoned record |
| `energy_kwh = 0` and duration < 5 min | Drop | Failed plug-in |
| Implied charging power > 1.1 × port kW | Cap energy, flag bit 1 | Physically impossible |
| Missing charge-end time | Impute charge end = connect + kWh / port kW (≤ disconnect), flag bit 2 | Needed for proration |
| User ID, zip code, MAC or card identifiers | Drop before storage; session id → salted SHA-256 | Privacy |
| Timestamps | Parse with source time zone → UTC | DST correctness |

## 4.2 Missing values

The hourly panel is reindexed to a complete grid per site. Hours without sessions are **true zeros** (0 kWh), not missing values. Weather gaps of ≤ 3 h are linearly interpolated; longer gaps are filled from the station's hour-of-year climatology computed on the training block only. In the live system, missing meter ticks are filled with the 4-week same-hour-of-week mean and flagged `is_imputed`.

## 4.3 Session-to-hourly proration and occupancy reconstruction

For a session i with energy E_i, charging interval [a_i, b_i) and connection interval [c_i, d_i), the energy assigned to hour τ is proportional to the overlap of that hour with the charging interval:

$$ y_{s,τ} = \sum_{i ∈ s}^{}{ E_{i} · \frac{|[a_{i}, b_{i}) ∩ [τ, τ+1)|}{b_{i} − a_{i}} } $$

Occupancy and idle share reuse the same overlap operator on the connection interval:

$$ occ_{s,τ} = \frac{1}{N_{s}} \sum_{i ∈ s}^{}{ |[c_{i}, d_{i}) ∩ [τ, τ+1)| } ,     idle_{s,τ} = \frac{\sum_{i}^{}{|[b_{i}, d_{i}) ∩ [τ, τ+1)|}}{\sum_{i}^{}{|[c_{i}, d_{i}) ∩ [τ, τ+1)|} + ε} $$

An hour is marked saturated when occ·N_s ≥ N_s − 0.5. For Palo Alto, b_i = a_i + Charging Time; for ACN, b_i = doneChargingTime.

## 4.4 Time handling

All timestamps are stored in UTC. Calendar features are derived in local time (America/Los_Angeles). DST transition days therefore have 23 or 25 local hours, while the UTC grid stays regular.

## 4.5 Encoding

Cyclical variables are encoded with sine/cosine pairs so that 23:00 and 00:00 are adjacent:

$$ hour_{sin} = sin(\frac{2π · hour}{24}) ,   hour_{cos} = cos(\frac{2π · hour}{24}) $$

The same applies to day-of-week (period 7) and day-of-year (period 365.25, one harmonic). Ridge additionally receives hour-of-week one-hot dummies and three weekly Fourier pairs, because a linear model cannot form hour × weekday interactions on its own. `station_id` is a native categorical for XGBoost/LightGBM, one-hot for Ridge and an 8-dimensional embedding for the LSTM. Target encoding is deliberately not used (leakage risk). Standardisation (fitted on the training block only) is applied to Ridge and LSTM inputs; tree models use raw values.

## 4.6 Feature catalogue

Every feature has exactly one **anchor class**: *target-anchored* (known in advance for the target hour t = T + h), *origin-anchored* (state at forecast origin T), or *static*.

| Group | Features | Anchor |
|---|---|---|
| Horizon | h | — |
| Calendar | hour sin/cos, dow sin/cos, doy sin/cos, is_weekend, is_holiday (US-CA), is_bridge_day, pricing_era* | Target |
| Seasonal lags (lag ≥ h) | y(t−24), y(t−48), y(t−168), mean of y(t−24k) for k = 1..7, mean of y(t−168k) for k = 1..4 | Target |
| Recent state | y(T), y(T−1), y(T−2), hours since last non-zero | Origin |
| Rolling | 24 h mean, 24 h std, 24 h max, EWMA (α = 0.3), 28-day level, level ratio | Origin |
| Occupancy | active sessions, occupied ports, idle ports, arrivals, port headroom, saturated hours in last 24 h | Origin |
| Weather (value at t) | temperature, precipitation, is_rain, cold_stress = max(0, 10 − T), heat_stress = max(0, T − 28) | Target |
| Static | station_id, total ports, poi_category, station age; rated kW (prior model only) | Static |

*Only if the window contains a documented price change.*

**Removed from the initial proposal (with reasons):** 3 h and 6 h rolling means (exact or near-exact linear combinations of the lags/EWMA), 6 h rolling std (too noisy), ramp rate (linear combination of lags), apparent temperature and humidity (collinear with temperature), cloud cover / solar radiation (no co-located PV), latitude/longitude as model inputs (station-ID proxies with few sites), and a fixed ToU regime (a deterministic function of hour and weekday).

**Weather-forecast realism.** Because training uses observed weather while serving uses forecasts, training temperatures are perturbed with Gaussian noise whose standard deviation grows with horizon, σ(h) = 0.8 + 0.05h °C. The value of weather is then measured directly by ablation (§7.6).

## 4.7 Leakage prevention

1. **Chronological split** inside the modelling window: 70% train, 15% validation, 15% test by timestamp. There is no shuffling, and a 168-hour embargo separates the blocks.
2. **Anchor discipline:** origin features use only y up to T; target-anchored lags satisfy lag ≥ h.
3. **Fit-on-train only:** scalers, imputation climatologies and drift references come from the training block. Conformal calibration uses only the validation block.
4. **Panel time-series cross-validation:** folds are cut on unique timestamps (not rows), with a gap measured in hours.
5. **Automated tests:** a future-poisoning test (overwrite all y after T with NaN or 10⁹ and assert features are unchanged), split-purity and weather-vintage tests run in CI.

# 5. Methodology: Machine-Learning Algorithms

## 5.1 Forecasting formulation

We use a **direct multi-horizon model with horizon as a feature**. One model per algorithm predicts every horizon:

$$ ŷ_{s,T+h} = f( x^{org}_{s,T} , x^{tgt}_{s,T+h} , x^{sta}_{s} , h ) ,   h = 1,…,24 $$

Training rows are (site, origin, horizon) triples. Compared with recursive forecasting, this avoids feeding predictions back as inputs, so errors do not compound and train-time and serve-time features are identical. A recursive LightGBM is kept as an ablation to show the difference empirically. The LSTM is natively multi-output (Seq2Seq / MIMO).

## 5.2 Baselines

$$ B0: ŷ_{T+h} = y_{T}    B1: ŷ_{t} = y_{t−24}    B2: ŷ_{t} = y_{t−168}    B3: ŷ_{t} = \frac{1}{|D_{s,w}|} \sum_{τ ∈ D_{s,w}}^{}{y_{τ}} $$

D(s, w) is the set of training hours at site s with the same hour-of-week w as t. B1 (seasonal naive) is the mandatory reference and the MASE denominator.

## 5.3 Ridge regression

Ridge minimises squared error with an L2 penalty. It is fitted on z = log(1 + y) to stabilise the variance of the zero-heavy, right-skewed target:

$$ β̂ = argmin_{β} \sum_{i}^{}{(z_{i} − x_{i}^{⊤}β)^{2}} + λ‖β‖_{2}^{2} = (X^{⊤}X + λI)^{−1}X^{⊤}z $$

The prediction is back-transformed with Duan's smearing correction, ŷ = (exp(x⊤β̂) − 1) · mean(exp(ε̂_i)), and clipped at 0. Coefficients are directly interpretable as multiplicative effects.

## 5.4 Random Forest

An average of B regression trees, each grown on a bootstrap sample with a random feature subset at each split:

$$ ŷ(x) = \frac{1}{B} \sum_{b=1}^{B}{T_{b}(x; Θ_{b})} $$

Splits minimise within-node squared error. Bagging reduces variance without heavy tuning, and `max_samples = 0.25` keeps training tractable on the multi-horizon table.

## 5.5 Gradient boosting: XGBoost and LightGBM

Boosting builds an additive model stage by stage, with each tree fitted to the gradient of the loss:

$$ F_{m}(x) = F_{m−1}(x) + η · f_{m}(x) ,   f_{m} = argmin_{f} \sum_{i}^{}{L(y_{i}, F_{m−1}(x_{i}) + f(x_{i}))} + Ω(f) $$

XGBoost uses a second-order Taylor expansion with gradients g_i and Hessians h_i and the regulariser Ω(f) = γK + ½λ‖w‖². For a tree with K leaves, the optimal leaf weight and the split gain are:

$$ w^{*}_{j} = − \frac{G_{j}}{H_{j} + λ} ,    Gain = \frac{1}{2}[ \frac{G_{L}^{2}}{H_{L}+λ} + \frac{G_{R}^{2}}{H_{R}+λ} − \frac{(G_{L}+G_{R})^{2}}{H_{L}+H_{R}+λ} ] − γ $$

XGBoost grows trees depth-wise. LightGBM grows them leaf-wise (best-first), with histogram binning, Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). Both support native categorical splits on `station_id` and L1/L2 regularisation. Leaf-wise growth is constrained through `num_leaves` and `min_data_in_leaf`. Both use a **Tweedie objective** (variance power 1 < p < 2), which suits a non-negative, zero-inflated target. It is a compound Poisson–Gamma model with a log link:

$$ L_{Tweedie}(y, μ) = −y · \frac{μ^{1−p}}{1−p} + \frac{μ^{2−p}}{2−p} $$

Monotone non-decreasing constraints on `heat_stress` and `cold_stress` give the model physically plausible behaviour under weather shocks.

## 5.6 Seq2Seq LSTM

An LSTM encoder reads the previous L = 168 hourly vectors (demand, occupancy, calendar, weather, station embedding). For each step t, with input xₜ, previous hidden state hₜ₋₁ and cell state cₜ₋₁:

$$ i_{t} = σ(W_{i}[h_{t−1}, x_{t}] + b_{i}) ,  f_{t} = σ(W_{f}[h_{t−1}, x_{t}] + b_{f}) ,  o_{t} = σ(W_{o}[h_{t−1}, x_{t}] + b_{o}) $$

$$ c̃_{t} = tanh(W_{c}[h_{t−1}, x_{t}] + b_{c}) ,  c_{t} = f_{t} ⊙ c_{t−1} + i_{t} ⊙ c̃_{t} ,  h_{t} = o_{t} ⊙ tanh(c_{t}) $$

The final hidden state is concatenated with the known future covariates (24 × calendar + weather) and mapped by a dense head to 24 outputs at once (MIMO). Training uses Adam, early stopping on validation loss, and gradient clipping at 1.0. The trained network is exported to ONNX for serving.

## 5.7 Weighted ensemble

The ensemble combines the predictions of the boosting models (and optionally the LSTM) with non-negative weights fitted on the validation block by non-negative least squares:

$$ ŷ^{ens} = \sum_{k}^{}{w_{k} ŷ^{(k)}} ,   w = argmin_{w ≥ 0, Σw = 1} \sum_{i ∈ val}^{}{(y_{i} − \sum_{k}^{}{w_{k}ŷ^{(k)}_{i}})^{2}} $$

## 5.8 Probabilistic forecasting

**Quantile regression.** LightGBM quantile heads for α ∈ {0.05, 0.50, 0.95} minimise the pinball loss:

$$ L_{α}(y, q) = max( α(y − q) , (α − 1)(y − q) ) $$

**Conformalized Quantile Regression (Romano et al., 2019).** On the validation block, separately for each horizon bucket B ∈ {1–3, 4–12, 13–24}, compute conformity scores and an adjustment:

$$ E_{i} = max( q̂_{lo}(x_{i}) − y_{i} , y_{i} − q̂_{hi}(x_{i}) ) ,    Q_{B} = Quantile_{(1−α)(1+1/n_{B})}( {E_{i}} ) $$

$$ PI(x) = [ max(0, q̂_{lo}(x) − Q_{B}) , min(P^{port}_{s}, q̂_{hi}(x) + Q_{B}) ] $$

Quantiles are sorted before calibration to remove crossing. Point models (Ridge, RF, XGBoost, LSTM) receive split-conformal intervals ŷ ± Q_B computed from absolute validation residuals, so every model is compared on the same probabilistic footing.

**Adaptive Conformal Inference (Gibbs and Candès, 2021)** keeps coverage on target online under drift. Here err_t = 1 if y_t falls outside the interval and γ = 0.005:

$$ α_{t+1} = α_{t} + γ(α − err_{t}) $$

## 5.9 Hyper-parameter optimisation

Optuna's Tree-structured Parzen Estimator (seed 42, median pruning) minimises mean WAPE across five expanding-window panel folds **inside the training block**. It runs 60 trials for GBDTs, 30 for RF and 20 for the LSTM. Search spaces are listed in Appendix B.

# 6. Experimental Setup and Evaluation Protocol

## 6.1 Data partitions

| Block | Share | Date range | Used for |
|---|---|---|---|
| Train | 70% | ⟨start → t₁⟩ | Fitting, Optuna CV |
| Embargo | 168 h | ⟨t₁ → t₁ + 7 d⟩ | Discarded |
| Validation | 15% | ⟨→ t₂⟩ | Early stopping, ensemble weights, conformal calibration |
| Embargo | 168 h | — | Discarded |
| Test | 15% | ⟨→ 2020-02-29⟩ | **Evaluated once** |
| OOD stress | — | 2020-03-01 → 2020-06-30 | Robustness / drift case study |

## 6.2 Cross-validation

`PanelTimeSeriesSplit` (5 expanding folds) splits on unique timestamps. It leaves a 24-hour gap (equal to the maximum horizon) plus a 168-hour embargo between each training fold and its validation fold, so no autocorrelated information crosses a boundary and no timestamp is shared between folds.

## 6.3 Metrics

With errors e = y − ŷ over n evaluation points:

$$ MAE = \frac{1}{n}\sum_{}^{}{|e|} ,   RMSE = \sqrt{\frac{1}{n}\sum_{}^{}{e^{2}}} ,   WAPE = \frac{\sum_{}^{}{|e|}}{\sum_{}^{}{y}} ,   R^{2} = 1 − \frac{\sum_{}^{}{e^{2}}}{\sum_{}^{}{(y − ȳ)^{2}}} $$

$$ MASE = \frac{MAE_{model}}{MAE_{seasonal naive, train}} ,    MAPE^{+} = \frac{100}{|𝒯^{+}|} \sum_{t ∈ 𝒯^{+}}^{}{|\frac{e_{t}}{y_{t}}|} ,  𝒯^{+} = {t : y_{t} ≥ 1 kWh} $$

$$ PICP = \frac{1}{n}\sum_{}^{}{1[ℓ ≤ y ≤ u]} ,   MPIW = \frac{1}{n}\sum_{}^{}{(u − ℓ)} ,   W_{α} = (u − ℓ) + \frac{2}{α}(ℓ − y)1[y < ℓ] + \frac{2}{α}(y − u)1[y > u] $$

WAPE is the primary metric because unfiltered MAPE is undefined on the many zero-demand hours. Peak-hour MAE (hours above the site's training 90th percentile) is reported because those hours drive transformer risk.

## 6.4 Statistical significance

For two models with absolute-error loss difference d_t, the Diebold–Mariano statistic uses a heteroskedasticity- and autocorrelation-consistent (HAC) variance with h − 1 lags:

$$ DM = \frac{d̄}{\sqrt{2π f̂_{d}(0) / n}} ∼ N(0,1) $$

The Harvey–Leybourne–Newbold small-sample correction is applied, and p-values are adjusted with Holm's method across model pairs.

## 6.5 Protocol rules

1. The test block is scored once by `make evaluate`, after all choices are frozen.
2. Every model uses identical rows, horizons and calibration data.
3. All runs use seed 42 and are logged to MLflow with data hash and git SHA.
4. Results are reported per horizon (1, 6, 12, 24), averaged over 1–24, and at the operational day-ahead issue time (14:00 local).

## 6.6 Environment

⟨CPU, RAM, OS⟩; Python 3.12; versions pinned in `uv.lock` (scikit-learn, xgboost, lightgbm, optuna, torch, shap); Node 20; PostgreSQL 16 + TimescaleDB.

# 7. Results and Model Comparison

*All cells marked ⟨ ⟩ are filled from `reports/metrics_test.json` and the CSV reports generated by `make evaluate`, `make ablations` and `make explain`. No value in this chapter may be typed by hand.*

## 7.1 Test-set leaderboard (average over h = 1…24)

| Model | WAPE | MASE | MAE (kWh) | RMSE (kWh) | R² | MAPE⁺ (%) | Peak MAE | Train time (s) | p50 latency (ms) |
|---|---|---|---|---|---|---|---|---|---|
| B0 Persistence | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | 0 | ⟨⟩ |
| B1 Seasonal naive 24 h | ⟨⟩ | 1.00* | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | 0 | ⟨⟩ |
| B2 Seasonal naive 168 h | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | 0 | ⟨⟩ |
| B3 Hour-of-week climatology | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| M1 Ridge | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| M2 Random Forest | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| M3 XGBoost | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| M4 LightGBM | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| M5 LSTM Seq2Seq | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| E1 Ensemble | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |

\*By definition when computed on training errors; the test-set MASE of B1 may differ from 1.

**How to write this up.** State the best model and its relative WAPE improvement over B1, whether the gap to the runner-up is significant (§7.5), and the accuracy-versus-training-time trade-off.

## 7.2 Accuracy by horizon

Figure 7.1 (`reports/figures/wape_by_horizon.png`) plots WAPE against h for all models, including the recursive ablation A1. Expected pattern to check: persistence (B0) is strong at h = 1 and degrades fast; seasonal models are flat; direct GBDTs stay below B1 at every h; recursive error grows with h.

| Model | h = 1 | h = 6 | h = 12 | h = 24 | Day-ahead (14:00 issue) |
|---|---|---|---|---|---|
| ⟨rows as 7.1⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |

## 7.3 Probabilistic performance (90% intervals)

| Model | Interval method | PICP | MPIW (kWh) | Winkler | Pinball q05 / q50 / q95 |
|---|---|---|---|---|---|
| M4 LightGBM | Raw quantiles (no calibration) | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| M4 LightGBM | CQR (Mondrian by horizon) | ⟨⟩ | ⟨⟩ | ⟨⟩ | ⟨⟩ |
| M3 / M2 / M1 / M5 | Split conformal | ⟨⟩ | ⟨⟩ | ⟨⟩ | — |
| Network aggregate | Aggregate conformal | ⟨⟩ | ⟨⟩ | ⟨⟩ | — |

Report the coverage gap of raw quantiles and the improvement from CQR. The target is PICP within 0.87–0.93 with the smallest MPIW.

## 7.4 Ensemble versus components (course §10 requirement)

| Model | WAPE | Training time | Inference latency | Interpretability |
|---|---|---|---|---|
| M3 XGBoost | ⟨⟩ | ⟨⟩ | ⟨⟩ | TreeSHAP exact |
| M4 LightGBM | ⟨⟩ | ⟨⟩ | ⟨⟩ | TreeSHAP exact |
| M5 LSTM | ⟨⟩ | ⟨⟩ | ⟨⟩ | Gradient-based / KernelSHAP (approx.) |
| E1 Ensemble (weights ⟨w⟩) | ⟨⟩ | sum of components | sum of components | Weighted SHAP of components |

## 7.5 Diebold–Mariano matrix

A matrix of Holm-adjusted p-values for every model pair at h = 1 and h = 24 (`reports/dm_tests.csv`). Interpretation: p < 0.05 means the row model's errors differ significantly from the column model's.

## 7.6 Ablations

| Ablation | WAPE | Δ vs M4 | Interpretation |
|---|---|---|---|
| A1 Recursive instead of direct | ⟨⟩ | ⟨⟩ | Error accumulation |
| A2 L2 loss instead of Tweedie | ⟨⟩ | ⟨⟩ | Value of zero-inflated loss |
| A3 No weather features | ⟨⟩ | ⟨⟩ | Value of weather |
| A4 No occupancy features | ⟨⟩ | ⟨⟩ | Value of state features |
| A5 Weather augmentation off | ⟨⟩ | ⟨⟩ | Robustness to forecast noise |
| DF sensitivity (0.5 / 0.7 / 0.9) | alerts ⟨n⟩ | — | Alert-count sensitivity to capacity assumption |

## 7.7 Secondary target: occupancy and idle share

Report MAE of port-utilisation forecasts and of idle share, together with a calibration plot of predicted vs observed occupancy deciles.

## 7.8 Overload-alert performance

Treat "actual hourly load ≥ 0.85 × C_s" as the event and the CRITICAL alert issued ≥ 6 h earlier as the prediction. Report precision, recall, F1 and the false-alarm rate per week on the test block.

# 8. Interpretation, Error Analysis and Practical Insights

## 8.1 Global explanations

TreeSHAP values for the champion model are computed on a 2,000-row stratified test sample. Report the mean |SHAP| bar chart and the beeswarm (`reports/figures/shap_*`). Questions to answer from the figures: is diurnal persistence (lag 24 h) dominant? How much do occupancy-state features add at short horizons? Does temperature matter beyond its interaction with season?

## 8.2 Local explanations

For selected peak hours, present waterfall plots and the auto-generated narrative, e.g. "Base 18.2 kWh + 14.5 evening rush + 6.1 heat + 4.2 yesterday's 18:00 load = 43.0 kWh". For Tweedie models, state that contributions are on the log scale.

## 8.3 Residual analysis

Produce the following views and comment on each:

- Residual heatmap by hour-of-day × day-of-week.
- Error by site (do small sites have higher WAPE?).
- Error on saturated vs unsaturated hours.
- Error by temperature tercile and on rainy hours.
- Holiday error.
- Residual autocorrelation (ACF of residuals; remaining structure indicates missed signal).

## 8.4 Censoring and idle occupancy

Quantify the share of saturated hours and the error on them. Explain why forecasts there are lower bounds on latent demand. Report the median idle time per session and the share of port-hours that are idle. These figures motivate idle fees or time limits.

## 8.5 COVID-19 out-of-distribution stress test

Apply frozen models to March–June 2020. Report the degradation in WAPE and coverage, the date on which the drift detector first fires, and how ACI widens the intervals. This shows the monitoring layer working on a real regime change.

## 8.6 Actionable findings (write from evidence)

Each finding must cite a figure or table. Typical forms:

1. "Site ⟨X⟩ exceeds 85% of assumed capacity in ⟨n⟩% of weekday evening hours. It is the first candidate for a transformer upgrade or load management."
2. "Idle port-hours make up ⟨p⟩% of occupancy at ⟨Y⟩. An idle fee after ⟨k⟩ hours would free ⟨m⟩ port-hours per day."
3. "Rerouting ⟨β⟩% of divertible demand from ⟨X⟩ to ⟨Z⟩ (⟨d⟩ km) removes ⟨n⟩ of ⟨N⟩ predicted CRITICAL events."
4. "Weather adds ⟨Δ⟩% accuracy in this climate. Operators can rely on calendar and state features."

## 8.7 Limitations

Single mild-climate city; synthetic transformer ratings; hourly resolution hides intra-hour peaks; censored demand; observed-weather training; replayed rather than true live telemetry; trees do not extrapolate beyond training ranges (handled with OOD flags); exchangeability assumptions of conformal prediction are approximated (mitigated by ACI).

# 9. System Implementation and Working Prototype

## 9.1 Architecture

The system is a monorepo with a shared Python feature library (`power_core`) used by the offline pipeline, the API and the workers, which removes train/serve skew. Offline, `power_ml` runs ingestion, resampling, tuning, training, calibration, evaluation, explanation and export to a versioned model registry. Online, a PostgreSQL 16 + TimescaleDB database stores stations, hourly telemetry, weather vintages, predictions, alerts and drift logs. A single worker process runs the autonomous agents and publishes events through PostgreSQL LISTEN/NOTIFY. A FastAPI service exposes REST and WebSocket endpoints and serves models (joblib/native/ONNX). A Next.js PWA is the operator interface, and an MCP server exposes read-only tools to AI assistants. See Figure 9.1 (`docs/02_system_blueprint.md`, §3.1).

## 9.2 Real-time pipeline

A simulation clock drives the stream in two modes. **REPLAY** walks through the held-out test period with archived weather, so live accuracy is a genuine out-of-sample measurement. **LIVE_SHADOW** re-stamps historical demand by whole weeks so weekday and hour align with today, and uses live Open-Meteo forecasts. Each tick closes an hour and triggers feature construction, inference for all models, calibration, SHAP, alerting and push to the browser. The latency budget is under 2 s per tick for all sites and models.

## 9.3 Autonomous agents

| Agent | Responsibility |
|---|---|
| Telemetry Replay / OCPP | Emits the closed hour per site (optionally as OCPP MeterValues) |
| Weather Sync | Batched Open-Meteo fetch, vintage storage, staleness fallback |
| Forecast | Features → all models → CQR → predictions + SHAP |
| Arena Accuracy | Joins actuals with past predictions; live metrics; ACI |
| Drift Detection | PSI on hour-of-week-standardised features, Page-Hinkley on residuals, coverage monitoring |
| Smart-Grid Alert & Reroute | Thresholds with hysteresis; greedy headroom-aware reroute; Web Push |
| Retrain (champion–challenger) | Sliding-window retrain; promotion gate on WAPE and coverage |
| Station Discovery | OpenChargeMap context stations with cold-start priors |
| Watchdog | Heartbeats, freshness, degraded-mode flags |

## 9.4 User interface

| Screen | Meaningful input → output |
|---|---|
| Command Center | Play/step clock, 24 h scrubber → KPI ribbon, risk-coloured map with halos and reroute arcs, alert feed |
| Station detail | Station, model → 48 h actual + 24 h forecast with 90% band, capacity line, occupancy forecast, SHAP waterfall |
| Model Arena | Metric, horizon, test/live → leaderboard, horizon curves, DM matrix |
| What-If Simulator | Temperature Δ, rain, holiday → baseline vs scenario, new alerts, OOD warnings |
| Explainability | Model → global SHAP |
| Data & Methodology | → data card, EDA, split diagram, leakage-test status |
| Ops | Mode, speed, retrain approval → agent health, drift, registry |

The PWA is installable, works offline with the last snapshot, and sends Web Push notifications for CRITICAL alerts.

## 9.5 MCP integration

The MCP server offers `get_station_forecast`, `get_network_status`, `explain_peak_shap`, `simulate_weather_shock`, `compare_models`, `get_alerts` and `get_drift_report`. It calls the REST API with a read-only token, so assistants see the same validated numbers as the UI and cannot change live state. An in-app copilot uses the same tools and is instructed to report only numbers returned by tools.

## 9.6 Testing and quality

Unit and property tests (pytest + hypothesis) cover features, splits, metrics and conformal code. Leakage tests run in CI. API tests run against a real PostgreSQL container. Frontend tests use Vitest, and end-to-end tests use Playwright running the viva demo script. A Lighthouse PWA audit is part of CI.

## 9.7 Deployment

`docker compose up` starts the database, API, worker, web app and MCP server. Configuration is via environment variables (README). Images are built in CI on tagged releases.

# 10. Ethics, Responsible AI, Limitations and Future Work

## 10.1 Privacy

Charging sessions reveal where and when individuals park. User IDs, card/MAC identifiers and zip codes are removed at ingestion; session IDs are salted hashes. Only site-level hourly aggregates are stored and served, and no API or MCP tool returns session-level records. With few users at a site, small aggregates could still be re-identifying; sites with fewer than ⟨k⟩ sessions per day are excluded from public views.

## 10.2 Bias and equity

The models learn from where chargers already exist and are used. Planning by predicted demand alone would keep investing in already well-served (often wealthier) areas and starve under-served neighbourhoods, where demand is suppressed by lack of supply (censoring). Planning outputs must therefore be paired with access and equity indicators, and saturation should be read as unmet demand, not as low interest. Training data from one Californian city may not represent other regions.

## 10.3 Harms of misprediction

Errors are asymmetric. Under-forecasting a peak risks transformer overload, outages and equipment ageing. Over-forecasting causes unnecessary curtailment, higher prices or needless reroutes. The system therefore alerts on the upper quantile, reports calibrated coverage, flags out-of-distribution scenarios and degraded inputs, and keeps a human operator in the loop: it never actuates chargers.

## 10.4 Responsible use

P.O.W.E.R. is a decision-support demonstration. Its outputs are not validated for operational grid control without utility data, real transformer ratings and field trials. All assumptions (diversity factor, driver willingness to reroute, tariff elasticity) are displayed in the UI and the data card.

## 10.5 Future work

- Censored-likelihood (Tobit) models and demand reconstruction from queueing.
- 15-minute resolution from ACN pilot signals.
- Hierarchical reconciliation (MinT) across sites and network.
- Graph neural networks over the station network.
- Coupling forecasts to smart-charging schedulers through OCPP charging profiles.
- Transfer to Indian city datasets.
- Real historical weather forecasts for training.

# 11. Reproducibility, Team Contribution and Viva Readiness

## 11.1 Reproducibility checklist

| Item | Location |
|---|---|
| Source code (modular) | `packages/`, `ml/`, `services/`, `apps/` |
| Pinned dependencies | `uv.lock`, `pnpm-lock.yaml`, Docker images |
| Dataset references + checksums | `data/manifest.json`, Chapter 3 |
| Preprocessing steps | `ml/src/power_ml/ingest`, `resample.py`, Chapter 4 |
| Configuration | `configs/*.yaml` |
| Seeds | `SEED=42` everywhere |
| Run instructions | `README.md` (`make data → train → evaluate → up`) |
| Experiment logs | `mlruns/` (MLflow) |
| Frozen results | `reports/metrics_test.json`, figures |

## 11.2 Individual contribution matrix

| Member | Primary ownership | Secondary | Evidence (commits / modules) |
|---|---|---|---|
| ⟨Name 1⟩ | Data ingestion, cleaning, proration, data card | EDA | `ml/ingest`, `resample.py` |
| ⟨Name 2⟩ | Models, tuning, evaluation, ablations | Conformal | `ml/models`, `evaluate.py` |
| ⟨Name 3⟩ | Database, API, workers, real-time | Drift | `services/api`, `services/worker` |
| ⟨Name 4⟩ | PWA, map, XAI, MCP | Report | `apps/web`, `services/mcp`, `explain.py` |

## 11.3 Viva demonstration script

1. Data & Methodology: counts, window, leakage tests.
2. Model Arena: leaderboard, horizon curve, DM, ensemble vs components.
3. Command Center: play the clock and watch live updates.
4. Scrub +1…+24 h and observe a CRITICAL station and its reroute arcs.
5. Station detail: interval band and SHAP narrative.
6. What-If: +10 °C and holiday, with OOD warning.
7. Ops: drift panel, champion–challenger gate, LIVE_SHADOW with live weather.
8. MCP: natural-language grid briefing.

# 12. Conclusion

⟨Summarise: the best model and its improvement over the seasonal baseline at short and day-ahead horizons; calibrated interval coverage; the most influential drivers from SHAP; the main operational insights (overload-prone sites, idle occupancy); and how the real-time system turns forecasts into explained, actionable alerts. Close with the main limitations and the most valuable next step.⟩

# References

Adnane, M., Khoumsi, A., & Trovão, J. P. F. (2023). Efficient management of energy consumption of electric vehicles using machine learning — A systematic and comprehensive survey. *Energies, 16*, 4897. https://doi.org/10.3390/en16134897

Amezquita, H., Guzman, C. P., & Morais, H. (2024). Forecasting electric vehicles' charging behavior at charging stations: A data science-based approach. *Energies, 17*, 3396. https://doi.org/10.3390/en17143396

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics, 13*(3), 253–263.

Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under distribution shift. *Advances in Neural Information Processing Systems, 34*.

Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting, 13*(2), 281–291.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735–1780.

Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems, 30*.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785–794.

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.

Lee, Z. J., Li, T., & Low, S. H. (2019). ACN-Data: Analysis and applications of an open EV charging dataset. *Proceedings of the Tenth ACM International Conference on Future Energy Systems (e-Energy)*, 139–149. https://doi.org/10.1145/3307772.3328313

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*.

Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized quantile regression. *Advances in Neural Information Processing Systems, 32*.

Shern, S. J., Sarker, M. T., Haram, M. H. S. M., et al. (2024). Artificial intelligence optimization for user prediction and efficient energy distribution in electric vehicle smart charging systems. *Energies, 17*, 5772. https://doi.org/10.3390/en17225772

Skala, R., Elgalhud, M. A. T. A., Grolinger, K., & Mir, S. (2023). Interval load forecasting for individual households in the presence of electric vehicle charging. *Energies, 16*(10), 4093. https://doi.org/10.3390/en16104093

Zhu, J., Yang, Z., Mourshed, M., et al. (2019). Electric vehicle charging load forecasting: A comparative study of deep learning approaches. *Energies, 12*, 2692. https://doi.org/10.3390/en12142692

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD*, 2623–2631.

# Appendix A. Feature Dictionary

| Feature | Type | Definition | Anchor | Models |
|---|---|---|---|---|
| h | int 1–24 | Forecast horizon | — | Ridge, RF, GBDT |
| hour_sin / hour_cos | float | sin/cos(2π·hour/24), local time of t | Target | All |
| dow_sin / dow_cos | float | sin/cos(2π·dow/7) | Target | All |
| doy_sin / doy_cos | float | sin/cos(2π·doy/365.25) | Target | Ridge, RF, GBDT |
| how_f1..3 sin/cos | float | sin/cos(2πk·how/168) | Target | Ridge |
| is_weekend, is_holiday, is_bridge_day | binary | Calendar flags (US-CA holidays) | Target | All / trees |
| lag_24h, lag_48h, lag_168h | kWh | y(t−24), y(t−48), y(t−168) | Target | Ridge, RF, GBDT |
| same_hour_7d_avg | kWh | mean of y(t−24k), k = 1..7 | Target | Ridge, RF, GBDT |
| same_how_4w_mean | kWh | mean of y(t−168k), k = 1..4 | Target | Ridge, RF, GBDT |
| y_org_0/1/2 | kWh | y(T), y(T−1), y(T−2) | Origin | Ridge, RF, GBDT |
| hours_since_nonzero | h | T − last hour with y > 0 | Origin | Trees |
| rolling_mean_24h, rolling_std_24h, rolling_max_24h | kWh | Statistics over y(T−23..T) | Origin | Trees (+Ridge mean) |
| ewma_6h | kWh | α = 0.3 exponential average up to T | Origin | Ridge, RF, GBDT |
| level_28d, level_ratio | kWh, ratio | 672 h mean; 24 h mean / (28 d mean + 0.1) | Origin | Ridge, RF, GBDT |
| active_sessions_T, occupied_ports_T, idle_ports_T, arrivals_T, headroom_ports_T | count | Occupancy state at end of hour T | Origin | Trees |
| saturation_hours_24h | count | Saturated hours in last 24 h | Origin | Trees |
| temperature_2m_c, precipitation_mm, is_rain | float/binary | Weather at t | Target | All |
| cold_stress, heat_stress | °C | max(0, 10 − T), max(0, T − 28) | Target | All (monotone in GBDT) |
| station_id | categorical | Native / one-hot / embedding | Static | All |
| total_ports, poi_category, station_age_days | int/cat/int | Station metadata | Static | Trees, prior |

# Appendix B. Hyper-parameter Search Spaces

| Model | Parameter | Range | Scale |
|---|---|---|---|
| Ridge | alpha | 1e−3 – 1e3 | log |
| Random Forest | n_estimators / max_depth / min_samples_leaf / max_features | 200–600 / 8–24 / 5–100 / {0.3, 0.5, sqrt} | int / int / int / cat |
| XGBoost | max_depth / eta / min_child_weight | 3–10 / 0.01–0.2 / 1–50 | int / log / log |
| XGBoost | subsample / colsample_bytree / lambda / alpha / tweedie_variance_power | 0.5–1 / 0.5–1 / 1e−3–10 / 1e−3–10 / 1.1–1.7 | uniform / uniform / log / log / uniform |
| LightGBM | num_leaves / min_data_in_leaf / learning_rate | 15–255 / 20–500 / 0.01–0.2 | int / log / log |
| LightGBM | feature_fraction / bagging_fraction / lambda_l1 / lambda_l2 | 0.5–1 / 0.5–1 / 1e−3–10 / 1e−3–10 | uniform / uniform / log / log |
| LSTM | hidden / dropout / lr / lookback | {32, 64, 128} / 0–0.3 / 1e−4–3e−3 / {72, 168} | cat / uniform / log / cat |

# Appendix C. API Summary

| Endpoint | Purpose |
|---|---|
| GET /api/v1/network/kpis | KPI ribbon |
| GET /api/v1/network/snapshot?h= | Map state for scrubber |
| GET /api/v1/stations/{id}/forecast | 24 h forecast with intervals |
| GET /api/v1/stations/{id}/explain | Local SHAP |
| GET /api/v1/models/leaderboard | Rubric comparison table |
| POST /api/v1/scenarios/simulate | What-if |
| GET /api/v1/alerts, /reroutes, /drift | Operations |
| WS /ws/stream | telemetry, forecast, alert, drift, sim events |

# Appendix D. Viva Question Bank

| Likely question | Prepared answer (outline) |
|---|---|
| Why not shuffle the data? | Autocorrelation; shuffling leaks future information into training and inflates accuracy |
| How do you know there is no leakage? | Anchor classes; lag ≥ h rule; embargoed splits; future-poisoning test in CI |
| Why WAPE instead of MAPE? | Many zero-demand hours make MAPE undefined; WAPE is scale-free and well defined |
| Why direct rather than recursive? | No error compounding, identical train/serve features; ablation A1 shows the difference |
| What does a 90% interval guarantee? | Under exchangeability, CQR gives ≥ 90% marginal coverage; ACI corrects under drift; empirical PICP is reported |
| Why Tweedie loss? | Non-negative, zero-inflated, right-skewed target; compound Poisson–Gamma fits it; ablation A2 |
| XGBoost vs LightGBM? | Same family; depth-wise vs leaf-wise growth; after tuning, accuracy is similar; we report both plus significance |
| What happens with a +15 °C shock? | Trees plateau outside training range; monotone constraints keep direction; UI flags OOD |
| Where do transformer ratings come from? | They are not public; synthetic DF × rated kW, with sensitivity analysis |
| How is real time achieved without real chargers? | Replay of held-out data on a simulation clock (true out-of-sample) and live-shadow mode with live weather; optional OCPP emulation |
| What if a meter tick is missing? | Seasonal imputation with flag; > 6 h gap → seasonal-naive fallback and degraded status |
| How do you detect drift? | PSI on hour-of-week-standardised features, Page-Hinkley on residuals, coverage monitoring; COVID case study |
| What are the ethical risks? | Location privacy; equity bias from censored demand; asymmetric harm of under-forecasting |

# Appendix E. Rubric Self-Assessment Checklist

| Criterion | Check before submission | Done |
|---|---|---|
| Problem (10) | Users, decisions, measurable objectives stated | ☐ |
| Dataset (15) | Exact counts, sources, licences, limitations, cleaning table, leakage tests green | ☐ |
| Implementation (20) | Equations for all models; code modular; configs versioned | ☐ |
| Evaluation (20) | Chronological split + embargo; ≥ 3 algorithms + baseline; per-horizon results; DM tests; ensemble vs components | ☐ |
| Insights (10) | Global + local SHAP; residual strata; censoring; OOD stress test; ≥ 3 evidence-backed findings | ☐ |
| Prototype (15) | `docker compose up` works on a clean machine; demo script rehearsed | ☐ |
| Report & viva (10) | Contribution matrix; reproducibility; ethics; viva Q&A rehearsed | ☐ |
