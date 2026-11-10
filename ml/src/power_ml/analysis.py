"""Interpretation, error analysis and the COVID stress test (report Chapter 8).

* SHAP: exact TreeSHAP from LightGBM (``pred_contrib=True``) for the champion, the q50 head. Its
  output is on the identity scale, so contributions add up in kWh: base + sum(contributions) = forecast.
* Error analysis: WAPE by hour x weekday, station, saturation, temperature, rain, holiday, horizon;
  residual autocorrelation; an example forecast with its 90% interval.
* Stress test: frozen models on Mar-Jun 2020 plus two drift alarms (Page-Hinkley on daily
  forecast bias, seasonal PSI on weekly demand), reporting when each first fires.

All outputs go to ``reports/`` (JSON/CSV) and ``reports/figures/`` (PNG).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from power_core import metrics  # noqa: E402
from power_core.config import data_config  # noqa: E402
from power_core.conformal import calibrator_from_dict  # noqa: E402
from power_core.drift import Cusum, seasonal_psi  # noqa: E402
from power_core.features.registry import BY_NAME  # noqa: E402
from power_ml.training import load_artifacts, load_split  # noqa: E402

CHAMPION = "M4_lightgbm_quantile"
MEAN_MODEL = "M4_lightgbm"
DISPLAY = {
    "h": "forecast horizon",
    "hour_sin": "hour of day (sin)",
    "hour_cos": "hour of day (cos)",
    "dow_sin": "day of week (sin)",
    "dow_cos": "day of week (cos)",
    "doy_sin": "season (sin)",
    "doy_cos": "season (cos)",
    "is_weekend": "weekend",
    "is_holiday": "public holiday",
    "is_bridge_day": "bridge day",
    "lag_24h": "same hour yesterday",
    "lag_48h": "same hour 2 days ago",
    "lag_168h": "same hour last week",
    "same_hour_7d_avg": "7-day same-hour average",
    "same_how_4w_mean": "4-week same-hour-of-week average",
    "y_org_0": "load in the last hour",
    "y_org_1": "load 2 hours ago",
    "y_org_2": "load 3 hours ago",
    "hours_since_nonzero": "hours since last charging",
    "rolling_mean_24h": "24 h average load",
    "rolling_std_24h": "24 h load volatility",
    "rolling_max_24h": "24 h peak load",
    "ewma_6h": "recent load trend (EWMA)",
    "level_28d": "28-day load level",
    "level_ratio": "today vs month level",
    "active_sessions_T": "cars charging now",
    "occupied_ports_T": "ports occupied now",
    "idle_ports_T": "plugged-in idle cars",
    "arrivals_T": "arrivals last hour",
    "headroom_ports_T": "free ports now",
    "saturation_hours_24h": "hours full in last day",
    "temperature_2m_c": "temperature",
    "precipitation_mm": "rain (mm)",
    "is_rain": "raining",
    "cold_stress": "cold",
    "heat_stress": "heat",
    "station_id": "station",
    "total_ports": "number of ports",
    "poi_category": "site type",
    "station_age_days": "station age",
}


def _fig_dir(root: Path) -> Path:
    d = root / "reports" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig: plt.Figure, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path.name)


# ============================================================================ SHAP
def _fmt(v: Any) -> str | float:
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return str(v)


def shap_contributions(model: Any, frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Exact TreeSHAP for the q50 head: (contributions per feature, base value per row)."""
    q50 = model.models_[0.50]
    raw = q50.predict(frame[model.columns_], pred_contrib=True)
    return pd.DataFrame(raw[:, :-1], columns=model.columns_, index=frame.index), raw[:, -1]


def run_shap(root: Path, source: str, n_sample: int = 4000, seed: int = 42) -> dict[str, Any]:
    models, _, _ = load_artifacts(root)
    model = models[CHAMPION]
    test = load_split(root, source, "test")
    sample = test.groupby(["station_id", "h"], observed=True, group_keys=False).sample(
        n=max(1, n_sample // (test["station_id"].nunique() * 24)), random_state=seed
    )
    contrib, base = shap_contributions(model, sample)
    pred = model.models_[0.50].predict(sample[model.columns_])
    additivity_err = float(np.max(np.abs(base + contrib.sum(axis=1).to_numpy() - pred)))

    imp = contrib.abs().mean().sort_values(ascending=False)
    groups = imp.groupby(lambda c: BY_NAME[c].group).sum().sort_values(ascending=False)
    imp.rename("mean_abs_shap_kwh").to_csv(root / "reports" / "shap_global.csv", index_label="feature")
    figs = _fig_dir(root)

    top = imp.head(20)[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([DISPLAY.get(c, c) for c in top.index], top.to_numpy(), color="#2E75B6")
    ax.set_xlabel("mean |SHAP| (kWh)")
    ax.set_title("Global feature importance - LightGBM q50 (test sample)")
    out = {"bar": _save(fig, figs / "shap_bar.png")}

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(groups.index[::-1], groups.to_numpy()[::-1], color="#1F3864")
    ax.set_xlabel("sum of mean |SHAP| (kWh)")
    ax.set_title("Importance by feature group")
    out["groups"] = _save(fig, figs / "shap_groups.png")

    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for i, c in enumerate(imp.head(15).index[::-1]):
        v = sample[c]
        col = v.cat.codes.to_numpy(float) if isinstance(v.dtype, pd.CategoricalDtype) else v.to_numpy(float)
        pct = pd.Series(col).rank(pct=True).to_numpy()
        ax.scatter(contrib[c], i + rng.uniform(-0.3, 0.3, len(contrib)), c=pct, cmap="coolwarm", s=4, alpha=0.6)
    ax.set_yticks(range(15), [DISPLAY.get(c, c) for c in imp.head(15).index[::-1]])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("SHAP value (kWh); colour = feature value (blue low, red high)")
    ax.set_title("SHAP beeswarm - LightGBM q50")
    out["beeswarm"] = _save(fig, figs / "shap_beeswarm.png")

    # ---- local: the three highest forecast peaks (distinct stations) at h = 6
    cand = test.loc[test["h"] == 6].copy()
    cand["pred"] = model.models_[0.50].predict(cand[model.columns_])
    picks = cand.sort_values("pred", ascending=False).drop_duplicates("station_id").head(3)
    c_loc, b_loc = shap_contributions(model, picks)
    tz = data_config(root).timezone
    examples = []
    for k, (idx, row) in enumerate(picks.iterrows()):
        cvec = c_loc.loc[idx].sort_values(key=np.abs, ascending=False)
        top6, rest = cvec.head(6), float(cvec.iloc[6:].sum())
        when = pd.Timestamp(row["target_time"]).tz_convert(tz)
        parts = [f"{v:+.1f} from {DISPLAY.get(f, f)}" for f, v in top6.items()]
        narrative = (
            f"{row['station_id']} at {when:%a %d %b %Y %H:%M}: base {b_loc[k]:.1f} kWh "
            + " ".join(parts)
            + f" {rest:+.1f} other = {row['pred']:.1f} kWh forecast (actual {row['y']:.1f})"
        )
        examples.append(
            {
                "station_id": str(row["station_id"]),
                "target_time_local": str(when),
                "base_kwh": round(float(b_loc[k]), 3),
                "forecast_kwh": round(float(row["pred"]), 3),
                "actual_kwh": round(float(row["y"]), 3),
                "top_contributions": [
                    {"feature": f, "display": DISPLAY.get(f, f), "value": _fmt(row[f]), "shap_kwh": round(float(v), 3)}
                    for f, v in top6.items()
                ],
                "other_kwh": round(rest, 3),
                "narrative": narrative,
            }
        )
        steps = [("base", b_loc[k]), *[(DISPLAY.get(f, f), v) for f, v in top6.items()], ("other", rest)]
        fig, ax = plt.subplots(figsize=(7, 3.8))
        cum = 0.0
        for j, (_label, v) in enumerate(steps):
            start = 0.0 if j == 0 else cum
            ax.barh(j, v, left=start, color="#7F7F7F" if j == 0 else ("#C00000" if v > 0 else "#2E75B6"))
            cum = start + v
        ax.axvline(cum, color="black", ls="--", lw=0.8)
        ax.set_yticks(range(len(steps)), [s[0] for s in steps])
        ax.invert_yaxis()
        ax.set_xlabel("kWh")
        ax.set_title(f"Why {row['pred']:.1f} kWh? {row['station_id']} {when:%a %d %b %H:%M}", fontsize=10)
        out[f"waterfall_{k + 1}"] = _save(fig, figs / f"shap_waterfall_{k + 1}.png")

    summary = {
        "model": CHAMPION,
        "sample_rows": len(sample),
        "additivity_max_abs_error_kwh": additivity_err,
        "top_features": {c: round(float(v), 4) for c, v in imp.head(15).items()},
        "group_importance": {g: round(float(v), 4) for g, v in groups.items()},
        "local_examples": examples,
        "figures": out,
    }
    (root / "reports" / "shap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# ============================================================================ error analysis
def _wape_table(df: pd.DataFrame, by: str | list[str], models: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, observed=True):
        r = {
            "group": key if not isinstance(key, tuple) else "|".join(map(str, key)),
            "n": len(g),
            "mean_y": round(float(g["y"].mean()), 3),
        }
        for m in models:
            r[f"wape_{m}"] = round(metrics.wape(g["y"], g[m]), 4)
        rows.append(r)
    return pd.DataFrame(rows)


def run_error_analysis(root: Path, source: str) -> dict[str, Any]:
    _, _, manifest = load_artifacts(root)
    p = pd.read_parquet(root / "reports" / "predictions_test.parquet")
    f = load_split(root, source, "test")
    if not (p["origin"].equals(f["origin"]) and (p["h"].to_numpy() == f["h"].to_numpy()).all()):
        raise ValueError("predictions_test.parquet is not row-aligned with the test features; re-run evaluate")
    tz = data_config(root).timezone
    df = p.copy()
    loc = pd.DatetimeIndex(df["target_time"]).tz_convert(tz)
    df["hour"], df["weekday"] = loc.hour, loc.dayofweek
    df["is_holiday"] = f["is_holiday"].to_numpy() > 0
    df["rain"] = np.where(f["is_rain"].to_numpy() > 0, "rain", "dry")
    df["temp_tercile"] = pd.qcut(f["temperature_2m_c"], 3, labels=["cool", "mild", "warm"]).to_numpy()
    n = f["total_ports"].to_numpy(float)
    df["saturated"] = np.where(f["occ_target"].to_numpy() * n >= n - 0.5, "saturated", "not saturated")
    df["h_bucket"] = pd.cut(df["h"], [0, 3, 12, 24], labels=["1-3", "4-12", "13-24"]).astype(str)
    models = (CHAMPION, MEAN_MODEL, "B1_seasonal_naive_24h", "B3_climatology")
    tables = {
        k: _wape_table(df, k, models)
        for k in ("station_id", "saturated", "temp_tercile", "rain", "is_holiday", "h_bucket", "weekday", "hour")
    }
    rep = root / "reports"
    for k, t in tables.items():
        t.to_csv(rep / f"error_by_{k}.csv", index=False)
    figs = _fig_dir(root)
    out: dict[str, str] = {}

    heat = df.assign(ae=(df["y"] - df[CHAMPION]).abs()).groupby(["weekday", "hour"]).agg(ae=("ae", "sum"), y=("y", "sum"))
    mat = (heat["ae"] / heat["y"].replace(0, np.nan)).unstack("hour")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="magma_r", vmin=0, vmax=min(2.0, np.nanmax(mat.to_numpy())))
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2), [f"{x:02d}" for x in range(0, 24, 2)])
    ax.set_xlabel("local hour of target")
    ax.set_title("WAPE by weekday x hour - LightGBM q50 (test)")
    fig.colorbar(im, ax=ax, label="WAPE")
    out["heatmap"] = _save(fig, figs / "error_heatmap_weekday_hour.png")

    st = tables["station_id"].sort_values(f"wape_{CHAMPION}")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(st))
    ax.bar(x - 0.2, st[f"wape_{CHAMPION}"], 0.4, label="LightGBM q50", color="#1F3864")
    ax.bar(x + 0.2, st["wape_B1_seasonal_naive_24h"], 0.4, label="seasonal naive", color="#BFBFBF")
    ax.set_xticks(x, [s.replace("PA_", "") for s in st["group"]], rotation=35, ha="right")
    ax.set_ylabel("WAPE")
    ax.legend()
    ax.set_title("Test WAPE by site")
    out["by_station"] = _save(fig, figs / "error_by_station.png")

    acf_lags = np.arange(1, 49)
    acfs = []
    for _, g in df.loc[df["h"] == 1].sort_values("target_time").groupby("station_id", observed=True):
        r = (g["y"] - g[CHAMPION]).to_numpy()
        r = r - r.mean()
        denom = np.dot(r, r)
        acfs.append([np.dot(r[k:], r[:-k]) / denom for k in acf_lags])
    acf = np.mean(acfs, axis=0)
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.bar(acf_lags, acf, color="#2E75B6")
    ax.axhline(1.96 / np.sqrt(len(df) / df["station_id"].nunique() / 24), color="red", ls="--", lw=0.8)
    ax.axhline(-1.96 / np.sqrt(len(df) / df["station_id"].nunique() / 24), color="red", ls="--", lw=0.8)
    ax.set_xlabel("lag (hours)")
    ax.set_title("Residual autocorrelation, h = 1 (mean over sites)")
    out["acf"] = _save(fig, figs / "residual_acf.png")

    # example: two weeks, one site, h = 1 forecasts with the calibrated 90% band
    site = tables["station_id"].sort_values("mean_y").iloc[-1]["group"]
    ex = df.loc[(df["station_id"] == site) & (df["h"] == 1)].sort_values("target_time")
    ex = ex.loc[(ex["target_time"] >= ex["target_time"].min() + pd.Timedelta(days=21))].head(24 * 14)
    cqr = calibrator_from_dict(manifest["calibrators"][f"{CHAMPION}_cqr"])
    lo, hi = cqr.predict(ex["q05"].to_numpy(), ex["q95"].to_numpy(), ex["h"].to_numpy(), lower=0.0)
    t = pd.DatetimeIndex(ex["target_time"]).tz_convert(tz)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.fill_between(t, lo, hi, color="#9DC3E6", alpha=0.6, label="90% interval (CQR)")
    ax.plot(t, ex["y"], color="black", lw=1, label="actual")
    ax.plot(t, ex[CHAMPION], color="#C00000", lw=1, label="LightGBM q50 (h=1)")
    ax.set_ylabel("kWh per hour")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.set_title(f"{site}: two weeks of 1-hour-ahead forecasts (test period)")
    out["example"] = _save(fig, figs / "forecast_example.png")

    summary = {
        "model": CHAMPION,
        "tables": {k: f"error_by_{k}.csv" for k in tables},
        "highlights": {
            k: tables[k].set_index("group")[f"wape_{CHAMPION}"].to_dict()
            for k in ("saturated", "temp_tercile", "rain", "is_holiday", "h_bucket")
        },
        "residual_acf_lag1": round(float(acf[0]), 3),
        "residual_acf_lag24": round(float(acf[23]), 3),
        "example_site": site,
        "figures": out,
    }
    (rep / "error_analysis.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


# ============================================================================ COVID stress test
def run_stress_test(
    root: Path, source: str, cusum_k: float = 0.5, cusum_h: float = 5.0, psi_threshold: float = 0.2
) -> dict[str, Any]:
    """Needs ``evaluate --split ood`` first (reports/predictions_ood.parquet, metrics_ood.json)."""
    tz = data_config(root).timezone
    preds = {s: pd.read_parquet(root / "reports" / f"predictions_{s}.parquet") for s in ("val", "test", "ood")}

    def daily(p: pd.DataFrame) -> pd.DataFrame:
        d = p.assign(
            day=pd.DatetimeIndex(p["target_time"]).tz_convert(tz).normalize(),
            e=p["y"] - p[CHAMPION],
            ae=(p["y"] - p[CHAMPION]).abs(),
        )
        g = d.groupby("day").agg(y=("y", "sum"), f=(CHAMPION, "sum"), e=("e", "sum"), ae=("ae", "sum"))
        g["bias"] = g["e"] / g["f"].clip(lower=1e-6)  # signed: negative = over-forecast
        g["wape"] = g["ae"] / g["y"].clip(lower=1e-6)
        return g

    ref = daily(preds["val"])
    stream = pd.concat([daily(preds["test"]).assign(split="test"), daily(preds["ood"]).assign(split="ood")])
    mu, sd = ref["bias"].mean(), ref["bias"].std(ddof=0)
    # Two-sided CUSUM on the daily bias standardised by the validation reference. Page-Hinkley
    # re-estimates its own mean from the stream and false-alarmed in the first test week, so the
    # known-reference CUSUM is the right detector here. It resets after each alarm.
    cusum = Cusum(k=cusum_k, h=cusum_h)
    alarms = []
    for day, b in stream["bias"].items():
        direction = cusum.update(float((b - mu) / sd))
        if direction:
            alarms.append(
                {
                    "date": str(day.date()),
                    "split": stream.loc[day, "split"],
                    "direction": "over-forecast" if direction == "down" else "under-forecast",
                }
            )
    first_ood = next((a["date"] for a in alarms if a["split"] == "ood"), None)

    # seasonal PSI of observed demand, weekly windows vs the training-period reference
    train = load_split(root, source, "train")
    ref_df = train.loc[train["h"] == 1, ["how", "y"]]
    obs = pd.concat(
        [load_split(root, source, s).loc[lambda d: d["h"] == 1, ["target_time", "how", "y"]] for s in ("test", "ood")]
    )
    obs["week"] = pd.DatetimeIndex(obs["target_time"]).tz_convert(tz).tz_localize(None).to_period("W").start_time
    psi_weeks = {str(w.date()): round(seasonal_psi(ref_df, g, "y", bins=10), 4) for w, g in obs.groupby("week")}
    alarm_psi = next((w for w, v in psi_weeks.items() if v > psi_threshold), None)

    m_test = json.loads((root / "reports" / "metrics_test.json").read_text(encoding="utf-8"))["leaderboard"]
    m_ood = json.loads((root / "reports" / "metrics_ood.json").read_text(encoding="utf-8"))["leaderboard"]
    compare = {
        n: {
            "wape_test": m_test[n]["wape"],
            "wape_ood": m_ood[n]["wape"],
            "picp_test": (m_test[n].get("interval_cqr") or m_test[n]["interval_split_conformal"])["picp"],
            "picp_ood": (m_ood[n].get("interval_cqr") or m_ood[n]["interval_split_conformal"])["picp"],
        }
        for n in m_test
    }

    figs = _fig_dir(root)
    fig, ax1 = plt.subplots(figsize=(11, 3.6))
    ax1.plot(stream.index, stream["y"], color="black", lw=1, label="actual daily energy (all sites)")
    ax1.plot(stream.index, stream["f"], color="#C00000", lw=1, label="forecast (sum of h=1..24)")
    ax1.set_ylabel("kWh per day (summed over horizons)")
    ood_start = stream.loc[stream["split"] == "ood"].index.min()
    ax1.axvline(ood_start, color="grey", ls=":", lw=1)
    ax1.text(ood_start, ax1.get_ylim()[1] * 0.95, " COVID period", fontsize=8, color="grey")
    for i, a in enumerate(alarms):
        ax1.axvline(
            pd.Timestamp(a["date"], tz=tz),
            color="#ED7D31" if a["direction"] == "over-forecast" else "#548235",
            lw=1.2,
            label="CUSUM alarm (over-/under-forecast)" if i == 0 else None,
        )
    if alarm_psi is not None:
        ax1.axvline(
            pd.Timestamp(alarm_psi, tz=tz), color="#7030A0", lw=1.5, ls="--", label=f"seasonal PSI alarm (week of {alarm_psi})"
        )
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("COVID-19 stress test: frozen model vs reality, with drift alarms")
    fig_name = _save(fig, figs / "covid_stress_test.png")

    summary = {
        "model": CHAMPION,
        "reference": "validation-period daily bias",
        "cusum": {"k": cusum_k, "h": cusum_h, "alarms": alarms, "first_alarm_in_covid_period": first_ood},
        "seasonal_psi": {"threshold": psi_threshold, "first_alarm_week": alarm_psi, "weekly": psi_weeks},
        "ood_start": str(ood_start.date()),
        "models": compare,
        "figure": fig_name,
    }
    (root / "reports" / "stress_test.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
