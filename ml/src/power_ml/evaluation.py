"""Evaluation (report Chapters 6-7): one function produces every results table.

Outputs (in ``reports/``):
* ``metrics_<split>.json``            leaderboard: point + interval metrics, train time, latency
* ``metrics_by_horizon_<split>.csv``  WAPE / MAE / RMSE per model and horizon
* ``metrics_by_station_<split>.csv``  WAPE per model and site
* ``dm_tests_<split>.csv``            Diebold-Mariano (HLN) + Holm, h = 1 and h = 24
* ``predictions_<split>.parquet``     row-level predictions for error analysis / SHAP (git-ignored)
Every test-split evaluation is appended to ``reports/test_evaluations.log`` so it stays auditable.
"""

from __future__ import annotations

import itertools
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from power_core import metrics
from power_core.config import data_config
from power_core.conformal import calibrator_from_dict
from power_core.splits import SplitBounds
from power_core.stats import diebold_mariano, holm
from power_ml.training import ALPHA, QUANTILE, load_artifacts, load_split, predict_all

LEADERBOARD_ORDER = ("B0_persistence", "B1_seasonal_naive_24h", "B2_seasonal_naive_168h", "B3_climatology",
                     "M1_ridge", "M2_random_forest", "M3_xgboost", "M4_lightgbm", QUANTILE, "E1_ensemble")
DM_MODELS = ("B1_seasonal_naive_24h", "B3_climatology", "M1_ridge", "M2_random_forest", "M3_xgboost",
             "M4_lightgbm", "E1_ensemble")


def _train_reference(root: Path, source: str) -> tuple[float, pd.Series]:
    """MASE scale and per-station p90 threshold, both from the *training* block of the hourly panel."""
    panel = pd.read_parquet(root / "data" / "processed" / f"panel_{source}.parquet",
                            columns=["station_id", "ts", "energy_kwh"])
    b = SplitBounds.from_dict(json.loads((root / "data" / "processed" / f"splits_{source}.json").read_text(encoding="utf-8")))
    tr = panel.loc[(panel["ts"] >= b.train_start) & (panel["ts"] <= b.train_end)]
    return metrics.seasonal_naive_scale(tr, 24), tr.groupby("station_id")["energy_kwh"].quantile(0.90)


def _latency_ms(model: Any, ens: Any, models: dict[str, Any], frame: pd.DataFrame, name: str, reps: int = 15) -> float:
    one = frame.loc[frame["origin"] == frame["origin"].iloc[len(frame) // 2]]
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        if name == ens.name:
            ens.predict({m: models[m].predict(one) for m in ens.members})
        else:
            model.predict(one)
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 2)


def evaluate(root: Path, source: str, split: str, log: Any = print) -> dict[str, Any]:
    models, ens, manifest = load_artifacts(root)
    frame = load_split(root, source, split)
    preds, quants = predict_all(models, ens, frame)
    y, h = frame["y"].to_numpy(), frame["h"].to_numpy()
    upper = frame["rated_kw_total"].to_numpy(float)
    scale, p90 = _train_reference(root, source)
    peak_thr = frame["station_id"].astype(str).map(p90).to_numpy(float)
    cal = manifest["calibrators"]

    board: dict[str, dict[str, Any]] = {}
    for name in [n for n in LEADERBOARD_ORDER if n in preds]:
        p = preds[name]
        point = metrics.point_report(y, p, scale=scale, peak_threshold=peak_thr)
        row: dict[str, Any] = {k: round(v, 4) for k, v in point.items()}
        lo, hi = calibrator_from_dict(cal[name]).predict(p, h, lower=0.0, upper=upper)
        row["interval_split_conformal"] = {k: round(v, 4) for k, v in metrics.interval_report(y, lo, hi, ALPHA).items()}
        if name == QUANTILE:
            row["interval_raw_quantiles"] = {k: round(v, 4) for k, v in
                                             metrics.interval_report(y, quants[0.05], quants[0.95], ALPHA).items()}
            lo, hi = calibrator_from_dict(cal[f"{QUANTILE}_cqr"]).predict(quants[0.05], quants[0.95], h,
                                                                          lower=0.0, upper=upper)
            row["interval_cqr"] = {k: round(v, 4) for k, v in metrics.interval_report(y, lo, hi, ALPHA).items()}
            row["pinball"] = {str(a): round(metrics.pinball(y, q, a), 4) for a, q in quants.items()}
        mi = manifest["models"].get(name, {})
        row["train_time_s"] = mi.get("train_time_s")
        row["latency_ms_one_origin"] = _latency_ms(models.get(name), ens, models, frame, name)
        board[name] = row

    # ---- by horizon / by station
    long = []
    for name, p in preds.items():
        df = pd.DataFrame({"h": h, "station_id": frame["station_id"].astype(str), "y": y, "p": p})
        for hh, g in df.groupby("h"):
            long.append({"model": name, "h": int(hh), "wape": metrics.wape(g.y, g.p), "mae": metrics.mae(g.y, g.p),
                         "rmse": metrics.rmse(g.y, g.p)})
    by_h = pd.DataFrame(long)
    by_st = pd.DataFrame([
        {"model": n, "station_id": s, "wape": metrics.wape(g["y"], g["p"]), "n": len(g)}
        for n, p in preds.items()
        for s, g in pd.DataFrame({"station_id": frame["station_id"].astype(str), "y": y, "p": p}).groupby("station_id")
    ])

    # ---- day-ahead issued at local midnight (origin 23:00 -> targets 00:00..23:00)
    tz = data_config(root).timezone
    midnight = (pd.DatetimeIndex(frame["origin"]).tz_convert(tz).hour == 23)
    for name in board:
        board[name]["wape_day_ahead_midnight"] = round(metrics.wape(y[midnight], preds[name][midnight]), 4)

    # ---- Diebold-Mariano on per-origin network absolute error (one series per horizon)
    dm_rows = []
    for hh in (1, 24):
        m = h == hh
        base = pd.DataFrame({"origin": frame["origin"].to_numpy()[m], "y": y[m]})
        errs = {n: base.assign(e=np.abs(y[m] - preds[n][m])).groupby("origin")["e"].sum().to_numpy()
                for n in DM_MODELS if n in preds}
        pairs = list(itertools.combinations(errs, 2))
        res = [diebold_mariano(errs[a], errs[b], h=hh, loss="abs") for a, b in pairs]
        adj = holm([r.p_value for r in res])
        for (a, b), r, pa in zip(pairs, res, adj, strict=True):
            dm_rows.append({"h": hh, "model_1": a, "model_2": b, "dm_stat": round(r.statistic, 3),
                            "p_value": r.p_value, "p_holm": pa, "mean_abs_err_diff": round(r.mean_loss_diff, 4),
                            "better": a if r.statistic < 0 else b, "significant_5pct": pa < 0.05})
    dm = pd.DataFrame(dm_rows)

    rep = root / "reports"
    rep.mkdir(exist_ok=True)
    summary = {
        "split": split, "model_version": manifest["version"], "git_sha_of_models": manifest["git_sha"],
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"), "rows": len(frame),
        "origin_range": [str(frame["origin"].min()), str(frame["origin"].max())],
        "mase_scale_kwh": round(scale, 4), "interval_level": 1 - ALPHA,
        "ensemble_weights": manifest["models"]["E1_ensemble"]["weights"], "leaderboard": board,
    }
    (rep / f"metrics_{split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    by_h.to_csv(rep / f"metrics_by_horizon_{split}.csv", index=False)
    by_st.to_csv(rep / f"metrics_by_station_{split}.csv", index=False)
    dm.to_csv(rep / f"dm_tests_{split}.csv", index=False)
    out = frame[["station_id", "origin", "target_time", "h", "y"]].copy()
    for n, p in preds.items():
        out[n] = p.astype(np.float32)
    for a, q in quants.items():
        out[f"q{int(a * 100):02d}"] = q.astype(np.float32)
    out.to_parquet(rep / f"predictions_{split}.parquet", index=False)
    if split == "test":
        with (rep / "test_evaluations.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{summary['evaluated_at']}\tmodels={manifest['version']}@{manifest['git_sha'][:10]}\trows={len(frame)}\n")
    log(leaderboard_text(summary))
    return summary


def leaderboard_text(summary: dict[str, Any]) -> str:
    lines = [f"{'model':<26}{'WAPE':>8}{'MASE':>8}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'peakMAE':>9}{'PICP':>7}{'train_s':>9}{'ms':>7}"]
    for n, r in summary["leaderboard"].items():
        iv = r.get("interval_cqr") or r["interval_split_conformal"]
        lines.append(f"{n:<26}{r['wape']:>8.3f}{r.get('mase', float('nan')):>8.3f}{r['mae']:>8.3f}{r['rmse']:>8.3f}"
                     f"{r['r2']:>8.3f}{r.get('peak_mae', float('nan')):>9.3f}{iv['picp']:>7.3f}"
                     f"{(r['train_time_s'] or 0):>9.1f}{r['latency_ms_one_origin']:>7.1f}")
    return "\n".join(lines)
