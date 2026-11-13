"""A7 champion-challenger retraining: sliding-window LightGBM quantile model, recalibrated, shadow-scored, gated.

Timeline, all strictly before ``as_of`` (the newest observed hour):

    |<----------- train: window_days ----------->|emb|<-- calib: calib_days -->|<-- shadow: shadow_days -->|
                                                                                                       as_of

* **train**    fits the challenger with the champion's frozen hyper-parameters (early stopping on a
               time holdout at the end of the train block).
* **calib**    refits CQR on the challenger's own quantiles (fresh calibration for the current regime).
* **shadow**   scores champion and challenger on the same unseen hours. Neither model saw them.

Promotion gate: challenger WAPE <= ``wape_gain`` x champion WAPE, PICP inside [picp_low, picp_high],
and no station more than ``max_station_regression`` worse than the champion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from power_core import metrics
from power_core.config import FeatureConfig, RetrainCfg
from power_core.conformal import CQR
from power_core.features import build_direct_frame
from power_core.splits import time_holdout
from power_ml.models.boosting import LGBMQuantile

ALPHA = 0.10
HOUR = pd.Timedelta(hours=1)


@dataclass
class ChallengerResult:
    model: LGBMQuantile
    cqr: CQR
    as_of: pd.Timestamp
    windows: dict[str, str]
    shadow: dict[str, Any]
    gate: dict[str, Any]
    passed: bool
    train_rows: int
    seconds: float
    per_station: dict[str, dict[str, float]] = field(default_factory=dict)


def _score(y: np.ndarray, q50: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    return {
        "wape": round(float(metrics.wape(y, q50)), 4),
        "mae": round(float(metrics.mae(y, q50)), 4),
        "picp": round(float(metrics.picp(y, lo, hi)), 4),
    }


def fit_challenger(
    panel: pd.DataFrame,
    stations: pd.DataFrame,
    fcfg: FeatureConfig,
    rcfg: RetrainCfg,
    *,
    as_of: pd.Timestamp,
    champion: Any,
    champion_cqr: Any,
    params: dict[str, Any],
    origin_stride: int = 2,
    early_stopping_rounds: int = 100,
    holdout_frac: float = 0.15,
    tz: str = "America/Los_Angeles",
    holidays: tuple[str, str | None] = ("US", "CA"),
    seed: int = 42,
    n_jobs: int = -1,
) -> ChallengerResult:
    """``panel`` uses the model's station ids and holds observed hours only (ts <= as_of)."""
    t0 = time.perf_counter()
    as_of = pd.Timestamp(as_of)
    panel = panel.loc[panel["ts"] <= as_of].copy()
    shadow_start = as_of - pd.Timedelta(days=rcfg.shadow_days)
    calib_start = shadow_start - pd.Timedelta(days=rcfg.calib_days)
    train_end = calib_start - pd.Timedelta(hours=rcfg.embargo_hours)
    train_start = train_end - pd.Timedelta(days=rcfg.window_days)
    warm = pd.Timedelta(hours=fcfg.warmup_hours)
    panel = panel.loc[panel["ts"] >= train_start - warm]
    if panel.empty or panel["ts"].min() > train_start:
        raise ValueError(f"not enough history before {train_start} for a {rcfg.window_days}-day window")

    common = {"tz": tz, "holidays": holidays, "apply_warmup": False, "drop_missing_target": True}
    tr = build_direct_frame(panel.loc[panel["ts"] <= train_end], stations, fcfg, origin_stride_hours=origin_stride, **common)
    tr = tr.loc[(tr["origin"] >= train_start) & (tr["target_time"] <= train_end)].reset_index(drop=True)
    # Calibration + shadow at every origin; only the lag warm-up of history is needed before them.
    ev = build_direct_frame(panel.loc[panel["ts"] >= calib_start - warm], stations, fcfg, origin_stride_hours=1, **common)
    calib = ev.loc[(ev["origin"] >= calib_start) & (ev["target_time"] < shadow_start)].reset_index(drop=True)
    shadow = ev.loc[(ev["origin"] >= shadow_start) & (ev["target_time"] <= as_of)].reset_index(drop=True)
    if tr.empty or calib.empty or shadow.empty:
        raise ValueError("a retrain block is empty; widen the window or wait for more observed hours")

    fit_idx, hold_idx = time_holdout(tr["origin"], tr["target_time"], holdout_frac, rcfg.embargo_hours)
    model = LGBMQuantile(params=params, early_stopping_rounds=early_stopping_rounds, seed=seed, n_jobs=n_jobs)
    model.fit(tr.iloc[fit_idx].reset_index(drop=True), tr.iloc[hold_idx].reset_index(drop=True))

    qc = model.predict_quantiles(calib)
    cqr = CQR(alpha=ALPHA).fit(calib["y"].to_numpy(), qc[0.05], qc[0.95], calib["h"].to_numpy())

    y, h = shadow["y"].to_numpy(), shadow["h"].to_numpy()
    upper = shadow["rated_kw_total"].to_numpy(float)
    q_new = model.predict_quantiles(shadow)
    lo_n, hi_n = cqr.predict(q_new[0.05], q_new[0.95], h, lower=0.0, upper=upper)
    q_old = champion.predict_quantiles(shadow)
    lo_o, hi_o = champion_cqr.predict(q_old[0.05], q_old[0.95], h, lower=0.0, upper=upper)
    scores = {"challenger": _score(y, q_new[0.50], lo_n, hi_n), "champion": _score(y, q_old[0.50], lo_o, hi_o)}

    per_station: dict[str, dict[str, float]] = {}
    worst = 0.0
    sid = shadow["station_id"].astype(str).to_numpy()
    for s in np.unique(sid):
        k = sid == s
        if y[k].sum() <= 0:
            continue
        wn, wo = float(metrics.wape(y[k], q_new[0.50][k])), float(metrics.wape(y[k], q_old[0.50][k]))
        per_station[s] = {"challenger": round(wn, 4), "champion": round(wo, 4)}
        worst = max(worst, wn / wo - 1.0 if wo > 0 else 0.0)

    ch, cm = scores["challenger"], scores["champion"]
    checks = {
        "wape": {"ok": ch["wape"] <= rcfg.wape_gain * cm["wape"], "rule": f"challenger <= {rcfg.wape_gain} x champion",
                 "challenger": ch["wape"], "champion": cm["wape"]},
        "picp": {"ok": rcfg.picp_low <= ch["picp"] <= rcfg.picp_high, "rule": f"in [{rcfg.picp_low}, {rcfg.picp_high}]",
                 "challenger": ch["picp"]},
        "stations": {"ok": worst <= rcfg.max_station_regression,
                     "rule": f"no station worse by more than {rcfg.max_station_regression:.0%}",
                     "worst_regression": round(worst, 4)},
    }
    passed = all(c["ok"] for c in checks.values())
    return ChallengerResult(
        model=model,
        cqr=cqr,
        as_of=as_of,
        windows={
            "train": f"{train_start.isoformat()} .. {train_end.isoformat()}",
            "calib": f"{calib_start.isoformat()} .. {shadow_start.isoformat()}",
            "shadow": f"{shadow_start.isoformat()} .. {as_of.isoformat()}",
        },
        shadow={**scores, "rows": len(shadow)},
        gate=checks,
        passed=passed,
        train_rows=len(tr),
        seconds=round(time.perf_counter() - t0, 1),
        per_station=per_station,
    )
