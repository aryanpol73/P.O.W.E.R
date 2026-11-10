"""Forecast metrics (report Section 6.3). All inputs are 1-D array-likes of equal length."""

from __future__ import annotations

import numpy as np
import pandas as pd

ArrayLike = np.ndarray | pd.Series | list[float]


def _arr(*xs: ArrayLike) -> list[np.ndarray]:
    out = [np.asarray(x, dtype=float) for x in xs]
    n = {len(a) for a in out}
    if len(n) != 1:
        raise ValueError("inputs must have equal length")
    return out


def mae(y: ArrayLike, yhat: ArrayLike) -> float:
    y_, f = _arr(y, yhat)
    return float(np.mean(np.abs(y_ - f)))


def rmse(y: ArrayLike, yhat: ArrayLike) -> float:
    y_, f = _arr(y, yhat)
    return float(np.sqrt(np.mean((y_ - f) ** 2)))


def wape(y: ArrayLike, yhat: ArrayLike) -> float:
    """Sum |e| / sum y. Well defined with zero-demand hours (unlike MAPE)."""
    y_, f = _arr(y, yhat)
    denom = np.sum(np.abs(y_))
    return float(np.sum(np.abs(y_ - f)) / denom) if denom > 0 else float("nan")


def r2(y: ArrayLike, yhat: ArrayLike) -> float:
    y_, f = _arr(y, yhat)
    ss_tot = np.sum((y_ - y_.mean()) ** 2)
    return float(1 - np.sum((y_ - f) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def mape_filtered(y: ArrayLike, yhat: ArrayLike, min_y: float = 1.0) -> float:
    """MAPE (%) restricted to hours with y >= min_y; always report the filter alongside."""
    y_, f = _arr(y, yhat)
    m = y_ >= min_y
    return float(100 * np.mean(np.abs((y_[m] - f[m]) / y_[m]))) if m.any() else float("nan")


def seasonal_naive_scale(train: pd.DataFrame, m: int = 24, value: str = "energy_kwh",
                         station: str = "station_id", ts: str = "ts") -> float:
    """MASE denominator: in-sample MAE of the seasonal-naive forecast y(t-m), pooled over stations."""
    errs = []
    for _, g in train.sort_values(ts).groupby(station, observed=True):
        s = g[value].to_numpy(dtype=float)
        if len(s) > m:
            errs.append(np.abs(s[m:] - s[:-m]))
    if not errs:
        raise ValueError("not enough history to compute the seasonal-naive scale")
    return float(np.mean(np.concatenate(errs)))


def mase(y: ArrayLike, yhat: ArrayLike, scale: float) -> float:
    return mae(y, yhat) / scale if scale > 0 else float("nan")


def peak_mae(y: ArrayLike, yhat: ArrayLike, threshold: ArrayLike | float) -> float:
    """MAE on hours where y >= threshold (e.g. each station's training p90)."""
    y_, f = _arr(y, yhat)
    thr = np.broadcast_to(np.asarray(threshold, dtype=float), y_.shape)
    m = y_ >= thr
    return float(np.mean(np.abs(y_[m] - f[m]))) if m.any() else float("nan")


def pinball(y: ArrayLike, q: ArrayLike, alpha: float) -> float:
    y_, q_ = _arr(y, q)
    d = y_ - q_
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def picp(y: ArrayLike, lo: ArrayLike, hi: ArrayLike) -> float:
    y_, lo_, hi_ = _arr(y, lo, hi)
    return float(np.mean((y_ >= lo_) & (y_ <= hi_)))


def mpiw(lo: ArrayLike, hi: ArrayLike) -> float:
    lo_, hi_ = _arr(lo, hi)
    return float(np.mean(hi_ - lo_))


def winkler(y: ArrayLike, lo: ArrayLike, hi: ArrayLike, alpha: float = 0.10) -> float:
    y_, lo_, hi_ = _arr(y, lo, hi)
    score = (hi_ - lo_) + (2 / alpha) * (lo_ - y_) * (y_ < lo_) + (2 / alpha) * (y_ - hi_) * (y_ > hi_)
    return float(np.mean(score))


def point_report(y: ArrayLike, yhat: ArrayLike, *, scale: float | None = None,
                 peak_threshold: ArrayLike | float | None = None) -> dict[str, float]:
    out = {"mae": mae(y, yhat), "rmse": rmse(y, yhat), "wape": wape(y, yhat), "r2": r2(y, yhat),
           "mape_filtered": mape_filtered(y, yhat), "n": float(len(np.asarray(y)))}
    if scale is not None:
        out["mase"] = mase(y, yhat, scale)
    if peak_threshold is not None:
        out["peak_mae"] = peak_mae(y, yhat, peak_threshold)
    return out


def interval_report(y: ArrayLike, lo: ArrayLike, hi: ArrayLike, alpha: float = 0.10) -> dict[str, float]:
    return {"picp": picp(y, lo, hi), "mpiw": mpiw(lo, hi), "winkler": winkler(y, lo, hi, alpha)}
