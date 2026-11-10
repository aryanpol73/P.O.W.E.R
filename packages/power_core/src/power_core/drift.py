"""Drift statistics (flaw F11): hour-of-week PSI on features and Page-Hinkley on residuals.

A plain KS test on raw load fires whenever the recent window samples a different part of the
daily cycle; comparing within hour-of-week buckets removes that seasonality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index with quantile bins taken from the reference sample."""
    ref = np.asarray(reference, float)
    cur = np.asarray(current, float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.allclose(cur, ref[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def seasonal_psi(ref: pd.DataFrame, cur: pd.DataFrame, feature: str, bucket: str = "how",
                 bins: int = 10, min_rows: int = 10) -> float:
    """PSI after removing the hour-of-week profile.

    Each value is standardised with its bucket's reference mean and std, then one PSI is computed
    on the pooled z-scores. A 7-day window holds only one sample per hour-of-week, so a separate
    PSI per bucket would be pure noise; standardising first uses every current sample.
    """
    stats_ = ref.groupby(bucket)[feature].agg(["mean", "std", "size"])
    stats_ = stats_.loc[stats_["size"] >= min_rows]
    stats_["std"] = stats_["std"].replace(0, np.nan)

    def z(df: pd.DataFrame) -> np.ndarray:
        s = stats_.reindex(df[bucket])
        return ((df[feature].to_numpy(float) - s["mean"].to_numpy()) / s["std"].to_numpy())

    return psi(z(ref), z(cur), bins=bins)


@dataclass
class Cusum:
    """Two-sided CUSUM on a stream already standardised against a reference (mean 0, sd 1).

    ``k`` is the allowance (half the shift to detect, in sd units), ``h`` the decision threshold.
    After an alarm both sums reset, so successive regime changes are each reported.
    """

    k: float = 0.5
    h: float = 5.0
    pos: float = 0.0
    neg: float = 0.0

    def update(self, z: float) -> str | None:
        self.pos = max(0.0, self.pos + z - self.k)
        self.neg = max(0.0, self.neg - z - self.k)
        if self.pos > self.h or self.neg > self.h:
            direction = "up" if self.pos > self.h else "down"
            self.pos = self.neg = 0.0
            return direction
        return None


@dataclass
class PageHinkley:
    """Detects an upward shift in the mean of a stream (e.g. |standardised residual|)."""

    delta: float = 0.05
    lam: float = 50.0
    n: int = 0
    mean: float = 0.0
    cum: float = 0.0
    min_cum: float = field(default=0.0)

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        return (self.cum - self.min_cum) > self.lam

    def reset(self) -> None:
        self.n, self.mean, self.cum, self.min_cum = 0, 0.0, 0.0, 0.0
