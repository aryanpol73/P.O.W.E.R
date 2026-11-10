"""Forecast-comparison statistics: Diebold-Mariano with HLN correction, Holm adjustment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class DMResult:
    statistic: float
    p_value: float
    mean_loss_diff: float
    n: int


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1, loss: str = "abs", hln: bool = True) -> DMResult:
    """Two-sided DM test of equal predictive accuracy.

    ``d_t = L(e1_t) - L(e2_t)``; a negative statistic means model 1 is more accurate. The long-run
    variance uses autocovariances up to lag h-1 (Diebold & Mariano, 1995); with ``hln`` the
    Harvey-Leybourne-Newbold small-sample correction and a t(n-1) reference are used.
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    if e1.shape != e2.shape:
        raise ValueError("error arrays must have the same shape")
    if loss == "abs":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "sq":
        d = e1**2 - e2**2
    else:
        raise ValueError("loss must be 'abs' or 'sq'")
    n = len(d)
    if n < 10:
        raise ValueError("need at least 10 paired errors")
    dbar = d.mean()
    dc = d - dbar
    gamma = [np.dot(dc[k:], dc[: n - k]) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        var = gamma[0] / n
    if var == 0:
        return DMResult(0.0, 1.0, float(dbar), n)
    dm = dbar / np.sqrt(var)
    if hln:
        dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        p = 2 * stats.t.sf(abs(dm), df=n - 1)
    else:
        p = 2 * stats.norm.sf(abs(dm))
    return DMResult(float(dm), float(p), float(dbar), n)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()
