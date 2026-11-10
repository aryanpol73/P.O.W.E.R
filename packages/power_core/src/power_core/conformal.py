"""Distribution-free prediction intervals (flaw F4).

* ``SplitConformal``  - symmetric intervals for point models from absolute residuals.
* ``CQR``             - conformalized quantile regression (Romano et al., 2019).
* ``AdaptiveConformal`` - online alpha update (Gibbs & Candes, 2021) for drift.

All calibrators are *Mondrian by horizon bucket*: a separate adjustment per bucket of h.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

Buckets = Sequence[tuple[int, int]]
DEFAULT_BUCKETS: tuple[tuple[int, int], ...] = ((1, 3), (4, 12), (13, 24))


def bucket_of(h: np.ndarray, buckets: Buckets = DEFAULT_BUCKETS) -> np.ndarray:
    h = np.asarray(h)
    out = np.full(h.shape, -1, dtype=int)
    for i, (lo, hi) in enumerate(buckets):
        out[(h >= lo) & (h <= hi)] = i
    if (out < 0).any():
        raise ValueError("some horizons are outside every bucket")
    return out


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-valid quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.inf) if k > n else float(s[k - 1])


def sort_quantiles(*qs: np.ndarray) -> list[np.ndarray]:
    """Remove quantile crossing by sorting per row."""
    stacked = np.sort(np.vstack([np.asarray(q, float) for q in qs]), axis=0)
    return [stacked[i] for i in range(stacked.shape[0])]


def _fit_buckets(scores: np.ndarray, h: np.ndarray, buckets: Buckets, alpha: float,
                 min_n: int = 30) -> tuple[dict[int, np.ndarray], dict[int, float]]:
    """Per-bucket scores and quantiles; buckets with fewer than ``min_n`` scores fall back to
    the pooled scores so a sparse bucket never produces an infinite or unstable interval."""
    b = bucket_of(h, buckets)
    out_s: dict[int, np.ndarray] = {}
    out_q: dict[int, float] = {}
    for i in range(len(buckets)):
        s = scores[b == i]
        out_s[i] = s if len(s) >= min_n else scores
        out_q[i] = conformal_quantile(out_s[i], alpha)
    return out_s, out_q


def _clip(lo: np.ndarray, hi: np.ndarray, lower: float | None, upper: np.ndarray | float | None) -> tuple[np.ndarray, np.ndarray]:
    if lower is not None:
        lo, hi = np.maximum(lo, lower), np.maximum(hi, lower)
    if upper is not None:
        lo, hi = np.minimum(lo, upper), np.minimum(hi, upper)
    return lo, hi


@dataclass
class SplitConformal:
    alpha: float = 0.10
    buckets: Buckets = DEFAULT_BUCKETS
    q_: dict[int, float] = field(default_factory=dict)
    scores_: dict[int, np.ndarray] = field(default_factory=dict)

    def fit(self, y: np.ndarray, yhat: np.ndarray, h: np.ndarray) -> SplitConformal:
        res = np.abs(np.asarray(y, float) - np.asarray(yhat, float))
        self.scores_, self.q_ = _fit_buckets(res, h, self.buckets, self.alpha)
        return self

    def predict(self, yhat: np.ndarray, h: np.ndarray, *, lower: float | None = 0.0,
                upper: np.ndarray | float | None = None, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        b = bucket_of(h, self.buckets)
        q = self._q(b, alpha)
        yhat = np.asarray(yhat, float)
        return _clip(yhat - q, yhat + q, lower, upper)

    def _q(self, b: np.ndarray, alpha: float | None) -> np.ndarray:
        if alpha is None:
            return np.array([self.q_[i] for i in b])
        return np.array([conformal_quantile(self.scores_[i], alpha) for i in b])

    def to_dict(self) -> dict[str, object]:
        return {"kind": "split", "alpha": self.alpha, "buckets": [list(x) for x in self.buckets],
                "q": {str(k): v for k, v in self.q_.items()}}


@dataclass
class CQR:
    alpha: float = 0.10
    buckets: Buckets = DEFAULT_BUCKETS
    q_: dict[int, float] = field(default_factory=dict)
    scores_: dict[int, np.ndarray] = field(default_factory=dict)

    def fit(self, y: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray, h: np.ndarray) -> CQR:
        q_lo, q_hi = sort_quantiles(q_lo, q_hi)
        y = np.asarray(y, float)
        scores = np.maximum(q_lo - y, y - q_hi)
        self.scores_, self.q_ = _fit_buckets(scores, h, self.buckets, self.alpha)
        return self

    def predict(self, q_lo: np.ndarray, q_hi: np.ndarray, h: np.ndarray, *, lower: float | None = 0.0,
                upper: np.ndarray | float | None = None, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        q_lo, q_hi = sort_quantiles(q_lo, q_hi)
        b = bucket_of(h, self.buckets)
        q = (np.array([self.q_[i] for i in b]) if alpha is None
             else np.array([conformal_quantile(self.scores_[i], alpha) for i in b]))
        lo, hi = q_lo - q, q_hi + q
        hi = np.maximum(hi, lo)  # a negative adjustment can never invert the interval
        return _clip(lo, hi, lower, upper)

    def to_dict(self) -> dict[str, object]:
        return {"kind": "cqr", "alpha": self.alpha, "buckets": [list(x) for x in self.buckets],
                "q": {str(k): v for k, v in self.q_.items()}}


def calibrator_from_dict(d: dict) -> SplitConformal | CQR:
    """Rebuild a fitted calibrator from ``to_dict()`` output (quantiles only, no raw scores)."""
    cls = CQR if d["kind"] == "cqr" else SplitConformal
    obj = cls(alpha=float(d["alpha"]), buckets=tuple(tuple(b) for b in d["buckets"]))
    obj.q_ = {int(k): float(v) for k, v in d["q"].items()}
    return obj


@dataclass
class AdaptiveConformal:
    """alpha_{t+1} = alpha_t + gamma * (alpha - err_t), err_t = 1 if y_t was not covered."""

    target_alpha: float = 0.10
    gamma: float = 0.005
    alpha_min: float = 0.01
    alpha_max: float = 0.50
    alpha_t: float = field(init=False)

    def __post_init__(self) -> None:
        self.alpha_t = self.target_alpha

    def update(self, covered: bool) -> float:
        err = 0.0 if covered else 1.0
        self.alpha_t = float(np.clip(self.alpha_t + self.gamma * (self.target_alpha - err), self.alpha_min, self.alpha_max))
        return self.alpha_t
