"""E1 weighted ensemble (report Section 5.7): non-negative weights summing to 1, fitted on validation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


@dataclass
class WeightedEnsemble:
    members: tuple[str, ...]
    name: str = "E1_ensemble"
    weights_: dict[str, float] = field(default_factory=dict, init=False)

    def fit(self, preds: dict[str, np.ndarray], y: np.ndarray) -> WeightedEnsemble:
        P = np.column_stack([preds[m] for m in self.members]).astype(float)
        y = np.asarray(y, float)
        k = P.shape[1]

        def loss(w: np.ndarray) -> float:
            return float(np.mean((y - P @ w) ** 2))

        res = minimize(loss, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                       constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},))
        w = np.clip(res.x, 0, None)
        w = w / w.sum()
        self.weights_ = {m: float(v) for m, v in zip(self.members, w, strict=True)}
        return self

    def predict(self, preds: dict[str, np.ndarray]) -> np.ndarray:
        return np.sum([self.weights_[m] * np.asarray(preds[m], float) for m in self.members], axis=0)
