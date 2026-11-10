"""M1 Ridge regression, solved in closed form (report Section 5.3).

A linear model cannot form interactions on its own, so the design adds:
* one-hot hour-of-week (168) and station,
* station x hour-of-day and station x weekend interactions (each site has its own daily shape),
* recent-state x horizon-bucket interactions (the weight on y(T) should fade with h).

The target is raw kWh by default. A log1p target with Duan smearing is available
(``log_target=True``) as an ablation: on this zero-heavy data it was clearly worse
(training-holdout WAPE 0.61 vs 0.55), because the back-transform misses the peaks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from power_core import metrics
from power_core.conformal import DEFAULT_BUCKETS, bucket_of
from power_core.features import feature_names
from power_ml.models.base import clip0

RECENT = ("y_org_0", "y_org_1", "y_org_2", "ewma_6h")
# Exact functions of the hour-of-week one-hot or of the station one-hot: keeping them makes
# X'X singular (ill-conditioned solve) without adding information.
REDUNDANT = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
             *(f"how_f{k}_{fn}" for k in (1, 2, 3) for fn in ("sin", "cos")), "total_ports")


@dataclass
class RidgeModel:
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    log_target: bool = False
    station_interactions: bool = True
    name: str = "M1_ridge"
    family: str = "ridge"
    alpha_: float = field(default=1.0, init=False)
    numeric_: list[str] = field(default_factory=list, init=False)
    mean_: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False)
    std_: np.ndarray = field(default_factory=lambda: np.ones(0), init=False)
    stations_: list[str] = field(default_factory=list, init=False)
    smear_: float = field(default=1.0, init=False)
    coef_: np.ndarray | None = field(default=None, init=False)
    intercept_: float = field(default=0.0, init=False)

    # ---------------------------------------------------------------- design
    def _numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in feature_names("ridge") if c in frame and c not in ("station_id", "how", *REDUNDANT)]
        num = frame[cols].astype(np.float32)
        b = bucket_of(frame["h"].to_numpy(), DEFAULT_BUCKETS)
        for c in RECENT:
            if c in frame:
                for i in range(len(DEFAULT_BUCKETS)):
                    num[f"{c}_x_b{i}"] = frame[c].to_numpy(np.float32) * (b == i)
        return num

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        n = len(frame)
        num = self._numeric(frame)[self.numeric_].to_numpy(np.float32)
        num = np.nan_to_num((num - self.mean_) / self.std_, nan=0.0)
        how = frame["how"].to_numpy(int)
        how_oh = np.zeros((n, 168), dtype=np.float32)
        how_oh[np.arange(n), how] = 1.0
        st = frame["station_id"].astype(str).to_numpy()
        st_oh = (st[:, None] == np.array(self.stations_)[None, :]).astype(np.float32)
        blocks = [num, how_oh[:, 1:], st_oh[:, 1:]]  # drop one level of each one-hot
        if self.station_interactions:
            hour, weekend = how % 24, (how >= 120).astype(np.float32)
            for j in range(1, len(self.stations_)):
                hh = np.zeros((n, 24), dtype=np.float32)
                hh[np.arange(n), hour] = st_oh[:, j]
                blocks += [hh[:, 1:], (st_oh[:, j] * weekend)[:, None]]
        return np.hstack(blocks)

    # ---------------------------------------------------------------- fit / predict
    def _back(self, eta: np.ndarray, smear: float) -> np.ndarray:
        """Raw target: clip at 0. Log1p target: Duan smearing, E[y|x] = exp(eta) * mean(exp(resid)) - 1."""
        return clip0(np.exp(eta) * smear - 1.0) if self.log_target else clip0(eta)

    def fit(self, train: pd.DataFrame, holdout: pd.DataFrame | None = None) -> RidgeModel:
        num = self._numeric(train)
        self.numeric_ = list(num.columns)
        self.mean_ = np.nanmean(num.to_numpy(np.float32), axis=0)
        sd = np.nanstd(num.to_numpy(np.float32), axis=0)
        self.std_ = np.where(sd > 0, sd, 1.0).astype(np.float32)
        self.stations_ = sorted(train["station_id"].astype(str).unique())
        Xd = self._design(train).astype(np.float64)
        y = train["y"].to_numpy(float)
        z = np.log1p(y) if self.log_target else y

        # One Gram matrix for every alpha: centre, then solve (X'X + alpha I) w = X'z.
        x_mean, z_mean = Xd.mean(axis=0), z.mean()
        Xc = Xd - x_mean
        G, b = Xc.T @ Xc, Xc.T @ (z - z_mean)
        eye = np.eye(G.shape[0])

        def solve(a: float) -> tuple[np.ndarray, float, float]:
            w = np.linalg.solve(G + a * eye, b)
            c = z_mean - x_mean @ w
            smear = float(np.mean(np.exp(z - (Xd @ w + c)))) if self.log_target else 1.0
            return w, c, smear

        if holdout is not None and len(self.alphas) > 1:
            Xh, yh = self._design(holdout).astype(np.float64), holdout["y"].to_numpy(float)
            scores = {}
            for a in self.alphas:
                w, c, s = solve(a)
                scores[a] = metrics.wape(yh, self._back(Xh @ w + c, s))
            self.alpha_ = min(scores, key=scores.get)
        else:
            self.alpha_ = self.alphas[0]
        self.coef_, self.intercept_, self.smear_ = solve(self.alpha_)
        return self

    def predict(self, frame: pd.DataFrame, chunk: int = 200_000) -> np.ndarray:
        assert self.coef_ is not None, "fit first"
        out = [self._back(self._design(frame.iloc[i:i + chunk]).astype(np.float64) @ self.coef_ + self.intercept_,
                          self.smear_) for i in range(0, len(frame), chunk)]
        return np.concatenate(out) if out else np.zeros(0)
