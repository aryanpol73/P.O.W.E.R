"""M1 Ridge regression on log1p(y) with Duan smearing (report Section 5.3).

A linear model cannot form interactions on its own, so the design adds:
* one-hot hour-of-week (168) and station,
* recent-state x horizon-bucket interactions (the weight on y(T) should fade with h).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from power_core import metrics
from power_core.conformal import DEFAULT_BUCKETS, bucket_of
from power_core.features import feature_names
from power_ml.models.base import clip0

RECENT = ("y_org_0", "y_org_1", "y_org_2", "ewma_6h")


@dataclass
class RidgeModel:
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    name: str = "M1_ridge"
    family: str = "ridge"
    alpha_: float = field(default=1.0, init=False)
    numeric_: list[str] = field(default_factory=list, init=False)
    mean_: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False)
    std_: np.ndarray = field(default_factory=lambda: np.ones(0), init=False)
    stations_: list[str] = field(default_factory=list, init=False)
    smear_: float = field(default=1.0, init=False)
    model_: Ridge | None = field(default=None, init=False)

    # ---------------------------------------------------------------- design
    def _numeric(self, frame: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in feature_names("ridge") if c in frame and c not in ("station_id", "how")]
        num = frame[cols].astype(np.float32)
        b = bucket_of(frame["h"].to_numpy(), DEFAULT_BUCKETS)
        for c in RECENT:
            if c in frame:
                for i in range(len(DEFAULT_BUCKETS)):
                    num[f"{c}_x_b{i}"] = frame[c].to_numpy(np.float32) * (b == i)
        return num

    def _design(self, frame: pd.DataFrame) -> np.ndarray:
        num = self._numeric(frame)[self.numeric_].to_numpy(np.float32)
        num = np.nan_to_num((num - self.mean_) / self.std_, nan=0.0)
        how = frame["how"].to_numpy(int)
        how_oh = np.zeros((len(frame), 168), dtype=np.float32)
        how_oh[np.arange(len(frame)), how] = 1.0
        st = frame["station_id"].astype(str).to_numpy()
        st_oh = (st[:, None] == np.array(self.stations_)[None, :]).astype(np.float32)
        return np.hstack([num, how_oh[:, 1:], st_oh[:, 1:]])  # drop one level of each one-hot

    # ---------------------------------------------------------------- fit / predict
    def _fit_alpha(self, Xd: np.ndarray, z: np.ndarray, alpha: float) -> Ridge:
        return Ridge(alpha=alpha, solver="cholesky").fit(Xd, z)

    def fit(self, train: pd.DataFrame, holdout: pd.DataFrame | None = None) -> RidgeModel:
        num = self._numeric(train)
        self.numeric_ = list(num.columns)
        self.mean_ = np.nanmean(num.to_numpy(np.float32), axis=0)
        sd = np.nanstd(num.to_numpy(np.float32), axis=0)
        self.std_ = np.where(sd > 0, sd, 1.0).astype(np.float32)
        self.stations_ = sorted(train["station_id"].astype(str).unique())
        Xd, z = self._design(train), np.log1p(train["y"].to_numpy(float))

        if holdout is not None and len(self.alphas) > 1:
            Xh, yh = self._design(holdout), holdout["y"].to_numpy(float)
            scores = {}
            for a in self.alphas:
                m = self._fit_alpha(Xd, z, a)
                smear = float(np.mean(np.exp(z - m.predict(Xd))))
                scores[a] = metrics.wape(yh, clip0(np.expm1(m.predict(Xh)) * smear))
            self.alpha_ = min(scores, key=scores.get)
        else:
            self.alpha_ = self.alphas[0]
        self.model_ = self._fit_alpha(Xd, z, self.alpha_)
        self.smear_ = float(np.mean(np.exp(z - self.model_.predict(Xd))))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None, "fit first"
        return clip0(np.expm1(self.model_.predict(self._design(frame))) * self.smear_)
