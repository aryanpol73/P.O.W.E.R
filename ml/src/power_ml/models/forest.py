"""M2 Random Forest (report Section 5.4). Categoricals are passed as integer codes."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from power_ml.models.base import X, clip0


def _codes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].cat.codes.astype(np.int16)
    return out


@dataclass
class RFModel:
    params: dict = field(default_factory=dict)
    seed: int = 42
    n_jobs: int = -1
    name: str = "M2_random_forest"
    family: str = "rf"
    columns_: list[str] = field(default_factory=list, init=False)
    model_: RandomForestRegressor | None = field(default=None, init=False)

    def fit(self, train: pd.DataFrame, holdout: pd.DataFrame | None = None) -> RFModel:
        Xt = _codes(X(train, self.family))
        self.columns_ = list(Xt.columns)
        self.model_ = RandomForestRegressor(random_state=self.seed, n_jobs=self.n_jobs, **self.params)
        self.model_.fit(Xt.to_numpy(np.float32), train["y"].to_numpy(float))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None, "fit first"
        return clip0(self.model_.predict(_codes(frame[self.columns_]).to_numpy(np.float32)))
