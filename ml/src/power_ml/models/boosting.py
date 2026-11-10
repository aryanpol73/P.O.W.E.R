"""M3 XGBoost, M4 LightGBM (Tweedie) and LightGBM quantile heads (report Sections 5.5, 5.8).

Both boosters use native categorical splits on ``station_id``/``poi_category``, monotone
non-decreasing constraints on the weather-stress features, and early stopping on a time-based
holdout taken from the end of the training block.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from power_ml.models.base import X, clip0


def _monotone(columns: list[str], increasing: tuple[str, ...]) -> list[int]:
    return [1 if c in increasing else 0 for c in columns]


@dataclass
class XGBModel:
    params: dict = field(default_factory=dict)
    early_stopping_rounds: int = 100
    monotone_increasing: tuple[str, ...] = ("heat_stress", "cold_stress")
    seed: int = 42
    n_jobs: int = -1
    name: str = "M3_xgboost"
    family: str = "gbdt"
    columns_: list[str] = field(default_factory=list, init=False)
    model_: xgb.XGBRegressor | None = field(default=None, init=False)
    best_iteration_: int | None = field(default=None, init=False)

    def fit(self, train: pd.DataFrame, holdout: pd.DataFrame | None = None) -> XGBModel:
        Xt = X(train, self.family)
        self.columns_ = list(Xt.columns)
        p = dict(self.params)
        self.model_ = xgb.XGBRegressor(
            tree_method="hist", enable_categorical=True, random_state=self.seed, n_jobs=self.n_jobs,
            monotone_constraints=tuple(_monotone(self.columns_, self.monotone_increasing)),
            eval_metric="mae", early_stopping_rounds=self.early_stopping_rounds if holdout is not None else None,
            **p,
        )
        eval_set = [(X(holdout, self.family)[self.columns_], holdout["y"])] if holdout is not None else None
        self.model_.fit(Xt, train["y"], eval_set=eval_set, verbose=False)
        self.best_iteration_ = getattr(self.model_, "best_iteration", None)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None, "fit first"
        return clip0(self.model_.predict(frame[self.columns_]))


@dataclass
class LGBMModel:
    params: dict = field(default_factory=dict)
    early_stopping_rounds: int = 100
    monotone_increasing: tuple[str, ...] = ("heat_stress", "cold_stress")
    seed: int = 42
    n_jobs: int = -1
    name: str = "M4_lightgbm"
    family: str = "gbdt"
    columns_: list[str] = field(default_factory=list, init=False)
    model_: lgb.LGBMRegressor | None = field(default=None, init=False)
    best_iteration_: int | None = field(default=None, init=False)

    def _make(self, extra: dict | None = None) -> lgb.LGBMRegressor:
        p = {**self.params, **(extra or {})}
        return lgb.LGBMRegressor(random_state=self.seed, n_jobs=self.n_jobs, verbose=-1,
                                 monotone_constraints=_monotone(self.columns_, self.monotone_increasing), **p)

    def fit(self, train: pd.DataFrame, holdout: pd.DataFrame | None = None) -> LGBMModel:
        Xt = X(train, self.family)
        self.columns_ = list(Xt.columns)
        self.model_ = self._make()
        kw: dict = {}
        if holdout is not None:
            kw = {"eval_X": (holdout[self.columns_],), "eval_y": (holdout["y"],), "eval_metric": "l1",
                  "callbacks": [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]}
        self.model_.fit(Xt, train["y"], **kw)
        self.best_iteration_ = self.model_.best_iteration_ or None
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.model_ is not None, "fit first"
        return clip0(self.model_.predict(frame[self.columns_]))


@dataclass
class LGBMQuantile:
    """Three LightGBM quantile regressors (default q05 / q50 / q95), minimising pinball loss."""

    params: dict = field(default_factory=dict)
    alphas: tuple[float, ...] = (0.05, 0.50, 0.95)
    early_stopping_rounds: int = 100
    seed: int = 42
    n_jobs: int = -1
    name: str = "M4_lightgbm_quantile"
    family: str = "gbdt"
    columns_: list[str] = field(default_factory=list, init=False)
    models_: dict[float, lgb.LGBMRegressor] = field(default_factory=dict, init=False)

    def fit(self, train: pd.DataFrame, holdout: pd.DataFrame | None = None) -> LGBMQuantile:
        Xt = X(train, self.family)
        self.columns_ = list(Xt.columns)
        for a in self.alphas:
            m = lgb.LGBMRegressor(objective="quantile", alpha=a, random_state=self.seed, n_jobs=self.n_jobs,
                                  verbose=-1, **self.params)
            kw: dict = {}
            if holdout is not None:
                kw = {"eval_X": (holdout[self.columns_],), "eval_y": (holdout["y"],), "eval_metric": "quantile",
                      "callbacks": [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]}
            m.fit(Xt, train["y"], **kw)
            self.models_[a] = m
        return self

    def predict_quantiles(self, frame: pd.DataFrame) -> dict[float, np.ndarray]:
        Xf = frame[self.columns_]
        return {a: clip0(m.predict(Xf)) for a, m in self.models_.items()}

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_quantiles(frame)[0.50]
