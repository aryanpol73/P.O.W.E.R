"""Common model interface and helpers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from power_core.features import feature_matrix


@runtime_checkable
class ForecastModel(Protocol):
    name: str
    family: str  # feature family in power_core.features.registry: ridge | rf | gbdt | prior | baseline

    def fit(self, train: pd.DataFrame, holdout: pd.DataFrame | None = None) -> ForecastModel: ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray: ...


def subsample_origins(frame: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Keep every ``stride``-th forecast origin (all stations and horizons of a kept origin stay together)."""
    if stride <= 1:
        return frame
    o = pd.DatetimeIndex(frame["origin"])
    k = ((o - o.min()) // pd.Timedelta(hours=1)).to_numpy() % stride == 0
    return frame.loc[k]


def X(frame: pd.DataFrame, family: str) -> pd.DataFrame:  # noqa: N802  (X = design matrix, ML convention)
    return feature_matrix(frame, family)


def clip0(a: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(a, dtype=float), 0.0)
