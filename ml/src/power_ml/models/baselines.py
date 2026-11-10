"""B0-B3 baselines (report Section 5.2). They read feature columns that are causal by construction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from power_ml.models.base import clip0


@dataclass
class LagBaseline:
    """B0 persistence (``y_org_0``), B1 seasonal naive 24 h (``lag_24h``), B2 weekly (``lag_168h``)."""

    name: str
    column: str
    family: str = "baseline"

    def fit(self, train: pd.DataFrame, holdout: pd.DataFrame | None = None) -> LagBaseline:
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return clip0(frame[self.column].to_numpy())


@dataclass
class Climatology:
    """B3: training mean of y per (station, hour-of-week of the target hour)."""

    name: str = "B3_climatology"
    family: str = "baseline"
    table_: pd.Series = field(default_factory=pd.Series)
    station_mean_: pd.Series = field(default_factory=pd.Series)
    global_mean_: float = 0.0

    def fit(self, train: pd.DataFrame, holdout: pd.DataFrame | None = None) -> Climatology:
        key = [train["station_id"].astype(str), train["how"].astype(int)]
        self.table_ = train.groupby(key, observed=True)["y"].mean()
        self.station_mean_ = train.groupby(train["station_id"].astype(str), observed=True)["y"].mean()
        self.global_mean_ = float(train["y"].mean())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([frame["station_id"].astype(str), frame["how"].astype(int)])
        out = self.table_.reindex(idx).to_numpy()
        fallback = frame["station_id"].astype(str).map(self.station_mean_).fillna(self.global_mean_).to_numpy()
        return clip0(np.where(np.isnan(out), fallback, out))


def default_baselines() -> list:
    return [
        LagBaseline("B0_persistence", "y_org_0"),
        LagBaseline("B1_seasonal_naive_24h", "lag_24h"),
        LagBaseline("B2_seasonal_naive_168h", "lag_168h"),
        Climatology(),
    ]
