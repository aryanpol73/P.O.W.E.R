"""Model zoo for the direct multi-horizon frame (report Chapter 5)."""

from power_ml.models.base import ForecastModel, subsample_origins
from power_ml.models.baselines import Climatology, LagBaseline
from power_ml.models.boosting import LGBMModel, LGBMQuantile, XGBModel
from power_ml.models.ensemble import WeightedEnsemble
from power_ml.models.forest import RFModel
from power_ml.models.linear import RidgeModel

__all__ = [
    "Climatology",
    "ForecastModel",
    "LGBMModel",
    "LGBMQuantile",
    "LagBaseline",
    "RFModel",
    "RidgeModel",
    "WeightedEnsemble",
    "XGBModel",
    "subsample_origins",
]
