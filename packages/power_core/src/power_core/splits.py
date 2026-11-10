"""Chronological splitting for panel (multi-station) data.

Fixes flaw F2: scikit-learn's ``TimeSeriesSplit`` splits by row position, so on a panel the
``gap`` is measured in rows and fold boundaries can cut through a timestamp. Everything here
splits on *time* and measures gaps in hours.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from power_core.timeutils import HOUR


@dataclass(frozen=True)
class SplitBounds:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_hours: int

    def to_dict(self) -> dict[str, str | int]:
        return {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict[str, str | int]) -> SplitBounds:
        kw = {k: (pd.Timestamp(v) if k != "embargo_hours" else int(v)) for k, v in d.items()}
        return cls(**kw)  # type: ignore[arg-type]


def chronological_bounds(
    start: pd.Timestamp, end: pd.Timestamp, train_frac: float, val_frac: float, embargo_hours: int
) -> SplitBounds:
    """Cut [start, end] (inclusive hours) into train / embargo / val / embargo / test."""
    start, end = pd.Timestamp(start).tz_convert("UTC").floor("h"), pd.Timestamp(end).tz_convert("UTC").floor("h")
    total = int((end - start) / HOUR) + 1
    usable = total - 2 * embargo_hours
    if usable <= 0:
        raise ValueError("window too short for the requested embargo")
    n_train = int(np.floor(usable * train_frac))
    n_val = int(np.floor(usable * val_frac))
    if n_train <= 0 or n_val <= 0 or usable - n_train - n_val <= 0:
        raise ValueError("window too short for the requested split fractions")
    train_end = start + (n_train - 1) * HOUR
    val_start = train_end + (embargo_hours + 1) * HOUR
    val_end = val_start + (n_val - 1) * HOUR
    test_start = val_end + (embargo_hours + 1) * HOUR
    return SplitBounds(start, train_end, val_start, val_end, test_start, end, embargo_hours)


def assign_split(origin: pd.Series, target_time: pd.Series, b: SplitBounds) -> np.ndarray:
    """Label each direct-frame row 'train' / 'val' / 'test' / 'gap'.

    * train: the *target* hour is inside the training block, so no training label ever
      falls inside validation or test;
    * val / test: both origin and target are inside the block.
    """
    def ts(x: pd.Timestamp) -> np.datetime64:
        return np.datetime64(x.tz_convert("UTC").tz_localize(None))

    o = pd.DatetimeIndex(origin).tz_convert("UTC").tz_localize(None).to_numpy()
    t = pd.DatetimeIndex(target_time).tz_convert("UTC").tz_localize(None).to_numpy()
    labels = np.full(len(o), "gap", dtype=object)
    labels[(o >= ts(b.train_start)) & (t <= ts(b.train_end))] = "train"
    labels[(o >= ts(b.val_start)) & (t <= ts(b.val_end))] = "val"
    labels[(o >= ts(b.test_start)) & (t <= ts(b.test_end))] = "test"
    return labels


class PanelTimeSeriesSplit:
    """Expanding-window CV on unique forecast origins, with an embargo measured in hours.

    For fold k, validation rows are those whose origin lies in chunk k+1 of the sorted unique
    origins; training rows are those whose *target* hour ends at least ``embargo_hours``
    before the first validation origin. No timestamp is ever shared between train and
    validation, whatever the number of stations.
    """

    def __init__(self, n_splits: int = 5, embargo_hours: int = 168,
                 origin_col: str = "origin", target_col: str = "target_time") -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo = pd.Timedelta(hours=embargo_hours)
        self.origin_col, self.target_col = origin_col, target_col

    def get_n_splits(self, X: pd.DataFrame | None = None, y: object = None, groups: object = None) -> int:
        return self.n_splits

    def split(self, X: pd.DataFrame, y: object = None, groups: object = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        origin = pd.DatetimeIndex(X[self.origin_col])
        target = pd.DatetimeIndex(X[self.target_col])
        uniq = np.sort(origin.unique())
        chunks = np.array_split(uniq, self.n_splits + 1)
        for k in range(1, self.n_splits + 1):
            v = chunks[k]
            if len(v) == 0:
                continue
            v_start, v_end = pd.Timestamp(v[0]), pd.Timestamp(v[-1])
            train_idx = np.flatnonzero(target <= v_start - self.embargo)
            val_idx = np.flatnonzero((origin >= v_start) & (origin <= v_end))
            if len(train_idx) and len(val_idx):
                yield train_idx, val_idx
