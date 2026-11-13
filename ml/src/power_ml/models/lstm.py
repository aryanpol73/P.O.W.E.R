"""M5 Seq2Seq LSTM, MIMO (report Section 5.6). Requires ``uv sync --extra dl``.

* Encoder: 2-layer LSTM over the last ``lookback`` hours of
  [load/site scale, occupancy, charging share, hour/day cycles, holiday, weekend, temperature, rain].
* Decoder: the final hidden state + an 8-d station embedding + the known future covariates for
  T+1..T+24 (calendar and weather) -> dense head -> 24 outputs at once (no recursion).
* Loss: L1 in kWh, so like the champion it targets the conditional median (WAPE-optimal).

The model reads hourly history from the panel. The panel is *not* pickled with the model;
``attach_panel`` rebuilds the arrays after loading. Only data up to the origin T enters the
encoder, and only calendar/weather (known in advance) enters the decoder.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from power_core.features.calendar import calendar_features
from power_ml.models.base import clip0

H = 24
ENC_COLS = ("y_scaled", "occ_ratio", "active_share", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "is_holiday", "is_weekend", "temp_z", "precip_log")
FUT_COLS = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_holiday", "is_weekend",
            "temp_z", "precip_log")


def _link(z: torch.Tensor) -> torch.Tensor:
    """Non-negative output link. Softplus, not ReLU: a ReLU output unit that goes negative gets zero
    gradient and never recovers (in the first trained model the h=21 output died and predicted 0
    for every row)."""
    return nn.functional.softplus(z)


class _Net(nn.Module):
    def __init__(self, c_enc: int, c_fut: int, n_st: int, hidden: int, layers: int, dropout: float, emb: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_st, emb)
        self.lstm = nn.LSTM(c_enc, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden + emb + c_fut * H, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, H))

    def forward(self, x_enc: torch.Tensor, x_fut: torch.Tensor, st: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x_enc)
        return self.head(torch.cat([h[-1], self.emb(st), x_fut.flatten(1)], dim=1))


@dataclass
class LSTMModel:
    lookback: int = 168
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.1
    emb: int = 8
    lr: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 20
    patience: int = 3
    max_minutes: float = 30.0
    tz: str = "America/Los_Angeles"
    holidays: tuple[str, str | None] = ("US", "CA")  # a deployment profile may override tz and holidays
    seed: int = 42
    name: str = "M5_lstm"
    family: str = "lstm"
    stations_: list[str] = field(default_factory=list, init=False)
    scale_: dict[str, float] = field(default_factory=dict, init=False)
    temp_mean_: float = field(default=0.0, init=False)
    temp_std_: float = field(default=1.0, init=False)
    state_: dict[str, Any] | None = field(default=None, init=False)
    history_: list[dict[str, float]] = field(default_factory=list, init=False)
    best_epoch_: int | None = field(default=None, init=False)

    # ------------------------------------------------------------------ arrays
    def attach_panel(self, panel: pd.DataFrame) -> LSTMModel:
        """Build per-hour encoder/future arrays for every station (concatenated with offsets)."""
        enc, fut, ys, sts, keys = [], [], [], [], {}
        offset = 0
        for k, sid in enumerate(self.stations_):
            p = panel.loc[panel["station_id"] == sid].sort_values("ts")
            if p.empty:
                continue
            ts = pd.DatetimeIndex(p["ts"])
            cal = calendar_features(ts, self.tz, holidays=getattr(self, "holidays", ("US", "CA")))
            df = pd.DataFrame(index=ts)
            df["y_scaled"] = p["energy_kwh"].to_numpy(float) / self.scale_[sid]
            df["occ_ratio"] = p["occ_ratio"].to_numpy(float)
            n_ports = max(1.0, float(np.nanmax(p["occupied_ports"].to_numpy(float))) if "occupied_ports" in p else 1.0)
            df["active_share"] = p["active_sessions"].to_numpy(float) / n_ports
            for c in ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "is_holiday", "is_weekend"):
                df[c] = cal[c].to_numpy(float)
            df["temp_z"] = (p["temperature_2m_c"].to_numpy(float) - self.temp_mean_) / self.temp_std_
            df["precip_log"] = np.log1p(np.clip(p["precipitation_mm"].to_numpy(float), 0, None))
            df = df.fillna(0.0)
            enc.append(df[list(ENC_COLS)].to_numpy(np.float32))
            fut.append(df[list(FUT_COLS)].to_numpy(np.float32))
            ys.append(p["energy_kwh"].to_numpy(np.float32))
            sts.append(np.full(len(p), k, dtype=np.int64))
            keys[sid] = (offset, pd.Series(np.arange(len(p)), index=ts.asi8))
            offset += len(p)
        self._enc = torch.from_numpy(np.concatenate(enc))
        self._fut = torch.from_numpy(np.concatenate(fut))
        self._y = torch.from_numpy(np.concatenate(ys))
        self._st = torch.from_numpy(np.concatenate(sts))
        self._scale = torch.tensor([self.scale_[s] for s in self.stations_], dtype=torch.float32)
        self._keys = keys
        self._last = torch.from_numpy(np.concatenate(
            [np.full(len(lc), off + len(lc) - 1, dtype=np.int64) for off, lc in keys.values()]))
        return self

    def __getstate__(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def _global_index(self, station: pd.Series, origin: pd.Series, need_future: bool = True) -> np.ndarray:
        out = np.empty(len(station), dtype=np.int64)
        o = pd.DatetimeIndex(origin).asi8
        st = station.astype(str).to_numpy()
        for sid in np.unique(st):
            m = st == sid
            off, local = self._keys[sid]
            idx = local.reindex(o[m]).to_numpy()
            if np.isnan(idx).any():
                raise KeyError(f"{sid}: some origins are not in the attached panel")
            idx = idx.astype(np.int64)
            # Windows are gathered from one concatenated array, so they must stay inside this station.
            if (idx < self.lookback - 1).any() or (need_future and (idx + H >= len(local)).any()):
                raise ValueError(f"{sid}: origin too close to the start/end of its history for lookback={self.lookback}")
            out[m] = off + idx
        return out

    def _batch(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        back = torch.arange(-self.lookback + 1, 1)
        fwd = torch.arange(1, H + 1)
        return (self._enc[g[:, None] + back], self._fut[g[:, None] + fwd], self._st[g], self._y[g[:, None] + fwd])

    @staticmethod
    def _complete_origins(frame: pd.DataFrame) -> pd.DataFrame:
        """(station, origin) pairs with all 24 horizons present in this block."""
        c = frame.groupby(["station_id", "origin"], observed=True)["h"].count()
        return c.loc[c == H].reset_index()[["station_id", "origin"]]

    # ------------------------------------------------------------------ fit / predict
    def prepare(self, panel: pd.DataFrame, train_frame: pd.DataFrame) -> LSTMModel:
        """Fit scalers on the training block only, then attach the panel."""
        self.stations_ = sorted(train_frame["station_id"].astype(str).unique())
        tr = panel.loc[panel["ts"] <= train_frame["target_time"].max()]
        means = tr.groupby("station_id")["energy_kwh"].mean()
        self.scale_ = {s: float(max(means.get(s, 1.0), 0.1)) for s in self.stations_}
        self.temp_mean_ = float(tr["temperature_2m_c"].mean())
        self.temp_std_ = float(tr["temperature_2m_c"].std() or 1.0)
        return self.attach_panel(panel)

    def _net(self) -> _Net:
        return _Net(len(ENC_COLS), len(FUT_COLS), len(self.stations_), self.hidden, self.layers, self.dropout, self.emb)

    def _forward(self, net: _Net, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_enc, x_fut, st, y = self._batch(g)
        return _link(net(x_enc, x_fut, st)) * self._scale[st][:, None], y

    def fit(self, train: pd.DataFrame, holdout: pd.DataFrame | None = None) -> LSTMModel:
        if not hasattr(self, "_enc"):
            raise RuntimeError("call prepare(panel, train_frame) before fit")
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        tr_pairs = self._complete_origins(train)
        g_tr = self._global_index(tr_pairs["station_id"], tr_pairs["origin"])
        g_ho = None
        if holdout is not None:
            ho_pairs = self._complete_origins(holdout)
            g_ho = torch.from_numpy(self._global_index(ho_pairs["station_id"], ho_pairs["origin"]))
        net = self._net()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        best, best_state, bad = float("inf"), None, 0
        t_start = time.perf_counter()
        for epoch in range(1, self.max_epochs + 1):
            net.train()
            order = torch.from_numpy(rng.permutation(g_tr))
            tot, n = 0.0, 0
            for i in range(0, len(order), self.batch_size):
                pred, y = self._forward(net, order[i:i + self.batch_size])
                loss = torch.mean(torch.abs(pred - y))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(y)
                n += len(y)
            rec = {"epoch": epoch, "train_l1": tot / n}
            if g_ho is not None:
                rec["holdout_wape"] = self._wape(net, g_ho)
                if rec["holdout_wape"] < best - 1e-4:
                    best, bad, self.best_epoch_ = rec["holdout_wape"], 0, epoch
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
            rec["elapsed_s"] = round(time.perf_counter() - t_start, 1)
            self.history_.append(rec)
            if bad >= self.patience or rec["elapsed_s"] > self.max_minutes * 60:
                break
        self.state_ = best_state or net.state_dict()
        return self

    @torch.no_grad()
    def _wape(self, net: _Net, g: torch.Tensor) -> float:
        net.eval()
        ae = tot = 0.0
        for i in range(0, len(g), 4096):
            pred, y = self._forward(net, g[i:i + 4096])
            ae += float(torch.sum(torch.abs(pred - y)))
            tot += float(torch.sum(y))
        return ae / max(tot, 1e-9)

    @torch.no_grad()
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert self.state_ is not None, "fit first"
        net = self._net()
        net.load_state_dict(self.state_)
        net.eval()
        pairs = frame[["station_id", "origin"]].drop_duplicates().reset_index(drop=True)
        g = torch.from_numpy(self._global_index(pairs["station_id"], pairs["origin"], need_future=False))
        outs = []
        for i in range(0, len(g), 4096):
            gi = g[i:i + 4096]
            x_enc, x_fut, st, _ = self._batch_x(gi)
            outs.append((_link(net(x_enc, x_fut, st)) * self._scale[st][:, None]).numpy())
        mat = np.concatenate(outs) if outs else np.zeros((0, H), np.float32)
        key = pd.MultiIndex.from_frame(pairs.astype({"station_id": str}))
        row = key.get_indexer(pd.MultiIndex.from_arrays([frame["station_id"].astype(str), frame["origin"]]))
        return clip0(mat[row, frame["h"].to_numpy(int) - 1])

    def _batch_x(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Encoder/future inputs only: valid even when targets after T are unknown (serving)."""
        back = torch.arange(-self.lookback + 1, 1)
        # Clamp future steps to the station's last row: only horizons that do not exist read the
        # clamped (repeated) covariates, and those outputs are never selected.
        fwd = torch.minimum(g[:, None] + torch.arange(1, H + 1), self._last[g][:, None])
        return self._enc[g[:, None] + back], self._fut[fwd], self._st[g], None
