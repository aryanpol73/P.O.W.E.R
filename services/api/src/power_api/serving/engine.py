"""Model serving: history + weather -> features -> every model -> calibrated intervals -> SHAP.

The serving path calls ``power_core.features.build_direct_frame``, the *same* function that built
the training data, on a panel slice that holds observed demand up to the origin T and rows for
T+1..T+24 with demand unknown (NaN) and forecast/archive weather. This keeps train/serve skew out.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from power_core.config import data_config, feature_config, threshold_config
from power_core.conformal import calibrator_from_dict
from power_core.features import build_direct_frame
from power_ml.analysis import DISPLAY, shap_contributions
from power_ml.training import QUANTILE, load_artifacts, predict_all

H = 24


@dataclass
class Forecast:
    origin: pd.Timestamp
    rows: pd.DataFrame  # station_id, model, h, target_time, y_mean, q05, q50, q95
    explanations: pd.DataFrame  # station_id, h, base_value, contributions(list)
    latency_ms: float
    degraded: dict[str, bool]


class ServingEngine:
    def __init__(self, root: Path, source: str, champion: str = QUANTILE, version: str | None = None) -> None:
        self.root, self.source, self.champion = root, source, champion
        self.models, self.ensemble, self.manifest = load_artifacts(root, version or None)
        self.version = self.manifest["version"]
        self.dcfg, self.fcfg, self.tcfg = data_config(root), feature_config(root), threshold_config(root)
        self.stations = pd.read_parquet(root / "data" / "interim" / f"stations_{source}.parquet")
        self.calibrators = {k: calibrator_from_dict(v) for k, v in self.manifest["calibrators"].items()}
        self.ranges = self._training_ranges()
        self.network_q = self._network_calibration()

    # ------------------------------------------------------------------ metadata
    def _network_calibration(self, alpha: float = 0.10) -> dict[int, float]:
        """Conformal half-width for the *network total* per horizon, from validation predictions.

        Summing station intervals would be wrong: station errors partly cancel, so the sum of
        q95s overstates the network q95 (audit flaw F10).
        """
        p = self.root / "reports" / "predictions_val.parquet"
        if not p.exists():
            return {}
        v = pd.read_parquet(p, columns=["origin", "h", "y", self.champion])
        net = v.groupby(["origin", "h"])[["y", self.champion]].sum()
        res = (net["y"] - net[self.champion]).abs()
        out = {}
        for h, r in res.groupby(level="h"):
            n = len(r)
            k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
            out[int(h)] = float(np.sort(r.to_numpy())[k - 1])
        return out

    def _training_ranges(self) -> dict[str, tuple[float, float]]:
        """Observed min/max of weather in the training block. Trees cannot extrapolate past the values they
        saw, so a scenario is flagged only beyond this support (p1/p99 flagged ordinary hot days)."""
        cache = self.root / "artifacts" / "models" / self.version / "training_support.json"
        if cache.exists():
            return {k: tuple(v) for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
        path = self.root / "data" / "processed" / f"features_{self.source}_train.parquet"
        out: dict[str, tuple[float, float]] = {}
        if path.exists():
            df = pd.read_parquet(path, columns=["temperature_2m_c", "precipitation_mm"])
            out = {c: (float(df[c].min()), float(df[c].max())) for c in df.columns}
            cache.write_text(json.dumps(out), encoding="utf-8")
        return out

    def leaderboard(self) -> dict[str, Any]:
        p = self.root / "reports" / "metrics_test.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    # ------------------------------------------------------------------ core
    def _frame(self, panel: pd.DataFrame, origin: pd.Timestamp, force_holiday: bool = False) -> pd.DataFrame:
        frame = build_direct_frame(
            panel,
            self.stations,
            self.fcfg,
            tz=self.dcfg.timezone,
            pricing_changes=self.dcfg.pricing_changes,
            origins=pd.DatetimeIndex([origin]),
            apply_warmup=False,
            drop_missing_target=False,
        )
        if force_holiday:
            frame["is_holiday"] = np.float32(1.0)
        return frame

    def _predict(
        self, frame: pd.DataFrame, panel: pd.DataFrame, only: tuple[str, ...] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[float, np.ndarray]]:
        """Run every model, or only ``only`` (the ensemble pulls in its members automatically)."""
        names = set(self.models) if not only else set(only)
        if self.ensemble.name in names:
            names |= set(self.ensemble.members)
        models = {n: m for n, m in self.models.items() if n in names}
        for m in models.values():
            if hasattr(m, "attach_panel"):
                m.attach_panel(panel)
        ens = self.ensemble if (not only or self.ensemble.name in only) else None
        return predict_all(models, ens, frame)

    def forecast(
        self, panel: pd.DataFrame, origin: pd.Timestamp, *, explain: bool = True, models: tuple[str, ...] | None = None
    ) -> Forecast:
        """``panel``: station_id, ts (UTC), demand columns up to ``origin``, NaN after; weather through origin+24h."""
        t0 = time.perf_counter()
        frame = self._frame(panel, origin)
        preds, quants = self._predict(frame, panel, only=models)
        h = frame["h"].to_numpy()
        upper = frame["rated_kw_total"].to_numpy(float)
        rows = []
        for name, p in preds.items():
            if models and name not in models:
                continue
            if name == QUANTILE:
                lo, hi = self.calibrators[f"{QUANTILE}_cqr"].predict(quants[0.05], quants[0.95], h, lower=0.0, upper=upper)
            else:
                lo, hi = self.calibrators[name].predict(p, h, lower=0.0, upper=upper)
            rows.append(
                pd.DataFrame(
                    {
                        "station_id": frame["station_id"].astype(str),
                        "model": name,
                        "h": h,
                        "target_time": frame["target_time"],
                        "y_mean": p,
                        "q05": lo,
                        "q50": p,
                        "q95": hi,
                    }
                )
            )
        out = pd.concat(rows, ignore_index=True)
        expl = self.explain(frame) if explain else pd.DataFrame()
        hist = panel.loc[panel["ts"] <= origin]
        degraded = {s: bool(g["is_imputed"].tail(6).all()) if "is_imputed" in g else False for s, g in hist.groupby("station_id")}
        return Forecast(origin, out, expl, (time.perf_counter() - t0) * 1000, degraded)

    def explain(self, frame: pd.DataFrame, top_k: int = 8) -> pd.DataFrame:
        model = self.models.get(self.champion)
        if model is None or not hasattr(model, "models_"):
            return pd.DataFrame()
        contrib, base = shap_contributions(model, frame)
        recs = []
        for i, (idx, row) in enumerate(frame.iterrows()):
            c = contrib.loc[idx]
            top = c.reindex(c.abs().sort_values(ascending=False).index[:top_k])
            items = [
                {"feature": f, "display": DISPLAY.get(f, f), "value": _num(row[f]), "shap": round(float(v), 4)}
                for f, v in top.items()
            ]
            items.append(
                {"feature": "__other__", "display": "other features", "value": None, "shap": round(float(c.sum() - top.sum()), 4)}
            )
            recs.append(
                {
                    "station_id": str(row["station_id"]),
                    "h": int(row["h"]),
                    "base_value": round(float(base[i]), 4),
                    "contributions": items,
                }
            )
        return pd.DataFrame(recs)

    def scenario(
        self,
        panel: pd.DataFrame,
        origin: pd.Timestamp,
        *,
        temperature_delta_c: float = 0.0,
        precipitation_mm: float | None = None,
        force_holiday: bool = False,
        demand_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        """What-if: shock the *future* covariates only; history and live state are untouched."""
        base = self.forecast(panel, origin, explain=False, models=(self.champion,)).rows
        shocked = panel.copy()
        fut = shocked["ts"] > origin
        shocked.loc[fut, "temperature_2m_c"] = shocked.loc[fut, "temperature_2m_c"] + temperature_delta_c
        if precipitation_mm is not None:
            shocked.loc[fut, "precipitation_mm"] = precipitation_mm
        frame = self._frame(shocked, origin, force_holiday=force_holiday)
        preds, quants = self._predict(frame, shocked, only=(self.champion,))
        h = frame["h"].to_numpy()
        lo, hi = self.calibrators[f"{QUANTILE}_cqr"].predict(
            quants[0.05], quants[0.95], h, lower=0.0, upper=frame["rated_kw_total"].to_numpy(float)
        )
        scen = pd.DataFrame(
            {
                "station_id": frame["station_id"].astype(str),
                "h": h,
                "target_time": frame["target_time"],
                "q05": lo * demand_multiplier,
                "q50": preds[self.champion] * demand_multiplier,
                "q95": hi * demand_multiplier,
            }
        )
        flags = []
        for col, lo_hi in self.ranges.items():
            vals = shocked.loc[fut, col]
            if len(vals) and (vals.max() > lo_hi[1] or vals.min() < lo_hi[0]):
                flags.append(
                    {
                        "feature": col,
                        "min": round(float(vals.min()), 2),
                        "max": round(float(vals.max()), 2),
                        "train_min": round(lo_hi[0], 2),
                        "train_max": round(lo_hi[1], 2),
                        "message": "Outside the training range: tree models plateau, so treat the effect as indicative.",
                    }
                )
        assumptions = []
        if demand_multiplier != 1.0:
            assumptions.append(f"demand multiplier {demand_multiplier} applied after the model (scenario assumption)")
        if force_holiday:
            assumptions.append("all forecast hours treated as a public holiday")
        return {"baseline": base, "scenario": scen, "ood_flags": flags, "assumptions": assumptions}


def _num(v: Any) -> float | str | None:
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return str(v)
