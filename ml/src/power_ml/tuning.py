"""Optuna hyper-parameter search for the boosters (report Section 5.9, Appendix B).

Objective: mean WAPE over ``PanelTimeSeriesSplit`` folds cut *inside the training block*
(validation and test are never seen). Trials use a higher learning rate for speed; the final
fit keeps the configured learning rate with early stopping, so only structural parameters
(depth/leaves, regularisation, sampling, Tweedie power) are transferred.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import optuna

from power_core import metrics
from power_core.splits import PanelTimeSeriesSplit
from power_ml.models import LGBMModel, XGBModel, subsample_origins
from power_ml.training import load_models_cfg, load_split

TRANSFER_EXCLUDE = {"learning_rate", "n_estimators"}


def _space(trial: optuna.Trial, model: str) -> dict[str, Any]:
    if model == "lgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 500, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.7),
        }
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.7),
    }


def tune(root: Path, source: str, log: Any = print) -> dict[str, Any]:
    cfg = load_models_cfg(root)
    tc = cfg["tune"]
    data = subsample_origins(load_split(root, source, "train"), cfg["origin_stride"]["tune"]).reset_index(drop=True)
    folds = list(PanelTimeSeriesSplit(n_splits=tc["n_splits"], embargo_hours=tc["embargo_hours"]).split(data))
    log(f"tuning on {len(data):,} rows, {len(folds)} folds")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    results: dict[str, Any] = {}
    for name in tc["models"]:
        base = dict(cfg[name])
        base["learning_rate"] = tc["learning_rate"]

        def objective(trial: optuna.Trial, name: str = name, base: dict = base) -> float:
            params = {**base, **_space(trial, name)}
            scores = []
            for k, (tr, va) in enumerate(folds):
                cls = LGBMModel if name == "lgbm" else XGBModel
                m = cls(params=params, early_stopping_rounds=50, seed=cfg["seed"], n_jobs=cfg["n_jobs"],
                        monotone_increasing=tuple(cfg["monotone_increasing"]))
                d_tr, d_va = data.iloc[tr], data.iloc[va]
                m.fit(d_tr, d_va)
                scores.append(metrics.wape(d_va["y"], m.predict(d_va)))
                trial.report(float(np.mean(scores)), k)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        study = optuna.create_study(direction="minimize", study_name=f"{source}_{name}",
                                    sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
        t0 = time.perf_counter()
        study.optimize(objective, n_trials=tc["n_trials"], timeout=tc["timeout_s"])
        best = {k: v for k, v in study.best_params.items() if k not in TRANSFER_EXCLUDE}
        results[name] = {"best_value_wape": round(study.best_value, 5), "best_params": best,
                         "n_trials": len(study.trials),
                         "n_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
                         "elapsed_s": round(time.perf_counter() - t0, 1)}
        log(f"  {name}: best CV WAPE {study.best_value:.4f} after {len(study.trials)} trials")
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports" / "tuning.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
