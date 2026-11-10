"""Train the model zoo, calibrate on validation, save versioned artefacts (report Chapters 5-6).

Data usage, block by block:
* train (minus its last ``holdout_frac`` of origins) -> fit every model;
* train tail (after an embargo)                      -> early stopping and Ridge alpha choice;
* validation                                        -> ensemble weights + conformal calibration;
* test                                              -> touched only by ``evaluate --split test``.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from power_core import metrics
from power_core.conformal import CQR, SplitConformal
from power_core.splits import time_holdout
from power_ml.models import (
    LGBMModel,
    LGBMQuantile,
    RFModel,
    RidgeModel,
    WeightedEnsemble,
    XGBModel,
    subsample_origins,
)
from power_ml.models.baselines import default_baselines

QUANTILE = "M4_lightgbm_quantile"
ALPHA = 0.10


def load_models_cfg(root: Path) -> dict[str, Any]:
    return yaml.safe_load((root / "configs" / "models.yaml").read_text(encoding="utf-8"))


def load_split(root: Path, source: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(root / "data" / "processed" / f"features_{source}_{split}.parquet")


def artifacts_dir(root: Path, version: str) -> Path:
    return root / "artifacts" / "models" / version


def git_sha(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def tuned_params(root: Path) -> dict[str, dict]:
    p = root / "reports" / "tuning.json"
    if not p.exists():
        return {}
    return {k: v["best_params"] for k, v in json.loads(p.read_text(encoding="utf-8")).items()}


def build_zoo(cfg: dict[str, Any], tuned: dict[str, dict]) -> list[tuple[Any, int, bool]]:
    """(model, origin_stride, uses_holdout) in training order."""
    seed, n_jobs = cfg["seed"], cfg["n_jobs"]
    es = cfg["early_stopping"]["rounds"]
    mono = tuple(cfg["monotone_increasing"])
    stride = cfg["origin_stride"]
    xgb_p = {**cfg["xgb"], **tuned.get("xgb", {})}
    lgb_p = {**cfg["lgbm"], **tuned.get("lgbm", {})}
    q = cfg["lgbm_quantile"]
    q_p = {k: v for k, v in q.items() if k != "alphas"}
    zoo: list[tuple[Any, int, bool]] = [(b, 1, False) for b in default_baselines()]
    zoo += [
        (RidgeModel(alphas=tuple(cfg["ridge"]["alphas"])), stride["ridge"], True),
        (RFModel(params=cfg["rf"], seed=seed, n_jobs=n_jobs), stride["rf"], False),
        (XGBModel(params=xgb_p, early_stopping_rounds=es, monotone_increasing=mono, seed=seed, n_jobs=n_jobs),
         stride["xgb"], True),
        (LGBMModel(params=lgb_p, early_stopping_rounds=es, monotone_increasing=mono, seed=seed, n_jobs=n_jobs),
         stride["lgbm"], True),
        (LGBMQuantile(params=q_p, alphas=tuple(q["alphas"]), early_stopping_rounds=es, seed=seed, n_jobs=n_jobs),
         stride["lgbm_quantile"], True),
    ]
    return zoo


def predict_all(models: dict[str, Any], ensemble: WeightedEnsemble | None, frame: pd.DataFrame,
                ) -> tuple[dict[str, np.ndarray], dict[float, np.ndarray]]:
    preds, quants = {}, {}
    for name, m in models.items():
        if name == QUANTILE:
            quants = m.predict_quantiles(frame)
            preds[name] = quants[0.50]
        else:
            preds[name] = m.predict(frame)
    if ensemble is not None:
        preds[ensemble.name] = ensemble.predict(preds)
    return preds, quants


def train_all(root: Path, source: str, log: Any = print) -> dict[str, Any]:
    cfg = load_models_cfg(root)
    tuned = tuned_params(root)
    out_dir = artifacts_dir(root, cfg["version"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(root, source, "train")
    es_cfg = cfg["early_stopping"]
    fit_idx, hold_idx = time_holdout(train["origin"], train["target_time"], es_cfg["holdout_frac"], es_cfg["embargo_hours"])
    fit, hold = train.iloc[fit_idx].reset_index(drop=True), train.iloc[hold_idx].reset_index(drop=True)
    del train
    log(f"fit rows {len(fit):,} | early-stopping holdout rows {len(hold):,}")

    models: dict[str, Any] = {}
    info: dict[str, dict[str, Any]] = {}
    for model, stride, uses_holdout in build_zoo(cfg, tuned):
        f = subsample_origins(fit, stride)
        h = subsample_origins(hold, stride) if uses_holdout else None
        t0 = time.perf_counter()
        model.fit(f, h)
        dt = time.perf_counter() - t0
        models[model.name] = model
        info[model.name] = {"train_time_s": round(dt, 2), "train_rows": len(f), "origin_stride": stride,
                            "best_iteration": getattr(model, "best_iteration_", None)}
        if isinstance(model, RidgeModel):
            info[model.name]["alpha"] = model.alpha_
        log(f"  {model.name:<28} {dt:8.1f}s  rows={len(f):,}")
    del fit, hold

    # ---- validation: ensemble weights and conformal calibration
    val = load_split(root, source, "val")
    preds, quants = predict_all(models, None, val)
    member_names = {"xgb": "M3_xgboost", "lgbm": "M4_lightgbm", "rf": "M2_random_forest", "ridge": "M1_ridge"}
    ens = WeightedEnsemble(tuple(member_names[m] for m in cfg["ensemble"]["members"]))
    ens.fit({m: preds[m] for m in ens.members}, val["y"].to_numpy())
    preds[ens.name] = ens.predict(preds)
    info[ens.name] = {"train_time_s": round(sum(info[m]["train_time_s"] for m in ens.members), 2),
                      "weights": ens.weights_}
    y, hz = val["y"].to_numpy(), val["h"].to_numpy()
    calibrators: dict[str, dict] = {}
    for name, p in preds.items():
        calibrators[name] = SplitConformal(alpha=ALPHA).fit(y, p, hz).to_dict()
    cqr = CQR(alpha=ALPHA).fit(y, quants[0.05], quants[0.95], hz)
    calibrators[f"{QUANTILE}_cqr"] = cqr.to_dict()
    val_wape = {n: round(metrics.wape(y, p), 4) for n, p in preds.items()}

    # ---- save
    for name, m in models.items():
        joblib.dump(m, out_dir / f"{name}.joblib", compress=3)
    joblib.dump(ens, out_dir / f"{ens.name}.joblib")
    manifest = {
        "version": cfg["version"], "source": source, "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(root), "models": info, "val_wape": val_wape, "calibrators": calibrators,
        "tuned_params": tuned, "config": cfg,
        "data_manifest": json.loads((root / "data" / "manifest.json").read_text(encoding="utf-8")).get(source, {}),
        "feature_columns": {n: getattr(m, "columns_", None) or getattr(m, "numeric_", None)
                            for n, m in models.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log(json.dumps({"val_wape": val_wape, "ensemble_weights": ens.weights_}, indent=2))
    return manifest


def load_artifacts(root: Path, version: str | None = None) -> tuple[dict[str, Any], WeightedEnsemble, dict[str, Any]]:
    cfg = load_models_cfg(root)
    d = artifacts_dir(root, version or cfg["version"])
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    models = {n: joblib.load(d / f"{n}.joblib") for n in manifest["models"] if n != "E1_ensemble"}
    ens = joblib.load(d / "E1_ensemble.joblib")
    return models, ens, manifest
