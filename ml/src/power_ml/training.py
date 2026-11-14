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
import warnings
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
    """HEAD commit, suffixed with '-dirty' when tracked source files have uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
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
    models = {}
    for n in manifest["models"]:
        if n == "E1_ensemble":
            continue
        try:
            models[n] = joblib.load(d / f"{n}.joblib")
        except ModuleNotFoundError as exc:  # e.g. the LSTM in an image built without PyTorch
            warnings.warn(f"skipping {n}: {exc}; install the missing package to serve it", stacklevel=2)
    ens = joblib.load(d / "E1_ensemble.joblib")
    needs_panel = [m for m in models.values() if hasattr(m, "attach_panel")]
    if needs_panel:  # sequence models read hourly history from the panel (not pickled with them)
        panel = pd.read_parquet(root / "data" / "processed" / f"panel_{manifest['source']}.parquet")
        for m in needs_panel:
            m.attach_panel(panel)
    return models, ens, manifest


def train_lstm(root: Path, source: str, log: Any = print) -> dict[str, Any]:
    """Add M5 (LSTM) to the existing artefact version without touching the other models."""
    from power_ml.models.lstm import LSTMModel

    cfg = load_models_cfg(root)
    lc = dict(cfg["lstm"])
    stride = lc.pop("origin_stride")
    out_dir = artifacts_dir(root, cfg["version"])
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    train = load_split(root, source, "train")
    es_cfg = cfg["early_stopping"]
    fit_idx, hold_idx = time_holdout(train["origin"], train["target_time"], es_cfg["holdout_frac"], es_cfg["embargo_hours"])
    fit, hold = train.iloc[fit_idx].reset_index(drop=True), train.iloc[hold_idx].reset_index(drop=True)
    del train
    panel = pd.read_parquet(root / "data" / "processed" / f"panel_{source}.parquet")
    model = LSTMModel(seed=cfg["seed"], tz=data_config_tz(root), **lc).prepare(panel, fit)
    t0 = time.perf_counter()
    model.fit(subsample_origins(fit, stride), subsample_origins(hold, stride))
    dt = time.perf_counter() - t0
    for rec in model.history_:
        hw = rec.get("holdout_wape", float("nan"))
        log(f"  epoch {rec['epoch']:>2}  train L1 {rec['train_l1']:.3f} kWh  holdout WAPE {hw:.4f}  ({rec['elapsed_s']}s)")
    val = load_split(root, source, "val")
    p = model.predict(val)
    manifest["calibrators"][model.name] = SplitConformal(alpha=ALPHA).fit(val["y"].to_numpy(), p, val["h"].to_numpy()).to_dict()
    manifest["val_wape"][model.name] = round(metrics.wape(val["y"], p), 4)
    manifest["models"][model.name] = {"train_time_s": round(dt, 2),
                                      "best_epoch": model.best_epoch_, "epochs_run": len(model.history_),
                                      "origin_stride": stride, "history": model.history_, "git_sha": git_sha(root)}
    joblib.dump(model, out_dir / f"{model.name}.joblib", compress=3)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    log(f"{model.name}: val WAPE {manifest['val_wape'][model.name]} (best epoch {model.best_epoch_}, {dt:.0f}s)")
    return manifest["models"][model.name]


def data_config_tz(root: Path) -> str:
    from power_core.config import data_config

    return data_config(root).timezone
