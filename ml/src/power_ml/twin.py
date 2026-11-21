"""Evaluate the frozen models on a deployment twin (e.g. Pune) instead of the training city.

The live console serves a deployment profile (``configs/deployments/<name>.yaml``): real local sites, the
city's own weather, its holidays and time zone, with demand replayed from paired training sites aligned on
local wall-clock time. Every analysis the console shows (leaderboard, per-horizon errors, SHAP, error strata,
stress test, drift references) is recomputed here on that twin, so nothing on screen describes the training
city. The models themselves are not refit: this measures how the frozen models behave in the deployment's
context.

``build_twin_root`` writes a self-contained root under ``data/twins/<name>/`` (configs, model artefacts,
panel, splits and features in the twin's time zone and holidays), so the existing ``evaluate`` and
``analysis`` functions run on it unchanged. ``run_twin`` then evaluates val/test/ood, runs the analyses and
copies the results to ``reports/<name>/``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from power_core.config import data_config, feature_config
from power_core.deployment import Deployment, align_to_grid, load_deployment
from power_core.features import build_direct_frame
from power_core.splits import SplitBounds, assign_split

WX = ["temperature_2m_c", "precipitation_mm"]
SOURCE = "palo_alto"  # the training source the models were fitted on


def _twin_panel(root: Path, dep: Deployment) -> pd.DataFrame:
    """Training panel moved to the twin: local wall-clock time, twin weather, donor ids kept for the models."""
    panel = pd.read_parquet(root / "data" / "processed" / f"panel_{SOURCE}.parquet")
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    value_cols = [c for c in panel.columns if c not in ("station_id", "ts")]
    tw = dep.retime_panel(panel, value_cols)  # site ids, twin timeline
    tw = tw.loc[tw["station_id"].isin(dep.rounds[0])]  # one site per training site: the models key on donor ids
    cache = root / dep.cfg.weather_cache
    if not cache.exists():
        raise FileNotFoundError(f"{cache} missing: run `power-api seed` once with POWER_DEPLOYMENT={dep.cfg.name}")
    wx = pd.read_parquet(cache)
    wx["ts"] = pd.to_datetime(wx["ts"], utc=True)
    wx = align_to_grid(wx, pd.DatetimeIndex(sorted(tw["ts"].unique())), WX)
    tw = tw.drop(columns=WX).merge(wx, on=["station_id", "ts"], how="left")
    for c in WX:
        tw[c] = tw.groupby("station_id")[c].transform(lambda x: x.interpolate(limit_direction="both"))
    tw["station_id"] = tw["station_id"].map(dep.site_to_donor)
    return tw.sort_values(["station_id", "ts"]).reset_index(drop=True)


def build_twin_root(root: Path, name: str, log: Any = print) -> Path:
    dep = load_deployment(root, name)
    if dep is None:
        raise ValueError(f"no deployment profile configs/deployments/{name}.yaml")
    twin = root / "data" / "twins" / name
    if twin.exists():
        shutil.rmtree(twin)
    (twin / "data" / "processed").mkdir(parents=True)
    (twin / "data" / "interim").mkdir(parents=True)
    (twin / "reports" / "figures").mkdir(parents=True)

    # configs: the twin's time zone, holidays and site names
    shutil.copytree(root / "configs", twin / "configs")
    dcfg_raw = yaml.safe_load((root / "configs" / "data.yaml").read_text(encoding="utf-8"))
    dcfg_raw["timezone"] = dep.timezone
    dcfg_raw["holidays"] = {"country": dep.cfg.holidays.country, "subdiv": dep.cfg.holidays.subdiv}
    dcfg_raw["display_names"] = {s.donor: s.name for s in dep.cfg.sites}
    (twin / "configs" / "data.yaml").write_text(yaml.safe_dump(dcfg_raw, sort_keys=False), encoding="utf-8")
    (twin / "pyproject.toml").write_text("[project]\nname = 'twin'\n", encoding="utf-8")

    # frozen model artefacts (read-only copy) and the data manifest
    models_cfg = yaml.safe_load((root / "configs" / "models.yaml").read_text(encoding="utf-8"))
    version = models_cfg["version"]
    shutil.copytree(root / "artifacts" / "models" / version, twin / "artifacts" / "models" / version,
                    ignore=shutil.ignore_patterns("live"))
    shutil.copy2(root / "data" / "manifest.json", twin / "data" / "manifest.json")

    # panel, stations and split bounds on the twin timeline (donor ids: the models' categories)
    panel = _twin_panel(root, dep)
    for nm in (SOURCE, name):  # the LSTM reads panel_<training source>; reports read panel_<twin>
        panel.drop(columns=["split"], errors="ignore").to_parquet(twin / "data" / "processed" / f"panel_{nm}.parquet")
    st = pd.read_parquet(root / "data" / "interim" / f"stations_{SOURCE}.parquet")
    st = st.loc[st["station_id"].isin(dep.donor_to_site)].copy()
    orig = pd.to_datetime(st["first_seen"], utc=True)
    st["first_seen"] = pd.Series(dep.retime(orig), index=st.index).fillna(orig)
    for nm in (SOURCE, name):
        st.to_parquet(twin / "data" / "interim" / f"stations_{nm}.parquet")
    b = SplitBounds.from_dict(json.loads((root / "data" / "processed" / f"splits_{SOURCE}.json").read_text(encoding="utf-8")))
    tb = SplitBounds.from_dict({k: (str(dep.retime_one(v)) if isinstance(v, pd.Timestamp) else v)
                                for k, v in b.__dict__.items()})
    (twin / "data" / "processed" / f"splits_{name}.json").write_text(json.dumps(tb.to_dict()), encoding="utf-8")
    ood_start = dep.retime_one(pd.Timestamp(data_config(root).window.ood_start).tz_convert("UTC"))

    # features in the twin's calendar (time zone + holidays); evaluation rows keep observed weather
    fcfg = feature_config(root)
    hol = (dep.cfg.holidays.country, dep.cfg.holidays.subdiv)
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": [], "ood": []}
    for sid in sorted(panel["station_id"].unique()):
        frame = build_direct_frame(panel.loc[panel["station_id"] == sid], st, fcfg, tz=dep.timezone, holidays=hol)
        if frame.empty:
            continue
        split = assign_split(frame["origin"], frame["target_time"], tb)
        split[(frame["origin"] >= ood_start).to_numpy()] = "ood"
        for nm in parts:
            p = frame.loc[split == nm]
            if nm == "train":  # only the drift/PSI reference reads training rows: keep h = 1
                p = p.loc[p["h"] == 1]
            if len(p):
                parts[nm].append(p)
        log(f"  {dep.donor_to_site[sid]}: {len(frame):,} rows")
    for nm, ps in parts.items():
        pd.concat(ps, ignore_index=True).to_parquet(twin / "data" / "processed" / f"features_{name}_{nm}.parquet",
                                                   index=False)
    return twin


def run_twin(root: Path, name: str, log: Any = print) -> dict[str, Any]:
    from power_ml.analysis import run_error_analysis, run_shap, run_stress_test
    from power_ml.evaluation import evaluate

    twin = build_twin_root(root, name, log=log)
    data_config.cache_clear()
    feature_config.cache_clear()
    out: dict[str, Any] = {}
    for split in ("val", "test", "ood"):
        out[split] = evaluate(twin, name, split, log=log)["leaderboard"]
    run_shap(twin, name)
    run_error_analysis(twin, name)
    run_stress_test(twin, name)
    dest = root / "reports" / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(twin / "reports", dest)
    (dest / "README.md").write_text(
        f"# {name} twin results\n\nGenerated by `power-ml twin --deployment {name}`: the frozen models evaluated on "
        f"the {name} deployment twin (local sites, weather, holidays and time zone; demand replayed from the paired "
        "training sites). The console serves these instead of the training-city results.\n",
        encoding="utf-8",
    )
    return {"reports": str(dest), "test_wape": {k: v["wape"] for k, v in out["test"].items()}}
