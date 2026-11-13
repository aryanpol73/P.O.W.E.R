"""Live model registry for champion-challenger retrains.

The frozen, test-scored artefacts in ``artifacts/models/<version>/`` are never modified. Retrained
challengers live beside them in ``artifacts/models/<version>/live/<run_id>/`` and ``live/registry.json``
records every run and which one (if any) currently serves as the live champion. The test leaderboard
therefore always describes the frozen models; live forecasts carry the challenger's version tag.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from power_core.conformal import calibrator_from_dict


class LiveRegistry:
    def __init__(self, root: Path, version: str) -> None:
        self.dir = root / "artifacts" / "models" / version / "live"
        self.path = self.dir / "registry.json"

    def read(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"active": None, "runs": []}

    def _write(self, reg: dict[str, Any]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(reg, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)

    def save_run(self, run_id: str, model: Any, cqr_dict: dict[str, Any], record: dict[str, Any]) -> None:
        d = self.dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, d / "model.joblib", compress=3)
        (d / "cqr.json").write_text(json.dumps(cqr_dict), encoding="utf-8")
        reg = self.read()
        reg["runs"] = [r for r in reg["runs"] if r["run_id"] != run_id] + [
            {**record, "run_id": run_id, "saved_at": datetime.now(UTC).isoformat(timespec="seconds")}
        ]
        self._write(reg)

    def set_active(self, run_id: str | None) -> None:
        reg = self.read()
        if run_id is not None and not any(r["run_id"] == run_id for r in reg["runs"]):
            raise KeyError(run_id)
        reg["active"] = run_id
        for r in reg["runs"]:
            r["active"] = r["run_id"] == run_id
        self._write(reg)

    def load(self, run_id: str) -> tuple[Any, Any]:
        d = self.dir / run_id
        model = joblib.load(d / "model.joblib")
        cqr = calibrator_from_dict(json.loads((d / "cqr.json").read_text(encoding="utf-8")))
        return model, cqr
