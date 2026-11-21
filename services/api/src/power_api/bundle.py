"""Runtime bundle: the trained models, seed data and reports a hosted API needs (git-ignored locally).

``scripts/make_bundle.py`` packs them into one ``.tar.gz``; a host whose disk starts empty (Render)
downloads it on start-up from ``POWER_BUNDLE_URL``. Secrets (``data/vapid.json``, ``.env``) are never packed.
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path

import httpx

# Paths relative to the repo root; {source} and {reports} are filled in by the packer.
FILES = [
    "configs",
    "artifacts/models",
    "data/manifest.json",
    "data/interim/stations_{source}.parquet",
    "data/processed/panel_{source}.parquet",
    "data/processed/splits_{source}.json",
    "data/external",
    "reports/{reports}",
    "reports/figures",
    "reports/metrics_test.json",
]
NEVER = {"vapid.json", ".env", "power.db"}


def needed(root: Path) -> bool:
    return not (root / "artifacts" / "models").is_dir() or not (root / "data" / "processed").is_dir()


def fetch(url: str, root: Path) -> int:
    """Download and unpack the bundle into ``root``; returns the number of files written."""
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "bundle.tar.gz"
        with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(30.0, read=300.0)) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        n = 0
        with tarfile.open(dest, "r:gz") as tar:
            for member in tar.getmembers():
                p = Path(member.name)
                if p.is_absolute() or ".." in p.parts or p.name in NEVER or not (member.isfile() or member.isdir()):
                    continue  # only plain files inside the root
                tar.extract(member, root, filter="data")
                n += member.isfile()
        return n
