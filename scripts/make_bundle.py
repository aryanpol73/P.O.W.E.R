"""Pack the runtime bundle for a hosted API (see services/api/src/power_api/bundle.py and docs/DEPLOY.md).

    uv run python scripts/make_bundle.py                    # -> dist/power-runtime.tar.gz
    uv run python scripts/make_bundle.py --reports pune

Upload the file as a GitHub release asset and set POWER_BUNDLE_URL to its download link.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from power_api.bundle import FILES, NEVER

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="palo_alto")
    ap.add_argument("--reports", default="pune", help="the reports folder the deployment serves")
    ap.add_argument("--out", default="dist/power-runtime.tar.gz")
    a = ap.parse_args()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(out, "w:gz") as tar:
        for rel in FILES:
            p = ROOT / rel.format(source=a.source, reports=a.reports)
            if not p.exists():
                raise SystemExit(f"missing {p.relative_to(ROOT)}: run the training pipeline first")
            for f in [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file()):
                if f.name in NEVER or f.suffix == ".log":
                    continue
                tar.add(f, arcname=f.relative_to(ROOT).as_posix())
                n += 1
    print(f"{out.relative_to(ROOT)}: {n} files, {out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
