"""``power-api`` command line: seed the database, run the server."""

from __future__ import annotations

import json

import typer

from power_api.db.session import make_sync_engine
from power_api.settings import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def seed(
    if_empty: bool = typer.Option(False, "--if-empty", help="do nothing when stations already exist (container start-up)"),
    wait: int = typer.Option(0, "--wait", help="seconds to wait for the database to accept connections"),
) -> None:
    """Create tables and load stations, the hourly panel and archive weather (replaces existing data)."""
    import time

    from sqlalchemy import func, inspect, select, text
    from sqlalchemy.exc import OperationalError

    from power_api.db import models as m
    from power_api.seed import seed as run_seed

    s = get_settings()
    deadline = time.monotonic() + wait
    while True:  # a fresh PostgreSQL container restarts once after initdb; do not race it
        try:
            with make_sync_engine(s.db_url).connect() as conn:
                conn.execute(text("SELECT 1"))
            break
        except OperationalError:
            if time.monotonic() > deadline:
                raise
            typer.echo("waiting for the database...")
            time.sleep(2)
    if if_empty:
        eng = make_sync_engine(s.db_url)
        if inspect(eng).has_table("stations"):
            with eng.connect() as conn:
                if conn.execute(select(func.count()).select_from(m.Station)).scalar_one() > 0:
                    typer.echo("database already seeded; skipping")
                    return
    typer.echo(f"seeding {_masked(s.db_url)}")
    typer.echo(json.dumps(run_seed(make_sync_engine(s.db_url), s.repo_root, s.source, s.deployment), indent=2))


def _masked(url: str) -> str:
    """The database URL without its password, safe for logs."""
    from sqlalchemy.engine import make_url

    return make_url(url).render_as_string(hide_password=True)


@app.command("fetch-bundle")
def fetch_bundle() -> None:
    """Download the runtime bundle (models, seed data, reports) from POWER_BUNDLE_URL if it is missing."""
    from power_api import bundle

    s = get_settings()
    if not s.bundle_url:
        typer.echo("POWER_BUNDLE_URL not set; using the files on disk")
        return
    if not bundle.needed(s.repo_root):
        typer.echo("runtime bundle already present")
        return
    typer.echo("downloading the runtime bundle...")
    typer.echo(f"{bundle.fetch(s.bundle_url, s.repo_root)} files unpacked")


@app.command("vapid-keys")
def vapid_keys() -> None:
    """Print a new Web Push key pair to paste into the host's env vars (nothing is written to disk)."""
    import py_vapid
    from cryptography.hazmat.primitives import serialization

    v = py_vapid.Vapid02()
    v.generate_keys()
    pub = v.public_key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    priv = v.private_key.private_numbers().private_value.to_bytes(32, "big")
    typer.echo(f"VAPID_PUBLIC_KEY={py_vapid.b64urlencode(pub)}")
    typer.echo(f"VAPID_PRIVATE_KEY={py_vapid.b64urlencode(priv)}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the API (with the embedded simulator unless POWER_EMBEDDED_WORKER=false)."""
    import uvicorn

    uvicorn.run("power_api.main:app", host=host, port=port, reload=reload, workers=1)


if __name__ == "__main__":
    app()
