"""``power-api`` command line: seed the database, run the server."""

from __future__ import annotations

import json

import typer

from power_api.db.session import make_sync_engine
from power_api.settings import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def seed() -> None:
    """Create tables and load stations, the hourly panel and archive weather (replaces existing data)."""
    from power_api.seed import seed as run_seed

    s = get_settings()
    typer.echo(f"seeding {s.db_url}")
    typer.echo(json.dumps(run_seed(make_sync_engine(s.db_url), s.repo_root, s.source, s.deployment), indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the API (with the embedded simulator unless POWER_EMBEDDED_WORKER=false)."""
    import uvicorn

    uvicorn.run("power_api.main:app", host=host, port=port, reload=reload, workers=1)


if __name__ == "__main__":
    app()
