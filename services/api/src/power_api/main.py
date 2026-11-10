"""ASGI entry point: ``uvicorn power_api.main:app`` (single worker; the simulator is in-process)."""

from power_api.app import create_app

app = create_app()
