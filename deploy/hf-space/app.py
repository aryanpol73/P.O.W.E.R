"""P.O.W.E.R. backend on a free Hugging Face Space (Gradio SDK, CPU basic).

The Space installs requirements.txt, then runs ``python app.py``. Nothing Gradio-specific is used: this
fetches the runtime bundle (models, seed data, configs), seeds SQLite, starts the copilot on
127.0.0.1:8765 and serves the API on port 7860, which passes /copilot through. See docs/DEPLOY.md.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.setdefault("POWER_ROOT", str(Path.cwd() / "power")))
ROOT.mkdir(parents=True, exist_ok=True)
for k, v in {
    "POWER_SOURCE": "palo_alto",
    "POWER_DEPLOYMENT": "maharashtra",
    "SIM_MODE": "LIVE_SHADOW",
    "POWER_TORCH_THREADS": "2",
    "POWER_COPILOT_UPSTREAM": "http://127.0.0.1:8765",
    "POWER_API_URL": "http://127.0.0.1:7860",
    "POWER_COPILOT_EFFORT": "low",
    "MPLBACKEND": "Agg",
}.items():
    os.environ.setdefault(k, v)

bindir = Path(sys.executable).parent


def tool(name: str) -> str:
    for exe in (bindir / name, bindir / f"{name}.exe"):
        if exe.exists():
            return str(exe)
    return name


subprocess.run([tool("power-api"), "fetch-bundle"], check=True, cwd=ROOT)
subprocess.run([tool("power-api"), "seed", "--if-empty"], check=True, cwd=ROOT)
subprocess.Popen([tool("power-mcp"), "--transport", "http", "--host", "127.0.0.1", "--port", "8765"], cwd=ROOT)
os.chdir(ROOT)
sys.exit(subprocess.call([tool("power-api"), "serve", "--host", "0.0.0.0", "--port", "7860"]))
