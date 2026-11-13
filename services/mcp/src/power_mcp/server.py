"""P.O.W.E.R. MCP server: natural-language grid questions without giving an AI database access.

A thin, read-only adapter over the REST API (``POWER_API_URL``), so every number an assistant quotes is the
same number the operator console shows. Tools return compact JSON summaries sized for a model context.

Run:   uv run power-mcp                      # stdio (Claude Desktop / Claude Code)
       uv run power-mcp --transport http     # streamable HTTP on :8765/mcp
"""

from __future__ import annotations

import argparse
import os
from typing import Annotated, Any, Literal

import httpx
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from power_mcp.guards import RateLimiter, clamp, scrub

API_URL = os.environ.get("POWER_API_URL", "http://127.0.0.1:8011").rstrip("/")
TOKEN = os.environ.get("POWER_API_TOKEN", "")
limiter = RateLimiter(int(os.environ.get("POWER_MCP_RATE_PER_MIN", "30")), 60.0)

mcp = MCPServer(
    "power",
    version="0.1.0",
    instructions=(
        "P.O.W.E.R. forecasts hourly EV charging energy (kWh) and grid risk for a city's charging sites. "
        "Only state numbers returned by a tool, and say which tool they came from. Times are UTC ISO-8601; "
        "convert to the deployment's time zone (get_network_status -> deployment.timezone) when talking to people. "
        "Risk compares the q95 forecast with an ASSUMED transformer rating. If a scenario returns OOD flags, say "
        "the result is outside the training range. The live deployment may be a digital twin: say so when asked "
        "about accuracy on real local drivers."
    ),
)

_client: httpx.Client | None = None


def client() -> httpx.Client:
    global _client
    if _client is None:
        headers = {"User-Agent": "power-mcp/0.1"}
        if TOKEN:
            headers["Authorization"] = f"Bearer {TOKEN}"
        _client = httpx.Client(base_url=f"{API_URL}/api/v1", timeout=60.0, headers=headers)
    return _client


def _call(method: str, path: str, **kw: Any) -> Any:
    limiter.check()
    r = client().request(method, path, **kw)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"API {r.status_code} on {path}: {detail}")
    return scrub(r.json())


def _get(path: str, **params: Any) -> Any:
    return _call("GET", path, params={k: v for k, v in params.items() if v is not None})


def resolve_station(ref: str) -> str:
    """Accept a station id or (part of) its display name, case-insensitive."""
    stations = _get("/stations", modeled=False)
    ids = {s["station_id"]: s for s in stations}
    if ref in ids:
        return ref
    low = ref.lower()
    hits = [s for s in stations if low in s["name"].lower() or low in s["station_id"].lower()]
    modeled = [s for s in hits if s.get("is_modeled")]
    pick = modeled or hits
    if len(pick) == 1:
        return pick[0]["station_id"]
    if not pick:
        raise ValueError(f"no station matches {ref!r}; call list_stations")
    raise ValueError(f"{ref!r} is ambiguous: " + ", ".join(f"{s['name']} ({s['station_id']})" for s in pick[:8]))


# ------------------------------------------------------------------ tools
@mcp.tool()
def list_stations(
    risk: Annotated[Literal["NORMAL", "WARNING", "CRITICAL"] | None, Field(description="only stations at this risk")] = None,
    include_unmodelled: Annotated[bool, Field(description="also list discovered chargers that get no forecast")] = False,
) -> list[dict[str, Any]]:
    """Charging sites with ports, assumed transformer rating and current grid risk (worst hour in the next 12 h)."""
    out = []
    for s in _get("/stations", modeled=not include_unmodelled):
        if risk and s.get("risk") != risk:
            continue
        out.append({k: s.get(k) for k in ("station_id", "name", "total_ports", "transformer_kw", "risk", "is_modeled",
                                          "latitude", "longitude")})
    return out


@mcp.tool()
def get_network_status() -> dict[str, Any]:
    """Network KPIs now (load, next-hour forecast with interval, occupancy, risk counts), the simulation clock,
    the deployment (city, time zone, whether it is a digital twin) and the live champion model."""
    return {
        "kpis": _get("/network/kpis"),
        "sim": _get("/sim/state"),
        "deployment": _get("/deployment"),
        "health": {k: v for k, v in _get("/health").items() if k in ("status", "model_version", "live_champion_version",
                                                                      "champion", "last_error")},
    }


@mcp.tool()
def get_station_forecast(
    station: Annotated[str, Field(description="station id or part of its name, e.g. 'Westend'")],
    horizon_hours: Annotated[int, Field(ge=1, le=24)] = 24,
    model: Annotated[str, Field(description="'champion' or a model name such as M3_xgboost")] = "champion",
) -> dict[str, Any]:
    """Hourly forecast for the next N hours: median (q50), calibrated 90% interval (q05-q95) and risk per hour."""
    sid = resolve_station(station)
    f = _get(f"/stations/{sid}/forecast", model=model, history_hours=0)
    pts = f["points"][:horizon_hours]
    peak = max(pts, key=lambda p: p["q50"]) if pts else None
    return {
        "station_id": sid,
        "model": f["model"],
        "model_version": f["model_version"],
        "forecast_origin": f["forecast_origin"],
        "transformer_kw_assumed": f["transformer_kw"],
        "peak": peak,
        "points": [{k: p[k] for k in ("target_time", "h", "q05", "q50", "q95", "risk")} for p in pts],
    }


@mcp.tool()
def explain_peak_shap(
    station: Annotated[str, Field(description="station id or part of its name")],
    horizon_h: Annotated[int | None, Field(ge=1, le=24, description="hours ahead; default = the forecast peak")] = None,
    top_k: Annotated[int, Field(ge=1, le=8)] = 6,
) -> dict[str, Any]:
    """Why the champion forecasts what it does: TreeSHAP contributions in kWh that add up to the forecast."""
    sid = resolve_station(station)
    e = _get(f"/stations/{sid}/explain", h=horizon_h)
    contrib = [c for c in e["contributions"] if c["feature"] != "__other__"][:top_k]
    other = sum(c["shap"] for c in e["contributions"]) - sum(c["shap"] for c in contrib)
    return {
        "station_id": sid,
        "target_time": e["target_time"],
        "h": e["h"],
        "base_value_kwh": e["base_value"],
        "prediction_kwh": e["prediction"],
        "top_contributions": [{"driver": c["display"], "value": c["value"], "kwh": c["shap"]} for c in contrib],
        "all_other_features_kwh": round(other, 3),
        "narrative": e["narrative"],
    }


@mcp.tool()
def simulate_weather_shock(
    temperature_delta_c: Annotated[float, Field(ge=-15, le=15, description="added to every forecast hour")] = 0.0,
    precipitation_mm: Annotated[float | None, Field(ge=0, le=50, description="rain every hour (mm)")] = None,
    force_holiday: bool = False,
    demand_multiplier: Annotated[float, Field(gt=0, le=2, description="scenario assumption applied after the model")] = 1.0,
    stations: Annotated[list[str] | None, Field(description="station ids or names; omit for the whole network")] = None,
) -> dict[str, Any]:
    """What-if on the next 24 h (runs on a copy; live forecasts and alerts are untouched). Returns baseline vs scenario
    peaks, stations that would breach the alert thresholds, and out-of-training-range flags."""
    clamp(temperature_delta_c, -15, 15, "temperature_delta_c")
    ids = [resolve_station(s) for s in stations] if stations else ["*"]
    body = {"station_ids": ids, "temperature_delta_c": temperature_delta_c, "precipitation_mm": precipitation_mm,
            "force_holiday": force_holiday, "demand_multiplier": demand_multiplier}
    r = _call("POST", "/scenarios/simulate", json=body)

    def peaks(points: list[dict[str, Any]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in points:
            out[p["station_id"]] = max(out.get(p["station_id"], 0.0), p["q50"])
        return out

    pb, ps = peaks(r["baseline"]), peaks(r["scenario"])
    worst: dict[str, str] = {}
    for a in r["alerts_in_scenario"]:
        if worst.get(a["station_id"]) != "CRITICAL":
            worst[a["station_id"]] = a["level"]
    return {
        "forecast_origin": r["forecast_origin"],
        "requested_by": "mcp",
        "inputs": body,
        "delta_network_peak_kwh": r["delta_network_peak_kwh"],
        "station_peaks_kwh": {s: {"baseline": round(pb.get(s, 0), 2), "scenario": round(v, 2)} for s, v in ps.items()},
        "stations_at_risk": worst,
        "ood_flags": r["ood_flags"],
        "assumptions": r["assumptions"],
    }


@mcp.tool()
def compare_models(
    source: Annotated[
        Literal["test", "live"], Field(description="frozen test-set leaderboard or rolling live accuracy")
    ] = "test",
    metric: Literal["wape", "mase", "rmse", "mae", "picp"] = "wape",
    live_hours: Annotated[int, Field(ge=24, le=1440)] = 168,
) -> list[dict[str, Any]]:
    """Model leaderboard, best first. 'test' = held-out test split (the reported results); 'live' = hours the console
    has replayed since the forecasts were made."""
    if source == "live":
        rows = [{"model": k, **v} for k, v in _get("/models/live-arena", hours=live_hours)["models"].items()]
    else:
        lb = _get("/models/leaderboard").get("leaderboard", {})
        rows = []
        for k, v in lb.items():
            row = {"model": k, "wape": v.get("wape"), "mase": v.get("mase"), "rmse": v.get("rmse"), "mae": v.get("mae")}
            iv = v.get("interval_cqr") or v.get("interval_split_conformal") or {}
            row["picp"] = iv.get("picp")
            rows.append(row)
    key = metric
    rows = [r for r in rows if r.get(key) is not None]
    reverse_good = metric == "picp"
    rows.sort(key=lambda r: abs(r[key] - 0.9) if reverse_good else r[key])
    return [{k: v for k, v in r.items() if k in ("model", "n", "wape", "mase", "rmse", "mae", "picp")} for r in rows]


@mcp.tool()
def get_alerts(status: Literal["open", "all"] = "open", limit: Annotated[int, Field(ge=1, le=100)] = 20) -> dict[str, Any]:
    """Grid alerts (WARNING >= 70%, CRITICAL >= 85% of the assumed transformer rating, with hysteresis) and the
    current reroute suggestions."""
    alerts = _get("/alerts", status=status, limit=limit)
    return {"alerts": alerts, "reroutes": _get("/reroutes")}


@mcp.tool()
def get_drift_report(days: Annotated[int, Field(ge=1, le=90)] = 7) -> dict[str, Any]:
    """Latest data/model drift checks (CUSUM on forecast bias, seasonal PSI, interval coverage, WAPE vs test) and
    whether a retrain is recommended."""
    d = _get("/drift", days=days)
    return {k: d[k] for k in ("window_days", "checks", "latest_at", "latest", "recommend_retrain")}


@mcp.tool()
def get_agent_status() -> dict[str, Any]:
    """Health of the background agents (pipeline, drift, retrain, discovery), buffer freshness and live champion."""
    st = _get("/agents/status")
    for a in st.get("agents", {}).values():
        a.pop("last_details", None)
    return st


# ------------------------------------------------------------------ resources
@mcp.resource("power://datacard")
def datacard() -> dict[str, Any]:
    """Data card: sources, cleaning, splits, assumptions."""
    return _get("/datacard")


@mcp.resource("power://deployment")
def deployment() -> dict[str, Any]:
    """Which city the console serves and how it relates to the training data."""
    return _get("/deployment")


@mcp.resource("power://models/leaderboard")
def leaderboard() -> dict[str, Any]:
    """Frozen test-set leaderboard."""
    return _get("/models/leaderboard")


@mcp.resource("power://stations/{station_id}")
def station(station_id: str) -> dict[str, Any]:
    """One station's metadata and current forecast summary."""
    meta = next((s for s in _get("/stations", modeled=False) if s["station_id"] == station_id), None)
    if meta is None:
        raise ValueError(f"unknown station {station_id}")
    return {"station": meta, "forecast": get_station_forecast(station_id) if meta.get("is_modeled") else None}


# ------------------------------------------------------------------ prompts
@mcp.prompt()
def grid_briefing() -> str:
    """Morning briefing for a grid operator."""
    return (
        "Prepare a short operator briefing for the next 24 hours. Call get_network_status, list_stations, get_alerts and "
        "get_drift_report. Then cover: (1) network load now and the next-hour forecast with its interval, (2) any site "
        "at WARNING or CRITICAL, its peak hour and what drives it (explain_peak_shap), (3) reroute suggestions, "
        "(4) whether drift checks recommend retraining. Give times in the deployment's local time zone. Quote only tool "
        "numbers and keep it under 200 words."
    )


@mcp.prompt()
def viva_explainer(station: str) -> str:
    """Explain one forecast to a non-expert examiner."""
    return (
        f"Explain the forecast for '{station}' to a non-expert. Call get_station_forecast and explain_peak_shap. Say what "
        "the median and the 90% interval mean, why the peak happens (top 3 SHAP drivers in plain words), and what the "
        "risk level means for the grid. State that transformer ratings are an assumption, and if get_network_status "
        "shows a digital twin, say that demand is replayed from the training data."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="P.O.W.E.R. MCP server")
    ap.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if args.transport == "http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
