from __future__ import annotations

import httpx
import numpy as np
import pandas as pd
import pytest

from power_core.config import AcnCfg, CleaningCfg, PaloAltoCfg
from power_ml import resample as rs
from power_ml import sessions as sx
from power_ml import weather as wx
from power_ml.ingest import acn, palo_alto

U = pd.Timestamp


def _sessions(rows: list[tuple[str, str, str, str, float]]) -> pd.DataFrame:
    """(station, connect, charge_end, disconnect, kWh) in UTC."""
    df = pd.DataFrame(rows, columns=["station_id", "connect_time", "charge_end_time", "disconnect_time", "energy_kwh"])
    for c in ("connect_time", "charge_end_time", "disconnect_time"):
        df[c] = pd.to_datetime(df[c], utc=True)
    df["session_hash"] = [f"h{i}" for i in range(len(df))]
    df["port_id"] = [f"{s}:{i}" for i, s in enumerate(df["station_id"])]
    df["port_kw"] = 6.6
    df["latitude"], df["longitude"], df["quality_flags"] = 37.44, -122.16, 0
    return df[list(sx.SESSION_COLUMNS)]


def test_proration_occupancy_and_states():
    s = _sessions([
        ("A", "2019-01-01 10:30", "2019-01-01 12:30", "2019-01-01 14:00", 12.0),  # 2 h charging @ 6 kW
        ("A", "2019-01-01 11:00", "2019-01-01 11:30", "2019-01-01 11:45", 3.0),
    ])
    st = sx.build_station_table(s, "x")
    p = rs.sessions_to_hourly(s, st, U("2019-01-01 15:00", tz="UTC")).set_index("ts")
    e = p["energy_kwh"]
    assert e[U("2019-01-01 10:00", tz="UTC")] == pytest.approx(3.0)
    assert e[U("2019-01-01 11:00", tz="UTC")] == pytest.approx(6.0 + 3.0)
    assert e[U("2019-01-01 12:00", tz="UTC")] == pytest.approx(3.0)
    assert e.sum() == pytest.approx(s["energy_kwh"].sum())       # energy is conserved
    assert p.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()  # gap-free
    r12 = p.loc[U("2019-01-01 12:00", tz="UTC")]
    assert r12["occupied_port_hours"] == pytest.approx(1.0)
    assert r12["idle_port_hours"] == pytest.approx(0.5)
    assert r12["occ_ratio"] == pytest.approx(0.5)                # 2 ports
    r11 = p.loc[U("2019-01-01 11:00", tz="UTC")]
    assert r11["active_sessions"] == 1 and r11["occupied_ports"] == 1   # state at 12:00
    assert r11["arrivals"] == 1
    assert p.loc[U("2019-01-01 13:00", tz="UTC"), "active_sessions"] == 0


def test_cleaning_rules_and_report():
    s = _sessions([
        ("A", "2019-01-01 10:00", "2019-01-01 11:00", "2019-01-01 12:00", 5.0),
        ("A", "2019-01-01 10:00", "2019-01-01 11:00", "2019-01-01 12:00", 5.0),   # duplicate (same port id below)
        ("A", "2019-01-02 10:00", "2019-01-02 10:01", "2019-01-02 10:02", 0.0),   # failed plug-in
        ("A", "2019-01-03 10:00", "2019-01-03 10:30", "2019-01-03 09:00", 2.0),   # negative duration
        ("A", "2019-01-04 10:00", "2019-01-04 11:00", "2019-01-04 13:00", 50.0),  # impossible power
        ("A", "2019-01-05 10:00", None, "2019-01-05 14:00", 6.6),                 # missing charge end
    ])
    s.loc[1, "port_id"] = s.loc[0, "port_id"]
    clean, rep = sx.clean_sessions(s, CleaningCfg())
    r = rep.to_dict()
    assert r["rules"]["duplicate"] == 1
    assert r["rules"]["failed_plug_in"] == 1
    assert r["rules"]["invalid_duration_or_energy"] == 1
    assert r["rules"]["capped_power"] == 1
    assert r["rules"]["imputed_charge_end"] == 1
    assert len(clean) == 3 and r["clean_sessions"] == 3
    capped = clean.loc[clean["quality_flags"] & sx.FLAG_CAPPED_POWER > 0, "energy_kwh"].iloc[0]
    assert capped == pytest.approx(1.1 * 6.6 * 1.0)
    imputed = clean.loc[clean["quality_flags"] & sx.FLAG_IMPUTED_END > 0].iloc[0]
    assert imputed["charge_end_time"] == U("2019-01-05 11:00", tz="UTC")


PA_CFG = PaloAltoCfg(raw_path="x", site_regex=r'^(?:.*/\s*)?(?P<site>[^#/]+?)\s*(?:#\s*\d+)?\s*$',
                     port_kw={"Level 2": 6.6, "Level 1": 1.9})


def test_palo_alto_parser_times_sites_and_pii():
    raw = pd.DataFrame({
        "Station Name": ["PALO ALTO CA / HAMILTON #1", "PALO ALTO CA / HAMILTON #2", "PALO ALTO CA / BRYANT # 3"],
        "MAC Address": ["00:11", "00:12", "00:13"],
        "Start Date": ["7/29/2019 20:17", "1/15/2019 08:00", "11/3/2019 01:30"],
        "Start Time Zone": ["PDT", "PST", "PST"],
        "Total Duration (hh:mm:ss)": ["02:00:00", "26:10:00", "01:00:00"],
        "Charging Time (hh:mm:ss)": ["01:30:00", "03:00:00", "00:45:00"],
        "Energy (kWh)": ["8.5", "12", "3.1"],
        "Port Type": ["Level 2", "Level 2", "Level 1"],
        "Port Number": ["1", "2", "1"],
        "EVSE ID": ["111", "112", "113"],
        "Latitude": ["37.444", "37.444", "37.446"],
        "Longitude": ["-122.160", "-122.160", "-122.162"],
        "User ID": ["u1", "u2", "u3"],
        "Driver Postal Code": ["94301", "94301", "94303"],
    })
    out = palo_alto.parse(raw, PA_CFG, "America/Los_Angeles")
    assert list(out.columns) == list(sx.SESSION_COLUMNS)
    assert out["station_id"].tolist() == ["PA_HAMILTON", "PA_HAMILTON", "PA_BRYANT"]
    assert out.loc[0, "connect_time"] == U("2019-07-30 03:17", tz="UTC")
    assert out.loc[1, "connect_time"] == U("2019-01-15 16:00", tz="UTC")
    assert out.loc[1, "disconnect_time"] - out.loc[1, "connect_time"] == pd.Timedelta(hours=26, minutes=10)
    assert out.loc[2, "port_kw"] == 1.9
    text = out.astype(str).to_string()
    for pii in ("00:11", "u1", "94301"):
        assert pii not in text


def test_acn_parser():
    recs = [{"sessionID": "s1", "stationID": "CA-1", "siteID": "caltech", "userID": "secret",
             "connectionTime": "Wed, 25 Apr 2018 11:08:04 GMT", "disconnectTime": "Wed, 25 Apr 2018 13:20:00 GMT",
             "doneChargingTime": None, "kWhDelivered": 7.5}]
    out = acn.parse(recs, AcnCfg(raw_path="x"))
    assert out.loc[0, "station_id"] == "ACN_CALTECH"
    assert pd.isna(out.loc[0, "charge_end_time"])
    assert "secret" not in out.astype(str).to_string()


def test_weather_merge_interpolation_and_climatology():
    ts = pd.date_range("2019-01-01", periods=48, freq="h", tz="UTC")
    panel = pd.DataFrame({"station_id": "A", "ts": ts, "energy_kwh": 0.0})
    temp = np.arange(48, dtype=float)
    temp[5:7] = np.nan      # short gap -> interpolated
    temp[20:30] = np.nan    # long gap -> climatology (same hour, earlier day not available) -> stays NaN or filled
    weather = pd.DataFrame({"station_id": "A", "ts": ts, "temperature_2m_c": temp, "precipitation_mm": 0.0})
    out, rep = wx.merge_weather(panel, weather, max_gap_hours=3)
    assert out.loc[5, "temperature_2m_c"] == pytest.approx(5.0)
    assert rep["missing_before"] == 12 and rep["missing_after_interp"] == 10
    assert out.loc[20, "temperature_2m_c"] == pytest.approx(44.0)  # hour 20 climatology = the day-2 value


def test_fetch_archive_batches_locations():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["latitude"].count(",") == 1
        assert request.url.params["timezone"] == "GMT"
        item = {"hourly": {"time": ["2019-01-01T00:00", "2019-01-01T01:00"],
                           "temperature_2m": [10.0, 11.0], "precipitation": [0.0, 0.2]}}
        return httpx.Response(200, json=[item, item])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    w = wx.fetch_archive({"A": (37.4, -122.1), "B": (37.5, -122.2)}, U("2019-01-01", tz="UTC"),
                         U("2019-01-01", tz="UTC"), "https://example.invalid/archive",
                         ["temperature_2m", "precipitation"], client=client)
    assert set(w["station_id"]) == {"A", "B"} and len(w) == 4
    assert w["ts"].dt.tz is not None and "temperature_2m_c" in w
