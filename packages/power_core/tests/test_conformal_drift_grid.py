from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from power_core.config import AlertCfg, RerouteCfg
from power_core.conformal import CQR, AdaptiveConformal, SplitConformal, bucket_of, sort_quantiles
from power_core.drift import PageHinkley, psi, seasonal_psi
from power_core.grid import AlertHysteresis, Risk, plan_reroutes


def _heteroscedastic(n, rng):
    h = rng.integers(1, 25, n)
    x = rng.uniform(0, 10, n)
    y = x + rng.normal(0, 0.5 + 0.1 * h, n)
    return h, x, y


def test_split_conformal_coverage_per_bucket():
    rng = np.random.default_rng(1)
    h, x, y = _heteroscedastic(20000, rng)
    cal = slice(0, 10000)
    test = slice(10000, None)
    sc = SplitConformal(alpha=0.1).fit(y[cal], x[cal], h[cal])
    lo, hi = sc.predict(x[test], h[test], lower=None)
    covered = (y[test] >= lo) & (y[test] <= hi)
    b = bucket_of(h[test])
    for i in range(3):
        assert 0.88 <= covered[b == i].mean() <= 0.92


def test_cqr_coverage_and_no_crossing():
    rng = np.random.default_rng(2)
    h, x, y = _heteroscedastic(20000, rng)
    q_lo = x - 0.3 * (0.5 + 0.1 * h)   # deliberately too narrow
    q_hi = x + 0.3 * (0.5 + 0.1 * h)
    q_lo[:50], q_hi[:50] = q_hi[:50], q_lo[:50]  # some crossing
    cqr = CQR(alpha=0.1).fit(y[:10000], q_lo[:10000], q_hi[:10000], h[:10000])
    lo, hi = cqr.predict(q_lo[10000:], q_hi[10000:], h[10000:], lower=None)
    assert (hi >= lo).all()
    cov = ((y[10000:] >= lo) & (y[10000:] <= hi)).mean()
    assert 0.88 <= cov <= 0.92
    a, b, c = sort_quantiles(np.array([3.0]), np.array([1.0]), np.array([2.0]))
    assert (a[0], b[0], c[0]) == (1.0, 2.0, 3.0)


def test_cqr_clipping_to_physical_bounds():
    cqr = CQR(alpha=0.1).fit(np.ones(100), np.zeros(100), np.full(100, 2.0), np.ones(100, dtype=int))
    lo, hi = cqr.predict(np.array([-5.0]), np.array([50.0]), np.array([1]), lower=0.0, upper=10.0)
    assert lo[0] == 0.0 and hi[0] == 10.0


def test_aci_widens_after_misses():
    aci = AdaptiveConformal(target_alpha=0.1, gamma=0.05)
    for _ in range(5):
        aci.update(covered=False)
    assert aci.alpha_t < 0.1
    for _ in range(50):
        aci.update(covered=True)
    assert aci.alpha_t > 0.1


def test_psi_and_bucketed_psi():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert psi(a, b) < 0.02
    assert psi(a, rng.normal(1.5, 1, 5000)) > 0.5
    # A strong daily cycle must not look like drift once the hour-of-week profile is removed,
    # even though the current week covers only part of the raw value range.
    how = np.tile(np.arange(168), 30)
    vals = np.sin(2 * np.pi * how / 24) * 10 + rng.normal(0, 1, len(how))
    ref = pd.DataFrame({"how": how, "y": vals}).iloc[: -168 * 2]
    fresh_how = np.arange(168 * 2) % 168
    cur = pd.DataFrame({"how": fresh_how, "y": np.sin(2 * np.pi * fresh_how / 24) * 10 + rng.normal(0, 1, 336)})
    night_only = cur.loc[cur["how"] % 24 < 6]
    assert psi(ref["y"], night_only["y"]) > 0.5                 # raw PSI is fooled by the cycle
    assert seasonal_psi(ref, night_only, "y") < 0.2             # seasonal PSI is not
    assert seasonal_psi(ref, cur.assign(y=cur["y"] + 2.0), "y") > 0.5   # a real level shift is caught


def test_page_hinkley():
    rng = np.random.default_rng(4)
    ph = PageHinkley(delta=0.05, lam=20)
    assert not any(ph.update(x) for x in np.abs(rng.normal(0, 1, 2000)))
    assert any(ph.update(x) for x in np.abs(rng.normal(3, 1, 200)))


def test_alert_hysteresis_sequence():
    a = AlertHysteresis(AlertCfg())
    seq = [0.5, 0.72, 0.86, 0.83, 0.79, 0.79, 0.66, 0.6, 0.6]
    got = [a.update(r) for r in seq]
    assert got == [Risk.NORMAL, Risk.WARNING, Risk.CRITICAL, Risk.CRITICAL, Risk.CRITICAL,
                   Risk.WARNING, Risk.WARNING, Risk.WARNING, Risk.NORMAL]


def test_reroute_respects_radius_and_headroom():
    snap = pd.DataFrame({
        "station_id": ["HOT", "NEAR", "FAR", "BUSY"],
        "lat": [37.4440, 37.4460, 37.6000, 37.4450],
        "lon": [-122.1600, -122.1610, -122.1600, -122.1590],
        "q50": [90.0, 10.0, 0.0, 60.0],
        "q95": [120.0, 15.0, 5.0, 90.0],
        "capacity_kw": [100.0, 50.0, 50.0, 100.0],
        "is_modeled": [True, True, True, True],
    })
    plan = plan_reroutes(snap, RerouteCfg(radius_km=3.0, divertible_share=0.5, target_safe_ratio=0.7), 0.85)
    assert {r.to_station for r in plan} == {"NEAR"}  # FAR outside radius, BUSY critical itself
    by_src = {r.from_station: r.shifted_kwh for r in plan}
    assert by_src["HOT"] == pytest.approx(0.5 * (120 - 85))       # 17.5 of NEAR's 25 kWh headroom
    assert by_src["BUSY"] == pytest.approx(0.5 * (90 - 85))       # the second critical site uses what is left
    assert plan[-1].to_headroom_kwh == pytest.approx(25 - 17.5 - 2.5)
