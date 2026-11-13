"""Unit tests for the queueing layer and the analytic cross-checks."""
from __future__ import annotations

import math

import numpy as np
import pytest

from carnot.climate import Environment, add_solar_geometry, load_tmy, select_design_day
from carnot.config import cfg_path, load_config, deep_merge
from carnot.metrics import summarise
from carnot.queueing import (Arrival, build_arrivals, erlang_c, fluid_queue,
                             pointwise_stationary_wait, simulate_station)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def env(cfg):
    tmy = load_tmy(cfg_path(cfg, "raw_tmy"), cfg["climate"]["utc_offset_h"])
    tmy = add_solar_geometry(
        tmy, cfg["climate"]["latitude_deg"], cfg["climate"]["longitude_deg"],
        cfg["climate"]["utc_offset_h"], cfg["climate"]["albedo_ground"])
    return Environment.from_day(select_design_day(tmy))


# ----------------------------------------------------------------------
# Erlang-C and fluid queue
# ----------------------------------------------------------------------
def test_erlang_c_single_server_matches_mm1():
    for a in (0.2, 0.5, 0.8, 0.95):
        assert erlang_c(1, a) == pytest.approx(a, rel=1e-9)


def test_erlang_c_saturated_and_large_offered_load():
    assert erlang_c(4, 4.0) == 1.0
    assert erlang_c(4, 9.0) == 1.0
    # numerically stable in the heavily loaded regime
    assert 0.5 < erlang_c(4, 3.9) < 1.0


def test_pointwise_wait_matches_erlang_c_definition():
    lam, mu, c = 3.0, 1.0, 4
    a = lam / mu
    wq = pointwise_stationary_wait(np.array([lam]), np.array([mu]), c)[0]
    assert wq == pytest.approx(erlang_c(c, a) / (c * mu - lam), rel=1e-9)


def test_fluid_queue_accumulates_exactly():
    dt = 15.0
    lam = np.full(400, 6.0)
    mu = np.full(400, 1.0)
    q, wait = fluid_queue(lam, mu, c=4, dt_s=dt)
    # overload 2 veh/h for 100 minutes -> ~3.33 vehicles
    assert q[-1] == pytest.approx(2.0 * (len(lam) - 1) * dt / 3600.0, rel=1e-9)
    assert np.all(q >= 0.0)
    assert wait[-1] == pytest.approx(q[-1] / lam[-1])


def test_fluid_queue_never_negative_under_load():
    lam = np.array([1.0, 1.0, 1.0, 1.0])
    mu = np.full(4, 5.0)
    q, _ = fluid_queue(lam, mu, c=4, dt_s=60.0)
    assert np.all(q == 0.0)


# ----------------------------------------------------------------------
# Station simulation invariants
# ----------------------------------------------------------------------
def test_station_conserves_customers(cfg, env):
    rng = np.random.default_rng(3)
    arrivals = build_arrivals(cfg, rng)
    res = simulate_station(cfg, env, arrivals)
    served = sum(1 for r in res.records if r["served"])
    balked = sum(1 for r in res.records if r["balked"])
    reneged = sum(1 for r in res.records if r["reneged"])
    censored = sum(1 for r in res.records if r.get("censored"))
    assert served + balked + reneged + censored == res.n_arrivals == len(arrivals)


def test_station_never_double_books_a_stall(cfg, env):
    rng = np.random.default_rng(4)
    res = simulate_station(cfg, env, build_arrivals(cfg, rng))
    served = sorted(res.served, key=lambda r: r["t_start_s"])
    # the DES assigns one session per stall at a time; a served record's start
    # must never precede the previous served record's end on the same stall
    for prev, nxt in zip(served, served[1:]):
        if nxt["t_start_s"] < prev["t_end_s"]:
            # only legal if they used different stalls
            assert nxt["stall"] != prev["stall"]


def test_summary_metrics_are_bounded(cfg, env):
    rng = np.random.default_rng(5)
    res = simulate_station(cfg, env, build_arrivals(cfg, rng))
    s = summarise(res, cfg)
    assert 0.0 <= s["utilization"] <= 1.0
    assert s["peak_load_kw"] >= s["mean_load_kw"] > 0.0
    assert 0.0 <= s["lost_demand_pct"] <= 100.0
    assert s["wait_p95_min"] >= s["wait_median_min"] >= 0.0
    assert s["energy_kwh_day"] > 0.0


def test_littles_law_on_the_waiting_line(cfg, env):
    """L_q = lambda_eff * W_q for the admitted, served customers."""
    rng = np.random.default_rng(11)
    res = simulate_station(cfg, env, build_arrivals(cfg, rng))
    served = res.served
    if len(served) < 30:
        pytest.skip("too few served customers for a stable Little's law check")
    admitted = res.n_arrivals - sum(1 for r in res.records if r["balked"])
    lam_eff = admitted / 86400.0
    wq = np.mean([(r["t_start_s"] - r["arrival_s"]) for r in served])
    lq = float(res.queue_len.mean())
    assert lq == pytest.approx(lam_eff * wq, rel=0.35, abs=0.15)


def test_heatwave_degrades_performance(cfg, env):
    """A warmer day must not improve any headline operational metric."""
    from carnot.climate import Environment as E
    hot = E(t_amb=env.t_amb + 6.0, ghi=env.ghi, ghi_cabinet=env.ghi_cabinet,
            t_sky=env.t_sky + 6.0, wind=env.wind)
    base, warmer = [], []
    for seed in range(4):
        rng = np.random.default_rng(100 + seed)
        arr = build_arrivals(cfg, rng)
        base.append(summarise(simulate_station(cfg, env, arr), cfg)["wait_mean_min"])
        warmer.append(summarise(simulate_station(cfg, hot, arr), cfg)["wait_mean_min"])
    assert np.mean(warmer) > np.mean(base)
