"""Unit tests for the thermal core.

These tests check the properties the report relies on: energy conservation of
the integrator, the shape of the derating laws, agreement between the scalar
hot-loop and the vectorised reference implementation, and convergence of RK4.
"""
from __future__ import annotations

import numpy as np
import pytest

from carnot import thermal
from carnot.climate import Environment, add_solar_geometry, load_tmy, select_design_day
from carnot.config import cfg_path, load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def day(cfg):
    tmy = load_tmy(cfg_path(cfg, "raw_tmy"), cfg["climate"]["utc_offset_h"])
    tmy = add_solar_geometry(
        tmy,
        cfg["climate"]["latitude_deg"],
        cfg["climate"]["longitude_deg"],
        cfg["climate"]["utc_offset_h"],
        cfg["climate"]["albedo_ground"],
    )
    return select_design_day(tmy, cfg["climate"]["design_day"])


# ----------------------------------------------------------------------
# Derating laws
# ----------------------------------------------------------------------
def test_piecewise_derating_anchor_points():
    f = thermal.derate_piecewise
    assert f(30.0, 40.0, 52.0, 0.3) == pytest.approx(1.0)
    assert f(40.0, 40.0, 52.0, 0.3) == pytest.approx(1.0)
    assert f(52.0, 40.0, 52.0, 0.3) == pytest.approx(0.3)
    assert f(80.0, 40.0, 52.0, 0.3) == pytest.approx(0.3)
    assert f(46.0, 40.0, 52.0, 0.3) == pytest.approx(0.65)


def test_piecewise_derating_is_monotone_nonincreasing():
    t = np.linspace(-10, 100, 500)
    f = thermal.derate_piecewise(t, 40.0, 52.0, 0.3)
    assert np.all(np.diff(f) <= 1e-12)
    assert np.all((f >= 0.3) & (f <= 1.0))


def test_smooth_derating_converges_to_piecewise():
    """The softplus clamp must reproduce the exact law away from the kinks."""
    t = np.linspace(20, 95, 2000)
    exact = thermal.derate_piecewise(t, 40.0, 52.0, 0.3)
    smooth = thermal.derate_smooth(t, 40.0, 52.0, 0.3, kappa=200.0)
    # Ignore a narrow band around each knee (the clamp has finite width)
    mask = (np.abs(t - 40.0) > 0.35) & (np.abs(t - 52.0) > 0.35)
    assert np.max(np.abs(smooth[mask] - exact[mask])) < 0.02


def test_smooth_derating_is_bounded_by_the_floor():
    t = np.linspace(-20, 120, 1000)
    smooth = thermal.derate_smooth(t, 40.0, 52.0, 0.3, kappa=30.0)
    assert smooth.min() >= 0.3 - 1e-9
    assert smooth.max() <= 1.0 + 1e-9


# ----------------------------------------------------------------------
# Scalar hot loop vs vectorised reference
# ----------------------------------------------------------------------
def test_scalar_and_vector_power_agree(cfg):
    for tc, tb, soc in [(30, 30, 0.2), (55, 45, 0.5), (70, 51, 0.7), (85, 60, 0.9)]:
        assert thermal._power_s(tc, tb, soc, cfg) == pytest.approx(
            float(thermal.available_power(tc, tb, soc, cfg)), rel=1e-9
        )


def test_scalar_and_vector_heat_balances_agree(cfg):
    env = {"t_amb": 42.0, "ghi": 800.0, "ghi_cabinet": 420.0, "t_sky": 26.0, "wind": 3.0}
    p = 110.0
    cab_scalar = thermal._cab_heat_s(48.0, p, env["t_amb"], env["ghi_cabinet"],
                                     env["t_sky"], env["wind"], cfg)
    cab_vector = float(thermal.cabinet_heat(48.0, p, env, env["ghi_cabinet"], cfg))
    assert cab_scalar == pytest.approx(cab_vector, rel=1e-6, abs=1e-9)

    bat_scalar = thermal._bat_heat_s(45.0, p, 0.35, env["t_amb"], env["ghi"], cfg)
    bat_vector = float(thermal.battery_heat(45.0, p, 0.35, env, env["ghi"], cfg))
    assert bat_scalar == pytest.approx(bat_vector, rel=1e-6, abs=1e-9)


# ----------------------------------------------------------------------
# Energy and SoC bookkeeping
# ----------------------------------------------------------------------
def test_reference_duration_matches_analytic_flat_power(cfg):
    """The reference duration equals a fine-grid integral of the taper law."""
    start = 0.10
    target = cfg["session"]["target_soc"]
    p = min(cfg["vehicle"]["peak_kW"], cfg["site"]["stall_rating_kw"])
    socs = np.linspace(start, target, 4001)
    expected = np.trapezoid(
        3600.0 * cfg["battery"]["capacity_kWh"] / (p * thermal.soc_taper(socs, cfg)), socs)
    assert thermal.reference_duration_s(cfg, start) == pytest.approx(expected, rel=2e-3)


def test_reference_duration_increases_with_taper(cfg):
    """Tapering above the knee makes the reference slower per unit SoC."""
    low = thermal.reference_duration_s(cfg, 0.10) - thermal.reference_duration_s(cfg, 0.20)
    high = thermal.reference_duration_s(cfg, 0.60) - thermal.reference_duration_s(cfg, 0.70)
    assert high > low


def test_session_reaches_target_and_conserves_energy(cfg, day):
    env = Environment.from_day(day)
    at = env.at(15 * 3600)
    res = thermal.simulate_session(
        cfg, env, 15 * 3600, 0.25,
        at["t_amb"] + cfg["battery"]["T_arrival_offset_K"],
        at["t_amb"] + 4.0, dt_s=10.0,
    )
    expected_energy = (cfg["session"]["target_soc"] - 0.25) * cfg["battery"]["capacity_kWh"]
    assert res.soc_end == pytest.approx(cfg["session"]["target_soc"], abs=1e-6)
    assert res.energy_kwh == pytest.approx(expected_energy, rel=0.02)
    # Mean power must be consistent with delivered energy and duration
    assert res.mean_power_kw == pytest.approx(
        res.energy_kwh / (res.duration_s / 3600.0), rel=1e-9)


def test_shorter_timestep_converges(cfg, day):
    env = Environment.from_day(day)
    at = env.at(16 * 3600)
    kw = dict(soc_start=0.25, t_bat0=at["t_amb"] + 5.0, t_cab0=at["t_amb"] + 4.0)
    coarse = thermal.simulate_session(cfg, env, 16 * 3600, dt_s=20.0, **kw)
    fine = thermal.simulate_session(cfg, env, 16 * 3600, dt_s=5.0, **kw)
    assert abs(coarse.duration_s - fine.duration_s) / fine.duration_s < 0.01


# ----------------------------------------------------------------------
# Steady state
# ----------------------------------------------------------------------
def test_equilibrium_residual_is_zero(cfg):
    tc, tb, p = thermal.equilibrium_power(cfg, 43.0, 800.0, 450.0, 2.5, 27.0, 0.3)
    env = {"t_amb": 43.0, "ghi": 800.0, "ghi_cabinet": 450.0, "t_sky": 27.0, "wind": 2.5}
    assert abs(float(thermal.cabinet_heat(tc, p, env, 450.0, cfg))) < 0.05
    assert abs(float(thermal.battery_heat(tb, p, 0.3, env, 800.0, cfg))) < 0.05


def test_equilibrium_power_falls_with_ambient(cfg):
    powers = [thermal.equilibrium_power(cfg, t, 850.0, 480.0, 2.5, t - 16.0, 0.3)[2]
              for t in (35.0, 40.0, 45.0, 50.0)]
    assert powers[0] >= powers[1] >= powers[2] >= powers[3]
    assert powers[0] == pytest.approx(150.0, abs=1e-6)
    assert powers[-1] < 0.7 * powers[0]


def test_cabinet_is_self_cooling_at_zero_load(cfg):
    """An idle cabinet above ambient must lose heat."""
    q = thermal._cab_heat_s(55.0, 0.0, 40.0, 0.0, 30.0, 3.0, cfg)
    assert q < 0.0
