"""Tests for the grey-box calibration layer.

The parameter-recovery test is the important one: it proves that the
estimation procedure can recover a known physical parameter set from noisy
observations, which is what licenses the calibration claims in the report.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from carnot import calibration as cal
from carnot.climate import add_solar_geometry, load_tmy, representative_days
from carnot.config import cfg_path, load_config, nested_get, nested_set


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def tmy(cfg):
    df = load_tmy(cfg_path(cfg, "raw_tmy"), cfg["climate"]["utc_offset_h"])
    return add_solar_geometry(
        df, cfg["climate"]["latitude_deg"], cfg["climate"]["longitude_deg"],
        cfg["climate"]["utc_offset_h"], cfg["climate"]["albedo_ground"])


def test_parameter_recovery(cfg, tmy):
    rng = np.random.default_rng(7)
    days = representative_days(tmy, 5)
    data = cal.generate_twin(cfg, days, [0.0] * len(days), rng, n_per_day=2)
    fit = cal.fit_physics(cfg, data, max_nfev=150)

    assert fit.success
    assert fit.rmse_cab < 2.0
    assert fit.rmse_bat < 2.0

    truth = np.array([nested_get(cfg, n) * cal.TRUTH_MULTIPLIERS[n] for n in fit.names])
    rel = np.abs(fit.theta - truth) / truth
    # The two convective parameters are the least identifiable from
    # temperature logs alone, so they get a wider tolerance.
    loose = {"cabinet.UA_conv_natural_W_per_K", "cabinet.fan_ua_ratio"}
    for name, r in zip(fit.names, rel):
        limit = 0.40 if name in loose else 0.20
        assert r < limit, f"{name} recovered with {100 * r:.1f}% error"


def test_residual_learner_improves_fit(cfg, tmy):
    rng = np.random.default_rng(8)
    days = representative_days(tmy, 4)
    data = cal.generate_twin(cfg, days, [0.0] * len(days), rng, n_per_day=2)
    fit = cal.fit_physics(cfg, data, max_nfev=80)

    trial = copy.deepcopy(cfg)
    for name, value in zip(fit.names, fit.theta):
        nested_set(trial, name, float(value))

    grid = data.obs[0]["t"]
    rows = []
    for spec, ws, o in zip(data.specs, data.weather, data.obs):
        sim = cal.simulate_observation(trial, spec, ws, grid)
        rows.append(pd.DataFrame({
            "t_amb": o["t_amb"], "g_cab": o["g_cab"], "power": o["power"],
            "wind": o["wind"], "t_cab": o["t_cab"], "phys": sim["t_cab"],
        }))
    frame = pd.concat(rows, ignore_index=True)
    resid = frame["t_cab"].to_numpy() - frame["phys"].to_numpy()
    ridge = cal.RidgeResidual(lam=1.0).fit(frame, resid)

    pred_phys = frame["phys"].to_numpy()
    pred_grey = pred_phys + ridge.predict(frame)
    rmse_phys = float(np.sqrt(np.mean((frame["t_cab"] - pred_phys) ** 2)))
    rmse_grey = float(np.sqrt(np.mean((frame["t_cab"] - pred_grey) ** 2)))
    assert rmse_grey < rmse_phys
    assert ridge.r2_ > 0.0
    assert np.all(np.isfinite(ridge.coef_))


def test_black_box_is_exact_on_its_own_training_features(cfg, tmy):
    """A degree-3 ridge must at least fit a smooth synthetic response."""
    rng = np.random.default_rng(9)
    n = 400
    frame = pd.DataFrame({
        "t_amb": rng.uniform(30, 50, n),
        "g_cab": rng.uniform(0, 700, n),
        "power": rng.uniform(20, 150, n),
        "wind": rng.uniform(0.5, 8.0, n),
    })
    y = 0.9 * frame["t_amb"] + 0.01 * frame["g_cab"] - 0.05 * frame["power"]
    box = cal.PolynomialBlackBox(degree=3, lam=1e-6).fit(frame, y.to_numpy())
    assert np.sqrt(np.mean((box.predict(frame) - y) ** 2)) < 0.05
