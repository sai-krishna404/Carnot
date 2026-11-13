"""Experiment suite: every number and figure in the report is produced here."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import calibration as cal
from . import metrics, plotting, thermal
from .climate import (Environment, add_solar_geometry, hottest_day_in_month, load_tmy,
                      representative_days, select_design_day)
from .config import PROJECT_ROOT, cfg_path, deep_merge, ensure_dirs, nested_get, nested_set
from .queueing import build_arrivals, simulate_station

SUMMER_MONTHS = [5, 6, 7, 8, 9]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _scenario_cfg(cfg, canopy=False, coating=False):
    c = copy.deepcopy(cfg)
    c["site"]["canopy"] = canopy
    if coating:
        c["site"]["alpha_cabinet"] = c["site"]["coating_alpha_cabinet"]
    return c


def _env(cfg, day, heatwave, canopy):
    return Environment.from_day(
        day,
        canopy=canopy,
        transmission=cfg["site"]["canopy_transmission"],
        delta_t=cfg["site"]["canopy_delta_T_K"],
        heatwave_delta_k=heatwave,
    )


def run_scenario(cfg, day, heatwave, canopy=False, shift=False, coating=False,
                 seeds=8, base_seed=1234):
    """Run the DES for several arrival seeds and aggregate the metrics."""
    scfg = _scenario_cfg(cfg, canopy=canopy, coating=coating)
    env = _env(scfg, day, heatwave, canopy)
    runs, last = [], None
    for s in range(seeds):
        rng = np.random.default_rng(base_seed + 97 * s)
        arrivals = build_arrivals(scfg, rng, shift=shift)
        res = simulate_station(scfg, env, arrivals)
        runs.append(metrics.summarise(res, scfg))
        last = res
    agg = {}
    keys = [k for k, v in runs[0].items() if isinstance(v, (int, float))]
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        agg[k] = float(np.nanmean(vals))
        agg[k + "_sd"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    agg["label"] = f"{'canopy' if canopy else 'no canopy'}/{'shift' if shift else 'no shift'}"
    return agg, last, env, scfg


# ----------------------------------------------------------------------
# E1  climate
# ----------------------------------------------------------------------
def experiment_climate(cfg, tmy, out):
    day = select_design_day(tmy, cfg["climate"]["design_day"])
    processed = PROJECT_ROOT / cfg["paths"]["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    cols = ["t_amb_c", "rh_pct", "ghi", "dni", "dhi", "wind_ms", "zenith_deg",
            "azimuth_deg", "ghi_cabinet", "t_sky_c"]
    tmy[cols].to_csv(processed / "abu_dhabi_tmy_processed.csv", float_format="%.3f")
    day[cols].to_csv(out["tables"] / "t02_design_day.csv", float_format="%.3f")

    h = np.arange(24)
    plotting.climate(day, out["figures"] / "fig01_climate.png",
                     canopy_day=day["t_amb_c"].to_numpy() - cfg["site"]["canopy_delta_T_K"])
    return {
        "design_day": str(day.index[0].date()),
        "t_max": float(day["t_amb_c"].max()),
        "t_min": float(day["t_amb_c"].min()),
        "ghi_max": float(day["ghi_horizontal"].max()),
        "ghi_cabinet_max": float(day["ghi_cabinet"].max()),
        "t_sky_min": float(day["t_sky_c"].min()),
        "mean_wind": float(day["wind_ms"].mean()),
    }, day


# ----------------------------------------------------------------------
# E2  single-session physics
# ----------------------------------------------------------------------
def experiment_physics(cfg, day, out, heatwave=6.0):
    rows, traces = [], []
    cases = [
        ("Typical TMY day (39 degC peak)", 0.0, False, plotting.C["grey"]),
        ("Heatwave (+6 K, 45 degC peak)", heatwave, False, plotting.C["base"]),
        ("Heatwave + canopy", heatwave, True, plotting.C["canopy"]),
    ]
    for label, delta, canopy, color in cases:
        scfg = _scenario_cfg(cfg, canopy=canopy)
        env = _env(scfg, day, delta, canopy)
        at = env.at(15 * 3600)
        res = thermal.simulate_session(
            scfg, env, 15 * 3600, 0.25,
            at["t_amb"] + scfg["battery"]["T_arrival_offset_K"],
            at["t_amb"] + 4.0, dt_s=10.0, record=True)
        traces.append((label, res.trace, color))
        rows.append({
            "case": label, "t_amb_C": at["t_amb"], "duration_min": res.duration_s / 60,
            "reference_min": res.nominal_duration_s / 60, "slowdown": res.slowdown,
            "mean_power_kW": res.mean_power_kw, "energy_kWh": res.energy_kwh,
            "t_cab_start": res.t_cab_start, "t_cab_end": res.t_cab_end,
            "t_bat_start": res.t_bat_start, "t_bat_end": res.t_bat_end,
        })
    table = pd.DataFrame(rows)
    table.to_csv(out["tables"] / "t03_session_scenarios.csv", index=False, float_format="%.2f")
    plotting.traces(traces, out["figures"] / "fig02_physics_traces.png")

    temp = np.linspace(20, 90, 500)
    dc, db = cfg["cabinet"]["derate"], cfg["battery"]["derate"]
    plotting.derating_curves(
        cfg,
        lambda t, a, b, f: thermal.derate_piecewise(t, a, b, f),
        lambda t, a, b, f, k: thermal.derate_smooth(t, a, b, f, k),
        out["figures"] / "fig03_derating_laws.png")

    # steady-state map
    amb = np.arange(30.0, 52.1, 0.5)
    curves = []
    for label, canopy, color, ls in (("Unshaded", False, plotting.C["base"], "-"),
                                     ("Reflective canopy", True, plotting.C["canopy"], "--")):
        scfg = _scenario_cfg(cfg, canopy=canopy)
        p = []
        for ta in amb:
            # representative mid-afternoon irradiance for this ambient level
            ghi = 850.0 * max(0.0, 1.0 - (ta - 40.0) / 25.0)
            ghi = 700.0 if label.startswith("Ref") else ghi
            tc, tb, pw = thermal.equilibrium_power(
                scfg, float(ta), ghi, ghi * 0.55, 2.5, ta - 16.0, 0.30)
            p.append(pw)
        curves.append((label, amb, np.array(p), color, ls))
    plotting.cliff(curves, out["figures"] / "fig04_cliff.png")
    return table


# ----------------------------------------------------------------------
# E3  thermal stacking
# ----------------------------------------------------------------------
def experiment_stacking(cfg, day, out, heatwave=6.0, n_sessions=4):
    rows = []
    for headway in (0, 10, 20, 40):
        scfg = _scenario_cfg(cfg, canopy=False)
        env = _env(scfg, day, heatwave, False)
        ws = thermal.WeatherSeries(env)
        t = 13.0 * 3600.0
        at = env.at(t)
        t_cab = at["t_amb"] + 4.0
        for k in range(1, n_sessions + 1):
            res = thermal.simulate_session(
                scfg, env, t, 0.25, at["t_amb"] + scfg["battery"]["T_arrival_offset_K"],
                t_cab, dt_s=20.0, weather=ws)
            rows.append({"headway_min": headway,
                         "headway_label": f"{headway} min",
                         "session": k, "mean_power_kw": res.mean_power_kw,
                         "duration_min": res.duration_s / 60,
                         "t_cab_start": res.t_cab_start, "t_cab_end": res.t_cab_end})
            # idle cooling until the next plug-in
            t_end = res.t_end_s
            t = t_end + headway * 60.0
            for tt in np.arange(t_end, t, 30.0):
                a = env.at(tt)
                t_cab = thermal.idle_step(t_cab, a["t_amb"], a["ghi_cabinet"],
                                          a["t_sky"], a["wind"], scfg, 30.0)
    df = pd.DataFrame(rows)
    df.to_csv(out["tables"] / "t04_stacking.csv", index=False, float_format="%.2f")
    plotting.stacking(df, out["figures"] / "fig05_stacking.png")
    return df


# ----------------------------------------------------------------------
# E4  interventions
# ----------------------------------------------------------------------
def experiment_interventions(cfg, day, out, heatwave=6.0, seeds=8):
    specs = [
        ("Typical day, no intervention", 0.0, False, False, False),
        ("Heatwave baseline", heatwave, False, False, False),
        ("Reflective canopy", heatwave, True, False, False),
        ("Canopy + white coating", heatwave, True, False, True),
        ("Time-shift (dynamic tariff)", heatwave, False, True, False),
        ("Canopy + time-shift", heatwave, True, True, False),
        ("Canopy + coating + time-shift", heatwave, True, True, True),
    ]
    rows, results = [], {}
    for label, delta, canopy, shift, coating in specs:
        agg, res, env, scfg = run_scenario(
            cfg, day, delta, canopy=canopy, shift=shift, coating=coating, seeds=seeds)
        agg["scenario"] = label
        rows.append(agg)
        results[label] = (agg, res, env, scfg)
    table = pd.DataFrame(rows)
    table.to_csv(out["tables"] / "t05_interventions.csv", index=False, float_format="%.3f")

    # hourly waits + ECDF
    hourly, ecdf = [], []
    colors = {"Heatwave baseline": plotting.C["base"], "Reflective canopy": plotting.C["canopy"],
              "Time-shift (dynamic tariff)": plotting.C["shift"],
              "Canopy + time-shift": plotting.C["both"]}
    for label, color in colors.items():
        agg, res, env, scfg = results[label]
        hourly.append((label, metrics.hourly_profile(res, scfg), color))
        ecdf.append((label, [(r["t_start_s"] - r["arrival_s"]) / 60.0
                             for r in res.served], color))
    plotting.waits(hourly, out["figures"] / "fig06_waits.png")
    plotting.wait_ecdf(ecdf, out["figures"] / "fig07_wait_ecdf.png")

    # 2x2 factorial on mean wait
    v00 = results["Heatwave baseline"][0]["wait_mean_min"]
    v10 = results["Reflective canopy"][0]["wait_mean_min"]
    v01 = results["Time-shift (dynamic tariff)"][0]["wait_mean_min"]
    v11 = results["Canopy + time-shift"][0]["wait_mean_min"]
    effects = {
        "Canopy main effect": 0.5 * ((v00 - v10) + (v01 - v11)),
        "Time-shift main effect": 0.5 * ((v00 - v01) + (v10 - v11)),
        "Interaction": (v11 - v10) - (v01 - v00),
    }
    plotting.factorial(effects, out["figures"] / "fig08_factorial.png")
    pd.DataFrame({"effect": list(effects), "reduction_min": list(effects.values())}).to_csv(
        out["tables"] / "t06_factorial.csv", index=False, float_format="%.3f")

    # grid profiles
    profiles = []
    for label, color in (("Heatwave baseline", plotting.C["base"]),
                         ("Canopy + time-shift", plotting.C["both"])):
        res = results[label][1]
        hourly_load = res.load_kw.reshape(-1, int(3600 / res.dt_s)).mean(axis=1)
        profiles.append((label, {"hour": np.arange(24), "kw": hourly_load}, color))
    plotting.grid_profile(profiles, out["figures"] / "fig09_grid.png")
    return table, effects, results


# ----------------------------------------------------------------------
# E5  calibration and extrapolation
# ----------------------------------------------------------------------
def _frame_from(cfg_fit, data: cal.TwinDataset) -> pd.DataFrame:
    rows = []
    grid = data.obs[0]["t"]
    for spec, ws, o in zip(data.specs, data.weather, data.obs):
        sim = cal.simulate_observation(cfg_fit, spec, ws, grid)
        rows.append(pd.DataFrame({
            "t_amb": o["t_amb"], "g_cab": o["g_cab"], "power": o["power"],
            "wind": o["wind"], "t_cab_obs": o["t_cab"], "t_cab_phys": sim["t_cab"],
            "t_amb_day": spec.t_start, "valid": o["valid"], "grid": grid,
        })[o["valid"]])
    return pd.concat(rows, ignore_index=True)


def experiment_calibration(cfg, tmy, out, seeds=None):
    rng = np.random.default_rng(int(cfg["meta"]["seed"]))
    train_days = representative_days(tmy, 10)
    test_days = representative_days(tmy, 4)[1:]  # avoid the identical first day
    train = cal.generate_twin(cfg, train_days, [0.0] * len(train_days), rng, n_per_day=3)
    test = cal.generate_twin(cfg, test_days, [8.0] * len(test_days), rng, n_per_day=3)

    fit = cal.fit_physics(cfg, train, max_nfev=250)
    fitted = copy.deepcopy(cfg)
    for name, value in zip(fit.names, fit.theta):
        nested_set(fitted, name, float(value))

    train_frame = _frame_from(fitted, train)
    resid = (train_frame["t_cab_obs"] - train_frame["t_cab_phys"]).to_numpy()
    ridge = cal.RidgeResidual(lam=cfg["calibration"]["ridge_lambda"])
    ridge.fit(train_frame, resid)

    blackbox = cal.PolynomialBlackBox(degree=cfg["calibration"]["blackbox_degree"], lam=1.0)
    blackbox.fit(train_frame, train_frame["t_cab_obs"].to_numpy())

    test_frame = _frame_from(fitted, test)
    test_frame["t_cab_grey"] = test_frame["t_cab_phys"] + ridge.predict(test_frame)
    test_frame["t_cab_black"] = blackbox.predict(test_frame)
    test_frame["resid_phys"] = test_frame["t_cab_obs"] - test_frame["t_cab_phys"]
    test_frame["resid_grey"] = test_frame["t_cab_obs"] - test_frame["t_cab_grey"]

    def rmse(col):
        e = test_frame["t_cab_obs"] - test_frame[col]
        return float(np.sqrt(np.mean(e ** 2)))

    comparison = pd.DataFrame([
        {"model": "Physics only (calibrated)", "rmse_K": rmse("t_cab_phys"),
         "max_abs_K": float(np.max(np.abs(test_frame['resid_phys'])))},
        {"model": "Grey-box (+ ridge residual)", "rmse_K": rmse("t_cab_grey"),
         "max_abs_K": float(np.max(np.abs(test_frame['resid_grey'])))},
        {"model": f"Polynomial black box (deg {cfg['calibration']['blackbox_degree']})",
         "rmse_K": rmse("t_cab_black"),
         "max_abs_K": float(np.max(np.abs(test_frame['t_cab_obs'] - test_frame['t_cab_black'])))},
    ])
    comparison.to_csv(out["tables"] / "t08_model_comparison.csv", index=False, float_format="%.3f")

    params = pd.DataFrame({
        "parameter": [cal.PARAM_LABELS[n] for n in fit.names],
        "prior": fit.theta0,
        "truth": fit.theta0 * np.array([cal.TRUTH_MULTIPLIERS[n] for n in fit.names]),
        "fitted": fit.theta,
        "std_error": fit.std_err,
        "bias_pct": 100.0 * (fit.theta - fit.theta0 * np.array(
            [cal.TRUTH_MULTIPLIERS[n] for n in fit.names])) / (fit.theta0 * np.array(
            [cal.TRUTH_MULTIPLIERS[n] for n in fit.names])),
    })
    params.to_csv(out["tables"] / "t07_calibration_params.csv", index=False, float_format="%.4f")

    ridge.table().to_csv(out["tables"] / "t08b_residual_coefficients.csv",
                         index=False, float_format="%.4f")

    # parity (test set, subsample for legibility)
    sub = test_frame.sample(n=min(900, len(test_frame)), random_state=1)
    plotting.parity(sub["t_cab_obs"].to_numpy(),
                    [("Grey-box", sub["t_cab_grey"].to_numpy(), plotting.C["canopy"]),
                     ("Black box", sub["t_cab_black"].to_numpy(), plotting.C["base"])],
                    out["figures"] / "fig10_calibration.png")

    plotting.residual_plot([("Physics only", test_frame.assign(resid=test_frame["resid_phys"]),
                             plotting.C["base"]),
                            ("Grey-box", test_frame.assign(resid=test_frame["resid_grey"]),
                             plotting.C["canopy"])],
                           out["figures"] / "fig11_residuals.png")

    # error versus ambient band
    bins = [30, 36, 40, 43, 45, 48, 60]
    test_frame["band"] = pd.cut(test_frame["t_amb"], bins)
    band = test_frame.groupby("band", observed=True).apply(
        lambda g: pd.Series({
            "t_amb_mid": float(g["t_amb"].mean()),
            "Grey-box (physics + ridge)": float(np.sqrt(np.mean(g["resid_grey"] ** 2))),
            "Polynomial black box": float(np.sqrt(np.mean((g["t_cab_obs"] - g["t_cab_black"]) ** 2))),
        }), include_groups=False)
    band.to_csv(out["tables"] / "t08c_error_by_band.csv", float_format="%.3f")
    plotting.extrapolation(band.reset_index(), out["figures"] / "fig12_extrapolation.png")

    return {
        "fit_success": fit.success,
        "rmse_cabinet_K": fit.rmse_cab,
        "rmse_pack_K": fit.rmse_bat,
        "n_obs": fit.n_obs,
        "params": params,
        "comparison": comparison,
        "ridge_r2": ridge.r2_,
        "ridge_table": ridge.table(),
        "band": band,
    }


# ----------------------------------------------------------------------
# E6  uncertainty
# ----------------------------------------------------------------------
UNCERTAIN = {
    "cabinet.UA_conv_natural_W_per_K": (0.85, 1.15),
    "battery.R0_ohm": (0.85, 1.20),
    "battery.chiller_kW_at_35C": (0.80, 1.15),
    "battery.derate.T1_C": (-2.0, 2.0),   # additive
    "cabinet.derate.T1_C": (-3.0, 3.0),
}


def experiment_uncertainty(cfg, day, out, n=18, heatwave=6.0):
    base_rng = np.random.default_rng(int(cfg["meta"]["seed"]) + 11)
    samples = []
    for i in range(n):
        c = copy.deepcopy(cfg)
        for name, (lo, hi) in UNCERTAIN.items():
            prior = nested_get(cfg, name)
            if "T1_C" in name:
                nested_set(c, name, prior + base_rng.uniform(lo, hi))
            else:
                nested_set(c, name, prior * base_rng.uniform(lo, hi))
        delta_t = base_rng.uniform(1.5, 4.0)
        c["site"]["canopy_delta_T_K"] = delta_t
        seed = int(base_rng.integers(1, 10_000))
        agg_b, _, _, _ = run_scenario(c, day, heatwave, canopy=False, seeds=3, base_seed=seed)
        agg_c, _, _, _ = run_scenario(c, day, heatwave, canopy=True, seeds=3, base_seed=seed)
        samples.append({
            "sample": i, "delta_T_canopy_K": delta_t,
            "wait_base": agg_b["wait_mean_min"], "wait_canopy": agg_c["wait_mean_min"],
            "wait_reduction": agg_b["wait_mean_min"] - agg_c["wait_mean_min"],
            "lost_base": agg_b["lost_demand_pct"], "lost_canopy": agg_c["lost_demand_pct"],
            "energy_base": agg_b["energy_kwh_day"], "energy_canopy": agg_c["energy_kwh_day"],
        })
    df = pd.DataFrame(samples)
    df.to_csv(out["tables"] / "t09_uncertainty.csv", index=False, float_format="%.3f")

    # one-at-a-time tornado around the baseline
    base, _, _, _ = run_scenario(cfg, day, heatwave, canopy=False, seeds=3)
    base_c, _, _, _ = run_scenario(cfg, day, heatwave, canopy=True, seeds=3)
    base_benefit = base["wait_mean_min"] - base_c["wait_mean_min"]
    labels, lows, highs = [], [], []
    for name, (lo, hi) in UNCERTAIN.items():
        labels.append(name.split(".")[-1])
        low_b, high_b = [], []
        for target, value in (("low", lo), ("high", hi)):
            c = copy.deepcopy(cfg)
            prior = nested_get(cfg, name)
            v = prior + value if "T1_C" in name else prior * value
            nested_set(c, name, v)
            a, _, _, _ = run_scenario(c, day, heatwave, canopy=False, seeds=3)
            b, _, _, _ = run_scenario(c, day, heatwave, canopy=True, seeds=3)
            (low_b if target == "low" else high_b).append(a["wait_mean_min"] - b["wait_mean_min"])
        lows.append(low_b[0])
        highs.append(high_b[0])
    plotting.tornado(labels, lows, highs, base_benefit, out["figures"] / "fig13_uncertainty.png")
    summary = {
        "wait_reduction_mean": float(df["wait_reduction"].mean()),
        "wait_reduction_p05": float(df["wait_reduction"].quantile(0.05)),
        "wait_reduction_p95": float(df["wait_reduction"].quantile(0.95)),
        "energy_gain_mean": float((df["energy_canopy"] - df["energy_base"]).mean()),
        "base_benefit": float(base_benefit),
    }
    return df, summary


# ----------------------------------------------------------------------
# E7  seasonality and economics
# ----------------------------------------------------------------------
def experiment_season(cfg, tmy, out, heatwave=6.0):
    rows = []
    for m in range(1, 13):
        day = hottest_day_in_month(tmy, m)
        delta = heatwave if m in SUMMER_MONTHS else 0.0
        b, _, _, _ = run_scenario(cfg, day, delta, canopy=False, seeds=4)
        c, _, _, _ = run_scenario(cfg, day, delta, canopy=True, seeds=4)
        rows.append({
            "month": m, "heatwave": delta,
            "wait_base": b["wait_mean_min"], "wait_canopy": c["wait_mean_min"],
            "lost_base": b["lost_demand_pct"], "lost_canopy": c["lost_demand_pct"],
            "energy_base": b["energy_kwh_day"], "energy_canopy": c["energy_kwh_day"],
            "peak_base": b["peak_load_kw"], "peak_canopy": c["peak_load_kw"],
            "derate_base_pct": 100 * (1 - b["power_ratio_mean"]),
            "derate_canopy_pct": 100 * (1 - c["power_ratio_mean"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out["tables"] / "t11_seasonality.csv", index=False, float_format="%.3f")
    plotting.monthly(df, out["figures"] / "fig14_seasonality.png")
    return df


def experiment_economics(cfg, season_df, out):
    eco = cfg["interventions"]["economics"]
    n_stalls = cfg["site"]["n_stalls"]
    summer = season_df[season_df["month"].isin(SUMMER_MONTHS)]
    days = 31 + 30 + 31 + 31 + 30  # May-Sep, conservative (no benefit off-season)

    d_served_lost = float((summer["lost_base"] - summer["lost_canopy"]).mean())
    d_wait = float((summer["wait_base"] - summer["wait_canopy"]).mean())
    sessions = cfg["queue"]["sessions_per_day"]

    extra_served = d_served_lost / 100.0 * sessions
    margin = extra_served * eco["lost_session_margin_AED"] * days
    time_value = (d_wait / 60.0) * sessions * days * eco["value_of_driver_time_AED_per_h"]
    annual = (margin + time_value) / n_stalls
    capex = eco["canopy_capex_per_stall_AED"]
    opex = eco["canopy_opex_per_stall_year_AED"]
    net = annual - opex
    payback = capex / net if net > 0 else float("nan")

    years = np.arange(0, eco["horizon_years"] + 1)
    disc = (1 + eco["discount_rate"]) ** years
    cash = np.where(years == 0, -capex, net) / disc
    cumulative = np.cumsum(cash)
    npv = float(cumulative[-1])
    econ = pd.DataFrame({"year": years, "cumulative_AED": cumulative})
    econ.to_csv(out["tables"] / "t10_economics.csv", index=False, float_format="%.1f")
    plotting.economics(econ, out["figures"] / "fig15_economics.png")
    return {
        "annual_benefit_per_stall_AED": float(annual),
        "payback_years": float(payback),
        "npv_AED": npv,
        "extra_sessions_per_day": float(extra_served),
        "wait_reduction_min": d_wait,
        "summer_days": days,
    }, econ


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------
def run_all(cfg) -> dict:
    ensure_dirs(cfg)
    out = {
        "figures": PROJECT_ROOT / cfg["paths"]["figures_dir"],
        "tables": PROJECT_ROOT / cfg["paths"]["tables_dir"],
    }
    base = PROJECT_ROOT / cfg["paths"]["results_dir"]
    base.mkdir(parents=True, exist_ok=True)

    tmy = load_tmy(cfg_path(cfg, "raw_tmy"), cfg["climate"]["utc_offset_h"])
    tmy = add_solar_geometry(tmy, cfg["climate"]["latitude_deg"], cfg["climate"]["longitude_deg"],
                             cfg["climate"]["utc_offset_h"], cfg["climate"]["albedo_ground"])

    summary = {}
    climate, day = experiment_climate(cfg, tmy, out)
    summary["climate"] = climate

    physics = experiment_physics(cfg, day, out)
    summary["physics"] = physics.set_index("case").to_dict(orient="index")

    stacking = experiment_stacking(cfg, day, out)
    summary["stacking"] = (stacking.groupby("headway_min")
                           .agg(mean_power_first=("mean_power_kw", "first"),
                                mean_power_last=("mean_power_kw", "last"),
                                power_drop_kw=("mean_power_kw", lambda s: s.iloc[0] - s.iloc[-1]))
                           .reset_index().to_dict(orient="records"))

    interventions, effects, scen = experiment_interventions(
        cfg, day, out, seeds=cfg.get("_seeds", 8))
    summary["interventions"] = interventions.set_index("scenario").to_dict(orient="index")
    summary["factorial"] = effects

    calib = experiment_calibration(cfg, tmy, out)
    summary["calibration"] = {
        "fit_success": calib["fit_success"],
        "rmse_cabinet_K": calib["rmse_cabinet_K"],
        "rmse_pack_K": calib["rmse_pack_K"],
        "ridge_r2": calib["ridge_r2"],
        "comparison": calib["comparison"].to_dict(orient="records"),
    }

    unc_df, unc = experiment_uncertainty(cfg, day, out, n=cfg.get("_mc", 18))
    summary["uncertainty"] = unc

    season = experiment_season(cfg, tmy, out)
    econ, _ = experiment_economics(cfg, season, out)
    summary["economics"] = econ

    # headline deltas for the abstract
    base_wait = interventions.loc[
        interventions["scenario"] == "Heatwave baseline", "wait_mean_min"].iloc[0]
    best = interventions.loc[
        interventions["scenario"] == "Canopy + coating + time-shift"]
    summary["headline"] = {
        "wait_baseline_min": float(base_wait),
        "wait_best_min": float(best["wait_mean_min"].iloc[0]),
        "wait_reduction_pct": float(100 * (1 - best["wait_mean_min"].iloc[0] / base_wait)),
        "lost_baseline_pct": float(interventions.loc[
            interventions["scenario"] == "Heatwave baseline", "lost_demand_pct"].iloc[0]),
        "lost_best_pct": float(best["lost_demand_pct"].iloc[0]),
        "energy_baseline_kwh": float(interventions.loc[
            interventions["scenario"] == "Heatwave baseline", "energy_kwh_day"].iloc[0]),
        "energy_best_kwh": float(best["energy_kwh_day"].iloc[0]),
        "peak_baseline_kw": float(interventions.loc[
            interventions["scenario"] == "Heatwave baseline", "peak_load_kw"].iloc[0]),
        "peak_best_kw": float(best["peak_load_kw"].iloc[0]),
    }

    (base / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    return summary
