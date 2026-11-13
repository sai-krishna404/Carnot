"""Grey-box thermal core: heat balance, derating law, single-session simulation.

Model structure (two coupled lumped nodes)
------------------------------------------
Cabinet (power electronics), temperature T_c:

    C_c dT_c/dt = (1/eta(P) - 1) P_dc + P_aux + alpha_c A_c G_c
                  - UA_conv(wind, T_c) (T_c - T_amb)
                  - eps_c sigma A_c [(T_c+273.15)^4 - (T_sky+273.15)^4]

Battery pack, temperature T_b:

    C_b dT_b/dt = I^2 R_int(T_b, SoC) + Q_solar
                  - Q_chiller(T_b, T_amb) - (T_b - T_amb)/R_passive

    dSoC/dt = +P_dc / (3600 E_pack)

Delivered power:

    P_dc = P_veh * taper(SoC) * f_c(T_c) * f_b(T_b)

The pack's chiller capacity is itself ambient-limited.  That single coupling is
what makes a 45 degC day qualitatively worse than a 40 degC day: the cooling
system loses authority exactly when it is needed.

Two implementations of the same equations are provided:
  * vectorised numpy functions  -> analysis, plotting, tests
  * scalar `math` functions     -> the hot loop (queue, calibration, Monte Carlo)
They are cross-checked against each other in tests/test_thermal.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SIGMA = 5.670374419e-8  # Stefan-Boltzmann, W m^-2 K^-4
R_GAS = 8.314462618  # J mol^-1 K^-1
T_REF_K = 298.15


# ======================================================================
# Derating laws
# ======================================================================
def derate_piecewise(temp_c, t1_c, t2_c, floor):
    """Exact threshold law: 1 below T1, linear ramp, floor above T2."""
    if t2_c <= t1_c:
        raise ValueError("T2 must exceed T1")
    x = (t2_c - np.asarray(temp_c, dtype=float)) / (t2_c - t1_c)
    return floor + (1.0 - floor) * np.clip(x, 0.0, 1.0)


def _softplus(z, kappa):
    return np.logaddexp(0.0, kappa * z) / kappa


def derate_smooth(temp_c, t1_c, t2_c, floor, kappa=30.0):
    """C-infinity clamp; converges to derate_piecewise as kappa -> infinity.

    S(x) = softplus(x) - softplus(x-1) is a smooth clamp onto [0, 1].
    """
    x = (t2_c - np.asarray(temp_c, dtype=float)) / (t2_c - t1_c)
    smooth = _softplus(x, kappa) - _softplus(x - 1.0, kappa)
    return floor + (1.0 - floor) * smooth


def _derate_s(temp_c, t1_c, t2_c, floor):
    x = (t2_c - temp_c) / (t2_c - t1_c)
    if x <= 0.0:
        return floor
    if x >= 1.0:
        return 1.0
    return floor + (1.0 - floor) * x


def converter_efficiency(load_fraction, cfg):
    cab = cfg["cabinet"]
    lf = np.clip(np.asarray(load_fraction, dtype=float), 0.0, 1.0)
    eta = cab["eta_peak"] - cab["eta_partload_penalty"] * (1.0 - lf) ** 2
    return np.clip(eta, 0.85, 0.995)


def soc_taper(soc, cfg):
    bat = cfg["battery"]
    s = np.asarray(soc, dtype=float)
    start = bat["soc_taper_start"]
    out = np.ones_like(s)
    mask = s > start
    out[mask] = np.clip((1.0 - s[mask]) / (1.0 - start), 0.0, 1.0) ** bat["taper_exponent"]
    return out


def pack_resistance(temp_c, soc, cfg):
    """Arrhenius + SoC dependence of the DC internal resistance (ohm)."""
    bat = cfg["battery"]
    tk = np.asarray(temp_c, dtype=float) + 273.15
    arr = np.exp(bat["Ea_kJ_per_mol"] * 1000.0 / R_GAS * (1.0 / tk - 1.0 / T_REF_K))
    soc_factor = 1.0 + bat["R_soc_slope"] * (0.5 - np.asarray(soc, dtype=float))
    return bat["R0_ohm"] * arr * np.clip(soc_factor, 0.5, 3.0)


def chiller_capacity(temp_amb_c, cfg):
    bat = cfg["battery"]
    excess = np.maximum(np.asarray(temp_amb_c, dtype=float) - bat["chiller_ref_ambient_C"], 0.0)
    return bat["chiller_kW_at_35C"] * np.clip(
        1.0 - bat["chiller_degradation_per_K"] * excess, 0.0, 1.0)


def chiller_power(temp_b_c, temp_amb_c, cfg):
    bat = cfg["battery"]
    driving = (np.asarray(temp_b_c, dtype=float)
               - (bat["chiller_setpoint_C"] + bat["chiller_on_margin_K"]))
    demand = np.clip(driving, 0.0, None) / bat["chiller_R_th_K_per_W"]
    return np.minimum(demand, chiller_capacity(temp_amb_c, cfg))


def cabinet_conductance(temp_c, wind_ms, cfg):
    cab = cfg["cabinet"]
    fan = 1.0 / (1.0 + np.exp(-(np.asarray(temp_c, dtype=float) - cab["T_fan_on_C"])
                              / cab["fan_width_K"]))
    ua_w_per_k = (cab["UA_conv_natural_W_per_K"] * (1.0 + (cab["fan_ua_ratio"] - 1.0) * fan)
                  + cab["k_wind_W_per_K"] * np.asarray(wind_ms, dtype=float) ** cab["wind_exponent"])
    return ua_w_per_k / 1000.0


# ======================================================================
# Vectorised heat balances (analysis / plots)
# ======================================================================
def cabinet_heat(temp_c, power_dc, env_t, env_g, cfg):
    cab = cfg["cabinet"]
    eta = converter_efficiency(np.asarray(power_dc) / cfg["site"]["stall_rating_kw"], cfg)
    eta = np.where(np.asarray(power_dc) > 1e-6, eta, 1.0)
    joule = np.where(np.asarray(power_dc) > 1e-6,
                     np.asarray(power_dc) * (1.0 / eta - 1.0), 0.0)
    aux = cab["aux_kW"] * np.where(np.asarray(power_dc) > 1e-6, 1.0, 0.15)
    solar = 0.001 * cfg["site"]["alpha_cabinet"] * cab["area_m2"] * np.asarray(env_g)
    conv = cabinet_conductance(temp_c, env_t["wind"], cfg) * (np.asarray(temp_c) - env_t["t_amb"])
    rad = (0.001 * cab["emissivity"] * SIGMA * cab["area_m2"]
           * ((np.asarray(temp_c) + 273.15) ** 4 - (np.asarray(env_t["t_sky"]) + 273.15) ** 4))
    return joule + aux + solar - conv - rad


def battery_heat(temp_b, power_dc, soc, env_t, env_g, cfg):
    bat = cfg["battery"]
    p = np.asarray(power_dc, dtype=float)
    voltage = bat["nominal_V"] * (0.94 + 0.06 * np.asarray(soc, dtype=float))
    current = 1000.0 * p / voltage
    ohmic = current ** 2 * pack_resistance(temp_b, soc, cfg) / 1000.0
    solar = bat["solar_gain_kW_per_1000_Wm2"] * np.asarray(env_g) / 1000.0
    cooling = chiller_power(temp_b, env_t["t_amb"], cfg)
    passive = (np.asarray(temp_b) - env_t["t_amb"]) / bat["passive_R_th_K_per_W"]
    return ohmic + solar - cooling - passive


def available_power(temp_c, temp_b, soc, cfg, smooth: bool = False, kappa: float = 30.0):
    cab = cfg["cabinet"]["derate"]
    bat = cfg["battery"]["derate"]
    law = derate_smooth if smooth else derate_piecewise
    if smooth:
        f_c = law(temp_c, cab["T1_C"], cab["T2_C"], cab["floor"], kappa)
        f_b = law(temp_b, bat["T1_C"], bat["T2_C"], bat["floor"], kappa)
    else:
        f_c = law(temp_c, cab["T1_C"], cab["T2_C"], cab["floor"])
        f_b = law(temp_b, bat["T1_C"], bat["T2_C"], bat["floor"])
    p_veh = min(cfg["vehicle"]["peak_kW"], cfg["site"]["stall_rating_kw"])
    return p_veh * np.asarray(f_c) * np.asarray(f_b) * soc_taper(soc, cfg)


# ======================================================================
# Scalar physics (hot loop)
# ======================================================================
def _cab_heat_s(tc, p, tamb, gcab, tsky, wind, cfg):
    cab = cfg["cabinet"]
    if p > 1e-9:
        lf = p / cfg["site"]["stall_rating_kw"]
        if lf > 1.0:
            lf = 1.0
        eta = cab["eta_peak"] - cab["eta_partload_penalty"] * (1.0 - lf) ** 2
        if eta < 0.85:
            eta = 0.85
        elif eta > 0.995:
            eta = 0.995
        joule = p * (1.0 / eta - 1.0)
        aux = cab["aux_kW"]
    else:
        joule = 0.0
        aux = 0.15 * cab["aux_kW"]
    fan = 1.0 / (1.0 + math.exp(-(tc - cab["T_fan_on_C"]) / cab["fan_width_K"]))
    ua = (cab["UA_conv_natural_W_per_K"] * (1.0 + (cab["fan_ua_ratio"] - 1.0) * fan)
          + cab["k_wind_W_per_K"] * wind ** cab["wind_exponent"]) / 1000.0
    tck = tc + 273.15
    tsk = tsky + 273.15
    solar = 0.001 * cfg["site"]["alpha_cabinet"] * cab["area_m2"] * gcab
    rad = 0.001 * cab["emissivity"] * SIGMA * cab["area_m2"] * (tck ** 4 - tsk ** 4)
    return joule + aux + solar - ua * (tc - tamb) - rad


def _bat_resistance_s(tb, soc, cfg):
    bat = cfg["battery"]
    tk = tb + 273.15
    arr = math.exp(bat["Ea_kJ_per_mol"] * 1000.0 / R_GAS * (1.0 / tk - 1.0 / T_REF_K))
    soc_factor = 1.0 + bat["R_soc_slope"] * (0.5 - soc)
    if soc_factor < 0.5:
        soc_factor = 0.5
    elif soc_factor > 3.0:
        soc_factor = 3.0
    return bat["R0_ohm"] * arr * soc_factor


def _chiller_s(tb, tamb, cfg):
    bat = cfg["battery"]
    excess = tamb - bat["chiller_ref_ambient_C"]
    cap = bat["chiller_kW_at_35C"] * (1.0 - bat["chiller_degradation_per_K"] * excess) \
        if excess > 0 else bat["chiller_kW_at_35C"]
    if cap < 0.0:
        cap = 0.0
    demand = (tb - (bat["chiller_setpoint_C"] + bat["chiller_on_margin_K"])) \
        / bat["chiller_R_th_K_per_W"]
    if demand < 0.0:
        demand = 0.0
    return cap if demand > cap else demand


def _bat_heat_s(tb, p, soc, tamb, g, cfg):
    bat = cfg["battery"]
    v = bat["nominal_V"] * (0.94 + 0.06 * soc)
    current = 1000.0 * p / v
    ohmic = current * current * _bat_resistance_s(tb, soc, cfg) / 1000.0
    solar = bat["solar_gain_kW_per_1000_Wm2"] * g / 1000.0
    return ohmic + solar - _chiller_s(tb, tamb, cfg) - (tb - tamb) / bat["passive_R_th_K_per_W"]


def _taper_s(soc, cfg):
    bat = cfg["battery"]
    start = bat["soc_taper_start"]
    if soc <= start:
        return 1.0
    v = (1.0 - soc) / (1.0 - start)
    if v <= 0.0:
        return 0.0
    return v ** bat["taper_exponent"]


def _power_s(tc, tb, soc, cfg):
    cab = cfg["cabinet"]["derate"]
    bat = cfg["battery"]["derate"]
    f_c = _derate_s(tc, cab["T1_C"], cab["T2_C"], cab["floor"])
    f_b = _derate_s(tb, bat["T1_C"], bat["T2_C"], bat["floor"])
    p_veh = min(cfg["vehicle"]["peak_kW"], cfg["site"]["stall_rating_kw"])
    return p_veh * f_c * f_b * _taper_s(soc, cfg)


# ======================================================================
# Weather series for the hot loop
# ======================================================================
class WeatherSeries:
    """Cyclic linear interpolation of hourly weather at low cost."""

    __slots__ = ("t_amb", "ghi", "ghi_cabinet", "t_sky", "wind")

    def __init__(self, env):
        self.t_amb = list(env.t_amb) + [env.t_amb[0]]
        self.ghi = list(env.ghi) + [env.ghi[0]]
        self.ghi_cabinet = list(env.ghi_cabinet) + [env.ghi_cabinet[0]]
        self.t_sky = list(env.t_sky) + [env.t_sky[0]]
        self.wind = list(env.wind) + [env.wind[0]]

    @staticmethod
    def _interp(arr, hours):
        i = int(hours)
        if i >= 24:
            i = 23
        frac = hours - i
        return arr[i] * (1.0 - frac) + arr[i + 1] * frac

    def at(self, t_s):
        hours = (t_s / 3600.0) % 24.0
        return (self._interp(self.t_amb, hours), self._interp(self.ghi, hours),
                self._interp(self.ghi_cabinet, hours), self._interp(self.t_sky, hours),
                self._interp(self.wind, hours))


# ======================================================================
# Session simulation
# ======================================================================
@dataclass
class SessionResult:
    t_start_s: float
    t_end_s: float
    duration_s: float
    energy_kwh: float
    soc_start: float
    soc_end: float
    t_bat_start: float
    t_bat_end: float
    t_cab_start: float
    t_cab_end: float
    mean_power_kw: float
    nominal_duration_s: float
    derating_loss_kwh: float
    trace: dict = field(default_factory=dict)

    @property
    def slowdown(self) -> float:
        return self.duration_s / self.nominal_duration_s if self.nominal_duration_s else float("nan")


def reference_duration_s(cfg, soc_start: float) -> float:
    """Session duration with no thermal limit (nameplate reference).

    Integrates dSoC/dt = P_veh * taper(SoC) / E over the SoC window: the time
    an identical session would take with an unthrottled charger.  This is the
    denominator of every slowdown ratio reported.
    """
    target = cfg["session"]["target_soc"]
    if soc_start >= target:
        return 0.0
    p = min(cfg["vehicle"]["peak_kW"], cfg["site"]["stall_rating_kw"])
    socs = np.linspace(soc_start, target, 241)
    power = np.maximum(p * soc_taper(socs, cfg), 1e-3)
    integrand = 3600.0 * cfg["battery"]["capacity_kWh"] / power
    return float(np.trapezoid(integrand, socs))


def advance_session(cfg, ws: WeatherSeries, t_s: float, tc: float, tb: float, soc: float,
                    dt_s: float):
    """One classical RK4 step of the coupled system.

    Power is re-evaluated at every stage, so the derating feedback is integrated
    implicitly rather than frozen over the step.  Returns (tc, tb, soc, p_avg).
    """
    cap = cfg["battery"]["capacity_kWh"]
    cc = cfg["cabinet"]["C_kJ_per_K"]
    cb = cfg["battery"]["C_kJ_per_K"]
    h = 0.5 * dt_s

    p0 = _power_s(tc, tb, soc, cfg)
    tamb, g, gc, tsky, wind = ws.at(t_s)
    k1c = _cab_heat_s(tc, p0, tamb, gc, tsky, wind, cfg) / cc
    k1b = _bat_heat_s(tb, p0, soc, tamb, g, cfg) / cb
    k1s = p0 / (3600.0 * cap)

    tc2, tb2, soc2 = tc + h * k1c, tb + h * k1b, soc + h * k1s
    p1 = _power_s(tc2, tb2, soc2, cfg)
    tamb, g, gc, tsky, wind = ws.at(t_s + h)
    k2c = _cab_heat_s(tc2, p1, tamb, gc, tsky, wind, cfg) / cc
    k2b = _bat_heat_s(tb2, p1, soc2, tamb, g, cfg) / cb
    k2s = p1 / (3600.0 * cap)

    tc3, tb3, soc3 = tc + h * k2c, tb + h * k2b, soc + h * k2s
    p2 = _power_s(tc3, tb3, soc3, cfg)
    k3c = _cab_heat_s(tc3, p2, tamb, gc, tsky, wind, cfg) / cc
    k3b = _bat_heat_s(tb3, p2, soc3, tamb, g, cfg) / cb
    k3s = p2 / (3600.0 * cap)

    tc4, tb4, soc4 = tc + dt_s * k3c, tb + dt_s * k3b, soc + dt_s * k3s
    p3 = _power_s(tc4, tb4, soc4, cfg)
    tamb, g, gc, tsky, wind = ws.at(t_s + dt_s)
    k4c = _cab_heat_s(tc4, p3, tamb, gc, tsky, wind, cfg) / cc
    k4b = _bat_heat_s(tb4, p3, soc4, tamb, g, cfg) / cb
    k4s = p3 / (3600.0 * cap)

    p_avg = (p0 + 2.0 * p1 + 2.0 * p2 + p3) / 6.0
    return (tc + dt_s / 6.0 * (k1c + 2 * k2c + 2 * k3c + k4c),
            tb + dt_s / 6.0 * (k1b + 2 * k2b + 2 * k3b + k4b),
            soc + dt_s / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s),
            p_avg)


def simulate_session(cfg, env, t_start_s: float, soc_start: float, t_bat0: float,
                     t_cab0: float, dt_s: float = 15.0, record: bool = False,
                     weather: WeatherSeries | None = None) -> SessionResult:
    """Charge one vehicle from ``soc_start`` to the configured target SoC."""
    ws = weather if weather is not None else WeatherSeries(env)
    target = cfg["session"]["target_soc"]
    cap = cfg["battery"]["capacity_kWh"]
    max_dur = cfg["session"]["max_duration_s"]
    tc, tb, soc = float(t_cab0), float(t_bat0), float(soc_start)
    energy = 0.0
    elapsed = 0.0
    trace = {"t": [], "power": [], "t_cab": [], "t_bat": [], "soc": []}
    last_rec = -1e9

    while soc < target - 1e-9 and elapsed < max_dur:
        tc, tb, soc, p_avg = advance_session(cfg, ws, t_start_s + elapsed, tc, tb, soc, dt_s)
        energy += dt_s * p_avg / 3600.0
        elapsed += dt_s
        soc = min(soc, target)
        if record and (elapsed - last_rec) >= 30.0:
            trace["t"].append(elapsed)
            trace["power"].append(p_avg)
            trace["t_cab"].append(tc)
            trace["t_bat"].append(tb)
            trace["soc"].append(soc)
            last_rec = elapsed

    soc = min(soc, target)
    energy = min(energy, (target - soc_start) * cap)
    nominal = reference_duration_s(cfg, soc_start)
    p_nom = min(cfg["vehicle"]["peak_kW"], cfg["site"]["stall_rating_kw"])
    extra_h = max(elapsed - nominal, 0.0) / 3600.0
    return SessionResult(
        t_start_s=float(t_start_s), t_end_s=float(t_start_s + elapsed),
        duration_s=float(elapsed), energy_kwh=float(energy),
        soc_start=float(soc_start), soc_end=float(soc),
        t_bat_start=float(t_bat0), t_bat_end=float(tb),
        t_cab_start=float(t_cab0), t_cab_end=float(tc),
        mean_power_kw=float(energy / (elapsed / 3600.0)) if elapsed > 0 else 0.0,
        nominal_duration_s=float(nominal),
        derating_loss_kwh=float(extra_h * p_nom),
        trace={k: np.array(v) for k, v in trace.items()},
    )


def idle_step(t_cab: float, t_amb: float, g_cab: float, t_sky: float, wind: float,
              cfg, dt_s: float) -> float:
    """Cool an idle cabinet for one step (explicit Euler, stable at dt <= 60 s)."""
    k = _cab_heat_s(t_cab, 0.0, t_amb, g_cab, t_sky, wind, cfg) / cfg["cabinet"]["C_kJ_per_K"]
    return t_cab + dt_s * k


# ======================================================================
# Steady-state map ("derating cliff")
# ======================================================================
def equilibrium_power(cfg, t_amb: float, ghi: float, ghi_cab: float, wind: float,
                      t_sky: float | None = None, soc: float = 0.30):
    """Steady state of the coupled system at fixed SoC.

    Returns (T_cab, T_bat, P_dc); temperatures are NaN when no equilibrium
    exists below the component limits (the 'thermal cliff').  Several starts
    are used and the coolest physical root is retained.
    """
    from scipy.optimize import root

    env_t = {"t_amb": t_amb, "ghi": ghi, "ghi_cabinet": ghi_cab, "wind": wind,
             "t_sky": t_sky if t_sky is not None else t_amb - 15.0}

    def residual(x):
        tc, tb = x
        p = float(available_power(tc, tb, soc, cfg))
        return [float(cabinet_heat(tc, p, env_t, ghi_cab, cfg)),
                float(battery_heat(tb, p, soc, env_t, ghi, cfg))]

    best = None
    for d_c, d_b in ((15, 5), (25, 10), (40, 20), (5, 2), (55, 30)):
        sol = root(residual, [t_amb + d_c, t_amb + d_b], method="hybr", tol=1e-6)
        if not sol.success:
            continue
        tc, tb = sol.x
        if tc < -10 or tb < -10 or tc > 200 or tb > 200:
            continue
        if np.max(np.abs(residual(sol.x))) > 1e-3:
            continue
        if best is None or tc < best[0]:
            best = (float(tc), float(tb))
    if best is None:
        return float("nan"), float("nan"), float(available_power(t_amb, t_amb, soc, cfg))
    return best[0], best[1], float(available_power(best[0], best[1], soc, cfg))
