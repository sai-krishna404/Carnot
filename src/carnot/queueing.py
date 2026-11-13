"""Queueing layer: thermally-coupled station simulation plus two analytic
approximations used for cross-validation.

The novel ingredient is that service time is not an exogenous random variable:
it is produced by the thermal state of the stall, which in turn depends on the
recent history of that stall.  Back-to-back sessions at peak hour therefore
compound derating ("thermal stacking"), a feedback that a standard M/M/c model
cannot represent.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import thermal


# ----------------------------------------------------------------------
# Arrival process
# ----------------------------------------------------------------------
@dataclass
class Arrival:
    time_s: float
    soc0: float
    patience_s: float
    flexible: bool = False


def _night_time(rng, cfg, day_s: float = 86400.0) -> float:
    ts = cfg["interventions"]["time_shift"]
    centre = ts["night_peak_hour"] * 3600.0
    sd = ts["night_sd_h"] * 3600.0
    lo = ts["night_window"][0] * 3600.0
    hi = ts["night_window"][1] * 3600.0
    if hi >= lo:
        span = hi - lo
        return (lo + (rng.normal(centre, sd) - lo) % span) % day_s
    span = (day_s - lo) + hi
    return (lo + (rng.normal(centre, sd) - lo) % span) % day_s


def build_arrivals(cfg, rng, shift: bool = False) -> list[Arrival]:
    """Non-homogeneous Poisson arrivals with a two-peak urban profile."""
    q = cfg["queue"]
    sess = cfg["session"]
    weights = np.asarray(q["profile_weights"], dtype=float)
    weights = weights / weights.sum()
    lam_hour = weights * q["sessions_per_day"]

    arrivals: list[Arrival] = []
    for hour in range(24):
        n = rng.poisson(lam_hour[hour])
        for _ in range(n):
            t = (hour * 3600.0 + rng.uniform(0, 3600.0)) % 86400.0
            flexible = False
            if shift:
                ts = cfg["interventions"]["time_shift"]
                lo, hi = ts["price_window"]
                if lo * 3600.0 <= t < hi * 3600.0:
                    flexible = rng.random() < ts["flexible_fraction"]
                    if flexible:
                        t = _night_time(rng, cfg)
            soc = float(np.clip(rng.normal(sess["soc_init_mean"], sess["soc_init_sd"]),
                                sess["soc_init_min"], sess["soc_init_max"]))
            patience = float(rng.exponential(q["patience_mean_min"] * 60.0))
            arrivals.append(Arrival(t, soc, patience, flexible))
    arrivals.sort(key=lambda a: a.time_s)
    return arrivals


# ----------------------------------------------------------------------
# Station simulation
# ----------------------------------------------------------------------
@dataclass
class Stall:
    t_cab: float
    session: dict | None = None
    soc: float = 0.0
    t_bat: float = 0.0
    energy_kwh: float = 0.0


@dataclass
class StationResult:
    records: list[dict]
    load_kw: np.ndarray
    queue_len: np.ndarray
    time_s: np.ndarray
    dt_s: float
    n_arrivals: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def served(self) -> list[dict]:
        return [r for r in self.records if r["served"]]


def simulate_station(cfg, env, arrivals: list[Arrival], dt_s: float | None = None,
                     record_series: bool = True) -> StationResult:
    site = cfg["site"]
    n_stalls = int(cfg["queue"].get("n_stalls", site["n_stalls"]))
    dt = float(dt_s if dt_s is not None else cfg["queue"]["dt_s"])
    n_steps = int(round(86400.0 / dt))
    target = cfg["session"]["target_soc"]
    offset = cfg["battery"]["T_arrival_offset_K"]

    t0 = env.at(0.0)
    stalls = [Stall(t_cab=float(t0["t_amb"]) + 3.0) for _ in range(n_stalls)]
    ws = thermal.WeatherSeries(env)
    waiting: deque[dict] = deque()
    records: list[dict] = []
    load = np.zeros(n_steps)
    qlen = np.zeros(n_steps)
    ai = 0
    svc_mean = 2400.0  # running mean service time (s), initialised at 40 min
    arrivals = sorted(arrivals, key=lambda a: a.time_s)

    for k in range(n_steps):
        t = k * dt
        env_t = env.at(t)

        # 1) admissions
        while ai < len(arrivals) and arrivals[ai].time_s < t + dt:
            a = arrivals[ai]
            ai += 1
            r = {
                "arrival_s": a.time_s,
                "soc0": a.soc0,
                "patience_s": a.patience_s,
                "flexible": a.flexible,
                "t_bat_start": float(env.at(a.time_s)["t_amb"]) + offset,
                "balked": False, "reneged": False, "served": False,
                "t_start_s": np.nan, "t_end_s": np.nan, "energy_kwh": 0.0,
                "stall": -1, "t_bat_end": np.nan, "t_cab_start": np.nan,
                "mean_power_kw": 0.0,
            }
            est_wait = (len(waiting) / n_stalls) * svc_mean
            if est_wait > a.patience_s:
                r["balked"] = True
                records.append(r)
                continue
            waiting.append(r)

        # 2) reneging
        if waiting:
            keep = deque()
            for r in waiting:
                if (t - r["arrival_s"]) > r["patience_s"]:
                    r["reneged"] = True
                    records.append(r)
                else:
                    keep.append(r)
            waiting = keep

        # 3) dispatch
        for idx, stall in enumerate(stalls):
            if stall.session is None and waiting:
                r = waiting.popleft()
                r["t_start_s"] = max(t, r["arrival_s"])
                r["stall"] = idx
                r["t_cab_start"] = stall.t_cab
                stall.session = r
                stall.soc = r["soc0"]
                stall.t_bat = r["t_bat_start"]
                stall.energy_kwh = 0.0

        # 4) thermal integration for every stall
        step_load = 0.0
        for stall in stalls:
            if stall.session is None:
                tamb, g, gc, tsky, wind = ws.at(t)
                stall.t_cab = thermal.idle_step(stall.t_cab, tamb, gc, tsky, wind, cfg, dt)
                step_load += cfg["cabinet"]["aux_kW"] * 0.15 / 0.96
            else:
                tc, tb, soc, p_avg = thermal.advance_session(
                    cfg, ws, t, stall.t_cab, stall.t_bat, stall.soc, dt)
                stall.t_cab, stall.t_bat, stall.soc = tc, tb, soc
                stall.energy_kwh += p_avg * dt / 3600.0
                eta = float(thermal.converter_efficiency(max(p_avg, 1e-3) / site["stall_rating_kw"], cfg))
                step_load += p_avg / eta + cfg["cabinet"]["aux_kW"]

                if stall.soc >= target - 1e-9:
                    r = stall.session
                    r["served"] = True
                    r["t_end_s"] = t + dt
                    r["energy_kwh"] = float(stall.energy_kwh)
                    r["t_bat_end"] = float(stall.t_bat)
                    dur_h = (r["t_end_s"] - r["t_start_s"]) / 3600.0
                    r["mean_power_kw"] = float(stall.energy_kwh / dur_h) if dur_h > 0 else 0.0
                    records.append(r)
                    svc_mean = 0.9 * svc_mean + 0.1 * (r["t_end_s"] - r["t_start_s"])
                    stall.session = None
                    stall.soc = 0.0

        load[k] = step_load
        qlen[k] = len(waiting)

    # customers still in the system at midnight are censored, not dropped
    for r in waiting:
        r["censored"] = True
        records.append(r)

    res = StationResult(records=records, load_kw=load, queue_len=qlen,
                        time_s=np.arange(n_steps) * dt, dt_s=dt,
                        n_arrivals=len(arrivals))
    res.meta["n_stalls"] = n_stalls
    return res


# ----------------------------------------------------------------------
# Analytic cross-checks
# ----------------------------------------------------------------------
def erlang_c(c: int, a: float) -> float:
    """Probability of waiting in an M/M/c queue with offered load a (erlang)."""
    if c <= 0:
        return 1.0
    if a >= c:
        return 1.0
    s = sum(a ** n / math.factorial(n) for n in range(c))
    top = a ** c / math.factorial(c) * c / (c - a)
    return top / (s + top)


def pointwise_stationary_wait(lam, mu, c: int) -> np.ndarray:
    """Erlang-C wait (hours) using instantaneous rates (time-varying M/M/c).

    This is the standard pointwise-stationary approximation; it is known to
    under-predict waits when the arrival rate changes faster than the queue
    relaxes, which is exactly the regime we test against the DES.
    """
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    a = lam / np.maximum(mu, 1e-9)
    out = np.zeros_like(a)
    for i, (ai, li, mi) in enumerate(zip(a, lam, mu)):
        if li <= 0:
            continue
        if ai >= c:
            out[i] = np.inf
        else:
            out[i] = erlang_c(c, float(ai)) / (c * mi - li)
    return out


def fluid_queue(lam, mu, c: int, dt_s: float):
    """Deterministic fluid queue q' = lam - c*mu on the positive orthant."""
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    q = np.zeros_like(lam)
    for i in range(1, len(lam)):
        dq = lam[i - 1] - c * mu[i - 1]
        q[i] = max(0.0, q[i - 1] + dq * dt_s / 3600.0)
    wait = np.divide(q, np.maximum(lam, 1e-9))
    return q, wait
