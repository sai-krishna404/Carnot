"""Grey-box calibration layer and the experiments that justify it.

Stage 1  physics parameters  -> nonlinear least squares with physical priors
Stage 2  structure error     -> ridge regression on physically meaningful
                               features, so every learned coefficient has an
                               interpretation a reviewer can check

A degree-3 polynomial black box is trained on the same data and evaluated on
an out-of-distribution heatwave.  The point is not that ML is bad; it is that
a model with the right invariants extrapolates and a flexible one does not.
"""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import thermal
from .climate import Environment
from .config import nested_get, nested_set

PARAM_NAMES = [
    "cabinet.UA_conv_natural_W_per_K",
    "cabinet.fan_ua_ratio",
    "cabinet.C_kJ_per_K",
    "cabinet.k_wind_W_per_K",
    "battery.R0_ohm",
    "battery.chiller_kW_at_35C",
    "battery.chiller_R_th_K_per_W",
]

PARAM_LABELS = {
    "cabinet.UA_conv_natural_W_per_K": "UA_conv natural (W/K)",
    "cabinet.fan_ua_ratio": "fan UA ratio (-)",
    "cabinet.C_kJ_per_K": "C_cabinet (kJ/K)",
    "cabinet.k_wind_W_per_K": "wind UA coeff (W/K)",
    "battery.R0_ohm": "R_int reference (ohm)",
    "battery.chiller_kW_at_35C": "chiller capacity (kW)",
    "battery.chiller_R_th_K_per_W": "pack-coolant R_th (K/W)",
}

# Plausible field values used to create the synthetic twin ("plant truth").
# These are deliberately different from the model defaults: the calibration
# has to find them, not start from them.
TRUTH_MULTIPLIERS = {
    "cabinet.UA_conv_natural_W_per_K": 0.86,
    "cabinet.fan_ua_ratio": 1.18,
    "cabinet.C_kJ_per_K": 0.91,
    "cabinet.k_wind_W_per_K": 1.32,
    "battery.R0_ohm": 1.20,
    "battery.chiller_kW_at_35C": 0.88,
    "battery.chiller_R_th_K_per_W": 0.93,
}

SAMPLE_DT = 60.0
OBS_WINDOW_S = 2400.0


@dataclass
class SessionSpec:
    day: int
    t_start: float
    soc0: float
    t_bat0: float
    t_cab0: float


@dataclass
class TwinDataset:
    specs: list[SessionSpec]
    envs: list[Environment]
    weather: list
    obs: list[dict] = field(default_factory=list)

    def frame(self) -> pd.DataFrame:
        rows = []
        for i, (spec, o) in enumerate(zip(self.specs, self.obs)):
            for k, t in enumerate(o["t"]):
                rows.append({
                    "spec": i, "day": spec.day, "t_s": t,
                    "t_amb": o["t_amb"][k], "g_cab": o["g_cab"][k],
                    "power": o["power"][k], "wind": o["wind"][k],
                    "t_cab": o["t_cab"][k], "t_bat": o["t_bat"][k],
                    "soc": o["soc"][k],
                })
        return pd.DataFrame(rows)


def _truth_config(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    for name, mult in TRUTH_MULTIPLIERS.items():
        nested_set(out, name, float(nested_get(cfg, name) * mult))
    return out


def _apply(cfg: dict, names: list[str], theta: np.ndarray) -> dict:
    out = copy.deepcopy(cfg)
    for name, value in zip(names, theta):
        nested_set(out, name, float(value))
    return out


def simulate_observation(cfg, spec, ws, grid):
    """Simulate one session and sample it on the observation grid."""
    res = thermal.simulate_session(cfg, None, spec.t_start, spec.soc0, spec.t_bat0,
                                   spec.t_cab0, dt_s=30.0, record=True, weather=ws)
    t = np.concatenate([[0.0], res.trace["t"]])
    out = {
        "t": grid,
        "t_cab": np.interp(grid, t, np.concatenate([[spec.t_cab0], res.trace["t_cab"]])),
        "t_bat": np.interp(grid, t, np.concatenate([[spec.t_bat0], res.trace["t_bat"]])),
        "soc": np.interp(grid, t, np.concatenate([[spec.soc0], res.trace["soc"]])),
        "power": np.interp(grid, t, np.concatenate([
            [thermal._power_s(spec.t_cab0, spec.t_bat0, spec.soc0, cfg)], res.trace["power"]])),
        "valid": grid <= res.duration_s,
    }
    return out


def generate_twin(cfg: dict, days, deltas, rng, n_per_day: int = 2) -> TwinDataset:
    """Create the synthetic 'field' dataset from the truth parameter set.

    Weather is real TMY; physics is the same structural model with perturbed
    parameters; observations carry sensor noise plus a smooth structure-error
    term that the physics does not contain.
    """
    truth = _truth_config(cfg)
    specs, envs, weather, obs = [], [], [], []
    grid = np.arange(0.0, OBS_WINDOW_S + 1e-9, SAMPLE_DT)

    for d, (day, delta) in enumerate(zip(days, deltas)):
        env = Environment.from_day(day, heatwave_delta_k=delta)
        ws = thermal.WeatherSeries(env)
        envs.append(env)
        hours = [12.5, 18.5] if n_per_day == 2 else list(np.linspace(9.0, 20.0, n_per_day))
        for h in hours:
            t_start = (h * 3600.0 + rng.uniform(-1200, 1200)) % 86400.0
            at = env.at(t_start)
            soc0 = float(np.clip(rng.normal(0.25, 0.06), 0.10, 0.42))
            spec = SessionSpec(day=d, t_start=t_start, soc0=soc0,
                               t_bat0=float(at["t_amb"] + cfg["battery"]["T_arrival_offset_K"]),
                               t_cab0=float(at["t_amb"] + 4.0))
            i = len(specs)
            specs.append(spec)
            weather.append(ws)
            sim = simulate_observation(truth, spec, ws, grid)
            e_tcab = rng.normal(0.0, cfg["calibration"]["noise_C"], grid.size)
            e_tcab += cfg["calibration"]["discrepancy_C"] * np.sin(2 * np.pi * grid / 3600.0 / 7.0)
            e_tbat = rng.normal(0.0, cfg["calibration"]["noise_C"] * 0.7, grid.size)
            local = [env.at(t_start + g) for g in grid]
            o = {
                "t": grid,
                "t_amb": np.array([x["t_amb"] for x in local]),
                "g_cab": np.array([x["ghi_cabinet"] for x in local]),
                "wind": np.array([x["wind"] for x in local]),
                "t_cab": sim["t_cab"] + e_tcab,
                "t_bat": sim["t_bat"] + e_tbat,
                "power": sim["power"] + rng.normal(0.0, cfg["calibration"]["noise_kW"], grid.size),
                "soc": sim["soc"],
                "valid": sim["valid"],
                "spec": i,
            }
            obs.append(o)
    return TwinDataset(specs=specs, envs=envs, weather=weather, obs=obs)


# ----------------------------------------------------------------------
# Stage 1: nonlinear least squares
# ----------------------------------------------------------------------
@dataclass
class FitResult:
    names: list[str]
    theta: np.ndarray
    theta0: np.ndarray
    std_err: np.ndarray
    rmse_cab: float
    rmse_bat: float
    n_obs: int
    success: bool
    message: str


def _residual_vector(cfg, data: TwinDataset, theta, names, lam_prior: float = 2.0):
    trial = _apply(cfg, names, theta)
    grid = data.obs[0]["t"]
    cab_res, bat_res = [], []
    for spec, ws, o in zip(data.specs, data.weather, data.obs):
        sim = simulate_observation(trial, spec, ws, grid)
        valid = o["valid"]  # fixed by the dataset, so the residual size is constant
        cab_res.append((o["t_cab"] - sim["t_cab"])[valid])
        bat_res.append((o["t_bat"] - sim["t_bat"])[valid])
    theta0 = np.array([nested_get(cfg, n) for n in names])
    scale = np.maximum(np.abs(theta0), 1e-6)
    prior = np.sqrt(lam_prior) * (theta - theta0) / scale
    return np.concatenate(cab_res + bat_res + [prior]), [len(c) for c in cab_res], \
        [len(b) for b in bat_res]


def fit_physics(cfg: dict, data: TwinDataset, names: list[str] | None = None,
                max_nfev: int = 400) -> FitResult:
    names = names or PARAM_NAMES
    theta0 = np.array([nested_get(cfg, n) for n in names])
    lo = np.array([0.45 * t for t in theta0])
    hi = np.array([2.2 * t for t in theta0])

    def fun(theta):
        return _residual_vector(cfg, data, theta, names)[0]

    sol = least_squares(fun, theta0, bounds=(lo, hi), method="trf",
                        x_scale="jac", max_nfev=max_nfev, ftol=1e-8, xtol=1e-8)

    # linearised covariance from the Jacobian at the solution
    jac = sol.jac
    dof = max(jac.shape[0] - jac.shape[1], 1)
    sigma2 = 2.0 * sol.cost / dof
    try:
        cov = np.linalg.inv(jac.T @ jac) * sigma2
        std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        std = np.full(len(names), np.nan)

    resid, nc, nb = _residual_vector(cfg, data, sol.x, names)
    n_cab = int(sum(nc))
    n_bat = int(sum(nb))
    rmse_cab = float(np.sqrt(np.mean(resid[:n_cab] ** 2)))
    rmse_bat = float(np.sqrt(np.mean(resid[n_cab:n_cab + n_bat] ** 2)))
    return FitResult(names=names, theta=sol.x, theta0=theta0, std_err=std,
                     rmse_cab=rmse_cab, rmse_bat=rmse_bat,
                     n_obs=int(n_cab + n_bat),
                     success=bool(sol.success), message=str(sol.message))


# ----------------------------------------------------------------------
# Stage 2: ridge residual learner
# ----------------------------------------------------------------------
class RidgeResidual:
    """Ridge regression on physics residuals with interpretable features."""

    FEATURES = ("intercept", "T_amb", "G_cab", "P", "wind", "hot hinge")

    def __init__(self, lam: float = 1.0):
        self.lam = lam
        self.coef_: np.ndarray | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.r2_ = np.nan

    def _raw(self, frame: pd.DataFrame) -> np.ndarray:
        t = frame["t_amb"].to_numpy(float)
        return np.column_stack([
            np.ones(len(frame)),
            (t - 40.0) / 5.0,
            frame["g_cab"].to_numpy(float) / 600.0,
            frame["power"].to_numpy(float) / 150.0,
            (frame["wind"].to_numpy(float) - 3.0) / 3.0,
            np.clip((t - 44.0) / 3.0, 0.0, None),
        ])

    def fit(self, frame: pd.DataFrame, y: np.ndarray, sample_weight=None) -> "RidgeResidual":
        X = self._raw(frame)
        self.feature_mean_ = X[:, 1:].mean(axis=0)
        self.feature_scale_ = X[:, 1:].std(axis=0) + 1e-9
        Xn = self._raw(frame)
        Xn[:, 1:] = (Xn[:, 1:] - self.feature_mean_) / self.feature_scale_
        lam = self.lam
        A = Xn.T @ Xn + lam * np.eye(Xn.shape[1])
        A[0, 0] -= lam  # do not penalise the intercept
        b = Xn.T @ y
        self.coef_ = np.linalg.solve(A, b)
        pred = Xn @ self.coef_
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        self.r2_ = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        X = self._raw(frame)
        X[:, 1:] = (X[:, 1:] - self.feature_mean_) / self.feature_scale_
        return X @ self.coef_

    def table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "feature": list(self.FEATURES),
            "coefficient_K": self.coef_,
        })


class PolynomialBlackBox:
    """Degree-3 polynomial ridge: a stand-in for a generic ML surrogate."""

    def __init__(self, degree: int = 3, lam: float = 1.0):
        self.degree = degree
        self.lam = lam
        self.terms_: list[tuple[int, ...]] = []
        self.coef_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _base(self, frame: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            frame["t_amb"].to_numpy(float),
            frame["g_cab"].to_numpy(float) / 600.0,
            frame["power"].to_numpy(float) / 150.0,
            frame["wind"].to_numpy(float),
        ])

    def _expand(self, Xb: np.ndarray) -> np.ndarray:
        cols = [np.ones(len(Xb))]
        for deg in range(1, self.degree + 1):
            for combo in itertools.combinations_with_replacement(range(Xb.shape[1]), deg):
                cols.append(np.prod(Xb[:, combo], axis=1))
        return np.column_stack(cols)

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "PolynomialBlackBox":
        Xb = self._base(frame)
        self.mean_ = Xb.mean(axis=0)
        self.scale_ = Xb.std(axis=0) + 1e-9
        X = self._expand((Xb - self.mean_) / self.scale_)
        A = X.T @ X + self.lam * np.eye(X.shape[1])
        A[0, 0] -= self.lam
        self.coef_ = np.linalg.solve(A, X.T @ y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        Xb = (self._base(frame) - self.mean_) / self.scale_
        return self._expand(Xb) @ self.coef_


# ----------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------
def evaluate(data: TwinDataset, sim_fn, label: str) -> dict:
    """RMSE / MAE / bias of a predictor of the cabinet temperature."""
    errs = []
    for spec, ws, o in zip(data.specs, data.weather, data.obs):
        pred = sim_fn(spec, ws, o)
        valid = o["valid"] & np.isfinite(pred)
        errs.append(o["t_cab"][valid] - pred[valid])
    e = np.concatenate(errs) if errs else np.array([np.nan])
    return {
        "model": label,
        "rmse_K": float(np.sqrt(np.mean(e ** 2))),
        "mae_K": float(np.mean(np.abs(e))),
        "bias_K": float(np.mean(e)),
        "max_abs_K": float(np.max(np.abs(e))),
        "n": int(e.size),
    }
