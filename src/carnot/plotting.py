"""Figure helpers (matplotlib, no seaborn dependency)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.autolayout": True,
})

C = {
    "base": "#B03A2E",
    "canopy": "#1F618D",
    "shift": "#B7950B",
    "both": "#117A65",
    "grey": "#7F8C8D",
    "dark": "#1C2833",
    "light": "#D5DBDB",
}


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def climate(day, path, canopy_day=None):
    fig, ax1 = plt.subplots(figsize=(7.0, 3.2))
    h = np.arange(len(day))
    ax1.plot(h, day["t_amb_c"], color=C["base"], lw=2, label="Air temperature")
    if canopy_day is not None:
        ax1.plot(h, canopy_day, color=C["canopy"], lw=1.6, ls="--", label="Under canopy (-2.5 K)")
    ax1.set_xlabel("Hour of day (GST)")
    ax1.set_ylabel("Air temperature (degC)", color=C["base"])
    ax1.tick_params(axis="y", labelcolor=C["base"])
    ax1.set_xlim(0, 23)
    ax1.set_xticks(range(0, 24, 3))

    ax2 = ax1.twinx()
    ax2.fill_between(h, day["ghi_horizontal"], color=C["shift"], alpha=0.18)
    ax2.plot(h, day["ghi_horizontal"], color=C["shift"], lw=1.2)
    ax2.set_ylabel("GHI (W/m$^2$)", color=C["shift"])
    ax2.tick_params(axis="y", labelcolor=C["shift"])
    ax2.grid(False)
    ax1.set_title("Design day: hottest July day in the PVGIS TMY for Abu Dhabi")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=7)
    return save(fig, path)


def traces(traces, path):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharex=True)
    for label, tr, color in traces:
        t = tr["t"] / 60.0
        axes[0, 0].plot(t, tr["power"], color=color, lw=1.8, label=label)
        axes[0, 1].plot(t, tr["t_cab"], color=color, lw=1.8)
        axes[1, 0].plot(t, tr["t_bat"], color=color, lw=1.8)
        axes[1, 1].plot(t, tr["soc"] * 100, color=color, lw=1.8)
    axes[0, 0].set_ylabel("Delivered power (kW)")
    axes[0, 1].set_ylabel("Cabinet temperature (degC)")
    axes[1, 0].set_ylabel("Pack temperature (degC)")
    axes[1, 1].set_ylabel("State of charge (%)")
    for ax in axes[1]:
        ax.set_xlabel("Minutes from plug-in")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_title("(a) Power", loc="left")
    axes[0, 1].set_title("(b) Power electronics", loc="left")
    axes[1, 0].set_title("(c) Battery pack", loc="left")
    axes[1, 1].set_title("(d) State of charge", loc="left")
    return save(fig, path)


def derating_curves(cfg, piece, smooth, path):
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    info = [
        ("Cabinet (power electronics)", cfg["cabinet"]["derate"], C["base"], axes[0]),
        ("Battery pack (BMS)", cfg["battery"]["derate"], C["canopy"], axes[1]),
    ]
    for title, d, color, ax in info:
        t = np.linspace(d["T1_C"] - 12, d["T2_C"] + 8, 400)
        ax.plot(t, piece(t, d["T1_C"], d["T2_C"], d["floor"]), color=color, lw=2)
        ax.plot(t, smooth(t, d["T1_C"], d["T2_C"], d["floor"], d["kappa"]),
                color=C["dark"], lw=1.0, ls="--", label=f"smooth clamp (kappa={d['kappa']:.0f})")
        ax.axvline(d["T1_C"], color=C["grey"], lw=0.8, ls=":")
        ax.axvline(d["T2_C"], color=C["grey"], lw=0.8, ls=":")
        ax.set_title(title)
        ax.set_xlabel("Temperature (degC)")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Power fraction $f(T)$")
    axes[0].legend(loc="lower left", fontsize=7)
    return save(fig, path)


def cliff(curves, path, marker=45.0):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for label, x, y, color, ls in curves:
        ax.plot(x, y, color=color, lw=2, ls=ls, label=label)
    ax.axvline(marker, color=C["grey"], ls=":", lw=1)
    ax.text(marker + 0.25, ax.get_ylim()[1] * 0.92, f"{marker:.0f} degC", color=C["grey"], fontsize=7)
    ax.set_xlabel("Sustained ambient air temperature (degC)")
    ax.set_ylabel("Equilibrium delivered power (kW)")
    ax.set_title("Steady-state charging power: the derating cliff")
    ax.legend(fontsize=7)
    return save(fig, path)


def stacking(df, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for (label, g), color in zip(df.groupby("headway_label", sort=False),
                                 [C["base"], C["shift"], C["canopy"], C["both"], C["grey"]]):
        ax.plot(g["session"], g["mean_power_kw"], marker="o", color=color, lw=1.8, label=label)
    ax.set_xlabel("Consecutive session at the same stall")
    ax.set_ylabel("Mean delivered power (kW)")
    ax.set_title("Thermal stacking: back-to-back sessions compound derating")
    ax.legend(fontsize=7, title="Idle gap between sessions")
    ax.set_xticks([1, 2, 3, 4])
    return save(fig, path)


def waits(hourly, path):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for label, hp, color in hourly:
        ax.plot(hp["hour"], np.nan_to_num(hp["wait_min"]), marker="o", ms=3,
                color=color, lw=1.6, label=label)
    ax.set_xlabel("Hour of arrival (GST)")
    ax.set_ylabel("Mean wait (min)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Mean queueing delay by hour (heatwave day)")
    ax.legend(fontsize=7)
    return save(fig, path)


def wait_ecdf(curves, path):
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for label, waits, color in curves:
        w = np.sort(np.asarray(waits))
        if len(w) == 0:
            continue
        ax.step(w, np.arange(1, len(w) + 1) / len(w), where="post", color=color, lw=1.8, label=label)
    ax.set_xlabel("Wait before plug-in (min)")
    ax.set_ylabel("Cumulative share of served drivers")
    ax.set_title("Distribution of queueing delay")
    ax.legend(fontsize=7)
    ax.set_xlim(0, None)
    return save(fig, path)


def factorial(effects, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    labels = list(effects.keys())
    vals = [effects[k] for k in labels]
    colors = [C["canopy"] if "anopy" in l else C["shift"] if "hift" in l else C["dark"] for l in labels]
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.axhline(0, color=C["dark"], lw=0.8)
    ax.set_ylabel("Reduction in mean wait (min)")
    ax.set_title("Main effects and interaction (Shapley-style decomposition)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    return save(fig, path)


def grid_profile(profiles, path):
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for label, series, color in profiles:
        ax.plot(series["hour"], series["kw"], color=color, lw=1.8, label=label)
    ax.set_xlabel("Hour of day (GST)")
    ax.set_ylabel("Station AC demand (kW)")
    ax.set_title("Station load profile and grid peak")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=7)
    return save(fig, path)


def parity(obs, preds, path):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    allv = [np.asarray(obs)]
    for _, p, _ in preds:
        allv.append(np.asarray(p))
    lo, hi = np.nanmin(np.concatenate(allv)), np.nanmax(np.concatenate(allv))
    ax.plot([lo, hi], [lo, hi], color=C["grey"], lw=1, ls="--")
    for label, p, color in preds:
        ax.scatter(obs, p, s=6, alpha=0.45, color=color, label=label, edgecolors="none")
    ax.set_xlabel("Measured temperature (degC)")
    ax.set_ylabel("Predicted temperature (degC)")
    ax.set_title("Calibration parity, held-out days")
    ax.legend(fontsize=7, markerscale=3)
    return save(fig, path)


def residual_plot(frames, path):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
    for ax, (label, df, color) in zip(axes, frames):
        ax.scatter(df["t_amb"], df["resid"], s=6, alpha=0.4, color=color, edgecolors="none")
        ax.axhline(0, color=C["dark"], lw=0.8)
        ax.set_xlabel("Ambient temperature (degC)")
        ax.set_title(label, loc="left")
    axes[0].set_ylabel("Residual (K)")
    return save(fig, path)


def extrapolation(df, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    for label, color, marker in (("Grey-box (physics + ridge)", C["canopy"], "o"),
                                 ("Polynomial black box", C["base"], "s")):
        ax.plot(df["t_amb_mid"], df[label], color=color, marker=marker, lw=1.8, label=label)
    ax.set_xlabel("Ambient temperature band (degC)")
    ax.set_ylabel("RMSE of cabinet temperature (K)")
    ax.set_title("Extrapolation to a 47 degC day the model never saw")
    ax.legend(fontsize=7)
    return save(fig, path)


def tornado(labels, low, high, base, path):
    order = np.argsort(np.abs(np.array(high) - np.array(low)))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(labels))
    ax.barh(y, np.array(high)[order] - base, left=base, color=C["canopy"], alpha=0.8)
    ax.barh(y, np.array(low)[order] - base, left=base, color=C["base"], alpha=0.8)
    ax.axvline(base, color=C["dark"], lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in order], fontsize=7)
    ax.set_xlabel("Reduction in mean wait from canopy (min)")
    ax.set_title("Sensitivity of the canopy benefit")
    return save(fig, path)


def economics(df, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(df["year"], df["cumulative_AED"], color=C["canopy"], lw=2.2)
    ax.axhline(0, color=C["dark"], lw=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative net benefit (AED per stall)")
    ax.set_title("Canopy economics (one stall, conservative benefit)")
    return save(fig, path)


def monthly(df, path):
    fig, ax1 = plt.subplots(figsize=(7.0, 3.2))
    ax1.bar(df["month"], df["derate_base_pct"], color=C["base"], alpha=0.55,
            label="Mean derating, unshaded (%)")
    ax1.set_ylabel("Mean power derating (%)", color=C["base"])
    ax1.set_xlabel("Month")
    ax1.set_xticks(range(1, 13))
    ax2 = ax1.twinx()
    ax2.plot(df["month"], df["wait_base"], color=C["dark"], marker="o", lw=1.6,
             label="Mean wait, unshaded (min)")
    ax2.plot(df["month"], df["wait_canopy"], color=C["canopy"], marker="o", lw=1.6,
             label="Mean wait, canopy (min)")
    ax2.set_ylabel("Mean wait (min)")
    ax2.grid(False)
    lines = ax2.get_lines() + ax1.get_lines()[0:1]
    ax2.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="upper left")
    ax1.set_title("Seasonality: the intervention matters only in the hot months")
    return save(fig, path)
