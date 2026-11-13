"""Aggregation of station simulations into operational and grid metrics."""
from __future__ import annotations

import numpy as np

from . import thermal


def _pct(values, q):
    return float(np.percentile(values, q)) if len(values) else float("nan")


def summarise(res, cfg, label: str = "") -> dict:
    served = res.served
    waits = np.array([(r["t_start_s"] - r["arrival_s"]) / 60.0 for r in served]) \
        if served else np.array([])
    svc = np.array([(r["t_end_s"] - r["t_start_s"]) / 60.0 for r in served]) \
        if served else np.array([])
    energy = np.array([r["energy_kwh"] for r in served]) if served else np.array([])
    mean_p = np.array([r["mean_power_kw"] for r in served]) if served else np.array([])

    refs = np.array([thermal.reference_duration_s(cfg, r["soc0"]) / 60.0 for r in served]) \
        if served else np.array([])
    slowdown = np.divide(svc, refs, out=np.full_like(svc, np.nan), where=refs > 0) \
        if len(svc) else np.array([])

    n_stalls = res.meta.get("n_stalls", cfg["queue"]["n_stalls"])
    busy_s = float(svc.sum() * 60.0) if len(svc) else 0.0
    utilization = busy_s / (n_stalls * 86400.0)

    dt = res.dt_s
    win = max(1, int(round(900.0 / dt)))  # 15-minute demand window
    if len(res.load_kw) >= win:
        kernel = np.ones(win) / win
        smooth = np.convolve(res.load_kw, kernel, mode="same")
        peak = float(smooth.max())
    else:
        peak = float(res.load_kw.max())
    mean_load = float(res.load_kw.mean())

    balked = sum(1 for r in res.records if r.get("balked"))
    reneged = sum(1 for r in res.records if r.get("reneged"))
    censored = sum(1 for r in res.records if r.get("censored"))
    n_arr = res.n_arrivals
    lost = balked + reneged
    rate = min(cfg["site"]["stall_rating_kw"], cfg["vehicle"]["peak_kW"])

    # queue-length weighted by time (mean number waiting)
    return {
        "label": label,
        "n_arrivals": n_arr,
        "served": len(served),
        "balked": balked,
        "reneged": reneged,
        "censored": censored,
        "service_rate_pct": 100.0 * len(served) / n_arr if n_arr else float("nan"),
        "lost_demand_pct": 100.0 * lost / n_arr if n_arr else float("nan"),
        "wait_mean_min": float(waits.mean()) if len(waits) else float("nan"),
        "wait_median_min": _pct(waits, 50),
        "wait_p95_min": _pct(waits, 95),
        "wait_max_min": float(waits.max()) if len(waits) else float("nan"),
        "service_mean_min": float(svc.mean()) if len(svc) else float("nan"),
        "slowdown_mean": float(np.nanmean(slowdown)) if len(slowdown) else float("nan"),
        "power_ratio_mean": float(mean_p.mean() / rate) if len(mean_p) else float("nan"),
        "energy_kwh_day": float(energy.sum()) if len(energy) else 0.0,
        "energy_kwh_session": float(energy.mean()) if len(energy) else float("nan"),
        "peak_load_kw": peak,
        "mean_load_kw": mean_load,
        "load_factor": mean_load / peak if peak > 0 else float("nan"),
        "utilization": utilization,
        "queue_mean": float(res.queue_len.mean()),
        "driver_hours_lost": float(waits.sum() / 60.0) if len(waits) else 0.0,
    }


def hourly_profile(res, cfg, key: str = "wait") -> dict:
    """Mean wait / throughput binned by hour of arrival (for figures)."""
    hours = np.arange(24)
    waits = np.full(24, np.nan)
    counts = np.zeros(24)
    energy = np.zeros(24)
    for r in res.served:
        h = int((r["arrival_s"] / 3600.0) % 24)
        waits[h] = np.nan if np.isnan(waits[h]) else waits[h]
        counts[h] += 1
        energy[h] += r["energy_kwh"]
    sums = np.zeros(24)
    for r in res.served:
        h = int((r["arrival_s"] / 3600.0) % 24)
        sums[h] += (r["t_start_s"] - r["arrival_s"]) / 60.0
    with np.errstate(invalid="ignore", divide="ignore"):
        waits = np.where(counts > 0, sums / counts, np.nan)
    return {"hour": hours, "wait_min": waits, "count": counts, "energy_kwh": energy}


def annualise(monthly: list[dict], days_in_month) -> dict:
    """Aggregate monthly representative-day results into annual totals."""
    keys = [k for k, v in monthly[0].items() if isinstance(v, (int, float))]
    out = {}
    for k in keys:
        out[k] = float(sum(m[k] * d for m, d in zip(monthly, days_in_month)))
    out["_days"] = float(sum(days_in_month))
    return out
