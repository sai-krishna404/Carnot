"""Climate and solar geometry.

The weather driver is a real Typical Meteorological Year (TMY) for Abu Dhabi
(24.454 N, 54.377 E) obtained from the PVGIS v5.3 API (SARAH-3 satellite
irradiance + ERA5 meteorology, 2005-2023).  See report/references.md.

PVGIS timestamps are UTC; the model runs in Gulf Standard Time (UTC+4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PVGIS_RENAME = {
    "time(UTC)": "time_utc",
    "T2m": "t_amb_c",
    "RH": "rh_pct",
    "G(h)": "ghi",
    "Gb(n)": "dni",
    "Gd(h)": "dhi",
    "IR(h)": "lwr",
    "WS10m": "wind_ms",
    "WD10m": "wind_dir_deg",
    "SP": "pressure_pa",
}


def load_tmy(path, utc_offset_h: float = 4.0) -> pd.DataFrame:
    """Parse the PVGIS TMY CSV into a clean hourly local-time frame.

    PVGIS concatenates twelve typical months taken from different years, so the
    raw timestamps jump at month boundaries (some 'days' have 4 or 20 hours).
    The values are a continuous 8760-hour year, so the index is rebuilt on a
    synthetic calendar year; only month/day/hour are meaningful.
    """
    df = pd.read_csv(path, skiprows=17, nrows=8760)
    df = df.rename(columns=PVGIS_RENAME)
    if len(df) != 8760:
        raise ValueError(f"expected 8760 TMY hours, found {len(df)}")
    df.index = pd.date_range("2001-01-01 00:00", periods=8760, freq="h", name="time_utc")
    df.index = df.index + pd.Timedelta(hours=utc_offset_h)
    df.index.name = "time_local"

    # Data hygiene (documented in data/processed/README.md)
    df["ghi"] = df["ghi"].clip(lower=0.0)
    df["dni"] = df["dni"].clip(lower=0.0)
    df["dhi"] = df["dhi"].clip(lower=0.0)
    df["wind_ms"] = df["wind_ms"].clip(lower=0.3)
    df["t_amb_c"] = df["t_amb_c"].astype(float)
    # NOTE: the PVGIS 'IR(h)' column in this product is unusable (it is
    # mostly 0 W/m2 with isolated ~430 W/m2 spikes, physically impossible for
    # downward longwave).  Sky temperature is instead derived from air
    # temperature and dew point with the Berdahl & Martin (1984) model.
    df = df.drop(columns=["lwr"], errors="ignore")
    return df


def dew_point_c(temp_c, rh_pct):
    """Magnus-Tetens dew point (degC)."""
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1.0, 100.0)
    t = np.asarray(temp_c, dtype=float)
    gamma = np.log(rh / 100.0) + 17.625 * t / (243.04 + t)
    return 243.04 * gamma / (17.625 - gamma)


def sky_temperature_c(temp_c, rh_pct):
    """Effective sky temperature from the Berdahl-Martin clear-sky emissivity."""
    t_k = np.asarray(temp_c, dtype=float) + 273.15
    td_k = dew_point_c(temp_c, rh_pct) + 273.15
    eps_sky = np.clip(0.787 + 0.764 * np.log(np.clip(td_k, 200.0, 330.0) / 273.0), 0.6, 1.0)
    return t_k * eps_sky ** 0.25 - 273.15


def solar_position(lat_deg: float, lon_deg: float, when_local: pd.DatetimeIndex,
                   tz_offset_h: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """NOAA solar position (zenith and azimuth from North, clockwise), degrees.

    Accuracy is ~0.02 deg for this application, far below the uncertainty in
    plane-of-array irradiance.
    """
    t = pd.DatetimeIndex(when_local)
    doy = t.dayofyear.to_numpy().astype(float)
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60.0 + t.second.to_numpy() / 3600.0

    gamma = 2.0 * np.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.001480 * np.sin(3 * gamma))

    time_offset = eqtime + 4.0 * lon_deg - 60.0 * tz_offset_h
    tst = hour * 60.0 + time_offset
    ha = np.deg2rad((tst / 4.0) - 180.0)

    lat = np.deg2rad(lat_deg)
    cos_zen = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    zen = np.arccos(cos_zen)

    denom = np.cos(lat) * np.sin(zen)
    cos_az = np.where(np.abs(denom) > 1e-9,
                      (np.sin(decl) - np.sin(lat) * cos_zen) / np.where(denom == 0, 1e-9, denom) * -1.0,
                      0.0)
    az = np.rad2deg(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    az = np.where(ha > 0, 360.0 - az, az)
    return np.rad2deg(zen), az


def plane_of_array(dni, dhi, ghi, zenith_deg, azimuth_deg, tilt_deg, surface_azimuth_deg,
                   albedo: float = 0.2):
    """Isotropic-sky plane-of-array irradiance on a tilted surface (W/m2)."""
    zen = np.deg2rad(zenith_deg)
    aoi_cos = (np.cos(zen) * np.cos(np.deg2rad(tilt_deg))
               + np.sin(zen) * np.sin(np.deg2rad(tilt_deg))
               * np.cos(np.deg2rad(azimuth_deg - surface_azimuth_deg)))
    aoi_cos = np.clip(aoi_cos, 0.0, None)
    sky = dhi * (1.0 + np.cos(np.deg2rad(tilt_deg))) / 2.0
    ground = ghi * albedo * (1.0 - np.cos(np.deg2rad(tilt_deg))) / 2.0
    return dni * aoi_cos + sky + ground


def cabinet_irradiance(df: pd.DataFrame, albedo: float = 0.2) -> np.ndarray:
    """Effective solar irradiance absorbed by a box-shaped outdoor cabinet.

    30 % of the area is the roof (horizontal, receives GHI); the remaining
    70 % is split evenly across the four vertical facades.  This avoids the
    common over-simplification of treating a cabinet as a horizontal panel.
    """
    zen, az = df["zenith_deg"].to_numpy(), df["azimuth_deg"].to_numpy()
    dni, dhi, ghi = df["dni"].to_numpy(), df["dhi"].to_numpy(), df["ghi"].to_numpy()
    roof = ghi
    walls = np.mean([plane_of_array(dni, dhi, ghi, zen, az, 90.0, a, albedo)
                     for a in (0.0, 90.0, 180.0, 270.0)], axis=0)
    return 0.30 * roof + 0.70 * walls


def add_solar_geometry(df: pd.DataFrame, lat: float, lon: float,
                       tz_offset_h: float = 4.0, albedo: float = 0.2) -> pd.DataFrame:
    df = df.copy()
    zen, az = solar_position(lat, lon, df.index, tz_offset_h)
    df["zenith_deg"] = zen
    df["azimuth_deg"] = az
    df["ghi_horizontal"] = df["ghi"]
    df["ghi_cabinet"] = cabinet_irradiance(df, albedo)
    df["t_sky_c"] = sky_temperature_c(df["t_amb_c"].to_numpy(), df["rh_pct"].to_numpy())
    return df


def _complete_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Normalised dates with a full 24 hourly rows.

    The local-time index is shifted by the UTC offset, so the first and last
    calendar dates of the synthetic year are truncated.  They are excluded
    from every day-selection routine.
    """
    norm = df.index.normalize()
    sizes = pd.Series(1, index=norm).groupby(level=0).size()
    return pd.DatetimeIndex(sizes[sizes == 24].index)


def select_design_day(df: pd.DataFrame, mode: str = "hottest_july") -> pd.DataFrame:
    """Return 24 hourly rows of the design day.

    ``hottest_july`` picks the July day with the highest daily mean air
    temperature, i.e. a typical (not record) extreme summer day.
    """
    if mode not in ("hottest_july", "hottest", "mean_july"):
        day = pd.Timestamp(mode).date()
        out = df[df.index.date == day]
        if out.empty:
            raise ValueError(f"design day {mode} not present in TMY")
        return out.copy()

    complete = _complete_dates(df)
    subset = df[(df.index.month == 7) & df.index.normalize().isin(complete)]
    daily = subset.groupby(subset.index.normalize())["t_amb_c"].mean()
    if mode == "mean_july":
        target = daily.index[len(daily) // 2]
    else:
        target = daily.idxmax()
    return df[df.index.normalize() == target].copy()


def day_arrays(day: pd.DataFrame) -> dict[str, np.ndarray]:
    """Hourly vectors used by the simulator, keyed by physical name."""
    return {
        "t_amb": day["t_amb_c"].to_numpy(dtype=float),
        "ghi": day["ghi_horizontal"].to_numpy(dtype=float),
        "ghi_cabinet": day["ghi_cabinet"].to_numpy(dtype=float),
        "t_sky": day["t_sky_c"].to_numpy(dtype=float),
        "wind": day["wind_ms"].to_numpy(dtype=float),
        "dni": day["dni"].to_numpy(dtype=float),
        "dhi": day["dhi"].to_numpy(dtype=float),
    }


def representative_days(df: pd.DataFrame, n: int, months=(6, 7, 8, 9)) -> list[pd.DataFrame]:
    """Pick ``n`` summer days spread across the daily-mean temperature range.

    Used for calibration (variety of weather) and for the monthly annualisation.
    """
    sub = df[df.index.month.isin(months)]
    sub = sub[sub.index.normalize().isin(_complete_dates(df))]
    daily = sub.groupby(sub.index.normalize())["t_amb_c"].mean().sort_values()
    idx = np.linspace(0, len(daily) - 1, n).astype(int)
    dates = [daily.index[i] for i in idx]
    return [df[df.index.normalize() == d].copy() for d in dates]


def month_days(df: pd.DataFrame, month: int, day: int = 15) -> pd.DataFrame:
    out = df[(df.index.month == month) & (df.index.day == day)]
    if out.empty:
        raise ValueError(f"no TMY data for month {month}")
    return out.copy()


def hottest_day_in_month(df: pd.DataFrame, month: int) -> pd.DataFrame:
    """The month's own extreme day (highest daily-mean air temperature).

    Using the hottest day of each month - rather than an arbitrary mid-month
    day - keeps the seasonality experiment consistent with the design-day
    choice and represents the kind of day on which derating actually bites.
    """
    sub = df[df.index.month == month]
    if sub.empty:
        raise ValueError(f"no TMY data for month {month}")
    sub = sub[sub.index.normalize().isin(_complete_dates(df))]
    daily = sub.groupby(sub.index.normalize())["t_amb_c"].mean()
    return df[df.index.normalize() == daily.idxmax()].copy()


@dataclass
class Environment:
    """Hourly weather interpolated to arbitrary seconds-of-day (cyclic)."""

    t_amb: np.ndarray
    ghi: np.ndarray
    ghi_cabinet: np.ndarray
    t_sky: np.ndarray
    wind: np.ndarray

    def at(self, t_s):
        """Interpolate to seconds-of-day ``t_s`` (scalar or array), cyclic."""
        hours = (np.asarray(t_s, dtype=float) / 3600.0) % 24.0
        out = {}
        for name in ("t_amb", "ghi", "ghi_cabinet", "t_sky", "wind"):
            values = getattr(self, name)
            ext = np.concatenate([values, values[:1]])
            out[name] = np.interp(hours, np.arange(0, 25, 1.0), ext)
        return out

    @classmethod
    def from_day(cls, day: pd.DataFrame, canopy: bool = False,
                 transmission: float = 0.10, delta_t: float = 2.5,
                 heatwave_delta_k: float = 0.0) -> "Environment":
        a = day_arrays(day)
        t_amb = a["t_amb"] - (delta_t if canopy else 0.0) + heatwave_delta_k
        t_sky = a["t_sky"] + heatwave_delta_k
        ghi = a["ghi"] * (transmission if canopy else 1.0)
        ghi_cab = a["ghi_cabinet"] * (transmission if canopy else 1.0)
        return cls(t_amb=t_amb, ghi=ghi, ghi_cabinet=ghi_cab, t_sky=t_sky, wind=a["wind"])

    def shifted(self, delta_k: float) -> "Environment":
        return Environment(t_amb=self.t_amb + delta_k, ghi=self.ghi,
                           ghi_cabinet=self.ghi_cabinet,
                           t_sky=self.t_sky + delta_k, wind=self.wind)
