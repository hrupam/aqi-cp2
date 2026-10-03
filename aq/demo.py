"""
Synthetic but physically-motivated data so the whole system can be run and demoed
offline (no API key / no internet). PM2.5 responds to mixing height, wind, rain,
season, traffic hours and occasional burning episodes - the same mechanisms the
real model has to learn. Clearly labelled as DEMO everywhere it is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ar1(n, phi, sigma, rng):
    x = np.zeros(n)
    e = rng.normal(0, sigma, n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def make_demo(start="2023-01-01", end=None, seed=7):
    rng = np.random.default_rng(seed)
    now = pd.Timestamp.now(tz="UTC").floor("h")
    end = pd.Timestamp(end, tz="UTC") if end else now + pd.Timedelta(days=4)
    t = pd.date_range(pd.Timestamp(start, tz="UTC"), end, freq="h")
    n = len(t)
    loc = t.tz_convert("Asia/Kolkata")
    hr = loc.hour.values + loc.minute.values / 60
    doy = loc.dayofyear.values
    month = loc.month.values

    winter = np.cos(2 * np.pi * (doy - 15) / 365.25)            # +1 mid-Jan, -1 mid-Jul
    monsoon = ((month >= 6) & (month <= 9)).astype(float)
    day = np.clip(np.sin(np.pi * (hr - 7) / 11), 0, None)        # daylight shape 07-18 h

    temp = 27 - 5 * winter + 5 * day + _ar1(n, 0.97, 0.35, rng)
    rh = np.clip(65 + 15 * monsoon - 18 * day + 5 * winter + _ar1(n, 0.95, 2, rng), 15, 100)
    dew = temp - (100 - rh) / 5
    wind = np.clip(2.2 + 1.6 * monsoon + 1.2 * day + _ar1(n, 0.96, 0.25, rng), 0.2, None)
    gust = wind * 1.6 + np.abs(rng.normal(0, 0.5, n))
    base_dir = np.where(winter > 0.2, 315, np.where(monsoon > 0, 220, 180))
    wdir = (base_dir + _ar1(n, 0.98, 6, rng)) % 360
    rain_prob = 0.02 + 0.13 * monsoon
    precip = np.where(rng.random(n) < rain_prob, rng.gamma(1.2, 2.0, n), 0.0)
    cloud = np.clip(30 + 50 * monsoon + 40 * (precip > 0) + _ar1(n, 0.95, 6, rng), 0, 100)
    rad = np.clip(900 * day * (1 - cloud / 140), 0, None)
    blh = np.clip((180 + 60 * monsoon) + (1700 - 700 * winter) * day * (1 - cloud / 200)
                  + 40 * wind + _ar1(n, 0.9, 40, rng), 50, 3500)
    pressure = 1008 + 6 * winter + _ar1(n, 0.99, 0.3, rng)

    # a stagnation spell in the coming days so the demo has something to warn about
    stag = (t >= now + pd.Timedelta(hours=16)) & (t <= now + pd.Timedelta(hours=64))
    ramp = np.clip(np.minimum((t - (now + pd.Timedelta(hours=16))) / pd.Timedelta(hours=10),
                              ((now + pd.Timedelta(hours=64)) - t) / pd.Timedelta(hours=10)), 0, 1)
    ramp = np.where(stag, ramp, 0.0)
    wind = wind * (1 - 0.7 * ramp)
    gust = gust * (1 - 0.6 * ramp)
    blh = np.clip(blh * (1 - 0.65 * ramp), 50, None)
    precip = np.where(stag, 0.0, precip)
    wdir = np.where(stag, (300 + _ar1(n, 0.9, 5, rng)) % 360, wdir)

    # ---------------- PM2.5 dynamics
    traffic = 1 + 0.45 * np.exp(-((hr - 9.5) ** 2) / 4) + 0.6 * np.exp(-((hr - 21) ** 2) / 6)
    weekday = loc.dayofweek.values
    traffic *= np.where(weekday == 6, 0.85, 1.0)
    season = np.exp(0.75 * winter)                               # ~2.1x in winter vs ~0.47x monsoon
    ventilation = wind * blh
    vent_factor = (np.median(ventilation) / np.clip(ventilation, 50, None)) ** 0.35
    transport = 1 + 0.25 * (np.cos(np.deg2rad(wdir - 315)) > 0.7) * (winter > 0)
    # burning episodes (post-monsoon / winter), each lasts 1-4 days
    episode = np.ones(n)
    for i in np.where((rng.random(n) < 0.0012) & ((month >= 10) | (month <= 2)))[0]:
        L = rng.integers(24, 96)
        episode[i:i + L] *= rng.uniform(1.4, 2.2)
    episode *= 1 + 0.6 * ramp                                      # + local burning during the spell
    target = 38 * season * traffic * vent_factor * transport * episode * np.exp(_ar1(n, 0.97, 0.05, rng))
    pm = np.zeros(n)
    pm[0] = target[0]
    for i in range(1, n):
        washout = min(0.5, 0.06 * precip[i])
        pm[i] = (0.82 * pm[i - 1] + 0.18 * target[i]) * (1 - washout)
    pm = np.clip(pm * np.exp(rng.normal(0, 0.08, n)), 2, 900)

    # CAMS-like coarse model: smoother, biased low in winter, a few hours out of phase
    cams = pd.Series(pm).rolling(12, center=True, min_periods=1).mean().shift(4).bfill().ffill().values
    cams = cams * (0.65 + 0.1 * monsoon) * np.exp(_ar1(n, 0.99, 0.06, rng) + rng.normal(0, 0.12, n)) + 6

    weather = pd.DataFrame({"time": t, "temp": temp, "rh": rh, "dewpoint": dew, "wind_speed": wind,
                            "wind_dir": wdir, "wind_gust": gust, "pressure": pressure, "precip": precip,
                            "cloud": cloud, "radiation": rad, "blh": blh})
    cams_df = pd.DataFrame({"time": t, "cams_pm25": cams})
    last_obs = now - pd.Timedelta(hours=2)                      # stations report with a small delay
    obs_mask = (t <= last_obs) & (rng.random(n) > 0.03)          # ~3 % random gaps
    pm_df = pd.DataFrame({"time": t[obs_mask], "pm25": pm[obs_mask], "coverage": 100.0})
    return pm_df.reset_index(drop=True), weather, cams_df
