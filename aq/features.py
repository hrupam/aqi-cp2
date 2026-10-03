"""
Feature engineering for direct multi-horizon forecasting.

Every training/forecast sample is a pair (issue time T, horizon h):
  * what we KNOW at T      -> recent PM2.5 history, current weather, current CAMS bias
  * what we EXPECT at T+h  -> forecast weather, CAMS model PM2.5, hour/day/season
  * h itself
and the target is PM2.5 at T+h. One model therefore produces the full 1..60 h curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

PM_LAGS = (1, 2, 3, 6, 12, 24, 48)
WEATHER_COLS = ["temp", "rh", "dewpoint", "wind_speed", "wind_gust", "pressure",
                "precip", "cloud", "radiation", "blh"]


def series_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-hour helper columns computed once on the merged hourly table."""
    f = pd.DataFrame(index=df.index)
    pm = df["pm25"]
    logpm = np.log1p(pm)
    # ---- what is known at issue time (p_)
    f["p_pm"] = pm
    for k in PM_LAGS:
        f[f"p_lag{k}"] = pm.shift(k)
    for w in (3, 6, 24, 72):
        f[f"p_mean{w}"] = pm.rolling(w, min_periods=max(1, w // 2)).mean()
    f["p_std24"] = pm.rolling(24, min_periods=12).std()
    f["p_max24"] = pm.rolling(24, min_periods=12).max()
    f["p_min24"] = pm.rolling(24, min_periods=12).min()
    f["p_trend3"] = logpm - logpm.shift(3)
    f["p_trend24"] = logpm - logpm.shift(24)

    # ---- weather (w_) – used at the target time, some also at issue time
    for c in WEATHER_COLS:                        # always create the column so train/live match
        f[f"w_{c}"] = df[c] if c in df else np.nan
    wd = df["wind_dir"] if "wind_dir" in df else np.nan
    f["w_wind_u"] = -df["wind_speed"] * np.sin(np.deg2rad(wd))
    f["w_wind_v"] = -df["wind_speed"] * np.cos(np.deg2rad(wd))
    blh = df["blh"] if "blh" in df else pd.Series(np.nan, index=df.index)
    f["w_vent"] = df["wind_speed"] * blh                         # ventilation index (m²/s)
    f["w_log_vent"] = np.log1p(f["w_vent"])
    f["w_rain6"] = df["precip"].rolling(6, min_periods=1).sum()
    f["w_rain24"] = df["precip"].rolling(24, min_periods=1).sum()
    f["w_wind12"] = df["wind_speed"].rolling(12, min_periods=1).mean()
    f["w_logvent12"] = f["w_log_vent"].rolling(12, min_periods=1).mean()
    f["w_blh_max24"] = blh.rolling(24, min_periods=1).max()
    f["w_dpress24"] = df["pressure"] - df["pressure"].shift(24) if "pressure" in df else np.nan
    f["w_dtemp24"] = df["temp"] - df["temp"].shift(24)

    # ---- CAMS model PM2.5 (c_)
    cams = df["cams_pm25"] if "cams_pm25" in df else pd.Series(np.nan, index=df.index)
    f["c_cams"] = cams
    f["c_cams24"] = cams.rolling(24, min_periods=6).mean()

    # ---- calendar in local time (k_)
    loc = df.index.tz_convert(config.LOCAL_TZ)
    hour = loc.hour + loc.minute / 60
    f["k_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["k_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["k_dow"] = loc.dayofweek
    f["k_doy_sin"] = np.sin(2 * np.pi * loc.dayofyear / 365.25)
    f["k_doy_cos"] = np.cos(2 * np.pi * loc.dayofyear / 365.25)
    return f


def make_samples(sf: pd.DataFrame, issue_pos: np.ndarray, horizons: np.ndarray,
                 with_target: bool = True) -> pd.DataFrame:
    """Build the (T, h) design matrix. issue_pos are integer row positions in sf."""
    n = len(sf)
    I = np.repeat(issue_pos, len(horizons))
    H = np.tile(horizons, len(issue_pos))
    V = I + H
    ok = V < n
    I, H, V = I[ok], H[ok], V[ok]

    p_cols = [c for c in sf.columns if c.startswith("p_")]
    tgt_cols = [c for c in sf.columns if c.startswith(("w_", "c_", "k_"))]
    A = sf[p_cols].to_numpy()[I]
    B = sf[tgt_cols].to_numpy()[V]
    X = pd.DataFrame(np.hstack([A, B]), columns=p_cols + tgt_cols)
    X.insert(0, "h", H)

    # issue-time context and changes between now and target time
    vent_now = sf["w_log_vent"].to_numpy()[I]
    X["d_log_vent"] = X["w_log_vent"].to_numpy() - vent_now
    X["d_temp"] = X["w_temp"].to_numpy() - sf["w_temp"].to_numpy()[I]
    X["d_wind"] = X["w_wind_speed"].to_numpy() - sf["w_wind_speed"].to_numpy()[I]
    cams_now = sf["c_cams"].to_numpy()[I]
    pm_now = sf["p_pm"].to_numpy()[I]
    with np.errstate(divide="ignore", invalid="ignore"):
        X["c_bias_now"] = np.log1p(pm_now) - np.log1p(cams_now)          # local bias of CAMS
        X["c_cams_adj"] = np.expm1(np.log1p(X["c_cams"].to_numpy()) + X["c_bias_now"].to_numpy())
    # mean ventilation and total rain between issue and target time (cumulative-sum trick)
    cs_v = np.concatenate([[0], np.nancumsum(np.nan_to_num(sf["w_log_vent"].to_numpy()))])
    cs_r = np.concatenate([[0], np.nancumsum(np.nan_to_num(sf["w_precip"].to_numpy()))])
    X["path_log_vent"] = (cs_v[V + 1] - cs_v[I + 1]) / H
    X["path_rain"] = cs_r[V + 1] - cs_r[I + 1]
    # same clock hour on the most recent fully observed day
    back = 24 * np.ceil(H / 24).astype(int)
    pm_arr = sf["p_pm"].to_numpy()
    X["p_same_hour"] = pm_arr[np.clip(V - back, 0, n - 1)]

    X["c_cams_T"] = cams_now                                         # CAMS at issue time

    X["issue_time"] = sf.index[I]
    X["target_time"] = sf.index[V]
    if with_target:
        X["y"] = pm_arr[V]
    return X


def feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c not in ("issue_time", "target_time", "y")]


# absolute-level columns that are replaced by relative versions in the model matrix
_ABSOLUTE = {"p_pm", "c_cams", "c_cams24", "c_cams_adj", "c_bias_now", "c_cams_T", "k_doy_sin", "k_doy_cos"}
_RATIO_FREE = {"p_trend3", "p_trend24", "p_std24"}


def design(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Scale-free model matrix + the log persistence baseline.

    The model predicts log(PM at T+h) - log(PM now), i.e. how much pollution will CHANGE.
    All past-PM features are expressed relative to the current level and CAMS enters only
    as its forecast change. This keeps the model anchored to what the station measures
    today, so it does not drift toward last year's levels when a season is cleaner or
    dirtier than before, or when a station is recalibrated. Day-of-year is left out
    because with 1-3 years of data it lets trees memorise specific dates."""
    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.log1p(X["p_pm"].fillna(X["p_mean3"])).to_numpy()
        R = pd.DataFrame(index=X.index)
        for c in feature_columns(X):
            if c in _ABSOLUTE:
                continue
            if c.startswith("p_") and c not in _RATIO_FREE:
                R[c] = np.log1p(X[c]).to_numpy() - base
            elif c == "p_std24":
                R[c] = X[c].to_numpy() / np.clip(np.expm1(base), 1, None)
            else:
                R[c] = X[c].to_numpy()
        lc_T = np.log1p(X["c_cams_T"]).to_numpy()
        R["c_cams_change"] = np.log1p(X["c_cams"]).to_numpy() - lc_T
        R["c_cams24_change"] = np.log1p(X["c_cams24"]).to_numpy() - lc_T
    R = R.replace([np.inf, -np.inf], np.nan)
    return R, base
