"""Quality-control PM2.5 readings and merge them with weather + CAMS on one hourly UTC grid."""
from __future__ import annotations

import numpy as np
import pandas as pd


def clean_pm25(pm: pd.DataFrame, min_coverage: float = 50) -> tuple[pd.Series, dict]:
    """Return an hourly PM2.5 series with bad readings removed, plus a QC report."""
    df = pm.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.floor("h")
    df = df.groupby("time", as_index=False).agg(pm25=("pm25", "mean"), coverage=("coverage", "mean"))
    report = {"raw_hours": int(len(df))}
    s = df.set_index("time")["pm25"].astype(float)

    bad_range = (s <= 0) | (s > 1000)
    bad_cov = df.set_index("time")["coverage"].fillna(100) < min_coverage
    # stuck sensor: same value repeated for >= 6 consecutive hours
    run_id = (s.diff() != 0).cumsum()
    run_len = s.groupby(run_id).transform("size")
    bad_stuck = run_len >= 6
    # isolated spikes: > 4x the 24 h rolling median and > 250 µg/m³
    med = s.rolling(25, center=True, min_periods=5).median()
    bad_spike = (s > 4 * med) & (s > 250)
    report.update({
        "out_of_range": int(bad_range.sum()), "low_coverage": int(bad_cov.sum()),
        "stuck_values": int(bad_stuck.sum()), "spikes": int(bad_spike.sum()),
    })
    s = s.mask(bad_range | bad_cov | bad_stuck | bad_spike)
    if s.notna().sum() == 0:
        raise SystemExit("No usable PM2.5 readings for this station after quality checks.")

    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full)
    missing_before = int(s.isna().sum())
    s = s.interpolate(limit=3, limit_area="inside")            # fill only short gaps
    report.update({"grid_hours": int(len(full)), "gaps_filled": missing_before - int(s.isna().sum()),
                   "still_missing": int(s.isna().sum()),
                   "completeness_pct": round(float(100 * s.notna().mean()), 1)})
    s.name = "pm25"
    s.index.name = "time"
    return s, report


def merge_sources(pm: pd.Series, weather: pd.DataFrame, cams: pd.DataFrame) -> pd.DataFrame:
    """One row per UTC hour: pm25 (may be NaN in the future), weather, CAMS pm2.5."""
    w = weather.copy()
    w["time"] = pd.to_datetime(w["time"], utc=True)
    w = w.drop_duplicates("time").set_index("time").sort_index()
    w = w.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")   # guard against object dtypes
    c = cams.copy()
    c["time"] = pd.to_datetime(c["time"], utc=True)
    c = c.drop_duplicates("time").set_index("time").sort_index()
    c = c.apply(pd.to_numeric, errors="coerce")
    start = pm.index.min()
    end = max(pm.index.max(), w.index.max())
    grid = pd.date_range(start, end, freq="h", name="time")
    df = pd.DataFrame(index=grid)
    df["pm25"] = pm.reindex(grid)
    df = df.join(w.reindex(grid))
    df["cams_pm25"] = c["cams_pm25"].reindex(grid) if "cams_pm25" in c else np.nan
    # weather has essentially no gaps; interpolate the odd missing hour
    wcols = [col for col in w.columns if col in df]
    df[wcols] = df[wcols].interpolate(limit=6, limit_direction="both")
    if "cams_pm25" in df:
        df["cams_pm25"] = df["cams_pm25"].interpolate(limit=6, limit_area="inside")
    return df


def bridge_gap(df: pd.DataFrame, now: pd.Timestamp | None = None, min_gap_h: int = 6,
               max_gap_days: int = 14) -> tuple[pd.DataFrame, dict]:
    """If the station has stopped reporting, estimate the missing recent hours from CAMS,
    scaled by the station/CAMS ratio of the last 7 days of overlap. The estimate goes in a
    separate column 'pm25_bridge' so it is used to issue forecasts but never for training."""
    df = df.copy()
    df["pm25_bridge"] = np.nan
    now = (now or pd.Timestamp.now(tz="UTC")).floor("h")
    last = df["pm25"].last_valid_index()
    info = {"last_observation": last, "gap_hours": float((now - last).total_seconds() / 3600)}
    if info["gap_hours"] < min_gap_h or "cams_pm25" not in df or df["cams_pm25"].isna().all():
        return df, info
    if info["gap_hours"] > max_gap_days * 24:
        info["note"] = "station offline too long to bridge - choose another sensor"
        return df, info
    recent = df.loc[last - pd.Timedelta(days=7): last, ["pm25", "cams_pm25"]].dropna()
    if len(recent) < 24:
        return df, info
    ratio = float(np.exp(np.median(np.log1p(recent.pm25) - np.log1p(recent.cams_pm25))))
    gap = (df.index > last) & (df.index <= now)
    df.loc[gap, "pm25_bridge"] = np.clip(np.expm1(np.log1p(df.loc[gap, "cams_pm25"]) + np.log(ratio)), 0, None)
    info.update({"bridged": True, "bridge_ratio": ratio,
                 "bridged_until": df.loc[gap, "pm25_bridge"].last_valid_index()})
    return df, info
