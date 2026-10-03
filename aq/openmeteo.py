"""Weather (Open-Meteo) and CAMS model PM2.5 (Open-Meteo Air-Quality API). No key needed."""
from __future__ import annotations

import re
import time
import requests
import pandas as pd

HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"   # archived forecasts, 2022+
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"             # ERA5 reanalysis fallback
FCST_URL = "https://api.open-meteo.com/v1/forecast"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "surface_pressure", "precipitation", "cloud_cover",
    "shortwave_radiation", "boundary_layer_height",
]

RENAME = {
    "temperature_2m": "temp", "relative_humidity_2m": "rh", "dew_point_2m": "dewpoint",
    "wind_speed_10m": "wind_speed", "wind_direction_10m": "wind_dir", "wind_gusts_10m": "wind_gust",
    "surface_pressure": "pressure", "precipitation": "precip", "cloud_cover": "cloud",
    "shortwave_radiation": "radiation", "boundary_layer_height": "blh", "pm2_5": "cams_pm25",
}


def _request(url: str, params: dict, variables: list[str]) -> pd.DataFrame:
    """Call Open-Meteo; if a variable is not supported by that endpoint, drop it and retry."""
    variables = list(variables)
    for attempt in range(8):
        p = dict(params, hourly=",".join(variables), timezone="GMT", wind_speed_unit="ms")
        r = requests.get(url, params=p, timeout=120)
        if r.status_code == 400 and variables:
            reason = r.json().get("reason", "")
            bad = [v for v in variables if re.search(rf"\b{v}\b", reason)]
            if bad:
                print(f"   Open-Meteo: '{bad[0]}' not available here, continuing without it")
                variables.remove(bad[0])
                continue
        if r.status_code == 429:
            time.sleep(30 * (attempt + 1))
            continue
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame(h)
        df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
        # JSON nulls (e.g. a variable missing for part of the archive) make object columns
        for c in df.columns.drop("time"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.rename(columns=RENAME)
    raise RuntimeError(f"Open-Meteo request kept failing: {url}")


def _yearly_chunks(start: str, end: str):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    while s <= e:
        ce = min(s + pd.DateOffset(years=1) - pd.Timedelta(days=1), e)
        yield s.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")
        s = ce + pd.Timedelta(days=1)


def fetch_weather_history(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Hourly archived weather forecasts (closest match to what the live model will see)."""
    parts = []
    for s, e in _yearly_chunks(start, end):
        print(f"   Open-Meteo weather {s} -> {e}")
        try:
            parts.append(_request(HIST_URL, {"latitude": lat, "longitude": lon,
                                             "start_date": s, "end_date": e}, WEATHER_VARS))
        except Exception as ex:                       # fall back to ERA5 reanalysis
            print(f"   historical-forecast API failed ({ex}); using ERA5 archive")
            parts.append(_request(ARCHIVE_URL, {"latitude": lat, "longitude": lon,
                                                "start_date": s, "end_date": e}, WEATHER_VARS))
    return pd.concat(parts).drop_duplicates("time").reset_index(drop=True)


def fetch_weather_forecast(lat: float, lon: float, past_days: int = 5, days: int = 4) -> pd.DataFrame:
    return _request(FCST_URL, {"latitude": lat, "longitude": lon,
                               "past_days": past_days, "forecast_days": days}, WEATHER_VARS)


def fetch_cams_history(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """CAMS global model PM2.5 (what most weather apps display) – used as a model input
    and as the 'generic app' benchmark."""
    parts = []
    for s, e in _yearly_chunks(max(start, "2022-08-01"), end):
        print(f"   CAMS PM2.5 {s} -> {e}")
        try:
            parts.append(_request(AQ_URL, {"latitude": lat, "longitude": lon,
                                           "start_date": s, "end_date": e}, ["pm2_5"]))
        except Exception as ex:
            print(f"   CAMS history unavailable for {s}..{e}: {ex}")
    if not parts:
        return pd.DataFrame(columns=["time", "cams_pm25"])
    return pd.concat(parts).drop_duplicates("time").reset_index(drop=True)


def fetch_cams_forecast(lat: float, lon: float, past_days: int = 5, days: int = 4) -> pd.DataFrame:
    return _request(AQ_URL, {"latitude": lat, "longitude": lon,
                             "past_days": past_days, "forecast_days": days}, ["pm2_5"])
