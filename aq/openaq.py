"""Download hourly PM2.5 from the OpenAQ v3 API."""
from __future__ import annotations

import time
import requests
import pandas as pd

import config

BASE = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2


def _get(path: str, params: dict | None = None, retries: int = 6) -> dict:
    if not config.OPENAQ_API_KEY:
        raise RuntimeError(
            "OPENAQ_API_KEY is not set. Get a free key at https://explore.openaq.org "
            "(Account -> Settings) and put it in a .env file:  OPENAQ_API_KEY=xxxx"
        )
    headers = {"X-API-Key": config.OPENAQ_API_KEY}
    for attempt in range(retries):
        r = requests.get(BASE + path, params=params, headers=headers, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:      # rate-limited / server hiccup
            wait = int(r.headers.get("x-ratelimit-reset", 0) or 0) or 2 ** attempt * 5
            time.sleep(min(wait, 60))
            continue
        r.raise_for_status()
        time.sleep(1.1)        # stay under 60 requests / minute
        return r.json()
    r.raise_for_status()
    return {}


def find_pm25_sensors(lat: float, lon: float, radius_m: int = 25_000) -> pd.DataFrame:
    """List PM2.5 sensors near a point, ranked by how much history they have."""
    js = _get("/locations", {
        "coordinates": f"{lat},{lon}", "radius": radius_m,
        "parameters_id": PM25_PARAMETER_ID, "limit": 100,
    })
    rows = []
    for loc in js.get("results", []):
        for s in loc.get("sensors", []):
            if s.get("parameter", {}).get("id") != PM25_PARAMETER_ID:
                continue
            first = (loc.get("datetimeFirst") or {}).get("utc")
            last = (loc.get("datetimeLast") or {}).get("utc")
            rows.append({
                "sensor_id": s["id"], "location_id": loc["id"], "location": loc.get("name"),
                "provider": (loc.get("provider") or {}).get("name"),
                "is_monitor": loc.get("isMonitor"),
                "lat": loc["coordinates"]["latitude"], "lon": loc["coordinates"]["longitude"],
                "distance_m": loc.get("distance"), "first": first, "last": last,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["first"] = pd.to_datetime(df["first"], utc=True)
    df["last"] = pd.to_datetime(df["last"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df["active"] = (now - df["last"]) < pd.Timedelta(days=3)
    df["years"] = (df["last"] - df["first"]).dt.days / 365.25
    # prefer active, reference-grade monitors with long history, then closest
    return df.sort_values(["active", "is_monitor", "years", "distance_m"],
                          ascending=[False, False, False, True]).reset_index(drop=True)


def fetch_sensor_hours(sensor_id: int, start: str, end: str | None = None,
                       chunk_days: int = 120) -> pd.DataFrame:
    """Hourly-averaged PM2.5 for a sensor between start and end (UTC), in chunks
    so that long histories don't hit the API's request timeout."""
    def _utc(t):
        t = pd.Timestamp(t)
        return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    t0 = _utc(start)
    t1 = _utc(end) if end else pd.Timestamp.now(tz="UTC")
    parts = []
    while t0 < t1:
        t_next = min(t0 + pd.Timedelta(days=chunk_days), t1)
        parts.append(_fetch_hours_range(sensor_id, t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                        t_next.strftime("%Y-%m-%dT%H:%M:%SZ")))
        t0 = t_next
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["time", "pm25", "coverage"])
    df = pd.concat(parts)
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def _fetch_hours_range(sensor_id: int, start: str, end: str) -> pd.DataFrame:
    out, page, limit = [], 1, 1000
    while True:
        try:
            js = _get(f"/sensors/{sensor_id}/hours", {
                "datetime_from": start, "datetime_to": end, "limit": limit, "page": page})
        except requests.HTTPError as e:            # some deployments cap limit at 100
            if e.response is not None and e.response.status_code == 422 and limit != 100:
                limit, page, out = 100, 1, []
                continue
            raise
        res = js.get("results", [])
        for m in res:
            out.append({
                "time": m["period"]["datetimeFrom"]["utc"],
                "pm25": m.get("value"),
                "coverage": (m.get("coverage") or {}).get("percentComplete"),
            })
        print(f"   OpenAQ sensor {sensor_id} {start[:10]}..{end[:10]}: {len(out)} hours", end="\r")
        if len(res) < limit:
            break
        page += 1
    print()
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
