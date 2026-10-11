"""
Command-line pipeline (Kolkata).

    python pipeline.py fetch      # download PM2.5 (OpenAQ) + weather & CAMS (Open-Meteo), merge, QC
    python pipeline.py train      # train quantile models + backtest report
    python pipeline.py forecast   # issue a 60-hour forecast from the latest data
    python pipeline.py all        # all three
    python pipeline.py update     # quick refresh: latest readings + forecast (no retraining)

Add --demo to run everything on built-in synthetic data (no internet / API key needed).
"""
from __future__ import annotations

import argparse
import json
import pandas as pd

import config
from aq import dataset, model

def hourly_file():
    return config.PROC_DIR / "hourly.csv"


def meta_file():
    return config.PROC_DIR / "meta.json"


def _save_hourly(df: pd.DataFrame, meta: dict):
    df.to_csv(hourly_file(), index_label="time")
    meta_file().write_text(json.dumps(meta, indent=2, default=str))


def load_hourly() -> pd.DataFrame:
    df = pd.read_csv(hourly_file(), parse_dates=["time"]).set_index("time")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _combine(hist: pd.DataFrame, recent: pd.DataFrame) -> pd.DataFrame:
    """History first, then forecast-API rows for times the archive doesn't cover yet."""
    if hist is None or hist.empty:
        return recent
    recent = recent[recent["time"] > hist["time"].max()]
    return pd.concat([hist, recent], ignore_index=True)


# ------------------------------------------------------------------ fetch
def fetch(demo: bool = False, history: bool = True):
    if demo:
        from aq.demo import make_demo
        print("[fetch] DEMO mode: generating synthetic data")
        pm, weather, cams = make_demo(config.TRAIN_START)
        meta = {"mode": "demo", "city": config.CITY, "station": "Synthetic demo station",
                "sensor_id": None, "provider": "demo", "lat": config.LAT, "lon": config.LON,
                "tz": config.LOCAL_TZ}
    else:
        from aq import openaq, openmeteo
        sensors_file = config.RAW_DIR / "sensors.csv"
        print(f"[fetch] searching OpenAQ PM2.5 sensors within {config.SEARCH_RADIUS_M/1000:.0f} km of {config.CITY}")
        sensors = openaq.find_pm25_sensors(config.LAT, config.LON, config.SEARCH_RADIUS_M)
        if sensors.empty:
            raise SystemExit("No PM2.5 sensors found nearby - try another city or a bigger radius.")
        sensors.to_csv(sensors_file, index=False)
        pm, st, sensor_ids = _station_pm25(openaq, sensors, history)
        print(f"   station: '{st.location}' ({st.provider}), {(st.distance_m or 0)/1000:.1f} km away, "
              f"sensor(s) {sensor_ids}; {len(sensors)} candidates in {sensors_file.name}")

        wfile, cfile = config.RAW_DIR / "weather_history.csv", config.RAW_DIR / "cams_history.csv"
        hist_end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        if history or not wfile.exists():
            print("[fetch] Open-Meteo weather + CAMS history")
            wh = openmeteo.fetch_weather_history(config.LAT, config.LON, config.TRAIN_START, hist_end)
            ch = openmeteo.fetch_cams_history(config.LAT, config.LON, config.TRAIN_START, hist_end)
            wh.to_csv(wfile, index=False)
            ch.to_csv(cfile, index=False)
        else:
            wh = pd.read_csv(wfile, parse_dates=["time"])
            ch = pd.read_csv(cfile, parse_dates=["time"])
            wh["time"] = pd.to_datetime(wh["time"], utc=True)
            ch["time"] = pd.to_datetime(ch["time"], utc=True)
            # Top up the saved archive to two days ago. Without this, an update run several days
            # after the last one leaves a gap between the archive and the 5 past days of the
            # forecast API, i.e. hours with no weather or CAMS at all.
            wh, ch = _top_up(wh, wfile, openmeteo.fetch_weather_history, hist_end), \
                     _top_up(ch, cfile, openmeteo.fetch_cams_history, hist_end)
        print("[fetch] Open-Meteo latest weather + CAMS forecast")
        weather = _combine(wh, openmeteo.fetch_weather_forecast(config.LAT, config.LON))
        cams = _combine(ch, openmeteo.fetch_cams_forecast(config.LAT, config.LON))
        meta = {"mode": "live", "city": config.CITY, "station": st.location, "sensor_id": sensor_ids,
                "location_id": int(st.location_id),
                "provider": st.provider, "lat": float(st.lat), "lon": float(st.lon),
                "tz": config.LOCAL_TZ}

    series, qc = dataset.clean_pm25(pm)
    df = dataset.merge_sources(series, weather, cams)
    df, freshness = dataset.bridge_gap(df)
    meta.update({"qc": qc, "fetched_at": pd.Timestamp.now(tz="UTC"), "freshness": freshness,
                 "last_observation": series.last_valid_index()})
    if freshness["gap_hours"] > 6:
        print(f"   WARNING: station's latest reading is {freshness['gap_hours']:.0f} h old "
              f"({freshness['last_observation']}).")
        if freshness.get("bridged"):
            print(f"   Bridging the gap with CAMS x {freshness['bridge_ratio']:.2f} so forecasts start from now.")
        elif freshness.get("note"):
            print(f"   {freshness['note']}")
        if not demo:
            _suggest_fresh_sensors()
    _save_hourly(df, meta)
    print(f"[fetch] {len(df):,} hourly rows -> {hourly_file().relative_to(config.ROOT)}  | QC: {qc}")
    return df


def _top_up(hist: pd.DataFrame, path, fetcher, hist_end: str) -> pd.DataFrame:
    """Append archive days missing between the saved history and `hist_end`, and save."""
    start = (hist["time"].max() + pd.Timedelta(hours=1)).strftime("%Y-%m-%d") if not hist.empty \
        else config.TRAIN_START
    if start > hist_end:
        return hist
    print(f"[fetch] topping up saved history {start} -> {hist_end}")
    try:
        new = fetcher(config.LAT, config.LON, start, hist_end)
    except Exception as e:                                  # keep going; the forecast API still covers 5 days
        print(f"   could not top up history: {e}")
        return hist
    out = pd.concat([hist, new], ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], utc=True)
    out = out.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
    out.to_csv(path, index=False)
    return out


def _fetch_location(openaq, loc_sensors: pd.DataFrame, history: bool) -> pd.DataFrame:
    """All PM2.5 sensors of one station, stitched into one series. OpenAQ often has an
    old sensor id (years of history) and a new one (recent data) for the same monitor;
    where both report the same hour, the newer sensor wins. Saved incrementally."""
    loc_id = int(loc_sensors.location_id.iloc[0])
    f = config.RAW_DIR / f"pm25_location_{loc_id}.csv"
    old = pd.read_csv(f) if f.exists() else None
    now = pd.Timestamp.now(tz="UTC")
    parts = [] if old is None else [old]
    for _, sn in loc_sensors.sort_values("sensor_id").iterrows():
        if old is not None and not old.empty and (old.sensor_id == sn.sensor_id).any():
            last = pd.to_datetime(old.loc[old.sensor_id == sn.sensor_id, "time"], utc=True).max()
            start = last - pd.Timedelta(days=3)
        else:
            start = pd.Timestamp(config.TRAIN_START, tz="UTC") if history else now - pd.Timedelta(days=10)
        if sn.get("last") is not None and pd.notna(sn["last"]) and pd.Timestamp(sn["last"]) < start:
            continue
        d = openaq.fetch_sensor_hours(int(sn.sensor_id), start.strftime("%Y-%m-%dT%H:%M:%SZ"))
        if not d.empty:
            parts.append(d.assign(sensor_id=int(sn.sensor_id)))
    if not parts:
        return pd.DataFrame(columns=["time", "pm25", "coverage", "sensor_id"])
    pm = pd.concat(parts, ignore_index=True)
    pm["time"] = pd.to_datetime(pm["time"], utc=True)
    pm = (pm.sort_values(["time", "sensor_id"])          # newer (larger) sensor id last -> kept
            .drop_duplicates("time", keep="last").reset_index(drop=True))
    pm.to_csv(f, index=False)
    return pm


def _station_pm25(openaq, sensors: pd.DataFrame, history: bool):
    """Pick the best station that actually has data: ranked stations are tried in turn
    until one gives at least 30 days of readings with the latest within 14 days.
    A station used before is tried first, so the history stays consistent."""
    sensors = sensors.copy()
    sensors["last"] = pd.to_datetime(sensors["last"], utc=True, format="mixed", errors="coerce")
    order = sensors.drop_duplicates("location_id")["location_id"].tolist()
    # if a station was used before, try it first so history stays consistent
    meta = meta_file()
    if meta.exists():
        m = json.loads(meta.read_text())
        prev = m.get("location_id")
        if prev in order:
            order.remove(prev)
            order.insert(0, prev)
    now = pd.Timestamp.now(tz="UTC")
    best = None
    for loc_id in order[:6]:
        ls = sensors[sensors.location_id == loc_id]
        st = ls.iloc[0]
        print(f"[fetch] OpenAQ hourly PM2.5 for '{st.location}' (sensors {ls.sensor_id.tolist()})")
        pm = _fetch_location(openaq, ls, history)
        n = int(pm["pm25"].notna().sum()) if not pm.empty else 0
        recent = (not pm.empty) and (now - pm["time"].max()) < pd.Timedelta(days=14)
        if n >= (30 * 24 if history else 24) and recent:
            return pm, st, ls.sensor_id.astype(int).tolist()
        print(f"   only {n} hours{'' if recent else ', nothing recent'} - trying the next station")
        if n and (best is None or n > best[3]):
            best = (pm, st, ls.sensor_id.astype(int).tolist(), n)
    if best:
        print("   no station met the data requirements; using the one with the most data")
        return best[:3]
    raise SystemExit("None of the nearby OpenAQ stations returned PM2.5 data for this period. "
                     "Try again later (the feed may be down).")


def _suggest_fresh_sensors():
    f = config.RAW_DIR / "sensors.csv"
    if f.exists():
        s = pd.read_csv(f, parse_dates=["first", "last"]).sort_values("last", ascending=False).head(5)
        print("   Most recently reporting PM2.5 sensors nearby:")
        for _, r in s.iterrows():
            print(f"     {r.sensor_id:>10}  {str(r.location)[:40]:40}  last {r['last']:%Y-%m-%d %H:%M}  "
                  f"since {r['first']:%Y-%m-%d}")


# ------------------------------------------------------------------ train / forecast
def train():
    print("[train] building samples and fitting quantile models")
    rep = model.train_and_backtest(load_hourly())
    for k in ("summary_0-24h", "summary_24-48h"):
        s = rep[k]
        print(f"   {k[8:]}: MAE model {s['mae_model']:.1f} | persistence {s['mae_persistence']:.1f} | "
              f"CAMS {s['mae_cams']:.1f} µg/m³ | skill vs persistence {s['skill_vs_persistence']:+.0%}")
    print(f"   alerts @90 µg/m³ (24-48 h lead): {rep['alerts'].get('90')}")
    return rep


def run_forecast():
    df = load_hourly()
    if "pm25_bridge" in df:                       # estimated hours only extend the input, never train
        df["pm25"] = df["pm25"].combine_first(df["pm25_bridge"])
    bundle = model.load_bundle()
    fc = model.forecast(bundle, df)
    drivers = model.driver_attribution(bundle, df, fc)
    fc.to_csv(config.REPORT_DIR / "forecast.csv", index=False)
    drivers.to_csv(config.REPORT_DIR / "drivers.csv", index=False)
    now = pd.Timestamp.now(tz="UTC")
    last = fc["issue_time"].iloc[0]
    print(f"[forecast] issued from last observation {last} ({(now - last).total_seconds()/3600:.1f} h ago)")
    peak = fc.loc[fc["q50"].idxmax()]
    print(f"   peak expected {peak.q50:.0f} µg/m³ (80% range {peak.q10:.0f}-{peak.q90:.0f}) at "
          f"{peak.target_time.tz_convert(config.LOCAL_TZ):%a %d %b %H:%M}")
    return fc


def run(command: str, demo: bool = False):
    print(f"========== {config.CITY} ==========")
    if command in ("fetch", "all"):
        fetch(demo=demo, history=True)
    if command == "update":                   # light refresh: latest obs + forecasts only
        fetch(demo=demo, history=False)
    if command in ("train", "all"):
        train()
    if command in ("forecast", "update", "all"):
        run_forecast()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "train", "forecast", "update", "all"])
    ap.add_argument("--demo", action="store_true", help="use synthetic data (offline)")
    a = ap.parse_args()
    run(a.command, a.demo)


if __name__ == "__main__":
    main()
