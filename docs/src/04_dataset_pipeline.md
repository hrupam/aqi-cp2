
---

## 7. `aq/dataset.py`: quality control, merging, outage bridging

Raw station data is messy. This module turns it into one clean, gap-aware hourly table.

### 7.1 `clean_pm25(pm, min_coverage=50) → (series, report)`
Steps, in order:
1. **Floor timestamps to the hour and average duplicates.** Readings stamped `13:30` and `13:00` both become `13:00` and are averaged.
2. **Flag bad readings.** Four rules mark readings as missing; nothing is deleted outright.

   | Rule | Condition | Why |
   |---|---|---|
   | Out of range | `pm25 ≤ 0` or `> 1000` | Physically impossible or a sensor fault |
   | Low coverage | `coverage < 50 %` | The "hourly average" came from too few sub-hourly readings |
   | Stuck sensor | the same value for **≥ 6 consecutive hours** | A frozen logger repeats its last value. Real PM2.5 almost never stays exactly constant that long |
   | Isolated spike | `> 4 ×` the centred 25-hour rolling median **and** `> 250` | One-off glitches. Both conditions are required, so genuine high episodes, which rise gradually, survive |

   *How the stuck-sensor rule works:* `run_id = (s.diff() != 0).cumsum()` gives every run of equal consecutive values the same id. `groupby(run_id).transform("size")` then gives each reading the length of its run.
3. **Stop if nothing is left.** If every reading is now missing, the function stops with a clear message.
4. **Reindex to a complete hourly grid**, from the first to the last reading.
5. **Interpolate only short gaps** (`limit=3`, `limit_area="inside"`). A missing hour or two is filled with a straight line; longer gaps stay empty. Inventing hours of pollution would teach the model fake dynamics, and gradient boosting handles missing values natively anyway.
6. **Return the report** used by the dashboard's data-quality panel:
   * `raw_hours`
   * counts for each rule
   * `grid_hours`, `gaps_filled`, `still_missing`, `completeness_pct`

### 7.2 `merge_sources(pm, weather, cams) → DataFrame`
* Converts all inputs to UTC-indexed, de-duplicated, sorted, numeric frames. The `apply(pd.to_numeric)` step guards against object columns, and weather columns that are entirely empty are dropped.
* Builds a single hourly grid from the **first PM2.5 reading** to the **later of** the last PM2.5 reading and the last weather hour. The grid therefore extends about 4 days into the future, where `pm25` is NaN but weather and CAMS forecasts exist. That future part is what the forecast uses.
* Left-joins weather and CAMS onto the grid.
* Interpolates weather gaps of up to 6 hours in both directions (weather is smooth), and CAMS gaps of up to 6 hours inside the series only.

### 7.3 `bridge_gap(df, now, min_gap_h=6, max_gap_days=14) → (df, info)`
This handles the real-world problem of a station going offline.
* It always adds a `pm25_bridge` column, set to NaN by default.
* `gap_hours` is the time from the last real reading to now.
* **The gap is not bridged when:**
  * the gap is under 6 h (normal reporting delay)
  * there is no CAMS data
  * the gap is over 14 days; `info["note"]` records that the station has been offline too long, because CAMS alone is too uncertain that long
  * there are fewer than 24 overlapping hours in the last week
* **Otherwise:**
  1. Compute the station/CAMS ratio over the last 7 days in **log space**: `ratio = exp(median(log1p(pm) − log1p(cams)))`. The median makes it robust to outliers, and log space makes it a multiplicative correction.
  2. Fill `pm25_bridge` for the gap hours with `cams × ratio` (computed in log1p space, clipped at 0).
* `info` records `last_observation`, `gap_hours`, `bridged`, `bridge_ratio` and `bridged_until`. It is saved in `meta.json` and shown in the dashboard.
* **The estimates are kept separate on purpose.** Training reads only `pm25`, so fabricated values never become training targets. `run_forecast` merges them in only to give the model a starting point near *now*.

---

## 8. `pipeline.py`: the command-line orchestrator

### 8.1 Small helpers
* **`hourly_file()`, `meta_file()`** return the paths of `hourly.csv` and `meta.json` in `data/kolkata/processed/`.
* **`_save_hourly(df, meta)`** writes `hourly.csv` (index label `time`) and `meta.json` (`default=str` serialises timestamps).
* **`load_hourly()`** reads `hourly.csv` back with a UTC `DatetimeIndex`.
* **`_combine(hist, recent)`** appends forecast-API rows only **after** the last archived hour. The archive is preferred wherever both exist, so training-like data is used for the past.

### 8.2 `_fetch_location(openaq, loc_sensors, history)`: join one station's sensors
Problem: OpenAQ often lists the same CPCB monitor under two sensor IDs.

| | Example (Anand Vihar) | Data it holds |
|---|---|---|
| Old sensor ID | `384` | History |
| New sensor ID | `12235610` | Current readings |

Picking the wrong one gives an empty series. The fix is to treat the **station** as the unit.

1. **Load existing data.** The station's saved file `pm25_location_<id>.csv` is read if present (incremental mode).
2. **Download each sensor**, in sensor-ID order:
   * **Start time:**
     * the sensor is already in the file → 3 days before its last saved hour (overlap catches late corrections)
     * otherwise, a full build (`history=True`) → `TRAIN_START`
     * otherwise, an update → 10 days ago
   * **Skip check:** a sensor whose `last` date is before the start time is skipped.
   * **Download:** `openaq.fetch_sensor_hours`, tagging each row with its `sensor_id`.
3. **Join:** `sort_values(["time","sensor_id"]).drop_duplicates("time", keep="last")`. Where two sensors report the same hour, the **larger (newer) sensor ID wins**.
4. **Save** the file and return the series.

### 8.3 `_station_pm25(openaq, sensors, history) → (pm, station_row, sensor_ids)`
This is the automatic station-selection policy.
1. **Order:** unique `location_id`s in rank order from `find_pm25_sensors`.
2. **Stickiness:** if `meta.json` shows a station used last time, that station goes first. History stays consistent, and the model isn't retrained on a different station by surprise.
3. **Try up to 6 stations** with `_fetch_location`. Accept the first with **≥ 30 days** of readings (≥ 24 h for an update) **and** a reading within the last **14 days**.
4. **Fallback:** if none qualifies, use the one with the most data. If none has any data, stop with a clear message.

### 8.4 `fetch(demo=False, history=True)`
* **Demo mode** builds data with `aq.demo.make_demo` and writes a `meta.json` marked `"mode": "demo"`.
* **Live mode:**
  1. List the sensors and save `sensors.csv`.
  2. Choose the station and get its PM2.5 (`_station_pm25`).
  3. **Weather and CAMS history up to two days ago** (the archive lags about 2 days):
     * downloaded when `history=True` or when no saved file exists
     * otherwise read from `weather_history.csv` and `cams_history.csv`
  4. Fetch the recent and future weather and CAMS, and join them with `_combine`.
  5. **Build `meta`** with these fields: `mode`, `city`, `station`, `sensor_id` (list), `location_id`, `provider`, station `lat`/`lon`, `tz`.
* **Both modes then run:**
  1. `clean_pm25` → `merge_sources` → `bridge_gap`.
  2. Add `qc`, `fetched_at`, `freshness` and `last_observation` to `meta`.
  3. If the station is late (more than 6 h), print a warning and list the 5 most recently reporting sensors (`_suggest_fresh_sensors`).
  4. Save everything.

### 8.5 `train()`, `run_forecast()`, `run()`, `main()`
* **`train()`** calls `model.train_and_backtest(load_hourly())` and prints a summary: MAE against persistence and CAMS, skill score, and alert statistics.
* **`run_forecast()`**:
  1. Loads the data and fills outage hours: `pm25 = pm25.combine_first(pm25_bridge)`.
  2. Loads the model and calls `model.forecast` and `model.driver_attribution`.
  3. Writes the CSVs and prints the forecast peak.
* **`run(command, demo)`** maps commands to steps:

  | Command | Steps |
  |---|---|
  | `fetch` | fetch with history |
  | `update` | fetch without history, then forecast |
  | `train` | train |
  | `forecast` | forecast |
  | `all` | fetch with history, train, forecast |

* **`main()`** reads the command and the optional `--demo` flag, then calls `run`.
