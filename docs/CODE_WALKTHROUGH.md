# Kolkata PM2.5 Forecasting & Early Warning System: Project and Code Walkthrough (Checkpoint 2)

This document explains the checkpoint-2 build of the project:
* what it does and why
* how the data flows
* every file and every function in the code
* the machine-learning design, including why gradient boosting was chosen

It is written so that someone who has never seen the code can understand it, demonstrate it and defend it in a viva.

**Scope of this build.** It is a focused subset of the full system:

| Included | Not included in this build |
|---|---|
| Kolkata, with the monitoring station chosen automatically | Other cities, place search, the city comparison view |
| Data collection, cleaning and merging (OpenAQ + Open-Meteo + CAMS) | Choosing the station by hand |
| The 60-hour ML forecast with uncertainty and probabilities | Telegram / e-mail alert notifications |
| The early-warning dashboard: warning levels, 48-hour strip, daily outlook, best outdoor time, explanations, track record, patterns | |

The code has 12 Python files and about 1,900 lines.


## Contents

- [1. The problem](#1-the-problem)
- [2. The system at a glance](#2-the-system-at-a-glance)
- [3. What happens when you run it (end-to-end flows)](#3-what-happens-when-you-run-it-end-to-end-flows)
- [4. Data sources in detail](#4-data-sources-in-detail)
- [5. `config.py`: settings](#5-config-py-settings)
- [6. Data clients](#6-data-clients)
- [7. `aq/dataset.py`: quality control, merging, outage bridging](#7-aq-dataset-py-quality-control-merging-outage-bridging)
- [8. `pipeline.py`: the command-line orchestrator](#8-pipeline-py-the-command-line-orchestrator)
- [9. Machine-learning design, and why gradient boosting](#9-machine-learning-design-and-why-gradient-boosting)
- [10. `aq/features.py`: feature engineering](#10-aq-features-py-feature-engineering)
- [11. `aq/model.py`: training, calibration, evaluation, forecasting, explanation](#11-aq-model-py-training-calibration-evaluation-forecasting-explanation)
- [12. `aq/insights.py`: from numbers to decisions](#12-aq-insights-py-from-numbers-to-decisions)
- [13. `aq/ui.py`: the visual layer](#13-aq-ui-py-the-visual-layer)
- [14. `app.py`: the Streamlit dashboard](#14-app-py-the-streamlit-dashboard)
- [15. `aq/demo.py`: synthetic data for offline demos](#15-aq-demo-py-synthetic-data-for-offline-demos)
- [16. Files the system produces (data contracts)](#16-files-the-system-produces-data-contracts)
- [17. Evaluation and results](#17-evaluation-and-results)
- [18. Design decisions and the bugs that shaped them](#18-design-decisions-and-the-bugs-that-shaped-them)
- [19. Limitations and possible extensions](#19-limitations-and-possible-extensions)
- [20. Glossary](#20-glossary)
- [21. Likely viva questions, with answers](#21-likely-viva-questions-with-answers)

---

## 1. The problem

### 1.1 What PM2.5 is and why it matters
PM2.5 is the mass concentration of airborne particles smaller than 2.5 micrometres, measured in µg/m³ (micrograms per cubic metre).
* The particles are small enough to reach deep into the lungs and the bloodstream.
* Long-term exposure is linked to heart disease, stroke, lung cancer and respiratory illness.
* In Kolkata, as in most Indian cities, PM2.5 is usually the pollutant that drives the Air Quality Index (AQI). This is especially true in winter, when cold, still air traps emissions near the ground.

**Standards used in this project:**

| Reference | 24-hour PM2.5 limit |
|---|---|
| India NAAQS (National Ambient Air Quality Standard) | 60 µg/m³ |
| WHO 2021 guideline | 15 µg/m³ |

### 1.2 Why forecast it?
Pollution episodes are largely driven by weather:
* low wind speed
* a shallow mixing layer (the "lid" of the atmosphere)
* no rain
* winds that bring smoke from the Indo-Gangetic plain

Weather can be forecast a few days ahead, and the station's recent readings show where pollution stands now. Together they make PM2.5 predictable 1–2 days in advance. That gives people time to act:
* reschedule a run
* keep a child with asthma indoors
* close windows and switch on a purifier

### 1.3 The project's three objectives
1. **Collect and combine data**: historical PM2.5 from OpenAQ, plus weather (temperature, wind, humidity and more) from Open-Meteo.
2. **Predict future air quality**: a machine-learning model that learns how weather and past pollution interact, and forecasts PM2.5 24–48 hours ahead.
3. **Early warning**: a visual dashboard that highlights when pollution will cross safe thresholds and issues warnings in advance.

### 1.4 What "Google Weather doesn't provide"
Google Weather and most weather apps show an AQI that comes from a **global chemistry model**, typically CAMS (Copernicus Atmosphere Monitoring Service), on a grid of roughly 10–45 km. They give:
* no uncertainty
* no probability
* no early warning
* no explanation
* no personal thresholds
* no accuracy record

This project adds all of these. Section 2.4 lists each gap and the code that fills it.

---

## 2. The system at a glance

### 2.1 Architecture

```
                 ┌──────────────────────── DATA SOURCES ────────────────────────┐
                 │ OpenAQ v3 API          Open-Meteo APIs                       │
                 │  • station list         • historical-forecast (weather)       │
                 │  • hourly PM2.5         • forecast (weather, next 4 days)     │
                 │                         • air-quality (CAMS PM2.5)            │
                 └───────────┬───────────────────────┬──────────────────────────┘
                             │ aq/openaq.py          │ aq/openmeteo.py
                             ▼                       ▼
   pipeline.py fetch ─► automatic station choice + sensor stitching ─► data/kolkata/raw/
                             │
                             ▼  aq/dataset.py
                  quality control ─► hourly merge (one UTC grid) ─► outage bridging
                             │                          data/kolkata/processed/hourly.csv
                             ▼  aq/features.py
                  (issue time T, lead time h) samples ─► scale-free design matrix
                             │
                             ▼  aq/model.py
   pipeline.py train ─► 3 quantile gradient-boosting models (q10, q50, q90)
                        ─► backtest ─► blending weights ─► conformal band scaling
                        ─► models/kolkata/pm25_quantile_gbm.joblib, reports/kolkata/backtest.json
                             │
   pipeline.py forecast ─► 60-hour forecast + driver attribution
                        ─► reports/kolkata/forecast.csv, drivers.csv
                             │
                             ▼
             app.py (Streamlit dashboard), using aq/insights.py and aq/ui.py
```

### 2.2 Folder layout

| Path | What it holds |
|---|---|
| `config.py` | Settings: Kolkata's coordinates, folders, thresholds, model settings |
| `pipeline.py` | Command-line entry point: fetch, train, forecast, update, all |
| `app.py` | Streamlit dashboard |
| `aq/openaq.py` | OpenAQ API client |
| `aq/openmeteo.py` | Open-Meteo weather and CAMS client |
| `aq/dataset.py` | Quality control, merging, outage bridging |
| `aq/features.py` | Feature engineering and the model's design matrix |
| `aq/model.py` | Training, calibration, evaluation, forecasting, explanations |
| `aq/insights.py` | AQI, warning levels, daily outlook, best windows, advice |
| `aq/ui.py` | Dashboard look: CSS, Plotly template, header strip, tiles |
| `aq/demo.py` | Synthetic offline data |
| `data/kolkata/raw/` | `sensors.csv`, `pm25_location_<id>.csv`, `weather_history.csv`, `cams_history.csv` |
| `data/kolkata/processed/` | `hourly.csv`, `meta.json` |
| `models/kolkata/` | `pm25_quantile_gbm.joblib` |
| `reports/kolkata/` | `forecast.csv`, `drivers.csv`, `backtest.json`, `backtest_predictions.csv` |
| `.streamlit/config.toml` | Dashboard theme |
| `.env` | OpenAQ API key (never committed) |

### 2.3 Technology choices

| Need | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | The standard for data science; every library needed exists |
| Tables and time series | pandas, numpy | Time-indexed resampling, rolling windows, joins and time-zone handling are first-class |
| Machine learning | scikit-learn `HistGradientBoostingRegressor` | Gradient-boosted trees with native quantile loss and native missing-value support; no extra dependency (see section 9) |
| Model storage | joblib | Fast, standard way to save scikit-learn objects |
| HTTP | requests | Simple, reliable API calls |
| Dashboard | Streamlit | A full interactive web app in pure Python; re-runs the script on every interaction |
| Charts | Plotly | Interactive (hover, zoom), works natively in Streamlit, has a polar chart type for the pollution rose |
| Station data | OpenAQ v3 | The largest open archive of government monitor data (including CPCB) with one API |
| Weather | Open-Meteo | Free, no key, hourly, has *archived forecasts* and mixing height |
| Regional model | CAMS via Open-Meteo | Gives the large-scale pollution signal and is the baseline to beat |

### 2.4 Gaps in weather apps and where the code fills them

| Gap | Where it's solved |
|---|---|
| Not your station | `pipeline._station_pm25`, `_fetch_location` (station choice); `features.design` (anchored to the station's current level) |
| No uncertainty | Quantile models in `model.py` (q10/q50/q90) plus `interval_scale` (calibration) |
| No probability | `model.exceed_prob` (split-normal in log space) |
| No early warning | `insights.alert_summary` (Watch/Warning/Emergency, start time, lead time, duration, peak) |
| One scale for everyone | `config.PROFILES` and the sidebar threshold slider |
| US AQI | `insights.sub_index`, `aqi`, `aqi_category` (CPCB), `config.NAAQS_24H`, `WHO_24H` |
| No explanation | `model.driver_attribution` (counterfactual), `permutation_importance`, `insights.driver_sentences` |
| No planning help | `insights.best_windows`, `daily_outlook` (× WHO, cigarettes) |
| No accuracy record | `model.evaluate` (MAE by lead time, POD/FAR/CSI, Brier, reliability), Track record tab |
| No local analysis | Patterns tab (heatmap, pollution rose, weekday effect, days above the limit) |


---

## 3. What happens when you run it (end-to-end flows)

### 3.1 `python pipeline.py all`
1. **Configuration loads** (`import config`): it reads `.env` into environment variables, fixes Kolkata's coordinates and time zone, and creates the four folders `data/kolkata/raw`, `data/kolkata/processed`, `models/kolkata` and `reports/kolkata`.
2. `main()` parses the command (and `--demo` if given).
3. `run("all")` prints a header and calls `fetch(history=True)`, `train()` and `run_forecast()` in that order.
4. **`fetch`**:
   1. `openaq.find_pm25_sensors` lists every PM2.5 sensor within 25 km and saves `sensors.csv`.
   2. `_station_pm25` chooses a station. It tries stations in ranked order and downloads all sensors of each station with `_fetch_location`, until one has at least 30 days of data and a reading in the last 14 days.
   3. Weather and CAMS history (2023 to two days ago) are downloaded year by year, then joined with the 5-days-past / 4-days-ahead forecast data (`_combine`).
   4. `dataset.clean_pm25` runs quality control. `dataset.merge_sources` puts everything on one hourly grid. `dataset.bridge_gap` fills a recent station outage with an estimate.
   5. `hourly.csv` and `meta.json` are written.
5. **`train`** → `model.train_and_backtest`:
   1. Builds about 300,000–1,000,000 (issue time, lead time) samples.
   2. Holds out the last 25 % of the timeline and trains three quantile models on the rest.
   3. Tunes the blending weights and the band calibration on alternate weeks of the held-out period, and scores on the remaining weeks.
   4. Computes permutation importance.
   5. Refits on all the data and saves the model bundle and `backtest.json`.
6. **`run_forecast`**:
   1. Loads `hourly.csv` and fills any outage with the bridge estimate.
   2. Builds the 60 samples for "now" and predicts q10/q50/q90.
   3. Computes the driver attribution.
   4. Writes `forecast.csv` and `drivers.csv`.

### 3.2 `python pipeline.py update`
`update` does the same as `all` **without** retraining and without re-downloading the weather history:
* `fetch(history=False)` reuses the saved weather and CAMS history files.
* OpenAQ is asked only for the last few days per sensor (incremental download).
* Then `run_forecast()` runs.

It takes about a minute. This is what the dashboard's **Refresh** button runs.

### 3.3 Opening the dashboard (`streamlit run app.py`)
Streamlit runs `app.py` from top to bottom **on every interaction**. Each run:
1. **Checks that a forecast exists.** If `pipeline.py all` hasn't been run yet, the page says so and stops.
2. **`load_all`** reads the CSV/JSON files. It is cached and only reloads when the files' modification time changes.
3. **Sidebar:** the user's profile, warning threshold and outdoor-window length, plus the **Refresh** button.
4. **`insights.alert_summary`, `upcoming` and `daily_outlook`** turn the forecast into decisions.
5. **`ui.hero`** draws the header with the 48-hour strip, and **`ui.tiles`** draws the four figures.
6. **Six tabs** show charts and tables.

### 3.4 When the station goes silent
CPCB feeds on OpenAQ sometimes stop for hours or days.
1. `bridge_gap` detects the gap (more than 6 hours).
2. If the gap is under 14 days, it estimates the missing hours as **CAMS × (the station/CAMS ratio over the last 7 days)**.
3. The estimates go into a separate column `pm25_bridge`:
   * `run_forecast` merges it into `pm25` so the forecast starts from *now*.
   * Training never sees it.
4. The dashboard shows a warning and plots the estimate as a dotted line.

---

## 4. Data sources in detail

### 4.1 OpenAQ v3 (station measurements)
* **Base URL** `https://api.openaq.org/v3`. Every request needs the header `X-API-Key`.
* **Concepts:**
  * A **location** is a monitoring site, for example "Victoria, Kolkata - WBPCB".
  * A **sensor** is one parameter measured at that location. PM2.5 is **parameter id 2**.
  * Many Indian locations have **two PM2.5 sensor IDs**: an old one (history up to about 2024) and a new one (current data). The code joins them (section 8.2).
* **Endpoints used:**
  * `GET /locations?coordinates=lat,lon&radius=25000&parameters_id=2&limit=100` returns locations near a point, each with its sensors, provider, `isMonitor`, `datetimeFirst`, `datetimeLast` and distance.
  * `GET /sensors/{id}/hours?datetime_from&datetime_to&limit&page` returns **hourly averages** (`value`, `period.datetimeFrom.utc`, `coverage.percentComplete`).
* **Limits:** about 60 requests per minute. The client sleeps 1.1 s after each call, and on HTTP 429 or 5xx it waits and retries.
* **Time:** all times are taken in **UTC**. Indian stations report on the half hour in UTC (for example 13:30 UTC = 19:00 IST), and the code floors them to the hour.

### 4.2 Open-Meteo (weather and CAMS): free, no key
| API | URL | Used for |
|---|---|---|
| Historical **forecast** | `historical-forecast-api.open-meteo.com/v1/forecast` | Training weather: archived short-range forecasts, about 2022 onward |
| ERA5 archive | `archive-api.open-meteo.com/v1/archive` | Fallback if the above fails |
| Forecast | `api.open-meteo.com/v1/forecast` | The last 5 days plus the next 4 days |
| Air quality (CAMS) | `air-quality-api.open-meteo.com/v1/air-quality` | CAMS PM2.5: history from Aug 2022 plus a 4-day forecast |

**Why archived forecasts rather than reanalysis for training?** At prediction time the model only has *forecast* weather, which contains forecast errors. ERA5 reanalysis is "perfect hindsight" weather. A model trained on it learns relationships that are too clean, and then performs worse live. Training on archived forecasts makes the inputs statistically like the live inputs. This is the principle known as *train–serve consistency*.

**Weather variables requested:**

| Variable | Column | Why it matters for PM2.5 |
|---|---|---|
| `temperature_2m` | `temp` | Cold nights mean inversions and more heating and burning emissions |
| `relative_humidity_2m` | `rh` | Humid air makes particles absorb water and grow, and helps secondary aerosol form |
| `dew_point_2m` | `dewpoint` | Moisture content; fog episodes |
| `wind_speed_10m` | `wind_speed` | Dilution and dispersion |
| `wind_direction_10m` | `wind_dir` | Where air comes from (for example the polluted Indo-Gangetic plain) |
| `wind_gusts_10m` | `wind_gust` | Turbulence |
| `surface_pressure` | `pressure` | High pressure means stagnant, sinking air |
| `precipitation` | `precip` | Rain washes particles out ("wet deposition") |
| `cloud_cover` | `cloud` | Affects heating of the ground and therefore mixing |
| `shortwave_radiation` | `radiation` | Daytime heating deepens the mixing layer |
| `boundary_layer_height` | `blh` | **The single most important meteorological factor**: the depth of air that pollution mixes into |

### 4.3 CAMS
CAMS is a physics-based global chemistry model run by ECMWF for the EU Copernicus programme. It simulates emissions, transport and chemistry.

**Strengths:**
* It knows about regional events, such as smoke being carried in from Punjab.
* It forecasts several days ahead.

**Weaknesses:**
* It is coarse (about 45 km over India).
* Its emission inventories are imperfect.
* It is often biased: the Kolkata error is about 12–14 µg/m³, against about 4 for our model.

In this project CAMS plays two roles:
1. **An input**: the model uses how much CAMS expects PM2.5 to *change*, which captures incoming regional pollution.
2. **A benchmark**: it is what weather apps show, so beating it is the point of the project.


---

## 5. `config.py`: settings

`config.py` is imported by every other module and holds all constants in one place.

### 5.1 `_load_dotenv()`
This is a tiny `.env` reader that avoids a dependency on `python-dotenv`.
* It reads `KEY=value` lines, skipping blank lines and `#` comments, and strips quotes.
* It uses `os.environ.setdefault`, so a real environment variable always wins over the file.

It runs at import, and then `OPENAQ_API_KEY` is read.

### 5.2 Constants
| Name | Value | Meaning |
|---|---|---|
| `ROOT` | folder of `config.py` | Project root; all paths are built from it |
| `OPENAQ_API_KEY` | from `.env` | Needed for station data |
| `CITY`, `LAT`, `LON` | Kolkata, 22.5726, 88.3639 | The city centre used to search for monitors and fetch weather |
| `LOCAL_TZ` | Asia/Kolkata | Used for hour-of-day features and every time shown to users |
| `SEARCH_RADIUS_M` | 25 000 | OpenAQ's maximum search radius |
| `RAW_DIR`, `PROC_DIR`, `MODEL_DIR`, `REPORT_DIR` | `data/kolkata/raw`, `data/kolkata/processed`, `models/kolkata`, `reports/kolkata` | Created at import if missing |
| `TRAIN_START` | `2023-01-01` (or `AQ_TRAIN_START`) | First day of history downloaded. Open-Meteo's archived forecasts and CAMS history start in 2022, so 2023 is safe |
| `MAX_HORIZON` | 60 | Forecast length in hours: 48 h of useful forecast plus 12 h of slack, because Indian stations often report hours late. If the last reading is 10 h old, a 60-h forecast still covers the next 48 h from *now* |
| `QUANTILES` | (0.1, 0.5, 0.9) | The 10th, 50th (median) and 90th percentiles forecast |
| `ISSUE_STRIDE` | 3 | Only every 3rd hour is used as a training "issue time". Neighbouring hours are nearly identical, so this cuts the data and training time by 3× with almost no loss of information |
| `TEST_FRACTION` | 0.25 | The last 25 % of time is held out for the backtest |
| `CATEGORIES` | CPCB PM2.5 bands with colours | Good ≤30, Satisfactory ≤60, Moderate ≤90, Poor ≤120, Very Poor ≤250, Severe above that |
| `NAAQS_24H`, `WHO_24H` | 60, 15 | 24-h standards |
| `PROFILES` | 5 profiles → threshold | General 90; Sensitive, Children & elderly, Outdoor workers 60; Strict (WHO-based) 35 |

The folder layout (`data/kolkata/…`, `models/kolkata/…`, `reports/kolkata/…`) is the same as the full multi-city version. Data and models built by either version can therefore be copied between them.

---

## 6. Data clients

### 6.1 `aq/openaq.py`: OpenAQ v3 client
Constants: `BASE = "https://api.openaq.org/v3"`, `PM25_PARAMETER_ID = 2`.

**`_get(path, params, retries=6)`** handles every call to OpenAQ.
1. Refuses to run without an API key, with a helpful message that says where to get one.
2. Sends the request with header `X-API-Key` and a 60 s timeout.
3. On **429 (rate-limited) or 5xx**, waits and retries, up to 6 attempts:
   * the wait is the `x-ratelimit-reset` header if present, otherwise exponential back-off (5 s, 10 s, 20 s…)
   * the wait is capped at 60 s
4. On other errors, raises (`raise_for_status`).
5. After every success it sleeps **1.1 s**, which keeps the client under about 55 requests per minute.

**`find_pm25_sensors(lat, lon, radius_m)`** builds the list of candidate sensors.
* Calls `/locations` with `coordinates`, `radius`, `parameters_id=2` and `limit=100`.
* Makes one row per PM2.5 sensor with these fields:
  * `sensor_id`, `location_id`, `location` (name), `provider` (for example CPCB, AirNow)
  * `is_monitor` (reference-grade vs low-cost), `lat`/`lon`, `distance_m`
  * `first` / `last` (the location's first and last data times)
* Adds two derived columns:
  * `active`: last data less than 3 days ago
  * `years`: `(last - first)` in years
* **Ranking:** active first, then reference monitors, then the longest history, then the closest.
* *Nitty-gritty:* `first` and `last` are **location-level** dates. That is why two sensors of the same station get the same rank, and why the pipeline joins all sensors of a station instead of trusting one sensor ID (see section 8.2).

**`fetch_sensor_hours(sensor_id, start, end=None, chunk_days=120)`** downloads a long history safely.
* Normalises `start` and `end` to UTC. `end` defaults to now.
* Splits the period into **120-day chunks**, because asking for years at once can hit the API's timeout (HTTP 408).
* Concatenates the chunks, drops duplicate timestamps and sorts.
* Returns an empty frame with the right columns if there is no data.

**`_fetch_hours_range(sensor_id, start, end)`** handles one chunk, with pagination.
* Requests `limit=1000` per page and keeps paging until a page returns fewer rows than the limit.
* If the server rejects `limit=1000` with **422**, it restarts the chunk with `limit=100`. Some API deployments cap the page size.
* For each item it keeps `time = period.datetimeFrom.utc`, `pm25 = value` and `coverage = coverage.percentComplete`. Coverage is the share of sub-hourly readings present in that hour, and quality control uses it.
* Prints progress on one line (`end="\r"`).

### 6.2 `aq/openmeteo.py`: weather and CAMS client
Constants: four endpoint URLs, `WEATHER_VARS` (the 11 hourly variables in section 4.2), and `RENAME`, which maps API names to short column names.

**`_request(url, params, variables)`** is the self-healing core of this module.
* Always asks for `timezone=GMT` (UTC timestamps) and `wind_speed_unit=ms` (metres per second).
* **Unsupported-variable recovery:** if the API answers 400 and its `reason` mentions one of the requested variables, that variable is removed and the request retried. The historical endpoints don't all offer `boundary_layer_height` for every period, and this keeps the pipeline running instead of crashing.
* On 429 it waits 30 s, 60 s, and so on.
* Converts the JSON `hourly` block to a DataFrame and localises the times to UTC.
* **Forces every value column to numeric** (`pd.to_numeric(errors="coerce")`). JSON `null`s for a missing variable otherwise produce *object* columns, which crashed `interpolate` in an early version.
* Renames the columns with `RENAME`.

**`_yearly_chunks(start, end)`** is a generator of (start, end) date strings, one calendar year at a time. Long requests are slower and more fragile.

**`fetch_weather_history(lat, lon, start, end)`** fetches yearly chunks from the **historical-forecast** API. If a chunk fails, that chunk is fetched from the **ERA5 archive** instead.

**`fetch_weather_forecast(lat, lon, past_days=5, days=4)`** fetches recent and future weather. The 5 past days cover the archive lag (it ends 2 days ago) plus some overlap.

**`fetch_cams_history(lat, lon, start, end)`** fetches CAMS PM2.5 history.
* The start is clipped to `2022-08-01`, where CAMS history begins.
* A failed year is skipped with a message rather than stopping the run.
* Returns an empty frame with the right columns if nothing came back.

**`fetch_cams_forecast(...)`** fetches recent and future CAMS PM2.5.


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


---

## 9. Machine-learning design, and why gradient boosting

### 9.1 Framing the problem
We want PM2.5 at every hour **T+h** for lead times **h = 1…60**, using only what is known at the issue time **T**:
* the station's readings up to T
* weather and CAMS forecasts for the coming hours

This is **supervised regression on a time series with exogenous (external) inputs**.

There are three standard ways to produce a multi-step forecast:

| Strategy | How | Problem |
|---|---|---|
| **Recursive** (iterated one-step) | Predict T+1, feed it back as a lag, predict T+2, … | Errors **compound** over 60 steps, and the model is never trained on its own imperfect inputs |
| **Direct, one model per horizon** | Train 60 separate models | 60× the training cost; each model sees only its own slice of data; the forecast curve can be jagged |
| **Direct multi-horizon, h as a feature** ✅ | One model; each row is a (T, h) pair with `h` as an input | One model shares what it learns across horizons; no compounding; smooth curve; trained once |

The project uses the third strategy. Each training row says, in effect:
> "At issue time T, the recent PM2.5 looked like this. At T+h the forecast weather is this, CAMS expects this change, and it will be this hour of the day. What was PM2.5 at T+h?"

**Size:** with an issue time every 3 hours (`ISSUE_STRIDE`) and 60 horizons, each year of data gives about 2,920 × 60 ≈ **175,000 rows**. Three to four years therefore give several hundred thousand rows.

### 9.2 Why gradient-boosted decision trees (GBDT)?

**The short answer:** this is a **tabular** problem, with a few dozen heterogeneous columns (lags, weather, calendar), hundreds of thousands of rows, many missing values, strongly **non-linear effects and interactions**, and a need for **uncertainty** and **explanations**. Gradient-boosted trees are the strongest, most practical tool for exactly this setting.

**The long answer, point by point:**

1. **State of the art on tabular data.** Benchmarks repeatedly find that tree ensembles, gradient boosting especially, match or beat deep learning on medium-sized tabular datasets. They also need far less tuning and compute. See, for example, Grinsztajn, Oyallon & Varoquaux, *"Why do tree-based models still outperform deep learning on tabular data?"*, NeurIPS 2022.
2. **Non-linearity and thresholds.** PM2.5 does not fall linearly with wind speed. Below about 1–2 m/s the air stagnates and pollution builds; above about 4 m/s extra wind changes little. Trees model such step-like, saturating responses naturally. A linear model would need hand-made transformations.
3. **Interactions found automatically.** The worst episodes need several conditions at once: a low mixing height **and** low wind **and** night **and** no rain **and** air arriving from the polluted plain. Each tree split conditions on the earlier splits, so trees learn these "AND" combinations without anyone specifying them.
4. **Missing values handled natively.** Station data has gaps, and some weather variables (mixing height) can be missing for whole periods. scikit-learn's `HistGradientBoostingRegressor` learns, at every split, which branch missing values should go to. No imputation is needed, and no rows are thrown away.
5. **No feature scaling, robust to odd inputs.** Trees only compare thresholds, so units (m, m/s, µg/m³, sines of hours) and outliers in the inputs don't matter.
6. **Built-in quantile regression.** Choosing `loss="quantile"` gives the 10th, 50th and 90th percentiles directly. This provides the uncertainty band and the exceedance probability, which a plain point forecast cannot.
7. **Fast and light.** Histogram-based training handles about 500,000 rows × 49 features in roughly a minute per model on a laptop CPU. No GPU is needed, so it runs on a student laptop and retrains in a few minutes.
8. **Explainable.** Permutation importance and counterfactual "what if the weather were typical" runs (section 11.7) show *why* a forecast is high. That matters for a public-health warning.
9. **Proven for air quality.** Gradient boosting (XGBoost, LightGBM and similar) is among the most common and best-performing methods in PM2.5 forecasting studies, especially when combined with meteorological forecasts and chemistry-model outputs, as here.

**Alternatives considered:**

| Model | Why not chosen (here) |
|---|---|
| Linear / ridge regression | Can't capture thresholds or interactions without heavy manual feature engineering; under-fits |
| ARIMA / SARIMAX | Essentially linear and (near-)univariate; weak with many non-linear weather drivers; fiddly for 60 horizons |
| Prophet | Built for trend and seasonality, not weather-driven episodes; no feature interactions |
| Random forest | Good, but averages deep trees (high variance, big models), usually less accurate than boosting, and scikit-learn has no native quantile loss for it |
| XGBoost / LightGBM | Same family of algorithm, equally good. scikit-learn's histogram GBDT is modelled on LightGBM. Using it keeps dependencies minimal (scikit-learn is already needed) and installs everywhere; swapping is a few lines |
| LSTM / GRU / Temporal Fusion Transformer | Need more data, careful tuning and often a GPU; harder to explain; with a few years of hourly data per station they rarely beat GBDT. **A good extension** for a comparison chapter |
| SVR / k-nearest neighbours | Scale poorly to hundreds of thousands of rows; need feature scaling; no native quantiles |

### 9.3 How gradient boosting works (in this model)
Gradient boosting builds an **additive model** of many small regression trees:

$$F_M(x) = F_0 + \eta \sum_{m=1}^{M} f_m(x)$$

* **Start:** $F_0$ is a constant (for quantile loss, the τ-quantile of the target).
* **Each round m:**
  1. Compute every row's **negative gradient** of the loss at the current prediction (the "pseudo-residual").
  2. Fit a small tree $f_m$ to these pseudo-residuals, so the tree learns *the direction in which the current model is wrong*.
  3. Set each leaf's output value by a line search (for quantile loss, the τ-quantile of the residuals in that leaf).
  4. Add the tree to the model, scaled down by the **learning rate** η.
* **Repeat** for `max_iter` rounds. Each tree corrects a little of what the previous ones got wrong, which is why it is called "boosting".

**Histogram GBDT tricks** (`HistGradientBoostingRegressor`):
* Each feature is binned into at most 255 bins. Split finding then scans bins instead of sorted values, which makes it much faster.
* Trees grow **leaf-wise** (best-first) up to `max_leaf_nodes` leaves.
* Missing values get their own learned direction at each split.

**Quantile (pinball) loss** for quantile level τ, with residual $u = y - \hat{y}$:

$$L_\tau(u) = \max(\tau u,\ (\tau - 1)u)$$

* For τ = 0.9, under-predicting costs 9 times more than over-predicting. The minimiser is therefore the value that 90 % of outcomes fall below, which is the 90th percentile.
* The gradient is just $\tau - \mathbb{1}[y < \hat{y}]$. This makes training robust to outliers in the target.

### 9.4 Hyperparameters (`_new_model`) and why

| Parameter | Value | Reason |
|---|---|---|
| `loss` | `"quantile"` | Gives percentiles (uncertainty) instead of a single mean |
| `quantile` | 0.1 / 0.5 / 0.9 | Lower band, median forecast, upper band. The 10–90 % band is an 80 % interval |
| `learning_rate` | 0.06 | Small steps generalise better; 0.05–0.1 is standard |
| `max_iter` | 300 | Number of trees. With η = 0.06 this reaches a good fit without long training |
| `max_leaf_nodes` | 24 | Fairly shallow trees, enough for 4–5-way interactions. Earlier, deeper trees (48 leaves) over-fitted one year's patterns |
| `min_samples_leaf` | 200 | Every leaf must be backed by at least 200 rows. This stops the model memorising rare situations, and matters because neighbouring (T, h) rows are highly correlated |
| `l2_regularization` | 1.0 | Shrinks leaf values; adds stability |
| `early_stopping` | False | Deterministic, full-length training. The held-out evaluation is done properly elsewhere (section 9.10) |
| `random_state` | 0 | Reproducible results |

### 9.5 Target transform: logarithm
* PM2.5 is **right-skewed** (mostly 10–80, sometimes 300+), and its dynamics are **multiplicative**: a stagnant night might *double* pollution whatever the starting level.
* Modelling `log1p(PM)` turns ratios into differences, stabilises the variance, and stops the model wasting effort on a few extreme values. `log1p(x) = log(1+x)` is safe at 0.
* **Quantiles survive monotone transforms.** The 90th percentile of log(PM) is the log of the 90th percentile of PM. The model can therefore be trained in log space and its quantiles converted back exactly with `expm1`.

### 9.6 The key design decision: predict the *change*, not the level
**The original version** predicted `log1p(PM at T+h)` from absolute features. On Kolkata's real data it was **worse than simple persistence** ("the next hours equal the latest reading"):

| | MAE at 1 h | MAE at 24 h |
|---|---|---|
| Original model | 7.0 | 7.3 |
| Persistence | 1.6 | 4.9 |

**The diagnosis** (done by re-training on the user's data):
* Summer 2026 in Kolkata was about **35 % cleaner than summer 2025**.
* The model had learned "in this season, with this weather, PM is about 27". It kept forecasting about 27 while the station read about 17.
* It showed a +0.21 log-unit (about +24 %) bias even at 1 hour ahead.
* Season features (day-of-year) let the trees memorise *last year's dates*, and absolute CAMS and lag levels did the same.

**The fix**, implemented in `features.design()`:
1. **Target = log change from now.** `target = log1p(PM at T+h) − log1p(PM at T)`. The forecast is `PM_now × exp(predicted change)`, so it is always anchored to what the station measures today.
2. **Scale-free inputs:**
   * every past-PM feature (lags, means, max, min, same hour yesterday) is expressed as **log-ratio to now**
   * variability is expressed relative to the current level
   * CAMS enters only as its **predicted change** (`c_cams_change`, `c_cams24_change`), never as a level
3. **Drop day-of-year.** With 1–3 years of data it only enables memorisation. Hour of day and weekday stay, because they repeat thousands of times.

**Result:** the model now beats persistence at every lead time and beats CAMS by about 3× (section 17).

**Why this works:** the model now learns **physics-like rules**, such as "a falling mixing height tonight plus weak wind → PM rises about 40 %". These transfer across years, stations and calibration changes. Absolute levels don't transfer.

### 9.7 Blending with persistence
Even after the fix, at **1–3 hours** ahead the latest reading is very hard to beat, and the model's small predicted changes add noise. The code therefore shrinks the predicted change separately for each lead-time bucket:

$$\log \hat{y}_{T+h} = \log y_T + w_b \cdot \widehat{\Delta}_{50}$$

* **Weights:** $w_b \in [0, 1]$ is chosen on held-out data from a grid of 21 values (0, 0.05, …, 1) to minimise MAE.
* **Buckets** (`H_EDGES`): 1, 2–3, 4–6, 7–12, 13–24, 25–36, 37–48 and 49–60 h.
* **Kolkata's learned weights:** about 0.15 at 1 h, 0.35 at 2–3 h, 0.6 at 4–6 h, 0.7 at 7–12 h, 0.9 for 13–36 h, and 1.0 beyond 36 h. So the model is trusted more the further ahead it looks, which is the right behaviour.

This is a simple, principled form of **forecast combination** (shrinkage toward a strong baseline).

### 9.8 Calibrated uncertainty: split-conformal scaling
Quantile models are often over-confident: their 80 % band holds the truth only about 60–70 % of the time. The code fixes this with **split-conformal calibration** on held-out data:
1. **Score every calibration row.** For each row, compute how many band-half-widths the truth fell from the median, in log space:
   * below the median: $s = (\log q50 - \log y)/(\log q50 - \log q10)$
   * above the median: $s = (\log y - \log q50)/(\log q90 - \log q50)$
2. **Pick the scale factor.** For each lead-time bucket, $k_b$ is the **80th percentile of s**, clipped to [0.5, 4].
3. **Rescale the band.** Live forecasts stretch (or shrink) the distance from median to q10 and q90 by $k_b$.

By construction, about 80 % of held-out outcomes then fall inside the band. The backtest confirms coverage of 0.79–0.82 on Kolkata.

### 9.9 From quantiles to a probability of exceeding a threshold
The dashboard needs P(PM > threshold) for any threshold the user picks. `exceed_prob()` fits a **split-normal distribution in log space** through the three quantiles:
* centre at log q50
* lower spread $\sigma_{lo} = (\log q50 - \log q10)/1.2816$ and upper spread $\sigma_{hi} = (\log q90 - \log q50)/1.2816$. Here 1.2816 is the z-score of the 90th percentile, and both spreads are floored at 0.02 to avoid dividing by zero.
* the spread on the threshold's side is used:

$$P(\text{PM} > t) = 1 - \Phi\left(\frac{\log(1+t) - \log(1+q50)}{\sigma}\right)$$

A split-normal allows **skew**: pollution can spike up much further than it can drop. The reliability diagram in the Track record tab checks the probabilities ("when we say 70 %, does it happen about 70 % of the time?").

### 9.10 Honest evaluation: no leakage
* **Time-ordered split.** The cut-off is the issue time at the 75th percentile:
  * **training** uses only rows whose *target time* is before the cut-off, so no future value leaks in through long horizons
  * **testing** uses rows whose *issue time* is on or after the cut-off
* **Calibration and scoring are separate.** Inside the test period, **alternate weeks** are used to fit the blending weights and conformal scales, and the **other weeks** are used for every reported number. Alternating weeks (rather than first half / second half) means both halves cover the same seasons.
* **Baselines:**
  * persistence (`p_pm`), the hardest to beat at short lead times
  * same hour on the last observed day (`p_same_hour`), which captures the daily cycle
  * raw CAMS, which is what apps show
* **Final model:** after evaluation, the models are **refit on all the data** for live forecasting. The calibration and blending learned out of sample are kept. This is standard practice: evaluate honestly, then use every bit of data.

### 9.11 Explainability
* **Global: permutation importance.** On up to 15,000 test rows, each feature is shuffled in turn and the increase in error is measured, three repeats per feature. A feature whose shuffling hurts most matters most.
* **Local: counterfactual driver attribution.** For *today's* forecast, one weather group at a time is replaced by its **typical value for this month and hour** (the station's own climatology, a median), and the model is re-run. The difference shows how many µg/m³ that factor adds compared with a normal day. The five groups are:
  * wind
  * mixing height
  * rain and clouds
  * temperature, humidity and pressure
  * regional CAMS

  This is a simple, model-agnostic, causal-style explanation that people understand: "low wind adds +23 µg/m³ to tomorrow night".


---

## 10. `aq/features.py`: feature engineering

The module works in three stages:
1. `series_features` computes helper columns **once per hour**.
2. `make_samples` turns them into **(T, h) rows**.
3. `design` converts those rows into the **scale-free model matrix**.

Column prefixes say where a feature comes from:

| Prefix | Meaning |
|---|---|
| `p_` | measured PM2.5, taken at issue time T |
| `w_` | weather, taken at target time T+h |
| `c_` | CAMS |
| `k_` | calendar |
| `d_` | change between T and T+h |
| `path_` | conditions along the way from T to T+h |

Constants: `PM_LAGS = (1, 2, 3, 6, 12, 24, 48)` and `WEATHER_COLS` (10 weather variables).

### 10.1 `series_features(df) → DataFrame`
Computed on the merged hourly table (index = UTC hour):

**PM history (`p_`), known at the row's own hour:**

| Feature | Definition | Why |
|---|---|---|
| `p_pm` | PM2.5 now | The anchor of the forecast |
| `p_lag1 … p_lag48` | PM 1, 2, 3, 6, 12, 24 and 48 h ago | Short memory and the daily cycle |
| `p_mean3/6/24/72` | Rolling means (need at least half the window present) | Smoothed recent levels; the 72-h mean shows how persistent an episode is |
| `p_std24`, `p_max24`, `p_min24` | 24-h variability and extremes | Volatile or stable regime |
| `p_trend3`, `p_trend24` | log change over 3 h and 24 h | Build-up or clean-out already under way |

**Weather (`w_`):**

| Feature | Definition | Why |
|---|---|---|
| `w_temp` … `w_blh` | Raw variables | Direct drivers |
| `w_wind_u`, `w_wind_v` | Wind as east/north components: −speed·sin(dir), −speed·cos(dir) | Direction is circular (359° ≈ 1°). Components make it continuous **and** weight it by speed |
| `w_vent` | **Ventilation index** = wind speed × mixing height (m²/s) | The classic air-pollution dispersion indicator: how much air is available to dilute emissions. Below about 2,000 m²/s means stagnant |
| `w_log_vent` | log(1 + ventilation) | Compresses its huge range |
| `w_rain6`, `w_rain24` | Rain over the last 6 h and 24 h | Wash-out has a lingering effect |
| `w_wind12`, `w_logvent12` | 12-h means | Has the air been stagnant for a while? |
| `w_blh_max24` | Highest mixing height in the last 24 h | How well yesterday's daytime mixing cleaned the air |
| `w_dpress24`, `w_dtemp24` | 24-h pressure and temperature change | Air-mass changes and fronts |

**Every weather column is always created** (filled with NaN if a variable is missing). The model's column list must be identical in training and live use.

**CAMS (`c_`):** `c_cams` (hourly) and `c_cams24` (24-h mean).

**Calendar (`k_`), in the city's local time:**
* `k_hour_sin`, `k_hour_cos`: hour of day on a circle, so 23:00 and 00:00 are close.
* `k_dow`: day of week, 0 = Monday.
* `k_doy_sin`, `k_doy_cos`: day of year. These are kept in the raw table but **dropped by `design`** (section 9.6).

### 10.2 `make_samples(sf, issue_pos, horizons, with_target=True) → DataFrame`
This builds the (T, h) rows with numpy integer indexing. There are no Python loops, which matters at about 1 million rows.

```python
I = np.repeat(issue_pos, len(horizons))   # issue row index, repeated for each h
H = np.tile(horizons, len(issue_pos))     # 1..60 for each issue time
V = I + H                                 # target row index
A = sf[p_cols].to_numpy()[I]              # PM history at the issue time
B = sf[tgt_cols].to_numpy()[V]            # weather / CAMS / calendar at the target time
```

**Rows whose target lies beyond the table are dropped.** The function then adds:

| Feature | Meaning |
|---|---|
| `h` | Lead time; this is how one model covers every horizon |
| `d_log_vent`, `d_temp`, `d_wind` | Weather at T+h minus weather at T. A *falling* ventilation is the classic sign that an episode is coming |
| `c_bias_now`, `c_cams_adj` | CAMS's current local bias and a bias-corrected CAMS (kept for reference; removed by `design`) |
| `path_log_vent` | **Mean** log-ventilation over (T, T+h] |
| `path_rain` | **Total** rain over (T, T+h] |
| `p_same_hour` | PM at the same clock hour on the most recent fully observed day |
| `c_cams_T` | CAMS at the issue time |
| `issue_time`, `target_time` | Bookkeeping |
| `y` | PM2.5 at T+h, the target |

**How the path features are computed:** with **cumulative sums**, `(cs[V+1] − cs[I+1])`, the sum over any window is one subtraction. This gives "conditions along the way" for every (T, h) pair at once. The intuition is that pollution at T+h depends on the whole journey of the air, not just the weather at the final hour.

**How `p_same_hour` is computed:** `back = 24·ceil(h/24)` ensures the value comes from a day that was *already observed at T*, so there is no leakage.

### 10.3 `feature_columns(X)`
All columns except `issue_time`, `target_time` and `y`.

### 10.4 `design(X) → (R, base)`
This applies the scale-free transformation (section 9.6).
* **`base`** is `log1p(PM now)`, falling back to the 3-h mean if the current hour is missing. It is the persistence anchor.
* **Columns dropped** (`_ABSOLUTE`): `p_pm`, `c_cams`, `c_cams24`, `c_cams_adj`, `c_bias_now`, `c_cams_T`, `k_doy_sin`, `k_doy_cos`.
* **Each `p_` feature** becomes `log1p(value) − base`, the log-ratio to now. The exceptions:
  * `p_trend3` and `p_trend24` are already log differences and are kept as they are.
  * `p_std24` becomes the coefficient of variation (std ÷ current level).
* **New features:**
  * `c_cams_change = log1p(CAMS at T+h) − log1p(CAMS at T)`
  * `c_cams24_change`, the same for the 24-h mean
* **±inf is replaced with NaN**, which the trees handle.

**The final matrix has 49 inputs:**
* `h`
* 16 relative PM features
* 21 weather features
* 3 calendar features (hour sine/cosine, weekday)
* 3 change features (`d_`)
* 2 path features
* `p_same_hour`
* 2 CAMS-change features

---

## 11. `aq/model.py`: training, calibration, evaluation, forecasting, explanation

### 11.1 Constants
* **`Z90 = 1.28155`** is the standard-normal z-score of the 90th percentile.
* **`DRIVER_GROUPS`** maps the five explanation groups to raw columns:

  | Group | Columns |
  |---|---|
  | Wind | `wind_speed`, `wind_dir`, `wind_gust` |
  | Mixing height | `blh` |
  | Rain & clouds | `precip`, `cloud` |
  | Temperature, humidity & pressure | `temp`, `rh`, `dewpoint`, `radiation`, `pressure` |
  | Regional pollution | `cams_pm25` |

* **`H_EDGES = [1, 2, 4, 7, 13, 25, 37, 49, ∞]`** defines the lead-time buckets used for blending and calibration. They are finer at short lead times, where behaviour changes quickly.

### 11.2 Helper functions
* **`_new_model(q)`** returns a configured `HistGradientBoostingRegressor` for quantile *q* (hyperparameters in section 9.4).
* **`_norm_cdf(z)`** is the standard normal cumulative distribution, computed with `math.erf` so scipy isn't needed. It returns an empty array for empty input. A dashboard crash taught us that `np.vectorize` fails on size-0 input unless `otypes` is given.
* **`exceed_prob(q10, q50, q90, threshold)`** is the split-normal probability from section 9.9. It is vectorised over any number of hours.
* **`_bucket(h)`** maps a lead time to its bucket index with `np.searchsorted(H_EDGES, h, side="right") − 1`.
* **`_raw_residuals(models, R)`** returns the three models' predicted *log changes* (n × 3), **sorted per row**, so q10 ≤ q50 ≤ q90 always holds. Three independent quantile models can otherwise "cross".
* **`_assemble(r, base, h, blend, scale)`** turns log changes into PM2.5 quantiles:

  ```python
  mid = r[:, [1]]                                            # median log change
  P = base + w * mid + (r - mid) * k                         # blend the centre, scale the band
  P = clip(expm1(P), 0)
  ```

  * `w` (blend) moves the centre toward persistence.
  * `k` (conformal) widens or narrows the band around the centre.
* **`predict_quantiles(bundle, X)`** chains `design` → `_raw_residuals` → `_assemble`, using the bundle's saved column order, blend weights and scales.
* **`fit_blend(r50, base, y, h)`** chooses, for each bucket, the weight from {0, 0.05, …, 1} that minimises MAE on the calibration weeks.
* **`climatology(df)`** is the median of every driver column by (local month, local hour). It is saved in the bundle as the "typical day" for driver attribution.

### 11.3 `build_training_set(df)`
1. `series_features(df)`.
2. Choose issue positions where PM is observed, skipping the first 72 h (so lags exist) and taking every 3rd hour.
3. `make_samples` for h = 1…60.
4. Keep only rows whose target `y` is observed.

### 11.4 `train_and_backtest(df, fit_final=True)`
1. **Build the samples.**
2. **Split by time.** The cut-off is the issue time at the 75 % point:
   * train = rows with `target_time < cutoff`
   * test = rows with `issue_time ≥ cutoff`
3. **Fit three quantile models** on `design(train)` with target `log1p(y) − base`.
4. **Predict the test set** (raw log changes).
5. **Split the test set into alternating weeks.** Odd weeks are used to fit the blend weights and then the conformal scales (computed on blended predictions); even weeks are used for scoring.
6. **Score** the even weeks with `evaluate` and attach `interval_scale`, `blend_weight` and the period dates to the report.
7. **Compute permutation importance** of the median model on up to 15,000 scored rows, scored by mean absolute error of the log change, 3 repeats. The top 20 are kept.
8. **Save** `backtest_predictions.csv` for h ∈ {1, 3, 6, 12, 24, 36, 48}. The Track record tab plots it.
9. **Refit** the three models on *all* samples for live use.
10. **Save the bundle** (joblib) with these fields:
    * `models` (q → fitted model), `cols` (feature order), `scale`, `blend`
    * `version: 2` (format marker), `climatology`, `trained_until`, `city`

    Also save `backtest.json`.

### 11.5 `interval_scale(cal, target=0.8)`
This is the conformal factor per bucket (section 9.8). The `np.clip(…, 1e-3)` guards against zero-width bands. The `[0.5, 4]` clip stops a tiny bucket from producing absurd bands.

### 11.6 `evaluate(test)` and `_mae`
* `_mae(a, b)` ignores non-finite pairs. CAMS can be missing in early periods.
* **For each lead time h** it computes:
  * MAE of the model median, persistence, same hour and CAMS
  * RMSE of the model
  * **coverage_80**: the share of outcomes inside [q10, q90]
* **Summaries** for 0–24 h and 24–48 h: mean metrics plus the **skill score** `1 − MAE_model / MAE_baseline`. Skill above 0 means the model beats the baseline; 0.24 means 24 % less error.
* **Warning verification**, at 24–48 h lead time, for thresholds 60, 90 and 120. A warning is issued when P ≥ 0.5.

  | Metric | Formula | Meaning |
  |---|---|---|
  | POD (probability of detection) | hits / (hits + misses) | Share of real exceedances that were warned about |
  | FAR (false-alarm ratio) | false alarms / (hits + false alarms) | Share of warnings that didn't happen |
  | CSI (critical success index) | hits / (hits + misses + false alarms) | Combined score, 1 = perfect |
  | Brier score | mean((p − event)²) | Accuracy of the probabilities; lower is better |

* **Reliability** (threshold 90): probabilities are binned into tenths, and for each bin with at least 30 cases the mean forecast probability is compared with the observed frequency.

### 11.7 Live forecasting
* **`load_bundle()`** reads the joblib file.
* **`_tail(df)`** keeps the window from 200 h before the last observation to 60 h after it. That is enough for every lag and rolling window, and fast.
* **`_forecast_frame(bundle, df)`** runs `series_features` on the tail, takes the last row with an observation as T, and calls `make_samples` for h = 1…60 without a target.
* **`forecast(bundle, df)`**:
  * refuses old bundles (`version < 2`) with an instruction to retrain, because the feature design changed
  * returns `issue_time`, `target_time`, `h`, `cams`, `q10`, `q50` and `q90`
* **`driver_attribution(bundle, df, fc)`**, for each driver group:
  1. Copy the tail.
  2. Replace the group's columns, from 24 h before T onwards, with the climatology value for each hour's (month, hour). The original value is kept where the climatology is missing.
  3. Rebuild the forecast and predict the median.
  4. `effect = actual forecast − counterfactual forecast`, averaged over hours 1–24 and 25–48 after T.

  A positive effect means the factor is making the air **worse** than a typical day.


---

## 12. `aq/insights.py`: from numbers to decisions

**`category(v)`** returns the CPCB PM2.5 category name and colour for a concentration. It uses the first band whose upper limit is ≥ v, and returns ("No data", grey) for NaN.

**India CPCB AQI:**
* **`_AQI_BP`** holds the breakpoints:
  * PM2.5 `[0, 30, 60, 90, 120, 250, 500]` maps to AQI `[0, 50, 100, 200, 300, 400, 500]`
  * PM10 `[0, 50, 100, 250, 350, 430, 600]` maps to the same AQI scale
* **`sub_index(conc, pollutant)`** does piecewise-linear interpolation between breakpoints (`np.interp`), capped at 500. This is the CPCB formula $I = I_{lo} + \frac{I_{hi}-I_{lo}}{C_{hi}-C_{lo}}(C - C_{lo})$.
* **`aqi(pm25, pm10)`** is the **maximum** of the available sub-indices, as CPCB defines it. It returns NaN if neither is available.
* **`aqi_category(a)`** maps an AQI value to its category and colour.
* **About averaging:** CPCB defines the AQI on **24-hour averages**, so the dashboard always passes 24-h means. The hourly colour strip uses PM2.5 concentration categories, labelled as such.

**`localize(fc)`** parses the forecast's time columns as UTC and adds `local_time` in the city's time zone.

**`upcoming(fc, now, hours=48)`** returns the forecast rows strictly after the current hour, up to 48 h ahead. Because the forecast starts at the *last observation*, which may be hours old, this trims any part already in the past.

**`alert_summary(fc, threshold, now)`** is the early-warning logic.
* **No upcoming rows:** returns level "Unknown" with a message to update.
* **Otherwise:**
  * `p` = hourly exceedance probability; `pmax` is its maximum
  * `peak` = the hour with the highest median, with its q10 and q90
  * "likely" hours are those with p ≥ 0.5; they give the start time, end time, lead time (hours from now to the first likely hour) and duration
* **Levels:**

  | Level | Rule |
  |---|---|
  | EMERGENCY | peak median > 250 and p ≥ 0.5 |
  | WARNING | max p ≥ 0.6 |
  | WATCH | max p ≥ 0.3 |
  | CLEAR | otherwise |

  The thresholds favour **early detection** (Watch at 30 %) while reserving Warning for likely events, the same trade-off weather services make.
* **Message templates:**
  * *clear*: "No exceedance … (highest chance x %)"
  * *likely*: start, lead time, duration, peak with its 80 % range, and the maximum chance
  * *possible*: the most likely hour

**`daily_outlook(fc, now)`** covers today, tomorrow and the day after (local dates).
* Averages q10, q50 and q90 over each day's hours, and keeps only days with at least 12 hours left.
* Adds:
  * `p_naaqs`, the chance the 24-h mean exceeds 60
  * `x_who`, the multiple of the WHO 15 guideline
  * `cigarettes = q50 / 22` (Berkeley Earth's rule of thumb that breathing 22 µg/m³ for a day is like smoking about one cigarette)
  * `category`
* *Caveat (documented):* averaging hourly quantiles makes the daily band slightly too wide, which is conservative.

**`best_windows(fc, duration=2, start_hour=6, end_hour=21, now, per_day=1)`** finds the best time to go out.
* Rolling mean of the median over `duration` hours (`shift(-(duration-1))` aligns each window with its start hour).
* Keeps only windows that start at or after 06:00 and end by 22:00.
* For each day, picks the window with the lowest mean, and also reports that day's worst window, so the dashboard can say "about X µg/m³ cleaner than the worst slot".

**`driver_sentences(drivers, col="effect_24_48h", min_effect=3)`** turns attribution numbers into plain sentences.
* Ignores effects smaller than 3 µg/m³.
* Sorts by absolute effect.
* Uses a physical explanation for each group and sign, for example "a low mixing layer traps pollution near the ground".

**`HEALTH_ADVICE`** gives a recommendation for each category, based on CPCB and WHO guidance. It escalates from "enjoy outdoor activities" to "stay indoors with filtered air".

---

## 13. `aq/ui.py`: the visual layer

**Design tokens:**

| Token | Value | Use |
|---|---|---|
| `INK` | #1B2A41 | Deep slate: headings and the header background |
| `SKY` | #EEF3F8 | Pale sky: page background |
| `PAPER` | #FFFFFF | White surfaces |
| `HAZE` | #7C8DA3 | Secondary text |
| `LINE` | #DCE3EC | Borders and grid |
| `SIGNAL` | #2F6FDE | The forecast line and interactive accent |

* **Fonts:** **Bricolage Grotesque** for headings and numbers; **Atkinson Hyperlegible** for body text. Atkinson Hyperlegible was designed by the Braille Institute for legibility, which suits health information.
* **Category colours** live in `config.CATEGORIES` and `insights.AQI_CATEGORIES`, so every chart uses the same palette.
* **`LEVEL_COLORS`** gives a chip colour for each alert level.

**`CSS`** is one stylesheet injected into the Streamlit page. It covers:
* fonts, the page background, and the sidebar
* the tab style, with an underline in the signal colour
* white bordered panels around charts and tables, and metric tiles
* a keyboard-focus outline
* the header block (`.hero`), the colour strip (`.strip`) with its axis and legend
* stat tiles (`.tiles`), day cards (`.day`)
* a mobile breakpoint

It targets Streamlit's `data-testid` attributes.

**`inject()`**:
1. Writes the CSS into the page.
2. Registers a **Plotly template** named `airwatch`: fonts, white background, light gridlines, a colour sequence starting with the forecast blue, and a matching hover style.
3. Sets it as the default (`plotly_white+airwatch`), so every chart matches without per-chart styling.

**`_esc(x)`** HTML-escapes text, because station names come from an external API.

**`hero(place, subline, aqi_value, …, hours, threshold, advice)`** builds the header block. It is the signature element:
* Left side: the city in large type, with the station and how long ago it reported. Right side: the AQI as a large number in its category colour.
* Below: an alert chip (Clear / Watch / Warning / Emergency) and the message.
* **The 48-hour strip:**
  * one `<div>` per hour, coloured by forecast category, with a hover tooltip ("Sat 03 Oct 21:30 · 118 µg/m³ · Poor")
  * the class `over` draws a **white tick** on hours above the user's threshold
  * axis labels: "now", day changes ("Sat 03") and "noon"
* A legend of the categories, and the health advice.

**`tiles(items)`** draws a responsive grid of stat tiles from (label, value-HTML, sub-text, colour) tuples.

**`brand()`** is the name block at the top of the sidebar ("Kolkata Air Quality Early Warning").


---

## 14. `app.py`: the Streamlit dashboard

### 14.1 How Streamlit runs this file
Streamlit re-executes `app.py` from top to bottom **every time the user interacts**: moving a slider, choosing a profile, clicking a button. Two mechanisms make this work:
* **`@st.cache_data`** caches function results by their arguments, so the CSV and JSON files are not re-read on every interaction. Parameters whose names start with `_` are *not* part of the cache key.
* **`st.rerun()`** restarts the script immediately, for example after a refresh.

### 14.2 Page set-up and data loading
* `st.set_page_config(layout="wide")`, then `ui.inject()` (CSS and chart template).
* **`_mtime(path)`** returns a file's modification time, or 0 if it is missing.
* **`load_all(_stamp)`** reads `hourly.csv`, `meta.json`, `forecast.csv`, `drivers.csv`, `backtest.json` and `backtest_predictions.csv`. `_stamp` is the sum of the files' modification times, and the cache is cleared after every refresh, so new data always appears.
* **First-run guard:** if `hourly.csv`, `forecast.csv` or the model file is missing, the page shows the two commands to build the forecast (real data or `--demo`) and stops with `st.stop()`.

### 14.3 Sidebar
* **Your warning:**
  * a profile selectbox (default: Sensitive)
  * a threshold slider (15–250 µg/m³, default from the profile)
  * the outdoor-activity length (1–4 h)
* **Data:**
  * the station in use (or "Demo data" in demo mode)
  * **Refresh data and forecast**, which runs `python pipeline.py update` with `subprocess.run` behind a spinner, then clears the cache and reruns. On failure it shows the error output.

### 14.4 Header
1. A demo-mode notice, if relevant.
2. **Calculations:**
   * `last_obs_t` and `last_obs` from the real readings
   * `lag_h`: how stale they are
   * `alert = alert_summary(...)`, `up = upcoming(...)`, `daily = daily_outlook(...)`
   * **AQI now** = sub-index of the mean of the **last 24 available hours** (bridge estimates included if the station is offline)
3. `ui.hero(...)`: the city, station and how long ago it reported; the AQI as a large coloured number; the warning chip and message; the **48-hour colour strip** from the forecast median, with white ticks on hours above the threshold; and the health advice for the peak.
4. **Warnings:**
   * bridged outage → "estimated from CAMS × ratio"
   * stale data (more than 12 h) → a warning
   * a forecast entirely in the past → an error explaining that Refresh will bridge the gap
5. **Four tiles** (`ui.tiles`): last reading; peak in the next 24 h; peak in 24–48 h; tomorrow's 24-h mean with the chance of breaching the national limit.

### 14.5 Tabs
**`add_category_bands(fig, ymax)`** draws light horizontal bands for the CPCB categories behind a chart, with labels. Several charts use it.

| Tab | Content and how it's built |
|---|---|
| **Forecast** | Measured PM2.5 for the last 96 h (solid), bridge estimates (dotted), the forecast median (blue) with the q10–q90 band (`fill="tonexty"` between two traces), CAMS (grey dotted), the threshold line, a "now" line and category bands. Below: an hourly **exceedance-probability bar chart** (red ≥ 60 %, amber ≥ 30 %) |
| **Plan your day** | Day cards from `daily_outlook`; the **best outdoor window** per day from `best_windows`, with how much cleaner it is than the worst slot; an hour-by-hour table; **CSV download** (`st.download_button`) |
| **Why** | Grouped horizontal bars of the driver effects (next 24 h vs 24–48 h) plus plain sentences; a **ventilation-index** line with the stagnant zone (< 2,000 m²/s) shaded and rain bars on a second axis; the top 12 **permutation importances** with a key to the prefixes |
| **Track record** | Error tiles (0–24 h, 24–48 h, skill against persistence and CAMS, band coverage); **MAE by lead time** for the model and the three baselines; warning verification table (POD, FAR, CSI, Brier); **reliability diagram** (marker size ∝ √n); last 30 days of test forecasts against measurements, for a selectable lead time |
| **Patterns** | **Heatmap** of mean PM2.5 by local hour × month; **pollution rose**: wind direction binned into 8 sectors with `((dir + 22.5) // 45) % 8`, mean PM2.5 per sector on a `Barpolar` chart, showing where dirty air comes from; days per month with a 24-h mean above 60; **weekday effect**; the data-quality JSON |
| **Compared with weather apps** | The gap table (section 2.4) |

---

## 15. `aq/demo.py`: synthetic data for offline demos

**`_ar1(n, phi, sigma, rng)`** generates an AR(1) noise process, $x_t = \phi x_{t-1} + \varepsilon_t$. This is realistic "weather-like" persistence: random, but correlated over time.

**`make_demo(start, end=None, seed=7)`** builds a physically motivated world from the start date to now + 4 days, with a fixed random seed.

**Seasonality:**
* `winter = cos(2π(doy − 15)/365.25)`, which is +1 in mid-January and −1 in mid-July
* a monsoon flag for June–September
* a daylight curve from 07:00 to 18:00 local time

**Weather:**
* temperature, humidity, dew point
* wind (stronger by day and in the monsoon)
* wind direction (north-west in winter, south-west in the monsoon)
* rain (Poisson-like events, frequent in the monsoon)
* cloud, radiation
* mixing height (deep on sunny days, shallow at night and in winter)
* pressure

**A stagnation spell is planted 16–64 h after now:** wind and mixing height are cut, rain is removed, the wind turns north-west, and local burning increases. This gives the demo something to warn about.

**PM2.5 dynamics:**

$$\text{target} = 38 \cdot \text{season} \cdot \text{traffic} \cdot \text{ventilation factor} \cdot \text{transport} \cdot \text{episode} \cdot e^{\text{noise}}$$

| Factor | Definition |
|---|---|
| Traffic | Morning and evening peaks (Gaussian bumps at 09:30 and 21:00) and a Sunday dip |
| Ventilation factor | (median ventilation ÷ ventilation)^0.35: low ventilation means more pollution |
| Transport | +25 % when winter winds come from the north-west |
| Episodes | Random 1–4-day burning episodes from October to February |

**Then:**
* **Smoothing:** `pm[t] = (0.82·pm[t−1] + 0.18·target[t]) · (1 − washout)`, so pollution builds and decays gradually, and rain removes up to 50 %.
* **CAMS-like series:** the truth smoothed, lagged 4 h, biased low and noisy, plus 6. This imitates a coarse model.
* **Station:** reports up to 2 h ago, with about 3 % random gaps.

> Demo scores are optimistic, because the data obey the model's assumptions exactly. **Never report demo results.**


---

## 16. Files the system produces (data contracts)

| File | Written by | Contents |
|---|---|---|
| `data/kolkata/raw/sensors.csv` | `fetch` | Candidate sensors: `sensor_id, location_id, location, provider, is_monitor, lat, lon, distance_m, first, last, active, years` |
| `data/kolkata/raw/pm25_location_<id>.csv` | `_fetch_location` | The station's joined hourly series: `time, pm25, coverage, sensor_id` |
| `data/kolkata/raw/weather_history.csv` | `fetch` | Archived hourly weather (renamed columns, UTC) |
| `data/kolkata/raw/cams_history.csv` | `fetch` | `time, cams_pm25` |
| `data/kolkata/processed/hourly.csv` | `fetch` | **The master table**: `time` (UTC), `pm25`, 11 weather columns, `cams_pm25`, `pm25_bridge`. About 4 days of future rows (weather and CAMS only) |
| `data/kolkata/processed/meta.json` | `fetch` | `mode, city, station, sensor_id[], location_id, provider, lat, lon, tz, qc{…}, fetched_at, freshness{last_observation, gap_hours, bridged, bridge_ratio, …}, last_observation` |
| `models/kolkata/pm25_quantile_gbm.joblib` | `train` | Bundle: `models{0.1,0.5,0.9}, cols, scale, blend, version, climatology, trained_until, city` |
| `reports/kolkata/backtest.json` | `train` | `by_horizon[]`, `summary_0-24h`, `summary_24-48h`, `alerts{60,90,120}`, `reliability[]`, `interval_scale`, `blend_weight`, `period`, `importance` |
| `reports/kolkata/backtest_predictions.csv` | `train` | Test-period forecasts at h ∈ {1,3,6,12,24,36,48}: `issue_time, target_time, h, y, q10, q50, q90, c_cams, p_pm, p_same_hour` |
| `reports/kolkata/forecast.csv` | `forecast` | 60 rows: `issue_time, target_time, h, cams, q10, q50, q90` |
| `reports/kolkata/drivers.csv` | `forecast` | `driver, effect_next24h, effect_24_48h` (µg/m³) |

---

## 17. Evaluation and results

### 17.1 Kolkata (Padmapukur, Howrah station), forecasts issued April–September 2026
These scores come from re-training on the project's downloaded Kolkata data. MAE is in µg/m³; lower is better.

| Lead time | 1 h | 3 h | 6 h | 12 h | 24 h | 36 h | 48 h | 60 h |
|---|---|---|---|---|---|---|---|---|
| **Model** | **1.62** | **2.65** | **3.39** | **3.86** | **4.15** | **4.35** | **4.39** | **4.56** |
| Persistence | 1.64 | 2.84 | 4.11 | 5.01 | 4.90 | 5.74 | 5.98 | 6.66 |
| Same hour yesterday | 4.71 | 4.81 | 4.83 | 4.86 | 4.90 | 5.77 | 5.98 | 6.42 |
| CAMS | 12.08 | 12.03 | 12.27 | 12.73 | 12.78 | 13.51 | 13.29 | 14.05 |
| 80 % band coverage | 0.80 | 0.81 | 0.79 | 0.79 | 0.80 | 0.81 | 0.81 | 0.82 |

**Summary:**
* **0–24 h:** MAE 3.6 against 4.4 for persistence, a skill of **+18 %**.
* **24–48 h:** MAE 4.4 against 5.7 for persistence (skill **+24 %**) and 13.3 for CAMS (**about 3× better**).

**Learned blend weights**, the trust in the model's predicted change: 0.15 at 1 h, 0.35 at 2–3 h, 0.6 at 4–6 h, 0.7 at 7–12 h, 0.9 at 13–36 h, 1.0 from 37 h onwards.

**Most important inputs:**
1. `c_cams_change` (incoming regional change)
2. `p_mean72` (how far the current level is from the 3-day mean, i.e. reversion toward it)
3. `w_wind_u` (east–west wind)
4. `d_temp`
5. `w_dewpoint`
6. hour of day
7. `p_mean24`
8. `w_wind12`
9. `w_dtemp24`
10. `p_same_hour`

### 17.2 How to read these results
* **At 1 hour the model equals persistence.** This is expected and correct, because the blend weight is small there. Nothing beats "it's about what it was an hour ago".
* **The model's advantage grows with lead time.** Persistence errors grow quickly, while the model uses forecast weather and CAMS's expected change.
* **CAMS has a large constant bias at this station.** That is exactly what a station-anchored model removes.
* **Warning scores (POD/FAR) are not meaningful for this test period.** Summer and monsoon had only a handful of hours above 60 µg/m³. Retrain after winter (November–February), when most exceedances occur, to evaluate the warnings properly.
* **Your exact numbers will change after every retrain.** The Track record tab always shows the current ones.

---

## 18. Design decisions and the bugs that shaped them

These came up while running the project on real data, and each one is worth knowing for a viva.

| Problem seen | Root cause | Fix (where) |
|---|---|---|
| `TypeError: cannot interpolate with object dtype` during fetch | Open-Meteo returned JSON `null` for a variable over part of the archive, which made a text (object) column | Force numeric in `_request`; coerce again in `merge_sources`; always create weather feature columns |
| Dashboard crashed: `vectorize on size 0 inputs` | All Kolkata stations on OpenAQ stopped reporting on 29 Sep, so the 60-h forecast lay entirely in the past and the probability function got an empty array | `otypes=[float]` and an empty-input guard in `_norm_cdf`; **`bridge_gap`** fills outages with an estimate; dashboard warnings |
| Model **worse than persistence** on real data | Absolute-level target plus day-of-year let the model memorise 2025 levels; 2026 was about 35 % cleaner | **Predict the change from now**, use scale-free features, drop day-of-year, blend with persistence (sections 9.6–9.7) |
| A station build failed with an empty PM2.5 series (`NaT`) | The chosen sensor ID was the station's **old** ID with no recent data | Treat the **station** as the unit, join all its sensor IDs, try the next station if the data are insufficient (`_fetch_location`, `_station_pm25`) |
| Deprecation warning `use_container_width` | Streamlit 1.4x+ renamed it | `width="stretch"`; `streamlit>=1.64` in `requirements.txt` |

Further deliberate decisions:
* **Train on archived forecasts, not reanalysis.** Training inputs then match live inputs (section 4.2).
* **One model with h as a feature**, not 60 models and not a recursive model (section 9.1).
* **Quantiles plus conformal calibration**, rather than a point forecast. Warnings need probabilities.
* **Interpolate only gaps of up to 3 h.** The trees handle missing values, so longer gaps are never invented.
* **A 60-h horizon.** It absorbs station reporting delays and still gives a full 48 h from now.
* **Automatic, sticky station choice.** The best reporting reference monitor is picked once and then kept, so the training history stays consistent.

---

## 19. Limitations and possible extensions

**Limitations:**
* Forecast weather contains errors that grow after about 48 h, and the model inherits them.
* Sudden local sources (firecrackers, a nearby fire, construction dust) aren't in the inputs. The uncertainty band and trend features help only once a build-up has started.
* Kolkata is represented by **one station**, not a city-wide average or map.
* Only a few years of history per station. Seasonal patterns are seen only 2–3 times.
* The daily band (an average of hourly quantiles) is conservative.

**Already built in the full version of the project** (left out of this checkpoint build): any city in India and searched places, a city comparison map, choosing the station from the dashboard, and Telegram / e-mail alert notifications.

**Further extensions** (good for a report or future work):
1. **Multiple stations or a spatial model:** forecast every station and interpolate (kriging, or a graph neural network over stations) to produce a city map.
2. **Fire hotspots** (NASA FIRMS) upwind for the stubble-burning season; **festival flags** (Diwali); a traffic proxy.
3. **Model comparison:** LightGBM/XGBoost, Random Forest, LSTM/GRU, Temporal Fusion Transformer, all on the same backtest.
4. **SHAP values** for exact attribution of each individual forecast.
5. **More pollutants** (PM10, NO₂, O₃, CO) for a complete CPCB AQI.
6. **Probabilistic daily aggregation** by simulation instead of quantile averaging.
7. **Deployment:** Streamlit Community Cloud plus a scheduled `update --all-cities` (GitHub Actions or Task Scheduler).
8. **Automatic retraining** when the error in recent weeks drifts upward.


---

## 20. Glossary

| Term | Meaning |
|---|---|
| **PM2.5** | Mass of particles smaller than 2.5 µm per cubic metre of air (µg/m³) |
| **AQI (CPCB)** | India's Air Quality Index, 0–500. Each pollutant's 24-h mean is converted with breakpoints and the highest sub-index is reported |
| **NAAQS** | India's National Ambient Air Quality Standards. For PM2.5: 60 µg/m³ as a 24-h mean |
| **CPCB / SPCB** | Central / State Pollution Control Board, the operators of India's official monitors |
| **OpenAQ** | A non-profit platform that collects and serves air-quality data from governments and other sources |
| **Location vs sensor** | An OpenAQ *location* is a monitoring site; a *sensor* is one parameter there, and can have more than one ID |
| **CAMS** | Copernicus Atmosphere Monitoring Service, a global atmospheric-composition model run by ECMWF |
| **ERA5** | ECMWF's global reanalysis: best-estimate historical weather |
| **Boundary layer / mixing height (BLH)** | The lowest part of the atmosphere, mixed by turbulence. Pollution spreads within it. Low at night and in winter |
| **Inversion** | Warm air above cool surface air, acting like a lid that traps pollution |
| **Ventilation index** | Wind speed × mixing height; low values mean stagnant air |
| **Wet deposition** | Rain removing particles from the air |
| **Issue time (T) / lead time (h)** | When a forecast is made, and how many hours ahead it looks |
| **Persistence** | The baseline forecast "future = latest observation" |
| **Direct multi-horizon** | One model predicts every lead time, with h as an input |
| **GBDT** | Gradient-boosted decision trees: an additive ensemble of small trees, each fitted to the errors of the previous ones |
| **Quantile / pinball loss** | A loss whose minimiser is a chosen percentile, used to forecast uncertainty bands |
| **log1p / expm1** | log(1 + x) and its inverse; safe at zero |
| **Skill score** | 1 − error(model) / error(baseline). Above 0 means better than the baseline |
| **MAE / RMSE** | Mean absolute error / root mean squared error |
| **Coverage** | The share of outcomes inside a forecast interval; should match the nominal level (80 %) |
| **Conformal calibration** | Rescaling intervals on held-out data so their coverage is correct |
| **POD / FAR / CSI** | Warning verification: probability of detection, false-alarm ratio, critical success index |
| **Brier score** | Mean squared error of probability forecasts |
| **Reliability diagram** | Forecast probability against observed frequency; a calibrated forecast lies on the diagonal |
| **Permutation importance** | The drop in accuracy when one feature is randomly shuffled |
| **Counterfactual attribution** | Re-running the model with one factor set to "typical" to measure its effect |
| **Data leakage** | Accidentally using information from the future during training, which makes scores look too good |

---

## 21. Likely viva questions, with answers

**Q1. Why did you use gradient boosting rather than a neural network such as an LSTM?**

* The data are tabular, about 50 heterogeneous features and a few hundred thousand rows. On this kind of data gradient-boosted trees are consistently as good as or better than deep networks, train in about a minute on a CPU, need little tuning, handle missing values natively, support quantile loss directly, and are easy to explain.
* An LSTM would need more data per station, normalisation, careful tuning and usually a GPU, with no evidence of better accuracy here.
* I list it as an extension for comparison.

**Q2. Why one model for all 60 horizons instead of a recursive model?**

* Recursive models feed their own predictions back in, so errors compound over 60 steps.
* Making h a feature lets one model learn how relationships change with lead time and share data across horizons.
* There is no compounding, and the forecast curve is smooth.

**Q3. How do you make sure there is no data leakage?**

* The split is by time: training rows must have their *target* time before the cut-off, and test rows have their *issue* time after it.
* Every feature at issue time T uses data up to T only:
  * rolling windows look backwards
  * "same hour yesterday" is taken from a day already observed
  * weather at T+h comes from forecasts, which are known in advance
* Calibration and scoring use different weeks.

**Q4. Why train on archived weather forecasts instead of observed weather?**

* Live predictions use forecast weather, which has errors.
* Training on archived forecasts teaches the model to cope with those errors. Training on perfect hindsight weather would make it over-confident.

**Q5. Your first model lost to persistence. What did you learn?**

* Absolute-level models tied to the season memorised last year's levels. Kolkata's 2026 summer was about 35 % cleaner, so the forecasts were biased upward by about 24 %.
* Predicting the **change** from the current reading, with features relative to now, removed the bias.
* The model then beat persistence by 18–24 % and CAMS by about 3×.
* The lesson is that a model must be anchored to the most reliable current information, and that you should always compare against simple baselines.

**Q6. How do you get a probability from a regression model?**

* Three quantile models give the 10th, 50th and 90th percentiles.
* Conformal scaling on held-out data corrects the band so that 80 % of outcomes fall inside it.
* A split-normal distribution in log space, fitted through the three quantiles, gives P(PM > threshold) for any threshold.
* The reliability diagram checks it.

**Q7. What does the conformal step actually do?**

* For each held-out case it measures how many half-widths the truth was from the median.
* For each lead-time bucket it takes the 80th percentile of that score as a scale factor and stretches the band by it.
* Coverage then becomes about 80 % by construction (0.79–0.82 observed).

**Q8. Why blend with persistence?**

* At 1–3 hours ahead the latest reading is almost unbeatable, and the model's small predicted changes add noise.
* A weight for each lead time, chosen on held-out data, shrinks the predicted change: about 0.15 at 1 h, rising to 1.0 by 37 h.
* The result is never worse than persistence, and the model's skill grows with lead time.

**Q9. Why is the ventilation index important?**

* Pollution concentration ≈ emissions ÷ the volume of air they mix into, per unit time.
* Wind speed × mixing height measures that ventilation.
* Low ventilation, typical of winter nights, traps pollution. This is the main physical reason winter episodes happen.

**Q10. How do you explain a specific forecast?**

* For each weather group, the model is re-run with that group's inputs set to their typical value for that month and hour.
* The difference is the group's contribution in µg/m³, for example "low wind +23 µg/m³ tomorrow night".
* Globally, permutation importance ranks the inputs.

**Q11. What happens when the station stops reporting?**

* For gaps of 6 h to 14 days, the missing hours are estimated as CAMS × the station/CAMS ratio over the last week, so forecasts still start from now.
* The estimates are flagged, shown dotted, and never used for training.
* Beyond 14 days, the warning tells the user the station has been offline too long, and the next `fetch` tries other stations automatically.

**Q12. How is the station chosen?**

1. Rank all PM2.5 monitors within 25 km: active, reference-grade, long history, close.
2. Join all sensor IDs of a station.
3. Accept the first station with at least 30 days of data and a reading within 14 days; keep using the same station later.

**Q13. What is the AQI and how do you compute it?**

* CPCB's index converts each pollutant's 24-h mean concentration into a sub-index by linear interpolation between breakpoints.
* For PM2.5, 0–30 maps to AQI 0–50, 30–60 to 51–100, and so on.
* The overall AQI is the highest sub-index. We use PM2.5 and PM10.

**Q14. What does the project offer that Google Weather doesn't?**

* A station-anchored ML forecast that is about 3× more accurate than the model behind such apps.
* Uncertainty bands and probabilities.
* Early warnings with lead time and duration, and personal thresholds.
* Indian standards.
* Explanations of why air will be bad.
* The best time to go out, and health equivalents.
* A published accuracy record and local pattern analytics.

**Q15. How would you scale this to all of India?**

* Make the city a setting and run the pipeline per station on a schedule (downloads are already incremental). The full version of this project already supports many cities.
* Store the data in a database instead of CSVs.
* Serve forecasts through an API.
* Use a spatial model (kriging or a graph neural network) to fill the gaps between stations.
* Retrain monthly, and alert when accuracy drifts.

**Q16. What are the main limitations?**

* Inherited weather-forecast errors.
* No information about sudden local sources.
* One station per city.
* Only a few years of data.
* Warning scores still need a winter season to be validated.
