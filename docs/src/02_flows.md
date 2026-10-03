
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
