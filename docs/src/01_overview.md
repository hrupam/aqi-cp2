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
