
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
