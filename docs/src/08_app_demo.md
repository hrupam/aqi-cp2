
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
