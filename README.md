# Kolkata PM2.5 Forecasting & Early Warning System (Checkpoint 2)

This dashboard forecasts PM2.5 in Kolkata hour by hour, up to 60 hours ahead, and warns you before the air turns unhealthy.

It downloads real monitoring-station readings from **OpenAQ** and combines them with weather and the CAMS air-quality model from **Open-Meteo**. A machine-learning model (gradient-boosted quantile trees) produces the forecast, and an interactive **Streamlit** dashboard presents it.

> **This is the checkpoint-2 build.** It covers Kolkata only. The monitoring station is chosen automatically, and the warnings are shown in the dashboard. Other cities, choosing the station by hand, and Telegram/e-mail notifications belong to the full version and are not part of this build.
>
> A complete explanation of every file, function and modelling choice is in **`docs/CODE_WALKTHROUGH.html`** (open it in a browser). The same content is in `docs/CODE_WALKTHROUGH.md`.
>
> New to air-quality science? **`docs/CHART_GUIDE.html`** explains every chart in the dashboard from scratch: PM2.5, AQI, mixing height, wind, Kolkata's seasons. It also has a short demo script.

---

## What the three objectives look like in this build

| Objective | In this build |
|---|---|
| **1. Collect and combine data** | `pipeline.py fetch` finds the best PM2.5 monitor within 25 km of Kolkata. It downloads its hourly history (2023 onwards) from OpenAQ, adds weather (temperature, humidity, wind, rain, cloud, radiation, pressure, mixing height) and CAMS PM2.5 from Open-Meteo, cleans the readings and merges everything onto one hourly timeline |
| **2. Predict future air quality** | `pipeline.py train` trains three gradient-boosting models (10th, 50th and 90th percentiles) that forecast every hour from 1 to 60 hours ahead. Each forecast has a calibrated 80 % range and the probability of crossing any threshold |
| **3. Early warning dashboard** | `streamlit run app.py` shows a colour strip of the next 48 hours, warning levels (Clear / Watch / Warning / Emergency) with start time and duration, a daily outlook, the best time to go outdoors, the reasons behind the forecast, the model's track record and local pollution patterns |

---

## Getting started

Run all commands in a terminal **inside this folder**.

### 1. Create a Python virtual environment
**Windows**
1. Create the environment (only once):
   ```bat
   python -m venv .venv
   ```
2. Activate it (every time you open a new terminal):
   ```bat
   .venv\Scripts\activate
   ```
   Your prompt now starts with `(.venv)`.

   If PowerShell says *"running scripts is disabled"*, run this once and try again:
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the requirements
```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Add your OpenAQ API key
1. Get a free key at <https://explore.openaq.org> (Account → Settings).
2. Copy the template and paste the key into it:
   ```bat
   copy .env.example .env
   ```
   (`cp .env.example .env` on macOS/Linux.) Then edit `.env`:
   ```
   OPENAQ_API_KEY=your-key-here
   ```

### 4. Build the forecast (first time only)
**Real data.** The first run downloads about 3 years of history and trains the model, which takes about 5–10 minutes:
```bat
python pipeline.py all
```

**No key or no internet?** Use synthetic demo data:
```bat
python pipeline.py all --demo
```

**Already built Kolkata in the full project?** You can skip the download and training. Copy these three folders from the full project into this folder, keeping the same paths:
* `data\kolkata`
* `models\kolkata`
* `reports\kolkata`

The layout is identical, so the dashboard picks them up directly.

### 5. Start the dashboard
```bat
streamlit run app.py
```
It opens at <http://localhost:8501>. Stop it with `Ctrl + C`.

### Next time
```bat
.venv\Scripts\activate
python pipeline.py update        # latest readings + fresh forecast (about 1 minute)
streamlit run app.py
```

---

## Commands

| Command | What it does |
|---|---|
| `python pipeline.py all` | Download data, train the model, issue a forecast |
| `python pipeline.py update` | Fetch the latest readings and weather forecast, then re-forecast without retraining. The dashboard's **Refresh** button does the same |
| `python pipeline.py train` | Retrain the model and refresh the track record (weekly or monthly) |
| `python pipeline.py fetch` / `forecast` | Run just one stage |
| add `--demo` | Use synthetic data |

---

## The dashboard

* **Sidebar:**
  * **Your warning:** the profile you're planning for (each sets a default threshold), the threshold slider, and the length of the outdoor activity.
  * **Data:** the station in use, and **Refresh data and forecast**.
* **Header:**
  * the AQI of the last 24 hours (India CPCB scale)
  * the warning level and message, for example *"PM2.5 likely above 60 µg/m³ from Sat 21:30 (in 30 h) for about 11 h…"*
  * a **48-hour colour strip**, one block per hour coloured by category, with white ticks on hours above your threshold
  * health advice
* **Tiles:** last reading, peak in the next 24 h, peak in 24–48 h, and tomorrow's 24-hour mean with the chance of breaching India's limit.
* **Tabs:**

  | Tab | What it shows |
  |---|---|
  | **Forecast** | Measured PM2.5, the forecast with its 80 % band, and CAMS for comparison. Below it, the hourly chance of exceeding your threshold |
  | **Plan your day** | Daily outlook cards, the best outdoor window each day, an hourly table and a CSV download |
  | **Why** | How wind, mixing height, rain, temperature/humidity and regional pollution push the forecast up or down; a ventilation and rain chart; the model's most important inputs |
  | **Track record** | Out-of-sample accuracy against simple baselines and CAMS, warning statistics, a reliability chart, and recent test forecasts against measurements |
  | **Patterns** | Hour × month heatmap, pollution rose, days per month above the limit, weekday effect, data-quality report |
  | **Compared with weather apps** | What this adds beyond Google Weather |

---

## How it works (short version)
1. **Station:** monitors within 25 km are ranked (active, reference-grade, long history, close). OpenAQ often has an old and a new sensor ID for the same CPCB station, so all of a station's sensor IDs are joined into one series. The first station with enough recent data is used, and kept afterwards.
2. **Cleaning:** impossible values, low-coverage hours, stuck sensors and isolated spikes are removed. Only gaps of up to 3 h are interpolated. If the station stops reporting (6 hours to 14 days), the missing recent hours are estimated from CAMS scaled to the station, so forecasts still start from now. The dashboard flags this, and these estimates are never used for training.
3. **Model:** one gradient-boosting model per quantile predicts how PM2.5 will **change** from the latest reading, for every lead time from 1 to 60 hours.
   * **Inputs:** recent PM2.5 relative to now, forecast weather, the ventilation index (wind × mixing height), rain along the way, CAMS's expected change, and the hour and weekday.
   * **Blending:** for short lead times the change is blended toward "same as now", with the weights learned on held-out data.
   * **Calibration:** the 80 % band is calibrated with conformal scaling.
4. **Evaluation:** the last 25 % of the timeline is held out. The model is compared with persistence, "same hour yesterday" and CAMS, and the warnings are scored with POD, FAR, CSI and the Brier score.

On Kolkata data (forecasts issued April–September 2026), the mean absolute error was 3.6 µg/m³ over 0–24 h and 4.4 µg/m³ over 24–48 h. That is 18–24 % better than persistence and about 3 times more accurate than CAMS. Section 17 of the walkthrough has the details.

---

## Project structure
```
aqews-kolkata/
├── app.py                  Streamlit dashboard
├── pipeline.py             fetch | train | forecast | update | all   (+ --demo)
├── config.py               Kolkata settings, folders, thresholds, model settings
├── requirements.txt
├── .env.example            Copy to .env and add your OpenAQ key
├── .streamlit/config.toml  Dashboard theme
├── aq/
│   ├── openaq.py           OpenAQ v3 client
│   ├── openmeteo.py        Weather + CAMS client
│   ├── dataset.py          Quality control, hourly merge, outage bridging
│   ├── features.py         Feature engineering
│   ├── model.py            Training, calibration, backtest, forecast, explanations
│   ├── insights.py         AQI, warning levels, daily outlook, best windows
│   ├── ui.py               Dashboard styling and header strip
│   └── demo.py             Synthetic offline data
├── docs/
│   ├── CODE_WALKTHROUGH.html   Full project and code walkthrough (open in a browser)
│   ├── CODE_WALKTHROUGH.md     Same content in Markdown
│   ├── CHART_GUIDE.html        Every chart explained for beginners (with screenshots)
│   ├── CHART_GUIDE.md          Same content in Markdown (images in chart_guide_images/)
│   └── src/                    Section files + build.py (python docs/src/build.py rebuilds both)
├── data/kolkata/           Downloaded and processed data   (created on first run)
├── models/kolkata/         Trained model                   (created on first run)
└── reports/kolkata/        Forecast, drivers, backtest     (created on first run)
```

---

## Demo-day checklist
1. The day before, run `python pipeline.py all` (or copy your Kolkata folders, see step 4) and check that `streamlit run app.py` works.
2. Just before the demo, run `python pipeline.py update` so the forecast starts from the latest reading.
3. If the internet is unreliable at the venue, keep a demo build ready: `python pipeline.py all --demo` (it shows a clear "Demo mode" notice). Run it in a **copy** of this folder, because it overwrites the real Kolkata data.
4. A good talking order:
   1. the header and warning
   2. the Forecast tab (band and probabilities)
   3. Why (drivers)
   4. Track record (beats persistence and CAMS)
   5. Compared with weather apps

---

## Troubleshooting
| Problem | Fix |
|---|---|
| `OPENAQ_API_KEY is not set` | Create `.env` with your key (step 3) |
| `'streamlit' is not recognized` | Activate the environment first: `.venv\Scripts\activate` |
| *"The station has not reported for N h"* | The OpenAQ feed for Kolkata is delayed or down. Gaps under 14 days are estimated automatically. Click **Refresh** later |
| *"None of the nearby OpenAQ stations returned PM2.5 data"* | The feed is down for the whole area. Try again later, or use `--demo` |
| Page says *"No forecast yet"* | Run `python pipeline.py all` (or `--demo`) once, then reload the page |
| Console warning about `use_container_width` | Upgrade Streamlit: `pip install -U streamlit` |

---

## Data sources
* **OpenAQ:** <https://openaq.org>, CC BY 4.0. Kolkata's data comes from WBPCB/CPCB monitors.
* **Open-Meteo** weather and air-quality APIs: <https://open-meteo.com>, CC BY 4.0.
* **CAMS:** the Copernicus Atmosphere Monitoring Service, delivered through Open-Meteo.
* **AQI method:** Central Pollution Control Board (CPCB), National Air Quality Index.
