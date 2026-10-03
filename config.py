"""
Central configuration for the Kolkata PM2.5 Forecasting & Early Warning System (checkpoint-2 build).

This build covers one city, Kolkata. The monitoring station is chosen automatically
(see pipeline._station_pm25). Your OpenAQ API key is read from OPENAQ_API_KEY
(or a .env file in this folder).
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent


def _load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "")

# ---------------------------------------------------------------- location (fixed: Kolkata)
CITY = "Kolkata"
LAT, LON = 22.5726, 88.3639        # city centre
LOCAL_TZ = "Asia/Kolkata"
SEARCH_RADIUS_M = 25_000           # look for PM2.5 monitors within 25 km (the OpenAQ maximum)

# ---------------------------------------------------------------- folders
RAW_DIR = ROOT / "data" / "kolkata" / "raw"            # downloaded station, weather and CAMS data
PROC_DIR = ROOT / "data" / "kolkata" / "processed"     # merged hourly table + metadata
MODEL_DIR = ROOT / "models" / "kolkata"                # trained model
REPORT_DIR = ROOT / "reports" / "kolkata"              # forecast, drivers, backtest
for _d in (RAW_DIR, PROC_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- history window
# Open-Meteo historical forecasts (with boundary-layer height) start ~2022,
# CAMS air-quality history starts Aug 2022.
TRAIN_START = os.environ.get("AQ_TRAIN_START", "2023-01-01")

# ---------------------------------------------------------------- model
MAX_HORIZON = 60          # hours; 48 h + slack for station reporting delay
QUANTILES = (0.1, 0.5, 0.9)
ISSUE_STRIDE = 3          # use every 3rd hour as a forecast issue time when training
TEST_FRACTION = 0.25      # last 25 % of the timeline is held out for the backtest

# ---------------------------------------------------------------- thresholds (µg/m³)
# CPCB (India) PM2.5 categories; upper bound of each band.
CATEGORIES = [
    ("Good",         30,  "#3FA34D"),
    ("Satisfactory", 60,  "#A3C94F"),
    ("Moderate",     90,  "#F2C230"),
    ("Poor",         120, "#F08A24"),
    ("Very Poor",    250, "#D9412B"),
    ("Severe",       10_000, "#7E3BA6"),
]
NAAQS_24H = 60      # India National Ambient Air Quality Standard, 24-h mean
WHO_24H = 15        # WHO 2021 guideline, 24-h mean

# Warning threshold by user profile (hourly PM2.5 that triggers a warning in the dashboard)
PROFILES = {
    "General public": 90,
    "Sensitive (asthma / heart / lung)": 60,
    "Children & elderly": 60,
    "Outdoor workers / athletes": 60,
    "Strict (WHO-based)": 35,
}
