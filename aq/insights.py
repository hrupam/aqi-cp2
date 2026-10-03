"""Turn a raw quantile forecast into decisions: alert level, timing, daily outlook,
best outdoor windows, exposure and plain-language reasons."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from aq.model import exceed_prob


def category(v: float) -> tuple[str, str]:
    if v is None or not np.isfinite(v):
        return "No data", "#9e9e9e"
    for name, upper, color in config.CATEGORIES:
        if v <= upper:
            return name, color
    return config.CATEGORIES[-1][0], config.CATEGORIES[-1][2]


# ---------------------------------------------------------------- India CPCB AQI
# (concentration breakpoints, AQI breakpoints); officially applied to 24-h means
_AQI_BP = {
    "pm25": ([0, 30, 60, 90, 120, 250, 500], [0, 50, 100, 200, 300, 400, 500]),
    "pm10": ([0, 50, 100, 250, 350, 430, 600], [0, 50, 100, 200, 300, 400, 500]),
}
AQI_CATEGORIES = [("Good", 50, "#3FA34D"), ("Satisfactory", 100, "#A3C94F"), ("Moderate", 200, "#F2C230"),
                  ("Poor", 300, "#F08A24"), ("Very Poor", 400, "#D9412B"), ("Severe", 10_000, "#7E3BA6")]


def sub_index(conc, pollutant: str = "pm25"):
    """CPCB sub-index for PM2.5 or PM10 concentration(s) in µg/m³ (capped at 500)."""
    c_bp, i_bp = _AQI_BP[pollutant]
    return np.interp(np.asarray(conc, float), c_bp, i_bp, right=500.0)


def aqi(pm25=None, pm10=None) -> float:
    """AQI = the worse of the available sub-indices."""
    vals = [float(sub_index(v, k)) for k, v in (("pm25", pm25), ("pm10", pm10))
            if v is not None and np.isfinite(v)]
    return max(vals) if vals else float("nan")


def aqi_category(a: float) -> tuple[str, str]:
    if a is None or not np.isfinite(a):
        return "No data", "#9e9e9e"
    for name, upper, color in AQI_CATEGORIES:
        if a <= upper:
            return name, color
    return AQI_CATEGORIES[-1][0], AQI_CATEGORIES[-1][2]


def localize(fc: pd.DataFrame) -> pd.DataFrame:
    fc = fc.copy()
    for c in ("issue_time", "target_time"):
        fc[c] = pd.to_datetime(fc[c], utc=True)
    fc["local_time"] = fc["target_time"].dt.tz_convert(config.LOCAL_TZ)
    return fc


def upcoming(fc: pd.DataFrame, now: pd.Timestamp | None = None, hours: int = 48) -> pd.DataFrame:
    """Forecast rows for the next `hours` from now (handles station reporting delay)."""
    now = (now or pd.Timestamp.now(tz="UTC")).floor("h")
    return fc[(fc["target_time"] > now) & (fc["target_time"] <= now + pd.Timedelta(hours=hours))].copy()


def alert_summary(fc: pd.DataFrame, threshold: float, now: pd.Timestamp | None = None) -> dict:
    now = now or pd.Timestamp.now(tz="UTC")
    f = upcoming(fc, now)
    if f.empty:
        return {"level": "Unknown", "message": "Forecast is out of date - run `python pipeline.py update`."}
    f["p"] = exceed_prob(f.q10, f.q50, f.q90, threshold)
    pmax = float(f["p"].max())
    peak = f.loc[f["q50"].idxmax()]
    likely = f[f["p"] >= 0.5]
    out = {"threshold": threshold, "max_prob": pmax, "peak_value": float(peak.q50),
           "peak_lo": float(peak.q10), "peak_hi": float(peak.q90), "peak_time": peak.local_time,
           "hours_above": int(len(likely))}
    if not likely.empty:
        first = likely.iloc[0]
        out["start_time"] = first.local_time
        out["lead_hours"] = float((first.target_time - now).total_seconds() / 3600)
        out["end_time"] = likely.iloc[-1].local_time
    if peak.q50 > 250 and pmax >= 0.5:
        level = "EMERGENCY"
    elif pmax >= 0.6:
        level = "WARNING"
    elif pmax >= 0.3:
        level = "WATCH"
    else:
        level = "CLEAR"
    out["level"] = level
    if level == "CLEAR":
        out["message"] = (f"No exceedance of {threshold:.0f} µg/m³ expected in the next 48 h "
                          f"(highest chance {pmax:.0%}).")
    elif "start_time" in out:
        out["message"] = (f"PM2.5 likely above {threshold:.0f} µg/m³ from "
                          f"{out['start_time']:%a %d %b, %H:%M} (in {out['lead_hours']:.0f} h) "
                          f"for about {out['hours_above']} h. Peak ≈ {peak.q50:.0f} µg/m³ "
                          f"(80% range {peak.q10:.0f}–{peak.q90:.0f}) around {peak.local_time:%a %H:%M}. "
                          f"Chance of exceedance up to {pmax:.0%}.")
    else:
        out["message"] = (f"Possible exceedance of {threshold:.0f} µg/m³ (up to {pmax:.0%} chance), "
                          f"most likely around {f.loc[f['p'].idxmax()].local_time:%a %d %b %H:%M}.")
    return out


def daily_outlook(fc: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """24-h mean for today / tomorrow / day after, compared with the 24-h NAAQS.
    Daily quantiles are averaged hourly quantiles (a slightly wide, conservative band)."""
    now = now or pd.Timestamp.now(tz="UTC")
    f = fc[fc["target_time"] > now - pd.Timedelta(hours=1)].copy()
    f["day"] = f["local_time"].dt.date
    d = f.groupby("day").agg(q10=("q10", "mean"), q50=("q50", "mean"), q90=("q90", "mean"),
                             peak=("q90", "max"), hours=("q50", "size")).reset_index()
    d = d[d["hours"] >= 12]
    d["p_naaqs"] = exceed_prob(d.q10, d.q50, d.q90, config.NAAQS_24H)
    d["x_who"] = d["q50"] / config.WHO_24H
    d["cigarettes"] = d["q50"] / 22.0          # Berkeley Earth rule of thumb: 22 µg/m³ for 24 h ≈ 1 cigarette
    d["category"] = d["q50"].apply(lambda v: category(v)[0])
    return d.head(3)


def best_windows(fc: pd.DataFrame, duration: int = 2, start_hour: int = 6, end_hour: int = 21,
                 now: pd.Timestamp | None = None, per_day: int = 1) -> pd.DataFrame:
    """Cleanest daytime windows for outdoor activity in the next 48 h."""
    f = upcoming(fc, now).reset_index(drop=True)
    if len(f) < duration:
        return pd.DataFrame()
    f["roll"] = f["q50"].rolling(duration).mean().shift(-(duration - 1))
    f["roll_hi"] = f["q90"].rolling(duration).mean().shift(-(duration - 1))
    h = f["local_time"].dt.hour
    f = f[(h >= start_hour) & (h + duration <= end_hour + 1) & f["roll"].notna()]
    f["day"] = f["local_time"].dt.date
    best = f.sort_values("roll").groupby("day").head(per_day).sort_values("local_time")
    worst = f.sort_values("roll", ascending=False).groupby("day").head(1).set_index("day")["roll"]
    best = best.assign(start=best["local_time"],
                       end=best["local_time"] + pd.Timedelta(hours=duration),
                       worst_same_day=best["day"].map(worst))
    return best[["day", "start", "end", "roll", "roll_hi", "worst_same_day"]]


def driver_sentences(drivers: pd.DataFrame, col: str = "effect_24_48h", min_effect: float = 3) -> list[str]:
    out = []
    if drivers is None or drivers.empty:
        return out
    for _, r in drivers.reindex(drivers[col].abs().sort_values(ascending=False).index).iterrows():
        e = r[col]
        if abs(e) < min_effect:
            continue
        name = r["driver"]
        if e > 0:
            why = {
                "Wind (speed & direction)": "weaker or polluted-direction winds will let pollution build up",
                "Mixing height (atmospheric lid)": "a low mixing layer traps pollution near the ground",
                "Rain & clouds": "little rain to wash particles out",
                "Temperature, humidity & pressure": "cool, humid, stable air favours particle formation & trapping",
                "Regional pollution (CAMS model)": "polluted air mass arriving from the region (transport)",
            }.get(name, "")
        else:
            why = {
                "Wind (speed & direction)": "stronger / cleaner winds will disperse pollution",
                "Mixing height (atmospheric lid)": "a deep mixing layer dilutes pollution",
                "Rain & clouds": "rain will wash particles out",
                "Temperature, humidity & pressure": "unstable, warm air helps dispersion",
                "Regional pollution (CAMS model)": "cleaner regional background air",
            }.get(name, "")
        out.append(f"**{name}**: {e:+.0f} µg/m³ vs a typical day - {why}.")
    return out


HEALTH_ADVICE = {
    "Good": "Air is clean. Enjoy outdoor activities.",
    "Satisfactory": "Unusually sensitive people should consider limiting prolonged heavy exertion.",
    "Moderate": "Sensitive groups (asthma, heart/lung disease, children, elderly) should reduce long or intense outdoor exertion.",
    "Poor": "Everyone should reduce prolonged outdoor exertion; sensitive groups avoid it. Keep windows closed during peaks.",
    "Very Poor": "Avoid outdoor exertion. Use an N95/FFP2 mask outdoors; run an air purifier indoors.",
    "Severe": "Health emergency conditions. Stay indoors with filtered air; avoid all outdoor activity.",
}
