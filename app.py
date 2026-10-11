"""
Kolkata PM2.5 Early Warning dashboard.     Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from aq import insights as I, ui
from aq.model import exceed_prob

st.set_page_config(page_title="Kolkata PM2.5 Early Warning", page_icon="🌫️", layout="wide")
ui.inject()


def _pipeline_env() -> dict:
    """Environment for the Refresh subprocess. Locally the key comes from .env; on Streamlit
    Community Cloud it is stored in the app's Secrets, which a child process does not see
    unless we pass it on explicitly."""
    env = os.environ.copy()
    try:
        key = st.secrets.get("OPENAQ_API_KEY")
    except Exception:                       # no secrets file at all (normal when running locally)
        key = None
    if key:
        env["OPENAQ_API_KEY"] = str(key).strip()
    return env


# ------------------------------------------------------------------ cached loaders
def _mtime(p):
    return p.stat().st_mtime if p.exists() else 0


@st.cache_data
def load_all(_stamp):
    hourly = pd.read_csv(config.PROC_DIR / "hourly.csv", parse_dates=["time"]).set_index("time")
    hourly.index = pd.to_datetime(hourly.index, utc=True)
    meta = json.loads((config.PROC_DIR / "meta.json").read_text())
    fc = I.localize(pd.read_csv(config.REPORT_DIR / "forecast.csv"))
    drivers = pd.read_csv(config.REPORT_DIR / "drivers.csv")
    bt_file = config.REPORT_DIR / "backtest.json"
    backtest = json.loads(bt_file.read_text()) if bt_file.exists() else None
    btp_file = config.REPORT_DIR / "backtest_predictions.csv"
    btp = pd.read_csv(btp_file, parse_dates=["issue_time", "target_time"]) if btp_file.exists() else None
    return hourly, meta, fc, drivers, backtest, btp


now = pd.Timestamp.now(tz="UTC")
files = [config.PROC_DIR / "hourly.csv", config.REPORT_DIR / "forecast.csv",
         config.MODEL_DIR / "pm25_quantile_gbm.joblib"]
if not all(f.exists() for f in files):
    st.title("Kolkata PM2.5 Early Warning")
    st.warning("No forecast yet. Build it once from a terminal in this folder, then reload this page:")
    st.code("python pipeline.py all          # real data (needs OPENAQ_API_KEY in .env)\n"
            "python pipeline.py all --demo   # offline demo with synthetic data")
    st.stop()

hourly, meta, fc, drivers, backtest, btp = load_all(sum(_mtime(f) for f in files))

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(ui.brand(), unsafe_allow_html=True)
    st.header("Your warning")
    profile = st.selectbox("Who are you planning for?", list(config.PROFILES), index=1)
    threshold = st.slider("Warn me when PM2.5 exceeds (µg/m³)", 15, 250, config.PROFILES[profile], step=5)
    win_len = st.slider("Outdoor activity length (h)", 1, 4, 2)
    st.divider()
    st.header("Data")
    st.caption(f"Station: {meta.get('station')} ({meta.get('provider')})" if meta.get("mode") != "demo"
               else "Demo data: synthetic station.")
    if st.button("Refresh data and forecast"):
        args = [sys.executable, "pipeline.py", "update"] + (["--demo"] if meta.get("mode") == "demo" else [])
        with st.spinner("Downloading latest observations & weather forecast ..."):
            r = subprocess.run(args, cwd=config.ROOT, capture_output=True, text=True, env=_pipeline_env())
        if r.returncode == 0:
            st.cache_data.clear()
            st.rerun()
        else:
            st.error((r.stderr or r.stdout)[-1500:])

# ------------------------------------------------------------------ header
if meta.get("mode") == "demo":
    st.info("Demo mode: these are synthetic data. Run `python pipeline.py all` with an OpenAQ key for real data.")

last_obs_t = hourly["pm25"].last_valid_index()
last_obs = float(hourly.loc[last_obs_t, "pm25"])
issue = fc["issue_time"].iloc[0]
lag_h = (now - last_obs_t).total_seconds() / 3600

fresh = meta.get("freshness", {})
fc_end = fc["target_time"].max()
alert = I.alert_summary(fc, threshold, now)
peak_cat = I.category(alert.get("peak_value", np.nan))[0]
up = I.upcoming(fc, now)
daily = I.daily_outlook(fc, now)
pm_all = hourly["pm25"].combine_first(hourly["pm25_bridge"]) if "pm25_bridge" in hourly else hourly["pm25"]
aqi_now = I.aqi(pm_all[pm_all.index <= now].dropna().tail(24).mean())      # last 24 available hours
aqi_name, aqi_col = I.aqi_category(aqi_now)

ago = f"{lag_h:.0f} hours ago" if lag_h >= 1.5 else "within the last hour"
st.markdown(ui.hero(
    place=meta.get("city"),
    subline=f"{meta.get('station')}, last measured {ago}",
    aqi_value=aqi_now, aqi_cat=aqi_name, aqi_color=aqi_col, aqi_label="AQI, last 24 hours",
    level=alert["level"], message=alert["message"],
    hours=up.head(48).rename(columns={"q50": "value"})[["local_time", "value"]],
    threshold=threshold,
    advice=f"At the peak ({peak_cat}): {I.HEALTH_ADVICE.get(peak_cat, '')}" if peak_cat != "No data" else ""),
    unsafe_allow_html=True)

if fresh.get("bridged") and lag_h > 6:
    st.warning(f"The station has not reported for {lag_h:.0f} h. The missing hours are estimated from the "
               f"CAMS model × {fresh['bridge_ratio']:.2f} (its recent local bias), so the forecast still starts from now. "
               "Treat it as lower-confidence until the station is back.")
elif lag_h > 12:
    st.warning(f"The station's latest reading is {lag_h:.0f} h old, so forecasts further ahead are less reliable. "
               "Use Refresh in the sidebar later.")
if fc_end <= now:
    st.error(f"This forecast ended at {fc_end.tz_convert(config.LOCAL_TZ):%a %d %b %H:%M} because the station stopped "
             "reporting. Click Refresh in the sidebar: outages under 14 days are bridged with CAMS.")

items = [("Last reading", f"{last_obs:.0f} <small>µg/m³</small>",
          f"{I.category(last_obs)[0]}, {last_obs_t.tz_convert(config.LOCAL_TZ):%a %H:%M}", I.category(last_obs)[1])]
nxt = up[up["target_time"] <= now + pd.Timedelta(hours=24)]
lat = up[up["target_time"] > now + pd.Timedelta(hours=24)]
for label, part in (("Peak in the next 24 h", nxt), ("Peak in 24–48 h", lat)):
    if not part.empty:
        r = part.loc[part.q50.idxmax()]
        items.append((label, f"{r.q50:.0f} <small>µg/m³</small>",
                      f"{I.category(r.q50)[0]}, {r.local_time:%a %H:%M}", I.category(r.q50)[1]))
if len(daily) > 1:
    d = daily.iloc[1]
    items.append((f"Tomorrow's 24-h mean", f"{d.q50:.0f} <small>µg/m³</small>",
                  f"{d.p_naaqs:.0%} chance above the national limit of {config.NAAQS_24H}", I.category(d.q50)[1]))
st.markdown(ui.tiles(items), unsafe_allow_html=True)

tabs = st.tabs(["Forecast", "Plan your day", "Why", "Track record", "Patterns", "Compared with weather apps"])


def add_category_bands(fig, ymax):
    lo = 0
    for name, hi, color in config.CATEGORIES:
        if lo >= ymax:
            break
        fig.add_hrect(y0=lo, y1=min(hi, ymax), fillcolor=color, opacity=0.08, line_width=0,
                      annotation_text=name, annotation_position="top left",
                      annotation_font_size=10, annotation_font_color="#666")
        lo = hi


# ------------------------------------------------------------------ tab 1: forecast
with tabs[0]:
    hist = hourly.loc[last_obs_t - pd.Timedelta(hours=96): last_obs_t, "pm25"]
    hist_local = hist.index.tz_convert(config.LOCAL_TZ)
    ymax = float(np.nanmax([fc.q90.max(), hist.max(), threshold]) * 1.15)
    fig = go.Figure()
    add_category_bands(fig, ymax)
    fig.add_trace(go.Scatter(x=fc.local_time, y=fc.q90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc.local_time, y=fc.q10, fill="tonexty", fillcolor="rgba(47,111,222,0.16)",
                             line=dict(width=0), name="80% range"))
    fig.add_trace(go.Scatter(x=fc.local_time, y=fc.q50, name="Our forecast (median)",
                             line=dict(color="#2F6FDE", width=3)))
    fig.add_trace(go.Scatter(x=hist_local, y=hist.values, name="Measured", line=dict(color="#1B2A41", width=2)))
    if "pm25_bridge" in hourly and hourly["pm25_bridge"].notna().any():
        b = hourly["pm25_bridge"].dropna()
        fig.add_trace(go.Scatter(x=b.index.tz_convert(config.LOCAL_TZ), y=b.values, name="Estimated (station offline)",
                                 line=dict(color="#1B2A41", width=2, dash="dot")))
    if fc["cams"].notna().any():
        fig.add_trace(go.Scatter(x=fc.local_time, y=fc.cams, name="CAMS global model (what weather apps show)",
                                 line=dict(color="#888", dash="dot")))
    fig.add_hline(y=threshold, line=dict(color="#c62828", dash="dash"),
                  annotation_text=f"your alert threshold {threshold}", annotation_position="bottom right")
    fig.add_vline(x=now.tz_convert(config.LOCAL_TZ), line=dict(color="#555", dash="dot"))
    fig.update_layout(height=470, yaxis_title="PM2.5 (µg/m³)", yaxis_range=[0, ymax], hovermode="x unified",
                      margin=dict(t=30, b=10), legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

    up["p"] = exceed_prob(up.q10, up.q50, up.q90, threshold)
    figp = go.Figure(go.Bar(x=up.local_time, y=up.p * 100,
                            marker_color=np.where(up.p >= 0.6, "#c62828", np.where(up.p >= 0.3, "#f9a825", "#90a4ae")),
                            hovertemplate="%{x|%a %H:%M}: %{y:.0f}%<extra></extra>"))
    figp.update_layout(height=230, title=f"Chance PM2.5 exceeds {threshold} µg/m³, hour by hour",
                       yaxis_title="%", yaxis_range=[0, 100], margin=dict(t=40, b=10))
    st.plotly_chart(figp, width="stretch")

# ------------------------------------------------------------------ tab 2: plan
with tabs[1]:
    st.subheader("Daily outlook")
    if daily.empty:
        st.info("No forecast hours left for the coming days - refresh the data first.")
    cols = st.columns(max(len(daily), 1))
    for col, (_, d) in zip(cols, daily.iterrows()):
        name, color = I.category(d.q50)
        col.markdown(
            f"""<div class="day" style="--c:{color}"><h4>{pd.Timestamp(d.day):%A %d %b}</h4>
            <div class="big">{d.q50:.0f} <small>µg/m³ 24-h mean</small></div>
            <p><b>{name}</b>, likely between {d.q10:.0f} and {d.q90:.0f}</p>
            <p>{d.x_who:.1f} times the WHO guideline, like smoking {d.cigarettes:.1f} cigarettes</p>
            <p>{d.p_naaqs:.0%} chance of breaching India's 24-hour limit ({config.NAAQS_24H})</p></div>""",
            unsafe_allow_html=True)
    st.caption("Cigarette equivalence: Berkeley Earth rule of thumb (22 µg/m³ over 24 h ≈ 1 cigarette). "
               "Partial days show only the remaining forecast hours.")

    st.subheader(f"Best {win_len}-hour window for outdoor activity (06:00–21:00)")
    bw = I.best_windows(fc, duration=win_len, now=now)
    if bw.empty:
        st.write("Not enough forecast hours left.")
    else:
        for _, r in bw.iterrows():
            gain = r.worst_same_day - r.roll
            st.markdown(f"- **{r.start:%a %d %b %H:%M}–{r.end:%H:%M}** · expected ≈ {r.roll:.0f} µg/m³ "
                        f"({I.category(r.roll)[0]}) – about {gain:.0f} µg/m³ cleaner than the worst slot that day")

    st.subheader("Hour by hour (next 48 h)")
    tbl = up.assign(Time=up.local_time.dt.strftime("%a %d %b %H:%M"),
                    Forecast=up.q50.round(0), Low=up.q10.round(0), High=up.q90.round(0),
                    Category=up.q50.apply(lambda v: I.category(v)[0]),
                    **{"Chance > threshold": (up.p * 100).round(0).astype(int).astype(str) + "%"})
    st.dataframe(tbl[["Time", "Forecast", "Low", "High", "Category", "Chance > threshold"]],
                 hide_index=True, width="stretch", height=360)
    st.download_button("⬇️ Download forecast (CSV)", fc.to_csv(index=False), "pm25_forecast.csv")

# ------------------------------------------------------------------ tab 3: why
with tabs[2]:
    st.subheader("What is driving the forecast?")
    st.caption("Each weather factor is swapped for its typical value for this month and hour, "
               "and the model is re-run. Bars show how much that factor raises (+) or lowers (−) PM2.5.")
    if not drivers.empty:
        dd = drivers.melt(id_vars="driver", var_name="period", value_name="effect")
        dd["period"] = dd["period"].map({"effect_next24h": "Next 24 h", "effect_24_48h": "24–48 h"})
        figd = go.Figure()
        for per, color in (("Next 24 h", "#90caf9"), ("24–48 h", "#2F6FDE")):
            s = dd[dd.period == per]
            figd.add_trace(go.Bar(y=s.driver, x=s.effect, name=per, orientation="h", marker_color=color))
        figd.update_layout(height=340, barmode="group", xaxis_title="µg/m³ vs a typical day", margin=dict(t=10))
        st.plotly_chart(figd, width="stretch")
        for s in I.driver_sentences(drivers) or ["No single weather factor stands out - conditions are near typical."]:
            st.markdown(f"- {s}")

    st.subheader("Weather behind it: ventilation & rain")
    w = hourly.loc[last_obs_t - pd.Timedelta(hours=24): last_obs_t + pd.Timedelta(hours=config.MAX_HORIZON)]
    wl = w.index.tz_convert(config.LOCAL_TZ)
    figw = go.Figure()
    if "blh" in w and w["blh"].notna().any():
        figw.add_trace(go.Scatter(x=wl, y=w.wind_speed * w.blh, name="Ventilation index (wind × mixing height, m²/s)",
                                  line=dict(color="#00897b")))
    figw.add_trace(go.Bar(x=wl, y=w.precip, name="Rain (mm/h)", yaxis="y2", marker_color="#5c6bc0"))
    figw.add_hrect(y0=0, y1=2000, fillcolor="#e53935", opacity=0.07, line_width=0,
                   annotation_text="stagnant (< 2000)", annotation_position="top left")
    figw.update_layout(height=320, yaxis=dict(title="m²/s"), margin=dict(t=10),
                       yaxis2=dict(title="mm/h", overlaying="y", side="right", showgrid=False),
                       legend=dict(orientation="h", y=1.12))
    st.plotly_chart(figw, width="stretch")

    if backtest and backtest.get("importance"):
        imp = pd.Series(backtest["importance"]).head(12)[::-1]
        figi = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#546e7a"))
        figi.update_layout(height=380, title="Most important model inputs overall (permutation importance)",
                           margin=dict(t=40))
        st.plotly_chart(figi, width="stretch")
        st.caption("The model predicts how PM2.5 will CHANGE from the latest reading. "
                   "Prefixes: p_ = recent measured PM2.5 (relative to now), w_ = weather at the target hour, "
                   "c_cams_change = how much CAMS expects regional PM2.5 to change, "
                   "k_ = calendar, d_ = change from now, path_ = conditions between now and the target hour, h = lead time.")

# ------------------------------------------------------------------ tab 4: track record
with tabs[3]:
    if not backtest:
        st.info("Run `python pipeline.py train` to produce the backtest.")
    else:
        p = backtest["period"]
        st.caption(f"Out-of-sample test: forecasts issued {p['test_start'][:10]} → {p['test_end'][:10]} "
                   "(the model never saw these hours while training).")
        a, b = backtest["summary_0-24h"], backtest["summary_24-48h"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Error 0–24 h (MAE)", f"{a['mae_model']:.1f} µg/m³",
                  f"{a['skill_vs_persistence']:+.0%} vs persistence")
        k2.metric("Error 24–48 h (MAE)", f"{b['mae_model']:.1f} µg/m³",
                  f"{b['skill_vs_persistence']:+.0%} vs persistence")
        if "skill_vs_cams" in b:
            k3.metric("Better than CAMS (24–48 h)", f"{b['skill_vs_cams']:+.0%}", "lower error than global model")
        k4.metric("80% range really contains truth", f"{b['coverage_80']:.0%}", "target 80%", delta_color="off")

        bh = pd.DataFrame(backtest["by_horizon"])
        fig = go.Figure()
        for col, name, style in (("mae_model", "Our model", dict(color="#2F6FDE", width=3)),
                                 ("mae_persistence", "Persistence (now = later)", dict(color="#999")),
                                 ("mae_same_hour", "Same hour yesterday", dict(color="#bbb", dash="dash")),
                                 ("mae_cams", "CAMS global model", dict(color="#f57c00", dash="dot"))):
            if bh[col].notna().any():
                fig.add_trace(go.Scatter(x=bh.h, y=bh[col], name=name, line=style))
        fig.update_layout(height=360, xaxis_title="Lead time (h)", yaxis_title="Mean absolute error (µg/m³)",
                          margin=dict(t=10), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

        st.subheader("How good are the warnings? (24–48 h ahead, warn when chance ≥ 50%)")
        al = pd.DataFrame(backtest["alerts"]).T.reset_index().rename(columns={"index": "Threshold µg/m³"})
        al = al.rename(columns={"events": "Event hours", "POD": "Caught (POD)", "FAR": "False alarms (FAR)",
                                "CSI": "CSI", "brier": "Brier score"})
        st.dataframe(al, hide_index=True, width="stretch")

        c1, c2 = st.columns(2)
        rel = pd.DataFrame(backtest.get("reliability", []))
        if not rel.empty:
            figr = go.Figure()
            figr.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(color="#aaa", dash="dash"), name="perfect"))
            figr.add_trace(go.Scatter(x=rel.forecast_prob, y=rel.observed_freq, mode="lines+markers",
                                      name="model", marker=dict(size=np.clip(np.sqrt(rel.n) / 3, 6, 22))))
            figr.update_layout(height=340, title="Reliability: when we say X%, does it happen X% of the time? (>90)",
                               xaxis_title="Forecast probability", yaxis_title="Observed frequency",
                               margin=dict(t=40))
            c1.plotly_chart(figr, width="stretch")
        if btp is not None:
            hsel = c2.selectbox("Lead time to inspect", sorted(btp.h.unique()), index=min(4, btp.h.nunique() - 1))
            s = btp[btp.h == hsel].sort_values("target_time")
            s = s[s.target_time >= s.target_time.max() - pd.Timedelta(days=30)]
            # the scored weeks alternate with calibration weeks: break the lines across the gaps
            gap = s.target_time.diff() > pd.Timedelta(hours=12)
            if gap.any():
                fill = s[gap].assign(target_time=s[gap].target_time - pd.Timedelta(hours=1),
                                     y=np.nan, q10=np.nan, q50=np.nan, q90=np.nan)
                s = pd.concat([s, fill]).sort_values("target_time")
            figb = go.Figure()
            figb.add_trace(go.Scatter(x=s.target_time, y=s.q90, line=dict(width=0), showlegend=False))
            figb.add_trace(go.Scatter(x=s.target_time, y=s.q10, fill="tonexty", line=dict(width=0),
                                      fillcolor="rgba(47,111,222,0.16)", name="80% range"))
            figb.add_trace(go.Scatter(x=s.target_time, y=s.q50, name=f"Forecast {hsel} h ahead",
                                      line=dict(color="#2F6FDE")))
            figb.add_trace(go.Scatter(x=s.target_time, y=s.y, name="Measured", line=dict(color="#1B2A41")))
            figb.update_layout(height=340, title="Last 30 days of the test period", margin=dict(t=40),
                               legend=dict(orientation="h", y=-0.2))
            c2.plotly_chart(figb, width="stretch")

# ------------------------------------------------------------------ tab 5: insights
with tabs[4]:
    h = hourly[hourly.pm25.notna()].copy()
    loc = h.index.tz_convert(config.LOCAL_TZ)
    h["month"], h["hour"], h["dow"] = loc.month, loc.hour, loc.dayofweek
    c1, c2 = st.columns(2)
    hm = h.pivot_table(index="hour", columns="month", values="pm25", aggfunc="mean")
    figh = go.Figure(go.Heatmap(z=hm.values, x=[pd.Timestamp(2000, m, 1).strftime("%b") for m in hm.columns],
                                y=hm.index, colorscale="YlOrRd", colorbar=dict(title="µg/m³")))
    figh.update_layout(height=420, title="Typical PM2.5 by hour of day and month", yaxis_title="Hour (IST)",
                       margin=dict(t=40))
    c1.plotly_chart(figh, width="stretch")

    if "wind_dir" in h and h["wind_dir"].notna().any():
        hw = h.dropna(subset=["wind_dir"])                  # hours without weather data can't be placed
        sector = ((hw.wind_dir + 22.5) // 45 % 8).astype(int)
        names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        rose = hw.pm25.groupby(sector).mean().reindex(range(8))
        figr = go.Figure(go.Barpolar(r=rose.values, theta=names, marker_color=rose.values,
                                     marker_colorscale="YlOrRd"))
        figr.update_layout(height=420, title="Pollution rose: mean PM2.5 by wind direction (where dirty air comes from)",
                           margin=dict(t=60),
                           polar=dict(angularaxis=dict(rotation=90, direction="clockwise")))   # compass: N up, E right
        c2.plotly_chart(figr, width="stretch")

    daily_obs = h.pm25.resample("D").mean()
    ex = (daily_obs > config.NAAQS_24H).groupby(daily_obs.index.to_period("M")).sum()
    fige = go.Figure(go.Bar(x=ex.index.astype(str), y=ex.values, marker_color="#e53935"))
    fige.update_layout(height=300, title=f"Days per month with 24-h mean above NAAQS ({config.NAAQS_24H} µg/m³)",
                       margin=dict(t=40))
    st.plotly_chart(fige, width="stretch")

    wk = h.groupby("dow").pm25.mean()
    c1, c2 = st.columns(2)
    c1.plotly_chart(go.Figure(go.Bar(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=wk.values,
                                     marker_color="#546e7a")).update_layout(
        height=280, title="Weekday effect (traffic & activity)", margin=dict(t=40)), width="stretch")
    c2.markdown("**Data quality report**")
    c2.json(meta.get("qc", {}))

# ------------------------------------------------------------------ tab 6: comparison
with tabs[5]:
    st.subheader("What this system adds beyond Google Weather / typical weather apps")
    st.markdown("""
| Need | Typical weather app (Google Weather etc.) | This system |
|---|---|---|
| Source of the number | Global/regional chemistry model (e.g. CAMS, ~10–45 km grid), often not your street | **Your nearest real monitoring station**, corrected by ML for local bias |
| Forecast detail | Current AQI and a single daily or hourly guess | **Hourly 1–60 h forecast with an 80% uncertainty band** |
| Early warning | None – you see it when it is already bad | **Alert levels (Watch / Warning / Emergency) with lead time, start, duration and peak** |
| Probability | Not given | **Chance (%) of crossing your threshold, every hour and per day** |
| Personal thresholds | One AQI scale for everyone | **Profiles for asthma, children/elderly, athletes; custom threshold** |
| Indian standards | US-EPA style AQI in many apps | **CPCB categories, 24-h NAAQS (60) and WHO (15) comparison** |
| Why will it be bad? | No explanation | **Driver attribution: wind, mixing height, rain, humidity, regional transport** |
| When to go out | Not offered | **Cleanest outdoor time window each day** |
| Health meaning | Generic text | **Cigarette-equivalent dose, ×WHO, category-specific advice** |
| Trustworthiness | Accuracy never shown | **Public track record: error vs lead time, alert hit/false-alarm rates, reliability** |
| Local patterns | — | **Diurnal/seasonal heatmap, pollution rose (source direction), weekday effect, exceedance days** |
| Open & reproducible | Closed | **Open data (OpenAQ, Open-Meteo), open code, retrainable at any time** |
""")
