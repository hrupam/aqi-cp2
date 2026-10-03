"""Visual layer for the Streamlit dashboard: fonts, colours, Plotly template, hero strip."""
from __future__ import annotations

import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

import config

INK = "#1B2A41"        # deep slate - headings, hero
SKY = "#EEF3F8"        # page background, pale clear sky
PAPER = "#FFFFFF"      # surfaces
HAZE = "#7C8DA3"       # secondary text, rules
LINE = "#DCE3EC"       # borders, grid
SIGNAL = "#2F6FDE"     # forecast line, interactive accent

DISPLAY = "'Bricolage Grotesque', 'Segoe UI', sans-serif"
BODY = "'Atkinson Hyperlegible', 'Segoe UI', sans-serif"

LEVEL_COLORS = {"CLEAR": "#3FA34D", "WATCH": "#E0A800", "WARNING": "#E8711C", "EMERGENCY": "#7E3BA6",
                "Unknown": HAZE, "CAMS": HAZE}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&display=swap');
html, body, .stApp, [data-testid="stMarkdownContainer"], .stTextInput input, .stSelectbox, button {{
  font-family: {BODY};
}}
.stApp {{ background: {SKY}; color: {INK}; }}
h1, h2, h3, h4 {{ font-family: {DISPLAY} !important; color: {INK}; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.45rem !important; }} h3 {{ font-size: 1.2rem !important; }}
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
#MainMenu, footer {{ visibility: hidden; }}
section[data-testid="stSidebar"] {{ background: {PAPER}; border-right: 1px solid {LINE}; }}
section[data-testid="stSidebar"] h2 {{ font-size: 1.05rem !important; margin-top: .4rem; }}
.brand {{ font-family: {DISPLAY}; font-weight: 800; font-size: 1.35rem; color: {INK}; line-height: 1.15; }}
.brand small {{ display: block; font-family: {BODY}; font-weight: 400; font-size: .82rem; color: {HAZE}; margin-top: 2px; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-family: {DISPLAY}; font-weight: 700; font-size: 1rem; color: {HAZE};
  padding: .55rem 0; background: transparent; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {SIGNAL}; height: 3px; }}

/* surfaces */
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stDeckGlJsonChart"] {{
  background: {PAPER}; border: 1px solid {LINE}; border-radius: 12px; padding: 6px; }}
[data-testid="stMetric"] {{ background: {PAPER}; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 16px; }}
[data-testid="stMetricValue"] {{ font-family: {DISPLAY}; font-weight: 700; color: {INK}; }}
[data-testid="stMetricLabel"] p {{ color: {HAZE}; font-size: .88rem; }}
.stButton button[kind="primary"] {{ background: {SIGNAL}; border-color: {SIGNAL}; }}
:focus-visible {{ outline: 2px solid {SIGNAL} !important; outline-offset: 2px; }}

/* hero */
.hero {{ background: {INK}; color: #fff; border-radius: 18px; padding: 26px 28px 18px; margin: 4px 0 18px; }}
.hero-top {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; flex-wrap: wrap; }}
.hero-place h1 {{ color: #fff !important; font-size: clamp(2rem, 4.2vw, 3.1rem) !important; font-weight: 800;
  margin: 0; padding: 0; line-height: 1.02; }}
.hero-place p {{ color: #B7C3D3; margin: 6px 0 0; font-size: .95rem; }}
.hero-aqi {{ text-align: right; }}
.hero-aqi .num {{ font-family: {DISPLAY}; font-weight: 800; font-size: clamp(2.8rem, 6vw, 4.4rem); line-height: .9;
  color: var(--cat); }}
.hero-aqi .cat {{ display: block; font-weight: 700; font-size: 1.05rem; color: #fff; margin-top: 4px; }}
.hero-aqi .lbl {{ display: block; color: #B7C3D3; font-size: .82rem; }}
.hero-msg {{ margin: 18px 0 16px; font-size: 1.06rem; line-height: 1.5; max-width: 78ch; color: #E8EEF6; }}
.chip {{ display: inline-block; border-radius: 999px; padding: 2px 11px; margin-right: 8px; font-weight: 700;
  font-size: .85rem; color: #fff; background: var(--lv); vertical-align: 1px; }}
.strip {{ display: flex; height: 46px; border-radius: 8px; overflow: hidden; gap: 1px; background: {INK}; }}
.strip div {{ flex: 1; position: relative; }}
.strip div.over::after {{ content: ""; position: absolute; left: 0; right: 0; top: 0; height: 5px; background: #fff; opacity: .9; }}
.strip-axis {{ position: relative; height: 22px; margin-top: 5px; font-size: .8rem; color: #B7C3D3; }}
.strip-axis span {{ position: absolute; transform: translateX(-1px); padding-left: 5px; border-left: 1px solid #50627C; }}
.hero-foot {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-top: 6px;
  font-size: .8rem; color: #B7C3D3; }}
.hero-foot i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 5px 0 12px; vertical-align: -1px; }}
.advice {{ color: #B7C3D3; font-size: .92rem; margin-top: 10px; max-width: 80ch; }}

/* stat tiles */
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 0 0 18px; }}
.tile {{ background: {PAPER}; border: 1px solid {LINE}; border-radius: 12px; padding: 14px 16px; }}
.tile .k {{ color: {HAZE}; font-size: .86rem; }}
.tile .v {{ font-family: {DISPLAY}; font-weight: 700; font-size: 1.75rem; color: {INK}; line-height: 1.15; margin-top: 2px; }}
.tile .v small {{ font-family: {BODY}; font-weight: 400; font-size: .85rem; color: {HAZE}; }}
.tile .s {{ font-size: .86rem; color: {INK}; margin-top: 2px; }}
.tile .dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; background: var(--c); }}

/* day cards */
.day {{ background: {PAPER}; border: 1px solid {LINE}; border-radius: 12px; padding: 16px 18px; border-top: 6px solid var(--c); }}
.day h4 {{ margin: 0 0 4px; font-size: 1.05rem; }}
.day .big {{ font-family: {DISPLAY}; font-weight: 800; font-size: 2.2rem; line-height: 1; }}
.day .big small {{ font-family: {BODY}; font-size: .85rem; font-weight: 400; color: {HAZE}; }}
.day p {{ margin: 6px 0 0; font-size: .9rem; color: #33445C; }}
.note {{ color: {HAZE}; font-size: .86rem; }}
@media (max-width: 640px) {{ .hero {{ padding: 18px; }} .hero-aqi {{ text-align: left; }} }}
</style>
"""


def inject():
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family="Atkinson Hyperlegible, Segoe UI, sans-serif", color=INK, size=13),
        title=dict(font=dict(family="Bricolage Grotesque, Segoe UI, sans-serif", size=16, color=INK), x=0.01),
        paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        colorway=[SIGNAL, INK, "#E8711C", "#3FA34D", "#7E3BA6", HAZE],
        xaxis=dict(gridcolor="#EDF1F6", linecolor=LINE, zeroline=False),
        yaxis=dict(gridcolor="#EDF1F6", linecolor=LINE, zeroline=False),
        hoverlabel=dict(font=dict(family="Atkinson Hyperlegible, sans-serif"), bgcolor=PAPER, bordercolor=LINE),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=40, b=40),
    )
    pio.templates["airwatch"] = tpl
    pio.templates.default = "plotly_white+airwatch"


def _esc(x) -> str:
    return html.escape(str(x))


def hero(place: str, subline: str, aqi_value: float, aqi_cat: str, aqi_color: str, aqi_label: str,
         level: str, message: str, hours: pd.DataFrame, threshold: float, advice: str = "") -> str:
    """Dark hero with the next 48 hours drawn as a strip of CPCB colours.
    hours: columns local_time (tz-aware) and value (µg/m³), one row per hour."""
    from aq.insights import category
    cells, ticks = [], []
    n = max(len(hours), 1)
    for i, (t, v) in enumerate(zip(hours["local_time"], hours["value"])):
        name, color = category(v)
        over = " class=\"over\"" if np.isfinite(v) and v > threshold else ""
        cells.append(f'<div{over} style="background:{color}" title="{t:%a %d %b %H:%M} · {v:.0f} µg/m³ · {name}"></div>')
        if i == 0:
            ticks.append((0, "now"))
        elif t.hour == 0 or (i > 0 and t.date() != hours["local_time"].iloc[i - 1].date()):
            ticks.append((i / n * 100, f"{t:%a %d}"))
        elif t.hour in (12,) and n <= 72:
            ticks.append((i / n * 100, "noon"))
    axis = "".join(f'<span style="left:{p:.2f}%">{_esc(lbl)}</span>' for p, lbl in ticks)
    from config import CATEGORIES
    legend = "".join(f'<i style="background:{c}"></i>{_esc(nm)}' for nm, _, c in CATEGORIES)
    lv = LEVEL_COLORS.get(level, HAZE)
    num = "–" if not np.isfinite(aqi_value) else f"{aqi_value:.0f}"
    return f"""
<div class="hero">
  <div class="hero-top">
    <div class="hero-place"><h1>{_esc(place)}</h1><p>{_esc(subline)}</p></div>
    <div class="hero-aqi" style="--cat:{aqi_color}"><span class="num">{num}</span>
      <span class="cat">{_esc(aqi_cat)}</span><span class="lbl">{_esc(aqi_label)}</span></div>
  </div>
  <p class="hero-msg"><span class="chip" style="--lv:{lv}">{_esc(level.title())}</span>{_esc(message)}</p>
  <div class="strip" role="img" aria-label="Forecast colour strip for the next {n} hours">{''.join(cells)}</div>
  <div class="strip-axis">{axis}</div>
  <div class="hero-foot"><span>Next {n} hours, coloured by CPCB category. A white tick marks hours above your
  {threshold:.0f} µg/m³ threshold.</span><span>{legend}</span></div>
  {f'<div class="advice">{_esc(advice)}</div>' if advice else ''}
</div>"""


def tiles(items: list[tuple]) -> str:
    """items: (label, value_html, sub_text, color or None)"""
    out = []
    for label, value, sub, color in items:
        dot = f'<span class="dot" style="--c:{color}"></span>' if color else ""
        out.append(f'<div class="tile"><div class="k">{_esc(label)}</div><div class="v">{value}</div>'
                   f'<div class="s">{dot}{_esc(sub)}</div></div>')
    return f'<div class="tiles">{"".join(out)}</div>'


def brand():
    return ('<div class="brand">Kolkata Air Quality Early Warning'
            '<small>PM2.5 forecasts, 60 hours ahead</small></div>')
