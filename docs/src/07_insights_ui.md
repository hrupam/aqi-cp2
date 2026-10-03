
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
