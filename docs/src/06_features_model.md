
---

## 10. `aq/features.py`: feature engineering

The module works in three stages:
1. `series_features` computes helper columns **once per hour**.
2. `make_samples` turns them into **(T, h) rows**.
3. `design` converts those rows into the **scale-free model matrix**.

Column prefixes say where a feature comes from:

| Prefix | Meaning |
|---|---|
| `p_` | measured PM2.5, taken at issue time T |
| `w_` | weather, taken at target time T+h |
| `c_` | CAMS |
| `k_` | calendar |
| `d_` | change between T and T+h |
| `path_` | conditions along the way from T to T+h |

Constants: `PM_LAGS = (1, 2, 3, 6, 12, 24, 48)` and `WEATHER_COLS` (10 weather variables).

### 10.1 `series_features(df) → DataFrame`
Computed on the merged hourly table (index = UTC hour):

**PM history (`p_`), known at the row's own hour:**

| Feature | Definition | Why |
|---|---|---|
| `p_pm` | PM2.5 now | The anchor of the forecast |
| `p_lag1 … p_lag48` | PM 1, 2, 3, 6, 12, 24 and 48 h ago | Short memory and the daily cycle |
| `p_mean3/6/24/72` | Rolling means (need at least half the window present) | Smoothed recent levels; the 72-h mean shows how persistent an episode is |
| `p_std24`, `p_max24`, `p_min24` | 24-h variability and extremes | Volatile or stable regime |
| `p_trend3`, `p_trend24` | log change over 3 h and 24 h | Build-up or clean-out already under way |

**Weather (`w_`):**

| Feature | Definition | Why |
|---|---|---|
| `w_temp` … `w_blh` | Raw variables | Direct drivers |
| `w_wind_u`, `w_wind_v` | Wind as east/north components: −speed·sin(dir), −speed·cos(dir) | Direction is circular (359° ≈ 1°). Components make it continuous **and** weight it by speed |
| `w_vent` | **Ventilation index** = wind speed × mixing height (m²/s) | The classic air-pollution dispersion indicator: how much air is available to dilute emissions. Below about 2,000 m²/s means stagnant |
| `w_log_vent` | log(1 + ventilation) | Compresses its huge range |
| `w_rain6`, `w_rain24` | Rain over the last 6 h and 24 h | Wash-out has a lingering effect |
| `w_wind12`, `w_logvent12` | 12-h means | Has the air been stagnant for a while? |
| `w_blh_max24` | Highest mixing height in the last 24 h | How well yesterday's daytime mixing cleaned the air |
| `w_dpress24`, `w_dtemp24` | 24-h pressure and temperature change | Air-mass changes and fronts |

**Every weather column is always created** (filled with NaN if a variable is missing). The model's column list must be identical in training and live use.

**CAMS (`c_`):** `c_cams` (hourly) and `c_cams24` (24-h mean).

**Calendar (`k_`), in the city's local time:**
* `k_hour_sin`, `k_hour_cos`: hour of day on a circle, so 23:00 and 00:00 are close.
* `k_dow`: day of week, 0 = Monday.
* `k_doy_sin`, `k_doy_cos`: day of year. These are kept in the raw table but **dropped by `design`** (section 9.6).

### 10.2 `make_samples(sf, issue_pos, horizons, with_target=True) → DataFrame`
This builds the (T, h) rows with numpy integer indexing. There are no Python loops, which matters at about 1 million rows.

```python
I = np.repeat(issue_pos, len(horizons))   # issue row index, repeated for each h
H = np.tile(horizons, len(issue_pos))     # 1..60 for each issue time
V = I + H                                 # target row index
A = sf[p_cols].to_numpy()[I]              # PM history at the issue time
B = sf[tgt_cols].to_numpy()[V]            # weather / CAMS / calendar at the target time
```

**Rows whose target lies beyond the table are dropped.** The function then adds:

| Feature | Meaning |
|---|---|
| `h` | Lead time; this is how one model covers every horizon |
| `d_log_vent`, `d_temp`, `d_wind` | Weather at T+h minus weather at T. A *falling* ventilation is the classic sign that an episode is coming |
| `c_bias_now`, `c_cams_adj` | CAMS's current local bias and a bias-corrected CAMS (kept for reference; removed by `design`) |
| `path_log_vent` | **Mean** log-ventilation over (T, T+h] |
| `path_rain` | **Total** rain over (T, T+h] |
| `p_same_hour` | PM at the same clock hour on the most recent fully observed day |
| `c_cams_T` | CAMS at the issue time |
| `issue_time`, `target_time` | Bookkeeping |
| `y` | PM2.5 at T+h, the target |

**How the path features are computed:** with **cumulative sums**, `(cs[V+1] − cs[I+1])`, the sum over any window is one subtraction. This gives "conditions along the way" for every (T, h) pair at once. The intuition is that pollution at T+h depends on the whole journey of the air, not just the weather at the final hour.

**How `p_same_hour` is computed:** `back = 24·ceil(h/24)` ensures the value comes from a day that was *already observed at T*, so there is no leakage.

### 10.3 `feature_columns(X)`
All columns except `issue_time`, `target_time` and `y`.

### 10.4 `design(X) → (R, base)`
This applies the scale-free transformation (section 9.6).
* **`base`** is `log1p(PM now)`, falling back to the 3-h mean if the current hour is missing. It is the persistence anchor.
* **Columns dropped** (`_ABSOLUTE`): `p_pm`, `c_cams`, `c_cams24`, `c_cams_adj`, `c_bias_now`, `c_cams_T`, `k_doy_sin`, `k_doy_cos`.
* **Each `p_` feature** becomes `log1p(value) − base`, the log-ratio to now. The exceptions:
  * `p_trend3` and `p_trend24` are already log differences and are kept as they are.
  * `p_std24` becomes the coefficient of variation (std ÷ current level).
* **New features:**
  * `c_cams_change = log1p(CAMS at T+h) − log1p(CAMS at T)`
  * `c_cams24_change`, the same for the 24-h mean
* **±inf is replaced with NaN**, which the trees handle.

**The final matrix has 49 inputs:**
* `h`
* 16 relative PM features
* 21 weather features
* 3 calendar features (hour sine/cosine, weekday)
* 3 change features (`d_`)
* 2 path features
* `p_same_hour`
* 2 CAMS-change features

---

## 11. `aq/model.py`: training, calibration, evaluation, forecasting, explanation

### 11.1 Constants
* **`Z90 = 1.28155`** is the standard-normal z-score of the 90th percentile.
* **`DRIVER_GROUPS`** maps the five explanation groups to raw columns:

  | Group | Columns |
  |---|---|
  | Wind | `wind_speed`, `wind_dir`, `wind_gust` |
  | Mixing height | `blh` |
  | Rain & clouds | `precip`, `cloud` |
  | Temperature, humidity & pressure | `temp`, `rh`, `dewpoint`, `radiation`, `pressure` |
  | Regional pollution | `cams_pm25` |

* **`H_EDGES = [1, 2, 4, 7, 13, 25, 37, 49, ∞]`** defines the lead-time buckets used for blending and calibration. They are finer at short lead times, where behaviour changes quickly.

### 11.2 Helper functions
* **`_new_model(q)`** returns a configured `HistGradientBoostingRegressor` for quantile *q* (hyperparameters in section 9.4).
* **`_norm_cdf(z)`** is the standard normal cumulative distribution, computed with `math.erf` so scipy isn't needed. It returns an empty array for empty input. A dashboard crash taught us that `np.vectorize` fails on size-0 input unless `otypes` is given.
* **`exceed_prob(q10, q50, q90, threshold)`** is the split-normal probability from section 9.9. It is vectorised over any number of hours.
* **`_bucket(h)`** maps a lead time to its bucket index with `np.searchsorted(H_EDGES, h, side="right") − 1`.
* **`_raw_residuals(models, R)`** returns the three models' predicted *log changes* (n × 3), **sorted per row**, so q10 ≤ q50 ≤ q90 always holds. Three independent quantile models can otherwise "cross".
* **`_assemble(r, base, h, blend, scale)`** turns log changes into PM2.5 quantiles:

  ```python
  mid = r[:, [1]]                                            # median log change
  P = base + w * mid + (r - mid) * k                         # blend the centre, scale the band
  P = clip(expm1(P), 0)
  ```

  * `w` (blend) moves the centre toward persistence.
  * `k` (conformal) widens or narrows the band around the centre.
* **`predict_quantiles(bundle, X)`** chains `design` → `_raw_residuals` → `_assemble`, using the bundle's saved column order, blend weights and scales.
* **`fit_blend(r50, base, y, h)`** chooses, for each bucket, the weight from {0, 0.05, …, 1} that minimises MAE on the calibration weeks.
* **`climatology(df)`** is the median of every driver column by (local month, local hour). It is saved in the bundle as the "typical day" for driver attribution.

### 11.3 `build_training_set(df)`
1. `series_features(df)`.
2. Choose issue positions where PM is observed, skipping the first 72 h (so lags exist) and taking every 3rd hour.
3. `make_samples` for h = 1…60.
4. Keep only rows whose target `y` is observed.

### 11.4 `train_and_backtest(df, fit_final=True)`
1. **Build the samples.**
2. **Split by time.** The cut-off is the issue time at the 75 % point:
   * train = rows with `target_time < cutoff`
   * test = rows with `issue_time ≥ cutoff`
3. **Fit three quantile models** on `design(train)` with target `log1p(y) − base`.
4. **Predict the test set** (raw log changes).
5. **Split the test set into alternating weeks.** Odd weeks are used to fit the blend weights and then the conformal scales (computed on blended predictions); even weeks are used for scoring.
6. **Score** the even weeks with `evaluate` and attach `interval_scale`, `blend_weight` and the period dates to the report.
7. **Compute permutation importance** of the median model on up to 15,000 scored rows, scored by mean absolute error of the log change, 3 repeats. The top 20 are kept.
8. **Save** `backtest_predictions.csv` for h ∈ {1, 3, 6, 12, 24, 36, 48}. The Track record tab plots it.
9. **Refit** the three models on *all* samples for live use.
10. **Save the bundle** (joblib) with these fields:
    * `models` (q → fitted model), `cols` (feature order), `scale`, `blend`
    * `version: 2` (format marker), `climatology`, `trained_until`, `city`

    Also save `backtest.json`.

### 11.5 `interval_scale(cal, target=0.8)`
This is the conformal factor per bucket (section 9.8). The `np.clip(…, 1e-3)` guards against zero-width bands. The `[0.5, 4]` clip stops a tiny bucket from producing absurd bands.

### 11.6 `evaluate(test)` and `_mae`
* `_mae(a, b)` ignores non-finite pairs. CAMS can be missing in early periods.
* **For each lead time h** it computes:
  * MAE of the model median, persistence, same hour and CAMS
  * RMSE of the model
  * **coverage_80**: the share of outcomes inside [q10, q90]
* **Summaries** for 0–24 h and 24–48 h: mean metrics plus the **skill score** `1 − MAE_model / MAE_baseline`. Skill above 0 means the model beats the baseline; 0.24 means 24 % less error.
* **Warning verification**, at 24–48 h lead time, for thresholds 60, 90 and 120. A warning is issued when P ≥ 0.5.

  | Metric | Formula | Meaning |
  |---|---|---|
  | POD (probability of detection) | hits / (hits + misses) | Share of real exceedances that were warned about |
  | FAR (false-alarm ratio) | false alarms / (hits + false alarms) | Share of warnings that didn't happen |
  | CSI (critical success index) | hits / (hits + misses + false alarms) | Combined score, 1 = perfect |
  | Brier score | mean((p − event)²) | Accuracy of the probabilities; lower is better |

* **Reliability** (threshold 90): probabilities are binned into tenths, and for each bin with at least 30 cases the mean forecast probability is compared with the observed frequency.

### 11.7 Live forecasting
* **`load_bundle()`** reads the joblib file.
* **`_tail(df)`** keeps the window from 200 h before the last observation to 60 h after it. That is enough for every lag and rolling window, and fast.
* **`_forecast_frame(bundle, df)`** runs `series_features` on the tail, takes the last row with an observation as T, and calls `make_samples` for h = 1…60 without a target.
* **`forecast(bundle, df)`**:
  * refuses old bundles (`version < 2`) with an instruction to retrain, because the feature design changed
  * returns `issue_time`, `target_time`, `h`, `cams`, `q10`, `q50` and `q90`
* **`driver_attribution(bundle, df, fc)`**, for each driver group:
  1. Copy the tail.
  2. Replace the group's columns, from 24 h before T onwards, with the climatology value for each hour's (month, hour). The original value is kept where the climatology is missing.
  3. Rebuild the forecast and predict the median.
  4. `effect = actual forecast − counterfactual forecast`, averaged over hours 1–24 and 25–48 after T.

  A positive effect means the factor is making the air **worse** than a typical day.
