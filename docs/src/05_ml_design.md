
---

## 9. Machine-learning design, and why gradient boosting

### 9.1 Framing the problem
We want PM2.5 at every hour **T+h** for lead times **h = 1…60**, using only what is known at the issue time **T**:
* the station's readings up to T
* weather and CAMS forecasts for the coming hours

This is **supervised regression on a time series with exogenous (external) inputs**.

There are three standard ways to produce a multi-step forecast:

| Strategy | How | Problem |
|---|---|---|
| **Recursive** (iterated one-step) | Predict T+1, feed it back as a lag, predict T+2, … | Errors **compound** over 60 steps, and the model is never trained on its own imperfect inputs |
| **Direct, one model per horizon** | Train 60 separate models | 60× the training cost; each model sees only its own slice of data; the forecast curve can be jagged |
| **Direct multi-horizon, h as a feature** ✅ | One model; each row is a (T, h) pair with `h` as an input | One model shares what it learns across horizons; no compounding; smooth curve; trained once |

The project uses the third strategy. Each training row says, in effect:
> "At issue time T, the recent PM2.5 looked like this. At T+h the forecast weather is this, CAMS expects this change, and it will be this hour of the day. What was PM2.5 at T+h?"

**Size:** with an issue time every 3 hours (`ISSUE_STRIDE`) and 60 horizons, each year of data gives about 2,920 × 60 ≈ **175,000 rows**. Three to four years therefore give several hundred thousand rows.

### 9.2 Why gradient-boosted decision trees (GBDT)?

**The short answer:** this is a **tabular** problem, with a few dozen heterogeneous columns (lags, weather, calendar), hundreds of thousands of rows, many missing values, strongly **non-linear effects and interactions**, and a need for **uncertainty** and **explanations**. Gradient-boosted trees are the strongest, most practical tool for exactly this setting.

**The long answer, point by point:**

1. **State of the art on tabular data.** Benchmarks repeatedly find that tree ensembles, gradient boosting especially, match or beat deep learning on medium-sized tabular datasets. They also need far less tuning and compute. See, for example, Grinsztajn, Oyallon & Varoquaux, *"Why do tree-based models still outperform deep learning on tabular data?"*, NeurIPS 2022.
2. **Non-linearity and thresholds.** PM2.5 does not fall linearly with wind speed. Below about 1–2 m/s the air stagnates and pollution builds; above about 4 m/s extra wind changes little. Trees model such step-like, saturating responses naturally. A linear model would need hand-made transformations.
3. **Interactions found automatically.** The worst episodes need several conditions at once: a low mixing height **and** low wind **and** night **and** no rain **and** air arriving from the polluted plain. Each tree split conditions on the earlier splits, so trees learn these "AND" combinations without anyone specifying them.
4. **Missing values handled natively.** Station data has gaps, and some weather variables (mixing height) can be missing for whole periods. scikit-learn's `HistGradientBoostingRegressor` learns, at every split, which branch missing values should go to. No imputation is needed, and no rows are thrown away.
5. **No feature scaling, robust to odd inputs.** Trees only compare thresholds, so units (m, m/s, µg/m³, sines of hours) and outliers in the inputs don't matter.
6. **Built-in quantile regression.** Choosing `loss="quantile"` gives the 10th, 50th and 90th percentiles directly. This provides the uncertainty band and the exceedance probability, which a plain point forecast cannot.
7. **Fast and light.** Histogram-based training handles about 500,000 rows × 49 features in roughly a minute per model on a laptop CPU. No GPU is needed, so it runs on a student laptop and retrains in a few minutes.
8. **Explainable.** Permutation importance and counterfactual "what if the weather were typical" runs (section 11.7) show *why* a forecast is high. That matters for a public-health warning.
9. **Proven for air quality.** Gradient boosting (XGBoost, LightGBM and similar) is among the most common and best-performing methods in PM2.5 forecasting studies, especially when combined with meteorological forecasts and chemistry-model outputs, as here.

**Alternatives considered:**

| Model | Why not chosen (here) |
|---|---|
| Linear / ridge regression | Can't capture thresholds or interactions without heavy manual feature engineering; under-fits |
| ARIMA / SARIMAX | Essentially linear and (near-)univariate; weak with many non-linear weather drivers; fiddly for 60 horizons |
| Prophet | Built for trend and seasonality, not weather-driven episodes; no feature interactions |
| Random forest | Good, but averages deep trees (high variance, big models), usually less accurate than boosting, and scikit-learn has no native quantile loss for it |
| XGBoost / LightGBM | Same family of algorithm, equally good. scikit-learn's histogram GBDT is modelled on LightGBM. Using it keeps dependencies minimal (scikit-learn is already needed) and installs everywhere; swapping is a few lines |
| LSTM / GRU / Temporal Fusion Transformer | Need more data, careful tuning and often a GPU; harder to explain; with a few years of hourly data per station they rarely beat GBDT. **A good extension** for a comparison chapter |
| SVR / k-nearest neighbours | Scale poorly to hundreds of thousands of rows; need feature scaling; no native quantiles |

### 9.3 How gradient boosting works (in this model)
Gradient boosting builds an **additive model** of many small regression trees:

$$F_M(x) = F_0 + \eta \sum_{m=1}^{M} f_m(x)$$

* **Start:** $F_0$ is a constant (for quantile loss, the τ-quantile of the target).
* **Each round m:**
  1. Compute every row's **negative gradient** of the loss at the current prediction (the "pseudo-residual").
  2. Fit a small tree $f_m$ to these pseudo-residuals, so the tree learns *the direction in which the current model is wrong*.
  3. Set each leaf's output value by a line search (for quantile loss, the τ-quantile of the residuals in that leaf).
  4. Add the tree to the model, scaled down by the **learning rate** η.
* **Repeat** for `max_iter` rounds. Each tree corrects a little of what the previous ones got wrong, which is why it is called "boosting".

**Histogram GBDT tricks** (`HistGradientBoostingRegressor`):
* Each feature is binned into at most 255 bins. Split finding then scans bins instead of sorted values, which makes it much faster.
* Trees grow **leaf-wise** (best-first) up to `max_leaf_nodes` leaves.
* Missing values get their own learned direction at each split.

**Quantile (pinball) loss** for quantile level τ, with residual $u = y - \hat{y}$:

$$L_\tau(u) = \max(\tau u,\ (\tau - 1)u)$$

* For τ = 0.9, under-predicting costs 9 times more than over-predicting. The minimiser is therefore the value that 90 % of outcomes fall below, which is the 90th percentile.
* The gradient is just $\tau - \mathbb{1}[y < \hat{y}]$. This makes training robust to outliers in the target.

### 9.4 Hyperparameters (`_new_model`) and why

| Parameter | Value | Reason |
|---|---|---|
| `loss` | `"quantile"` | Gives percentiles (uncertainty) instead of a single mean |
| `quantile` | 0.1 / 0.5 / 0.9 | Lower band, median forecast, upper band. The 10–90 % band is an 80 % interval |
| `learning_rate` | 0.06 | Small steps generalise better; 0.05–0.1 is standard |
| `max_iter` | 300 | Number of trees. With η = 0.06 this reaches a good fit without long training |
| `max_leaf_nodes` | 24 | Fairly shallow trees, enough for 4–5-way interactions. Earlier, deeper trees (48 leaves) over-fitted one year's patterns |
| `min_samples_leaf` | 200 | Every leaf must be backed by at least 200 rows. This stops the model memorising rare situations, and matters because neighbouring (T, h) rows are highly correlated |
| `l2_regularization` | 1.0 | Shrinks leaf values; adds stability |
| `early_stopping` | False | Deterministic, full-length training. The held-out evaluation is done properly elsewhere (section 9.10) |
| `random_state` | 0 | Reproducible results |

### 9.5 Target transform: logarithm
* PM2.5 is **right-skewed** (mostly 10–80, sometimes 300+), and its dynamics are **multiplicative**: a stagnant night might *double* pollution whatever the starting level.
* Modelling `log1p(PM)` turns ratios into differences, stabilises the variance, and stops the model wasting effort on a few extreme values. `log1p(x) = log(1+x)` is safe at 0.
* **Quantiles survive monotone transforms.** The 90th percentile of log(PM) is the log of the 90th percentile of PM. The model can therefore be trained in log space and its quantiles converted back exactly with `expm1`.

### 9.6 The key design decision: predict the *change*, not the level
**The original version** predicted `log1p(PM at T+h)` from absolute features. On Kolkata's real data it was **worse than simple persistence** ("the next hours equal the latest reading"):

| | MAE at 1 h | MAE at 24 h |
|---|---|---|
| Original model | 7.0 | 7.3 |
| Persistence | 1.6 | 4.9 |

**The diagnosis** (done by re-training on the user's data):
* Summer 2026 in Kolkata was about **35 % cleaner than summer 2025**.
* The model had learned "in this season, with this weather, PM is about 27". It kept forecasting about 27 while the station read about 17.
* It showed a +0.21 log-unit (about +24 %) bias even at 1 hour ahead.
* Season features (day-of-year) let the trees memorise *last year's dates*, and absolute CAMS and lag levels did the same.

**The fix**, implemented in `features.design()`:
1. **Target = log change from now.** `target = log1p(PM at T+h) − log1p(PM at T)`. The forecast is `PM_now × exp(predicted change)`, so it is always anchored to what the station measures today.
2. **Scale-free inputs:**
   * every past-PM feature (lags, means, max, min, same hour yesterday) is expressed as **log-ratio to now**
   * variability is expressed relative to the current level
   * CAMS enters only as its **predicted change** (`c_cams_change`, `c_cams24_change`), never as a level
3. **Drop day-of-year.** With 1–3 years of data it only enables memorisation. Hour of day and weekday stay, because they repeat thousands of times.

**Result:** the model now beats persistence at every lead time and beats CAMS by about 3× (section 17).

**Why this works:** the model now learns **physics-like rules**, such as "a falling mixing height tonight plus weak wind → PM rises about 40 %". These transfer across years, stations and calibration changes. Absolute levels don't transfer.

### 9.7 Blending with persistence
Even after the fix, at **1–3 hours** ahead the latest reading is very hard to beat, and the model's small predicted changes add noise. The code therefore shrinks the predicted change separately for each lead-time bucket:

$$\log \hat{y}_{T+h} = \log y_T + w_b \cdot \widehat{\Delta}_{50}$$

* **Weights:** $w_b \in [0, 1]$ is chosen on held-out data from a grid of 21 values (0, 0.05, …, 1) to minimise MAE.
* **Buckets** (`H_EDGES`): 1, 2–3, 4–6, 7–12, 13–24, 25–36, 37–48 and 49–60 h.
* **Kolkata's learned weights:** about 0.15 at 1 h, 0.35 at 2–3 h, 0.6 at 4–6 h, 0.7 at 7–12 h, 0.9 for 13–36 h, and 1.0 beyond 36 h. So the model is trusted more the further ahead it looks, which is the right behaviour.

This is a simple, principled form of **forecast combination** (shrinkage toward a strong baseline).

### 9.8 Calibrated uncertainty: split-conformal scaling
Quantile models are often over-confident: their 80 % band holds the truth only about 60–70 % of the time. The code fixes this with **split-conformal calibration** on held-out data:
1. **Score every calibration row.** For each row, compute how many band-half-widths the truth fell from the median, in log space:
   * below the median: $s = (\log q50 - \log y)/(\log q50 - \log q10)$
   * above the median: $s = (\log y - \log q50)/(\log q90 - \log q50)$
2. **Pick the scale factor.** For each lead-time bucket, $k_b$ is the **80th percentile of s**, clipped to [0.5, 4].
3. **Rescale the band.** Live forecasts stretch (or shrink) the distance from median to q10 and q90 by $k_b$.

By construction, about 80 % of held-out outcomes then fall inside the band. The backtest confirms coverage of 0.79–0.82 on Kolkata.

### 9.9 From quantiles to a probability of exceeding a threshold
The dashboard needs P(PM > threshold) for any threshold the user picks. `exceed_prob()` fits a **split-normal distribution in log space** through the three quantiles:
* centre at log q50
* lower spread $\sigma_{lo} = (\log q50 - \log q10)/1.2816$ and upper spread $\sigma_{hi} = (\log q90 - \log q50)/1.2816$. Here 1.2816 is the z-score of the 90th percentile, and both spreads are floored at 0.02 to avoid dividing by zero.
* the spread on the threshold's side is used:

$$P(\text{PM} > t) = 1 - \Phi\left(\frac{\log(1+t) - \log(1+q50)}{\sigma}\right)$$

A split-normal allows **skew**: pollution can spike up much further than it can drop. The reliability diagram in the Track record tab checks the probabilities ("when we say 70 %, does it happen about 70 % of the time?").

### 9.10 Honest evaluation: no leakage
* **Time-ordered split.** The cut-off is the issue time at the 75th percentile:
  * **training** uses only rows whose *target time* is before the cut-off, so no future value leaks in through long horizons
  * **testing** uses rows whose *issue time* is on or after the cut-off
* **Calibration and scoring are separate.** Inside the test period, **alternate weeks** are used to fit the blending weights and conformal scales, and the **other weeks** are used for every reported number. Alternating weeks (rather than first half / second half) means both halves cover the same seasons.
* **Baselines:**
  * persistence (`p_pm`), the hardest to beat at short lead times
  * same hour on the last observed day (`p_same_hour`), which captures the daily cycle
  * raw CAMS, which is what apps show
* **Final model:** after evaluation, the models are **refit on all the data** for live forecasting. The calibration and blending learned out of sample are kept. This is standard practice: evaluate honestly, then use every bit of data.

### 9.11 Explainability
* **Global: permutation importance.** On up to 15,000 test rows, each feature is shuffled in turn and the increase in error is measured, three repeats per feature. A feature whose shuffling hurts most matters most.
* **Local: counterfactual driver attribution.** For *today's* forecast, one weather group at a time is replaced by its **typical value for this month and hour** (the station's own climatology, a median), and the model is re-run. The difference shows how many µg/m³ that factor adds compared with a normal day. The five groups are:
  * wind
  * mixing height
  * rain and clouds
  * temperature, humidity and pressure
  * regional CAMS

  This is a simple, model-agnostic, causal-style explanation that people understand: "low wind adds +23 µg/m³ to tomorrow night".
