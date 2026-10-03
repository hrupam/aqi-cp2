
---

## 20. Glossary

| Term | Meaning |
|---|---|
| **PM2.5** | Mass of particles smaller than 2.5 µm per cubic metre of air (µg/m³) |
| **AQI (CPCB)** | India's Air Quality Index, 0–500. Each pollutant's 24-h mean is converted with breakpoints and the highest sub-index is reported |
| **NAAQS** | India's National Ambient Air Quality Standards. For PM2.5: 60 µg/m³ as a 24-h mean |
| **CPCB / SPCB** | Central / State Pollution Control Board, the operators of India's official monitors |
| **OpenAQ** | A non-profit platform that collects and serves air-quality data from governments and other sources |
| **Location vs sensor** | An OpenAQ *location* is a monitoring site; a *sensor* is one parameter there, and can have more than one ID |
| **CAMS** | Copernicus Atmosphere Monitoring Service, a global atmospheric-composition model run by ECMWF |
| **ERA5** | ECMWF's global reanalysis: best-estimate historical weather |
| **Boundary layer / mixing height (BLH)** | The lowest part of the atmosphere, mixed by turbulence. Pollution spreads within it. Low at night and in winter |
| **Inversion** | Warm air above cool surface air, acting like a lid that traps pollution |
| **Ventilation index** | Wind speed × mixing height; low values mean stagnant air |
| **Wet deposition** | Rain removing particles from the air |
| **Issue time (T) / lead time (h)** | When a forecast is made, and how many hours ahead it looks |
| **Persistence** | The baseline forecast "future = latest observation" |
| **Direct multi-horizon** | One model predicts every lead time, with h as an input |
| **GBDT** | Gradient-boosted decision trees: an additive ensemble of small trees, each fitted to the errors of the previous ones |
| **Quantile / pinball loss** | A loss whose minimiser is a chosen percentile, used to forecast uncertainty bands |
| **log1p / expm1** | log(1 + x) and its inverse; safe at zero |
| **Skill score** | 1 − error(model) / error(baseline). Above 0 means better than the baseline |
| **MAE / RMSE** | Mean absolute error / root mean squared error |
| **Coverage** | The share of outcomes inside a forecast interval; should match the nominal level (80 %) |
| **Conformal calibration** | Rescaling intervals on held-out data so their coverage is correct |
| **POD / FAR / CSI** | Warning verification: probability of detection, false-alarm ratio, critical success index |
| **Brier score** | Mean squared error of probability forecasts |
| **Reliability diagram** | Forecast probability against observed frequency; a calibrated forecast lies on the diagonal |
| **Permutation importance** | The drop in accuracy when one feature is randomly shuffled |
| **Counterfactual attribution** | Re-running the model with one factor set to "typical" to measure its effect |
| **Data leakage** | Accidentally using information from the future during training, which makes scores look too good |

---

## 21. Likely viva questions, with answers

**Q1. Why did you use gradient boosting rather than a neural network such as an LSTM?**

* The data are tabular, about 50 heterogeneous features and a few hundred thousand rows. On this kind of data gradient-boosted trees are consistently as good as or better than deep networks, train in about a minute on a CPU, need little tuning, handle missing values natively, support quantile loss directly, and are easy to explain.
* An LSTM would need more data per station, normalisation, careful tuning and usually a GPU, with no evidence of better accuracy here.
* I list it as an extension for comparison.

**Q2. Why one model for all 60 horizons instead of a recursive model?**

* Recursive models feed their own predictions back in, so errors compound over 60 steps.
* Making h a feature lets one model learn how relationships change with lead time and share data across horizons.
* There is no compounding, and the forecast curve is smooth.

**Q3. How do you make sure there is no data leakage?**

* The split is by time: training rows must have their *target* time before the cut-off, and test rows have their *issue* time after it.
* Every feature at issue time T uses data up to T only:
  * rolling windows look backwards
  * "same hour yesterday" is taken from a day already observed
  * weather at T+h comes from forecasts, which are known in advance
* Calibration and scoring use different weeks.

**Q4. Why train on archived weather forecasts instead of observed weather?**

* Live predictions use forecast weather, which has errors.
* Training on archived forecasts teaches the model to cope with those errors. Training on perfect hindsight weather would make it over-confident.

**Q5. Your first model lost to persistence. What did you learn?**

* Absolute-level models tied to the season memorised last year's levels. Kolkata's 2026 summer was about 35 % cleaner, so the forecasts were biased upward by about 24 %.
* Predicting the **change** from the current reading, with features relative to now, removed the bias.
* The model then beat persistence by 18–24 % and CAMS by about 3×.
* The lesson is that a model must be anchored to the most reliable current information, and that you should always compare against simple baselines.

**Q6. How do you get a probability from a regression model?**

* Three quantile models give the 10th, 50th and 90th percentiles.
* Conformal scaling on held-out data corrects the band so that 80 % of outcomes fall inside it.
* A split-normal distribution in log space, fitted through the three quantiles, gives P(PM > threshold) for any threshold.
* The reliability diagram checks it.

**Q7. What does the conformal step actually do?**

* For each held-out case it measures how many half-widths the truth was from the median.
* For each lead-time bucket it takes the 80th percentile of that score as a scale factor and stretches the band by it.
* Coverage then becomes about 80 % by construction (0.79–0.82 observed).

**Q8. Why blend with persistence?**

* At 1–3 hours ahead the latest reading is almost unbeatable, and the model's small predicted changes add noise.
* A weight for each lead time, chosen on held-out data, shrinks the predicted change: about 0.15 at 1 h, rising to 1.0 by 37 h.
* The result is never worse than persistence, and the model's skill grows with lead time.

**Q9. Why is the ventilation index important?**

* Pollution concentration ≈ emissions ÷ the volume of air they mix into, per unit time.
* Wind speed × mixing height measures that ventilation.
* Low ventilation, typical of winter nights, traps pollution. This is the main physical reason winter episodes happen.

**Q10. How do you explain a specific forecast?**

* For each weather group, the model is re-run with that group's inputs set to their typical value for that month and hour.
* The difference is the group's contribution in µg/m³, for example "low wind +23 µg/m³ tomorrow night".
* Globally, permutation importance ranks the inputs.

**Q11. What happens when the station stops reporting?**

* For gaps of 6 h to 14 days, the missing hours are estimated as CAMS × the station/CAMS ratio over the last week, so forecasts still start from now.
* The estimates are flagged, shown dotted, and never used for training.
* Beyond 14 days, the warning tells the user the station has been offline too long, and the next `fetch` tries other stations automatically.

**Q12. How is the station chosen?**

1. Rank all PM2.5 monitors within 25 km: active, reference-grade, long history, close.
2. Join all sensor IDs of a station.
3. Accept the first station with at least 30 days of data and a reading within 14 days; keep using the same station later.

**Q13. What is the AQI and how do you compute it?**

* CPCB's index converts each pollutant's 24-h mean concentration into a sub-index by linear interpolation between breakpoints.
* For PM2.5, 0–30 maps to AQI 0–50, 30–60 to 51–100, and so on.
* The overall AQI is the highest sub-index. We use PM2.5 and PM10.

**Q14. What does the project offer that Google Weather doesn't?**

* A station-anchored ML forecast that is about 3× more accurate than the model behind such apps.
* Uncertainty bands and probabilities.
* Early warnings with lead time and duration, and personal thresholds.
* Indian standards.
* Explanations of why air will be bad.
* The best time to go out, and health equivalents.
* A published accuracy record and local pattern analytics.

**Q15. How would you scale this to all of India?**

* Make the city a setting and run the pipeline per station on a schedule (downloads are already incremental). The full version of this project already supports many cities.
* Store the data in a database instead of CSVs.
* Serve forecasts through an API.
* Use a spatial model (kriging or a graph neural network) to fill the gaps between stations.
* Retrain monthly, and alert when accuracy drifts.

**Q16. What are the main limitations?**

* Inherited weather-forecast errors.
* No information about sudden local sources.
* One station per city.
* Only a few years of data.
* Warning scores still need a winter season to be validated.
