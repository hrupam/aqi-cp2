"""Quantile gradient-boosting forecaster, backtest, exceedance probabilities, driver attribution."""
from __future__ import annotations

import json
import math
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import config
from aq.features import series_features, make_samples, design

Z90 = 1.2815515655446004   # z-score of the 90th percentile

DRIVER_GROUPS = {
    "Wind (speed & direction)": ["wind_speed", "wind_dir", "wind_gust"],
    "Mixing height (atmospheric lid)": ["blh"],
    "Rain & clouds": ["precip", "cloud"],
    "Temperature, humidity & pressure": ["temp", "rh", "dewpoint", "radiation", "pressure"],
    "Regional pollution (CAMS model)": ["cams_pm25"],
}


# ------------------------------------------------------------------ helpers
def _new_model(q: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, learning_rate=0.06, max_iter=300,
        max_leaf_nodes=24, min_samples_leaf=200, l2_regularization=1.0,
        early_stopping=False, random_state=0)


def _norm_cdf(z):
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        return z
    return 0.5 * (1 + np.vectorize(math.erf, otypes=[float])(z / math.sqrt(2)))


def exceed_prob(q10, q50, q90, threshold) -> np.ndarray:
    """P(PM2.5 > threshold) from three quantiles, using a split-normal in log space."""
    l10, l50, l90 = (np.log1p(np.asarray(a, float)) for a in (q10, q50, q90))
    lt = np.log1p(threshold)
    s_lo = np.clip((l50 - l10) / Z90, 0.02, None)
    s_hi = np.clip((l90 - l50) / Z90, 0.02, None)
    sigma = np.where(lt >= l50, s_hi, s_lo)
    return 1 - _norm_cdf((lt - l50) / sigma)


H_EDGES = np.array([1, 2, 4, 7, 13, 25, 37, 49, 10_000])     # lead-time buckets: 1,2-3,4-6,7-12,13-24,...


def _bucket(h) -> np.ndarray:
    return np.searchsorted(H_EDGES, np.asarray(h, int), side="right") - 1


def _raw_residuals(models: dict, R: pd.DataFrame) -> np.ndarray:
    """Predicted log-change quantiles (n x 3), sorted to avoid crossing."""
    return np.sort(np.column_stack([models[q].predict(R) for q in config.QUANTILES]), axis=1)


def _assemble(r: np.ndarray, base: np.ndarray, h, blend: dict | None, scale: dict | None,
              index=None) -> pd.DataFrame:
    """Turn log-change quantiles into PM2.5 quantiles.
    blend[b]  in [0,1]: how much of the model's predicted change to trust at that lead time
              (0 = pure persistence) - fitted out-of-sample.
    scale[b]: conformal widening/narrowing of the 10-90 % band."""
    b_idx = _bucket(h)
    w = np.array([blend.get(int(i), 1.0) for i in b_idx]) if blend else np.ones(len(r))
    k = np.array([scale.get(int(i), 1.0) for i in b_idx]) if scale else np.ones(len(r))
    mid = r[:, [1]]
    P = base[:, None] + w[:, None] * mid + (r - mid) * k[:, None]
    P = np.clip(np.expm1(P), 0, None)
    return pd.DataFrame(P, columns=[f"q{int(q*100)}" for q in config.QUANTILES], index=index)


def predict_quantiles(bundle: dict, X: pd.DataFrame) -> pd.DataFrame:
    R, base = design(X)
    r = _raw_residuals(bundle["models"], R[bundle["cols"]])
    return _assemble(r, base, X["h"].to_numpy(), bundle.get("blend"), bundle.get("scale"), X.index)


def fit_blend(r50: np.ndarray, base: np.ndarray, y: np.ndarray, h) -> dict:
    """Per lead-time bucket, the weight on the model's predicted change that minimises MAE."""
    out, b_idx = {}, _bucket(h)
    grid = np.linspace(0, 1, 21)
    for b in np.unique(b_idx):
        m = b_idx == b
        errs = [np.mean(np.abs(np.expm1(base[m] + g * r50[m]) - y[m])) for g in grid]
        out[int(b)] = float(grid[int(np.argmin(errs))])
    return out


def climatology(df: pd.DataFrame) -> pd.DataFrame:
    loc = df.index.tz_convert(config.LOCAL_TZ)
    cols = [c for g in DRIVER_GROUPS.values() for c in g if c in df]
    return df[cols].groupby([loc.month, loc.hour]).median().rename_axis(["month", "hour"])


# ------------------------------------------------------------------ training
def build_training_set(df: pd.DataFrame):
    sf = series_features(df)
    pos = np.where(sf["p_pm"].notna().to_numpy())[0]
    pos = pos[pos >= 72][::config.ISSUE_STRIDE]
    X = make_samples(sf, pos, np.arange(1, config.MAX_HORIZON + 1))
    return X[X["y"].notna()].reset_index(drop=True)


def train_and_backtest(df: pd.DataFrame, fit_final: bool = True, verbose: bool = True) -> dict:
    X = build_training_set(df)
    times = X["issue_time"].sort_values().unique()
    cutoff = times[int(len(times) * (1 - config.TEST_FRACTION))]
    train = X[X["target_time"] < cutoff]                 # no target leaks into the test period
    test = X[X["issue_time"] >= cutoff].copy()
    if verbose:
        print(f"   samples: train {len(train):,}  test {len(test):,}")
        print(f"   backtest period: {pd.Timestamp(cutoff):%Y-%m-%d} -> {X['target_time'].max():%Y-%m-%d}")

    Rtr, btr = design(train)
    cols = list(Rtr.columns)
    models = {}
    for q in config.QUANTILES:
        if verbose:
            print(f"   fitting quantile {q} ...")
        models[q] = _new_model(q).fit(Rtr, np.log1p(train["y"]) - btr)

    # Out-of-sample tuning on alternate weeks of the test period ("cal"); the other weeks
    # are used for every reported score. Both halves span all seasons in the test period.
    Rte, bte = design(test)
    r_te = _raw_residuals(models, Rte[cols])
    odd_week = (((test["issue_time"] - test["issue_time"].min()).dt.days // 7) % 2 == 1).to_numpy()
    blend = fit_blend(r_te[odd_week, 1], bte[odd_week], test["y"].to_numpy()[odd_week], test["h"].to_numpy()[odd_week])
    cal = test[odd_week].copy()
    cal[["q10", "q50", "q90"]] = _assemble(r_te[odd_week], bte[odd_week], cal["h"].to_numpy(), blend, None).to_numpy()
    scale = interval_scale(cal)
    test = test[~odd_week].copy()
    test[["q10", "q50", "q90"]] = _assemble(r_te[~odd_week], bte[~odd_week], test["h"].to_numpy(),
                                            blend, scale).to_numpy()
    report = evaluate(test)
    report["interval_scale"] = {str(H_EDGES[k]): v for k, v in scale.items()}
    report["blend_weight"] = {str(H_EDGES[k]): v for k, v in blend.items()}
    report["period"] = {"train_start": str(X["issue_time"].min()), "test_start": str(test["issue_time"].min()),
                        "test_end": str(X["target_time"].max())}
    if verbose:
        print(f"   trust in model's predicted change by lead time (h>=): {report['blend_weight']}")

    # permutation importance (median model) on a sample of the backtest
    samp = test.sample(min(15000, len(test)), random_state=0)
    Rs, bs = design(samp)
    pi = permutation_importance(models[0.5], Rs[cols], np.log1p(samp["y"]) - bs, n_repeats=3,
                                random_state=0, scoring="neg_mean_absolute_error")
    report["importance"] = (pd.Series(pi.importances_mean, index=cols)
                            .sort_values(ascending=False).head(20).round(5).to_dict())

    keep_h = [1, 3, 6, 12, 24, 36, 48]
    bt = test[test["h"].isin(keep_h)][["issue_time", "target_time", "h", "y", "q10", "q50", "q90",
                                       "c_cams", "p_pm", "p_same_hour"]]
    bt.to_csv(config.REPORT_DIR / "backtest_predictions.csv", index=False)

    if fit_final:                                         # refit on everything for live use
        if verbose:
            print("   refitting on full history for live forecasting ...")
        Ra, ba = design(X)
        models = {q: _new_model(q).fit(Ra[cols], np.log1p(X["y"]) - ba) for q in config.QUANTILES}

    bundle = {"models": models, "cols": cols, "scale": scale, "blend": blend, "version": 2, "climatology": climatology(df),
              "trained_until": str(df["pm25"].last_valid_index()), "city": config.CITY}
    joblib.dump(bundle, config.MODEL_DIR / "pm25_quantile_gbm.joblib")
    (config.REPORT_DIR / "backtest.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def interval_scale(cal: pd.DataFrame, target: float = 0.8) -> dict:
    """Factor per lead-time bucket that makes the q10-q90 band cover `target` of outcomes."""
    l10, l50, l90, ly = (np.log1p(cal[c].to_numpy()) for c in ("q10", "q50", "q90", "y"))
    need = np.where(ly < l50, (l50 - ly) / np.clip(l50 - l10, 1e-3, None),
                    (ly - l50) / np.clip(l90 - l50, 1e-3, None))
    out = {}
    for b in np.unique(_bucket(cal["h"])):
        m = _bucket(cal["h"]) == b
        out[int(b)] = float(np.clip(np.quantile(need[m], target), 0.5, 4.0))
    return out


def _mae(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else float("nan")


def evaluate(test: pd.DataFrame) -> dict:
    y = test["y"].to_numpy()
    out = {"by_horizon": [], "alerts": {}, "reliability": []}
    for h, g in test.groupby("h"):
        yy = g["y"].to_numpy()
        row = {"h": int(h),
               "mae_model": _mae(g["q50"].to_numpy(), yy),
               "mae_persistence": _mae(g["p_pm"].to_numpy(), yy),
               "mae_same_hour": _mae(g["p_same_hour"].to_numpy(), yy),
               "mae_cams": _mae(g["c_cams"].to_numpy(), yy),
               "rmse_model": float(np.sqrt(np.nanmean((g["q50"] - g["y"]) ** 2))),
               "coverage_80": float(((g["y"] >= g["q10"]) & (g["y"] <= g["q90"])).mean())}
        out["by_horizon"].append(row)
    bh = pd.DataFrame(out["by_horizon"])
    for name, lo, hi in (("0-24h", 1, 24), ("24-48h", 25, 48)):
        s = bh[(bh.h >= lo) & (bh.h <= hi)].mean(numeric_only=True)
        out[f"summary_{name}"] = {k: round(float(v), 3) for k, v in s.items() if k != "h"}
        out[f"summary_{name}"]["skill_vs_persistence"] = round(1 - s.mae_model / s.mae_persistence, 3)
        if np.isfinite(s.mae_cams):
            out[f"summary_{name}"]["skill_vs_cams"] = round(1 - s.mae_model / s.mae_cams, 3)

    # alert verification at 24-48 h lead time
    lead = test[(test.h >= 24) & (test.h <= 48)]
    for thr in sorted({config.NAAQS_24H, 90, 120}):
        p = exceed_prob(lead.q10, lead.q50, lead.q90, thr)
        warn, event = p >= 0.5, lead["y"].to_numpy() > thr
        hits, misses = int((warn & event).sum()), int((~warn & event).sum())
        fa = int((warn & ~event).sum())
        out["alerts"][str(thr)] = {
            "events": int(event.sum()),
            "POD": round(hits / max(hits + misses, 1), 3),          # probability of detection
            "FAR": round(fa / max(hits + fa, 1), 3),                # false-alarm ratio
            "CSI": round(hits / max(hits + misses + fa, 1), 3),     # critical success index
            "brier": round(float(np.mean((p - event) ** 2)), 4),
        }
        if thr == 90:
            bins = np.linspace(0, 1, 11)
            idx = np.digitize(p, bins[1:-1])
            for b in range(10):
                m = idx == b
                if m.sum() >= 30:
                    out["reliability"].append({"forecast_prob": float(p[m].mean()),
                                               "observed_freq": float(event[m].mean()), "n": int(m.sum())})
    return out


# ------------------------------------------------------------------ live forecast
def load_bundle():
    return joblib.load(config.MODEL_DIR / "pm25_quantile_gbm.joblib")


def _tail(df: pd.DataFrame) -> pd.DataFrame:
    """Enough history before the last observation for all lag features."""
    last_obs = df["pm25"].last_valid_index()
    return df.loc[last_obs - pd.Timedelta(hours=200): last_obs + pd.Timedelta(hours=config.MAX_HORIZON)]


def _forecast_frame(bundle, df: pd.DataFrame) -> pd.DataFrame:
    tail = _tail(df)
    sf = series_features(tail)
    last = int(np.where(sf["p_pm"].notna().to_numpy())[0].max())
    X = make_samples(sf, np.array([last]), np.arange(1, config.MAX_HORIZON + 1), with_target=False)
    return X


def forecast(bundle, df: pd.DataFrame) -> pd.DataFrame:
    X = _forecast_frame(bundle, df)
    if bundle.get("version", 1) < 2:
        raise SystemExit("Model file is from an older version - run `python pipeline.py train` once.")
    out = X[["issue_time", "target_time", "h", "c_cams"]].join(predict_quantiles(bundle, X))
    return out.rename(columns={"c_cams": "cams"})


def driver_attribution(bundle, df: pd.DataFrame, fc: pd.DataFrame) -> pd.DataFrame:
    """How much each weather factor pushes the forecast above/below a 'typical' day.
    Replaces one group of inputs with its climatological value for that month & hour,
    re-runs the model, and reports the change in the median forecast (µg/m³)."""
    clim = bundle["climatology"]
    tail = _tail(df).copy()
    loc = tail.index.tz_convert(config.LOCAL_TZ)
    key = pd.MultiIndex.from_arrays([loc.month, loc.hour])
    window = tail.index >= fc["issue_time"].iloc[0] - pd.Timedelta(hours=24)
    base = fc.set_index("target_time")["q50"]
    rows = []
    for group, cols in DRIVER_GROUPS.items():
        cols = [c for c in cols if c in tail and c in clim]
        if not cols:
            continue
        alt = tail.copy()
        typical = clim.reindex(key)[cols].to_numpy()
        vals = alt[cols].to_numpy(dtype=float, copy=True)
        vals[window] = np.where(np.isnan(typical[window]), vals[window], typical[window])
        alt[cols] = vals
        X = _forecast_frame(bundle, alt)
        q = predict_quantiles(bundle, X)["q50"].to_numpy()
        diff = base.to_numpy() - q
        rows.append({"driver": group,
                     "effect_next24h": float(np.mean(diff[:24])),
                     "effect_24_48h": float(np.mean(diff[24:48]))})
    return pd.DataFrame(rows)
