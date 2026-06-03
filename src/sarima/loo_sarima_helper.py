"""
loo_sarima_helper.py
Standalone SARIMA helper — runs ONE donor fold at a time.
Called as a subprocess by loo_full_pipeline.py.

Usage:
  python loo_sarima_helper.py <donor_name> <output_json>

<donor_name>  : key in DONORS dict (e.g. 'qatar', 'russia')
<output_json> : path where results are written as JSON

Keeps imports minimal to avoid MKL/BLAS conflicts with PyTorch.
"""

import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ── Import only what we need from the extraction script ──────────────────────
from sarima_residual_extraction import DONORS, load_donor_data, maybe_log, maybe_exp

np.random.seed(42)


def build_covid_dummy(index: pd.DatetimeIndex) -> pd.Series:
    dummy = pd.Series(0.0, index=index, name="covid")
    for m, val in [("2020-04-01", 1.0), ("2020-05-01", 1.0),
                   ("2020-06-01", 1.0), ("2020-07-01", 0.5)]:
        ts = pd.Timestamp(m)
        if ts in dummy.index:
            dummy.loc[ts] = val
    return dummy


def fit_sarima(train_t, exog_train, iso):
    candidates = [
        ((1, 1, 1), (1, 1, 1, 12)),
        ((1, 1, 1), (0, 1, 1, 12)),
        ((1, 1, 0), (0, 1, 1, 12)),
        ((0, 1, 1), (0, 1, 1, 12)),
        ((1, 1, 0), (0, 1, 0, 12)),
        ((0, 1, 1), (0, 1, 0, 12)),
    ]
    for order, s_order in candidates:
        try:
            mod = SARIMAX(
                train_t, exog=exog_train,
                order=order, seasonal_order=s_order,
                enforce_stationarity=False, enforce_invertibility=False,
            )
            res = mod.fit(disp=False, maxiter=300, method='lbfgs')
            print(f"  [{iso}] SARIMA{order}x{s_order}  AIC={res.aic:.1f}", flush=True)
            return res, order, s_order
        except Exception as exc:
            print(f"  [{iso}] SARIMA{order}x{s_order} failed: {exc}", flush=True)
            continue
    raise RuntimeError(f"All SARIMA candidates failed for {iso}")


def run_fold(country: str, output_path: str):
    cfg = DONORS[country]
    iso = cfg["iso"]
    wc_start = cfg["wc_start"]
    cutoff = cfg["training_cutoff"]
    win_s, win_e = cfg["residual_window"]

    print(f"\n[{iso}] Loading data...", flush=True)
    series = load_donor_data(country, cfg).asfreq("MS")
    train = series[series.index <= cutoff]

    covid_dummy = build_covid_dummy(series.index)
    use_exog = bool(covid_dummy.sum() > 0)
    exog_train = covid_dummy[train.index].to_frame("covid") if use_exog else None

    print(f"[{iso}] Training on {len(train)} obs | exog={use_exog}", flush=True)
    train_t = maybe_log(train, cfg)

    sarima_res, best_order, best_seasonal = fit_sarima(train_t, exog_train, iso)

    # Forecast
    n_fc = (win_e.year - cutoff.year) * 12 + (win_e.month - cutoff.month)
    fc_idx = pd.date_range(cutoff + pd.DateOffset(months=1), periods=n_fc, freq="MS")
    exog_f = covid_dummy.reindex(fc_idx).to_frame("covid") if use_exog else None

    fc_log = pd.Series(sarima_res.forecast(steps=n_fc, exog=exog_f).values, index=fc_idx)
    fc_gwh = maybe_exp(fc_log, cfg)

    actual_win = series[(series.index >= win_s) & (series.index <= win_e)]
    fc_win = fc_gwh.reindex(actual_win.index)
    raw_resid = actual_win - fc_win

    # Normalize using event-window peak (by design, consistent with training set)
    event_ts = [wc_start, wc_start + pd.DateOffset(months=1)]
    mask = raw_resid.index.isin(event_ts)
    peak = raw_resid[mask].abs().max() if mask.any() else np.nan
    if pd.isna(peak) or peak == 0:
        peak = raw_resid.abs().max()
    if pd.isna(peak) or peak == 0:
        peak = 1.0
    resid_norm = raw_resid / peak

    months_to_wc = [
        (d.year - wc_start.year) * 12 + (d.month - wc_start.month)
        for d in actual_win.index
    ]

    # Save results as JSON
    result = {
        "iso": iso,
        "peak_gwh": float(peak),
        "months_to_wc": months_to_wc,
        "ds": [str(d.date()) for d in actual_win.index],
        "actual_gwh": actual_win.values.tolist(),
        "sarima_gwh": fc_win.values.tolist(),
        "raw_resid_gwh": raw_resid.values.tolist(),
        "resid_norm": resid_norm.values.tolist(),
        "best_order": list(best_order),
        "best_seasonal": list(best_seasonal),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[{iso}] Saved -> {output_path}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python loo_sarima_helper.py <country_key> <output_json>")
        sys.exit(1)
    run_fold(sys.argv[1], sys.argv[2])
