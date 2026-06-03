"""
loo_neuralprophet_logspace.py
=============================================================
Leave-One-Out validation for a GLOBAL NeuralProphet panel model.

Design principles:
  - Target: log1p(consumption_GWh) — additive in log-space, multiplicative in GWh-space
  - Event regressors: 11 binary dummies wc_step_t for t in [-5, +5]  (n=11, covering -5 to +5 inclusive)
    NOTE: "months −5 to +6" = 12 calendar months; the step dummies cover the
    half-open interval [-5, +5] = 11 steps inside the model; month +6 is the
    trailing post-event month, captured by the dummy for t=+5 shifted one step.
    We follow the spec literally: range(-5, 6) → t ∈ {-5,-4,-3,-2,-1,0,1,2,3,4,5}
    which is exactly 11 dummies.  Month +6 is NOT a regressor (no data to train on).
  - trend_global_local='local', seasonality_global_local='local'
  - n_lags=0 (no AR context)
  - Fixed learning_rate=1e-3 — avoids the LR-finder which crashes on PyTorch 2.6
    due to its internal checkpoint save/restore using weights_only=True.
  - Counterfactual baseline fitted on log1p data; back-transformed with expm1.
  - Pulse injection: predicted(t) = counterfactual(t) × (1 + consensus_pulse_pct(t))
=============================================================
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\TS-project"
DATA_DIR    = r"C:\Users\Hp\OneDrive\Desktop\Data\Data Time series project"

BASELINE_SUMMARY_CSV = os.path.join(PROJECT_DIR, "loo_baseline_summary.csv")

# ── Event anchors ──────────────────────────────────────────────────────────────
EVENT_START_DATES = {
    "QAT": "2022-11-01",
    "EGY": "2019-06-01",
    "RUS": "2018-06-01",
    "CMR": "2022-01-01",
    "ZAF": "2010-06-01",
}

DONORS = ["QAT", "EGY", "RUS", "CMR", "ZAF"]

# Event window: months -5 to +5 as binary dummies (range(-5, 6) → 11 steps)
# The evaluation window is months -5 to +6 (12 calendar months).
EVENT_T_RANGE = list(range(-5, 6))   # [-5,-4,...,+5]
REGRESSOR_COLS = [f"wc_step_{t}" for t in EVENT_T_RANGE]

# ── Data loading ───────────────────────────────────────────────────────────────
def load_donor_series(iso: str) -> pd.Series:
    """Return a monthly GWh pd.Series indexed by period-start dates."""
    if iso == "ZAF":
        path = os.path.join(DATA_DIR, "south_africa_donor_electricity_demand.csv")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], format="%Y-%B")
        df = df.set_index("date").sort_index()
        return df["Consumption_GWh"].asfreq("MS").rename(iso)

    elif iso == "CMR":
        path = os.path.join(DATA_DIR, "cameroon_monthly_electricity _consumption.csv")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], format="%Y-%B")
        df = df.set_index("date").sort_index()
        return df["Consumption_GWh"].asfreq("MS").rename(iso)

    elif iso == "RUS":
        path = os.path.join(DATA_DIR, "Russia_data.csv")
        df = pd.read_csv(path, sep=";")
        df.columns = ["date_str", "consumption_kwh_bn"]
        df["date"] = pd.to_datetime(df["date_str"].str.replace("'", "20", regex=False), format="%b %Y")
        df = df.set_index("date").sort_index()
        series_local = (df["consumption_kwh_bn"] * 1000).rename(iso)
        try:
            iea_path = os.path.join(DATA_DIR, "monthly_full_release_long_format.csv")
            iea = pd.read_csv(iea_path)
            iea_sub = iea[(iea["ISO 3 code"] == "RUS") & (iea["Variable"] == "Demand")].copy()
            iea_sub["date"] = pd.to_datetime(iea_sub["Date"])
            iea_sub = iea_sub.set_index("date").sort_index()
            series_iea = (iea_sub["Value"] * 1000).rename(iso)
        except Exception:
            series_iea = pd.Series(dtype=float, name=iso)
        combined_idx = series_local.index.union(series_iea.index)
        series = pd.Series(index=combined_idx, dtype=float, name=iso)
        series.update(series_iea)
        series.update(series_local)
        return series.asfreq("MS").rename(iso)

    elif iso == "QAT":
        path = os.path.join(DATA_DIR, "qatar_electricity_transmitted (1).csv")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], format="%Y-%B")
        df = df.set_index("date").sort_index()
        series_local = (df["Total_MWh"] / 1000).rename(iso)
        try:
            iea_path = os.path.join(DATA_DIR, "monthly_full_release_long_format.csv")
            iea = pd.read_csv(iea_path)
            iea_sub = iea[(iea["ISO 3 code"] == "QAT") & (iea["Variable"] == "Demand")].copy()
            iea_sub["date"] = pd.to_datetime(iea_sub["Date"])
            iea_sub = iea_sub.set_index("date").sort_index()
            series_iea = (iea_sub["Value"] * 1000).rename(iso)
        except Exception:
            series_iea = pd.Series(dtype=float, name=iso)
        combined_idx = series_local.index.union(series_iea.index)
        series = pd.Series(index=combined_idx, dtype=float, name=iso)
        series.update(series_iea)
        series.update(series_local)
        return series.asfreq("MS").rename(iso)

    elif iso == "EGY":
        iea_path = os.path.join(DATA_DIR, "monthly_full_release_long_format.csv")
        iea = pd.read_csv(iea_path)
        iea_sub = iea[(iea["ISO 3 code"] == "EGY") & (iea["Variable"] == "Demand")].copy()
        iea_sub["date"] = pd.to_datetime(iea_sub["Date"])
        iea_sub = iea_sub.set_index("date").sort_index()
        return (iea_sub["Value"] * 1000).asfreq("MS").rename(iso)

    else:
        raise ValueError(f"Unknown donor ISO: {iso}")


def build_np_df(iso: str, series: pd.Series, cutoff_date=None) -> pd.DataFrame:
    """
    Build a NeuralProphet-ready dataframe for one donor.
    Adds all wc_step_{t} regressor columns (all zeros by default).
    If cutoff_date is given, only rows up to and including that date are kept.
    """
    s = series.dropna()
    if cutoff_date is not None:
        s = s[s.index <= cutoff_date]

    df = pd.DataFrame({"ds": s.index, "y": np.log1p(s.values), "ID": iso})

    # Initialise all regressor columns to 0
    for col in REGRESSOR_COLS:
        df[col] = 0

    # Stamp each event-month with a 1
    event_start = pd.Timestamp(EVENT_START_DATES[iso])
    for t in EVENT_T_RANGE:
        col = f"wc_step_{t}"
        target_month = event_start + pd.DateOffset(months=t)
        mask = df["ds"] == target_month
        df.loc[mask, col] = 1

    return df.reset_index(drop=True)


# ── Beta extraction ────────────────────────────────────────────────────────────
def extract_betas(model) -> np.ndarray:
    """
    Extract the linear coefficient for each wc_step_t regressor.
    In NP 0.9.0, all additive future regressors share a single tensor:
    'future_regressors.regressor_params.additive' of shape (1, num_regressors).
    The order matches the OrderedDict model.config_regressors.regressors.
    """
    state = model.model.state_dict()
    additive_weights = state.get("future_regressors.regressor_params.additive")
    
    if additive_weights is None:
        raise KeyError("Could not find 'future_regressors.regressor_params.additive' in state_dict.")
        
    additive_weights = additive_weights.flatten() # shape (num_regressors,)
    
    reg_names = list(model.config_regressors.regressors.keys())
    
    betas = []
    for t in EVENT_T_RANGE:
        name = f"wc_step_{t}"
        try:
            idx = reg_names.index(name)
            beta_val = additive_weights[idx].item()
            betas.append(beta_val)
        except ValueError:
            raise KeyError(f"Regressor '{name}' not found in model.config_regressors.regressors")
            
    return np.array(betas)


# ── Weighted-average baseline forecast loader ──────────────────────────────────
def load_weighted_avg_pulse(left_out: str) -> np.ndarray | None:
    """Load the saved per-fold weighted-average pulse from weighted_avg_pulse_per_fold.csv."""
    csv_path = os.path.join(PROJECT_DIR, "weighted_avg_pulse_per_fold.csv")
    if not os.path.exists(csv_path):
        return None
    df_pulse = pd.read_csv(csv_path)
    row = df_pulse[df_pulse["fold_left_out"] == left_out]
    if row.empty:
        return None
    # Columns: fold_left_out, month_-5, month_-4, ..., month_6
    months = list(range(-5, 7))  # 12 months
    vals = []
    for m in months:
        col = f"month_{m}"
        if col in row.columns:
            vals.append(row[col].values[0])
        else:
            vals.append(0.0)
    return np.array(vals)


# ── Main LOO pipeline ──────────────────────────────────────────────────────────
def main():
    from neuralprophet import NeuralProphet

    print("Loading loo_baseline_summary.csv...")
    df_baseline = pd.read_csv(BASELINE_SUMMARY_CSV).set_index("donor")

    # Pre-load all raw series
    print("Pre-loading all donor series...")
    all_series = {iso: load_donor_series(iso) for iso in DONORS}

    results = []

    # ── Plotting setup ──
    fig_f, axes_f = plt.subplots(1, 5, figsize=(22, 4), sharey=False)
    fig_s, ax_s   = plt.subplots(figsize=(10, 6))
    COLORS = plt.cm.tab10(np.linspace(0, 1, 5))

    for fold_idx, left_out in enumerate(DONORS):
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx+1}/5  —  Left-out donor: {left_out}")
        print(f"{'='*60}")

        event_start   = pd.Timestamp(EVENT_START_DATES[left_out])
        cutoff_date   = event_start - pd.DateOffset(months=6)
        eval_months   = [event_start + pd.DateOffset(months=m - 5) for m in range(12)]
        # eval_months[0]  = event_start - 5 months  (month -5)
        # eval_months[11] = event_start + 6 months  (month +6)

        # ── Step 1: Build global training panel (4 donors) ──────────────────
        print("  [1] Building training panel...")
        train_dfs = []
        for iso in DONORS:
            if iso == left_out:
                continue
            df_d = build_np_df(iso, all_series[iso])
            train_dfs.append(df_d)

        train_df = pd.concat(train_dfs, ignore_index=True)

        assert left_out not in train_df["ID"].values, \
            f"ASSERTION FAILED: {left_out} still present in training data!"
        print(f"    Verified: {left_out} is absent from training panel.")
        print(f"    Training donors: {sorted(train_df['ID'].unique())}")
        print(f"    Training rows: {len(train_df)}")

        # ── Step 2: Train global NeuralProphet model ─────────────────────────
        print("  [2] Training global NeuralProphet model...")
        model = NeuralProphet(
            growth="linear",
            n_forecasts=1,
            n_lags=0,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            trend_global_local="local",
            season_global_local="local",      # NP 0.9.0 param name
            learning_rate=1e-3,               # Fixed LR — skips LR finder (avoids PyTorch 2.6 crash)
            epochs=300,
            batch_size=16,
            trend_reg=0.1,
            seasonality_reg=0.1,
        )

        for t in EVENT_T_RANGE:
            model.add_future_regressor(name=f"wc_step_{t}", regularization=0.1)

        model.fit(train_df, freq="MS")
        print("    Training complete.")

        # Save model
        model_path = os.path.join(PROJECT_DIR, f"neuralprophet_logspace_fold_{left_out}.pkl")
        with open(model_path, "wb") as fh:
            pickle.dump(model, fh)
        print(f"    Saved model → {os.path.basename(model_path)}")

        # ── Step 3: Extract consensus pulse ──────────────────────────────────
        print("  [3] Extracting consensus pulse (beta coefficients)...")
        try:
            beta_log = extract_betas(model)
        except KeyError as e:
            print(f"    ERROR extracting betas: {e}")
            print("    State dict keys:")
            for k in model.model.state_dict():
                print(f"      {k}")
            raise

        consensus_pulse_pct = np.expm1(beta_log)   # dimensionless fraction (e.g. 0.05 = 5% lift)

        # Pad to 12 months: the +6 month has no trained dummy → treat as 0%
        beta_log_12       = np.append(beta_log, 0.0)          # shape (12,)
        pulse_pct_12      = np.append(consensus_pulse_pct, 0.0)

        pulse_df = pd.DataFrame({
            "months_to_wc":       list(range(-5, 7)),
            "beta_log":           beta_log_12,
            "consensus_pulse_pct": pulse_pct_12,
        })
        pulse_path = os.path.join(PROJECT_DIR, f"consensus_pulse_logspace_fold_{left_out}.csv")
        pulse_df.to_csv(pulse_path, index=False)
        print(f"    Saved pulse → {os.path.basename(pulse_path)}")
        print(f"    Beta (log-space):  {np.round(beta_log, 4)}")
        print(f"    Pulse (% lift):    {np.round(consensus_pulse_pct * 100, 2)} %")

        # ── Step 4: Fit counterfactual baseline for left-out donor ───────────
        print("  [4] Fitting counterfactual baseline for left-out donor...")
        lo_series    = all_series[left_out]
        train_lo_ser = lo_series[lo_series.index <= cutoff_date]

        df_cf = pd.DataFrame({
            "ds": train_lo_ser.index,
            "y":  np.log1p(train_lo_ser.values),
        })

        baseline_model = NeuralProphet(
            growth="linear",
            n_forecasts=1,
            n_lags=0,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            trend_reg=0.1,
            seasonality_reg=0.1,
            learning_rate=1e-3,               # Fixed LR — skips LR finder
            epochs=300,
            batch_size=16,
        )
        baseline_model.fit(df_cf, freq="MS")

        # Forecast 12 months (months -5 to +6)
        # make_future_dataframe for n_forecasts=1, n_lags=0: periods = steps forward
        future_df = baseline_model.make_future_dataframe(df_cf, periods=12, n_historic_predictions=True)
        cf_forecast = baseline_model.predict(future_df)

        # Extract only the 12 target months from the forecast
        cf_forecast_ds = pd.to_datetime(cf_forecast["ds"])
        target_ds_series = pd.Series(eval_months)
        mask_cf = cf_forecast_ds.isin(target_ds_series)
        cf_log_vals = cf_forecast.loc[mask_cf, "yhat1"].values

        if len(cf_log_vals) < 12:
            print(f"    WARNING: Got only {len(cf_log_vals)}/12 counterfactual months.")

        counterfactual_GWh = np.expm1(cf_log_vals)     # shape up to (12,)

        # ── Step 5: Inject consensus pulse ───────────────────────────────────
        print("  [5] Injecting consensus pulse...")
        n = len(counterfactual_GWh)
        pulse_pct_use = pulse_pct_12[:n]

        # Multiplicative injection
        predicted_GWh = counterfactual_GWh * (1.0 + pulse_pct_use)

        # ── Step 6: Evaluate ─────────────────────────────────────────────────
        print("  [6] Evaluating...")
        actual_all = lo_series[lo_series.index.isin(eval_months)]
        actual_GWh = actual_all.reindex(pd.DatetimeIndex(eval_months[:n])).values

        # Remove any NaN rows where actual is missing
        valid_mask = ~np.isnan(actual_GWh)
        actual_GWh_v      = actual_GWh[valid_mask]
        predicted_GWh_v   = predicted_GWh[valid_mask]
        cf_GWh_v          = counterfactual_GWh[valid_mask]
        pulse_pct_v       = pulse_pct_use[valid_mask]
        month_labels      = list(range(-5, -5 + n))
        month_labels_v    = np.array(month_labels)[valid_mask]

        mae  = float(np.mean(np.abs(actual_GWh_v - predicted_GWh_v)))
        rmse = float(np.sqrt(np.mean((actual_GWh_v - predicted_GWh_v) ** 2)))
        mape = float(np.mean(np.abs((actual_GWh_v - predicted_GWh_v) / actual_GWh_v)) * 100)

        sarima_rmse   = float(df_baseline.loc[left_out, "SARIMA_RMSE"])
        baseline_rmse = float(df_baseline.loc[left_out, "RMSE"])

        improvement_vs_sarima   = (sarima_rmse   - rmse) / sarima_rmse   * 100
        improvement_vs_baseline = (baseline_rmse - rmse) / baseline_rmse * 100

        # Shape metric
        actual_residual = actual_GWh_v - cf_GWh_v
        denom_actual = np.max(np.abs(actual_residual))
        denom_pulse  = np.max(np.abs(pulse_pct_v))

        if denom_actual > 0 and denom_pulse > 0:
            actual_res_norm   = actual_residual / denom_actual
            pulse_norm        = pulse_pct_v      / denom_pulse
            shape_mse = float(np.mean((pulse_norm - actual_res_norm) ** 2))
        else:
            actual_res_norm = np.zeros_like(actual_residual)
            pulse_norm      = np.zeros_like(pulse_pct_v)
            shape_mse       = np.nan

        print(f"    MAE={mae:.1f} GWh | RMSE={rmse:.1f} GWh | MAPE={mape:.2f}%")
        print(f"    vs SARIMA  : {improvement_vs_sarima:+.2f}%")
        print(f"    vs Weighted: {improvement_vs_baseline:+.2f}%")
        print(f"    Shape MSE  : {shape_mse:.4f}")

        results.append({
            "donor":                    left_out,
            "MAE":                      mae,
            "RMSE":                     rmse,
            "MAPE":                     mape,
            "shape_MSE":                shape_mse,
            "SARIMA_RMSE":              sarima_rmse,
            "baseline_RMSE":            baseline_rmse,
            "improvement_vs_sarima_pct":   improvement_vs_sarima,
            "improvement_vs_baseline_pct": improvement_vs_baseline,
        })

        # ── Forecast subplot ──────────────────────────────────────────────────
        ax = axes_f[fold_idx]
        color = COLORS[fold_idx]
        month_x = month_labels_v

        ax.plot(month_x, actual_GWh_v,     "-",  color="black",     lw=1.8, label="Actual")
        ax.plot(month_x, predicted_GWh_v,  "-",  color="tab:red",   lw=1.5, label="NP Combined")
        ax.plot(month_x, cf_GWh_v,         "--", color="tab:blue",  lw=1.2, label="Counterfactual")

        # Weighted-average baseline forecast (from stored pulse)
        wa_pulse = load_weighted_avg_pulse(left_out)
        if wa_pulse is not None:
            # Weighted avg baseline: SARIMA baseline × (1 + wa_pulse)
            # We don't have the SARIMA GWh array here; skip plotting WA forecast
            # but note wa_pulse is a normalized fractional lift in GWh (see loo_baseline_validation.py)
            # The WA baseline script stores it already in absolute GWh form; skip this layer
            pass  # Would need the SARIMA GWh baseline to reconstruct; omit for cleanliness

        ax.axvspan(-0.5, 1.5, color="tab:red", alpha=0.08, zorder=0)
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlim(-5, 6)
        ax.set_title(f"{left_out}\nRMSE={rmse:.0f} | MAPE={mape:.1f}%", fontsize=9)
        ax.set_xlabel("Months rel. to WC", fontsize=8)
        if fold_idx == 0:
            ax.set_ylabel("GWh", fontsize=8)
            ax.legend(fontsize=7)

        # ── Pulse-shape subplot ───────────────────────────────────────────────
        ax_s.plot(month_x, actual_res_norm, "-",  color=color, lw=1.5,
                  label=f"{left_out} actual res")
        ax_s.plot(month_x, pulse_norm,      "--", color=color, lw=1.2, alpha=0.85,
                  label=f"{left_out} NP pulse")

    # ── Save forecast plot ────────────────────────────────────────────────────
    fig_f.tight_layout(pad=1.5)
    fig_f.savefig(os.path.join(PROJECT_DIR, "loo_neuralprophet_logspace_forecast.png"), dpi=150)
    plt.close(fig_f)
    print(f"\nSaved forecast plot → loo_neuralprophet_logspace_forecast.png")

    # ── Save pulse-shape plot ─────────────────────────────────────────────────
    ax_s.axvline(0, color="black", ls=":", lw=0.9)
    ax_s.set_xlim(-5, 6)
    ax_s.set_ylim(-1.2, 1.2)
    ax_s.set_xlabel("Months relative to WC start", fontsize=10)
    ax_s.set_ylabel("Normalized value", fontsize=10)
    ax_s.set_title("NeuralProphet LOO — Pulse shape vs Actual normalized residual", fontsize=10)
    ax_s.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(PROJECT_DIR, "loo_neuralprophet_logspace_pulse_shape.png"), dpi=150)
    plt.close(fig_s)
    print(f"Saved pulse-shape plot → loo_neuralprophet_logspace_pulse_shape.png")

    # ── Save summary CSV ──────────────────────────────────────────────────────
    df_out = pd.DataFrame(results)
    df_out = df_out[[
        "donor", "MAE", "RMSE", "MAPE", "shape_MSE",
        "SARIMA_RMSE", "baseline_RMSE",
        "improvement_vs_sarima_pct", "improvement_vs_baseline_pct",
    ]]
    out_csv = os.path.join(PROJECT_DIR, "loo_neuralprophet_logspace_summary.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Saved summary CSV  → loo_neuralprophet_logspace_summary.csv")

    # ── Interpretation block ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  INTERPRETATION BLOCK")
    print("=" * 60)

    n_beat_baseline = 0   # donors where NP beats weighted-average baseline

    for row in results:
        donor   = row["donor"]
        r_np    = row["RMSE"]
        r_sar   = row["SARIMA_RMSE"]
        r_base  = row["baseline_RMSE"]
        i_sar   = row["improvement_vs_sarima_pct"]
        i_base  = row["improvement_vs_baseline_pct"]

        best = "NeuralProphet"
        if r_base <= r_np and r_base <= r_sar:
            best = "Weighted Average"
        elif r_sar <= r_np:
            best = "SARIMA-only"
        else:
            n_beat_baseline += 1

        print(f"\n[{donor}]")
        print(f"  1. NP vs SARIMA-only   : {i_sar:+.2f}%  | ΔGWh = {r_sar - r_np:+.1f}")
        print(f"  2. NP vs Weighted Avg  : {i_base:+.2f}%  | ΔGWh = {r_base - r_np:+.1f}")
        print(f"  3. Best model: {best}")

    mean_mape = df_out["MAPE"].mean()
    n_underperforming = len(DONORS) - n_beat_baseline

    print("\n" + "=" * 60)
    print("  GLOBAL SUMMARY")
    print("=" * 60)
    print(f"  Mean MAPE across 5 folds        : {mean_mape:.2f}%")
    print(f"  NP beats Weighted Average on    : {n_beat_baseline}/5 donors")
    print(f"  NP underperforms on             : {n_underperforming}/5 donors")
    print()
    if n_underperforming >= 3:
        print("  WARNING: NeuralProphet global panel does NOT outperform")
        print("           the weighted-average baseline on ≥3/5 donors.")
        print("  Recommend deploying WEIGHTED AVERAGE as Layer 2 for Morocco 2030.")
    else:
        print("  Recommended Layer 2 for Morocco 2030: NeuralProphet Global Panel")

    print("\nDone.")


if __name__ == "__main__":
    main()
