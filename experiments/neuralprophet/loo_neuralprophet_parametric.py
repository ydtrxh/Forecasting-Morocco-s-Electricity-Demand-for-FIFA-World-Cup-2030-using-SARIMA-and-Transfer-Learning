"""
loo_neuralprophet_parametric.py
=============================================================
Leave-One-Out validation for a GLOBAL NeuralProphet panel model
using a Parametric Gaussian Pulse for event regression.

Key design principles:
  - Parametric Gaussian Pulse: β_t = a * exp( - (t - μ)^2 / (2 * σ^2) )
  - μ and σ optimized via grid search on training loss
  - Amplitude 'a' learned by NP global model
  - Target: log1p(consumption_GWh)
  - PyTorch 2.6 compat: fixed learning_rate=1e-3
  - Confounders: eskom_2008, covid_pulse
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
import logging

# Suppress heavy logging from pytorch_lightning/neuralprophet
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("neuralprophet").setLevel(logging.WARNING)
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
    """Build NeuralProphet dataframe for one donor."""
    s = series.dropna()
    if cutoff_date is not None:
        s = s[s.index <= cutoff_date]
    df = pd.DataFrame({"ds": s.index, "y": np.log1p(s.values), "ID": iso})
    return df.reset_index(drop=True)


def get_relative_month(ds_series: pd.Series, iso: str) -> pd.Series:
    """Compute relative month distance from event start for a donor."""
    event_start = pd.Timestamp(EVENT_START_DATES[iso])
    return (ds_series.dt.year - event_start.year) * 12 + (ds_series.dt.month - event_start.month)


def extract_amplitude_a(model, reg_name: str) -> float:
    """Extract learned amplitude weight 'a' for a regressor in NP 0.9.0."""
    state = model.model.state_dict()
    additive_weights = state.get("future_regressors.regressor_params.additive")
    if additive_weights is None:
        raise KeyError("Could not find 'future_regressors.regressor_params.additive' in state_dict.")
    
    additive_weights = additive_weights.flatten()
    reg_names = list(model.config_regressors.regressors.keys())
    
    try:
        idx = reg_names.index(reg_name)
        return additive_weights[idx].item()
    except ValueError:
        raise KeyError(f"Regressor '{reg_name}' not found in model.config_regressors.regressors")


# ── Grid Search Definitions ────────────────────────────────────────────────────
MU_GRID    = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
SIGMA_GRID = [0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2]
REG_SCALES = [1.0, 5.0, 10.0]


# ── Main LOO pipeline ──────────────────────────────────────────────────────────
def main():
    from neuralprophet import NeuralProphet

    print("Loading loo_baseline_summary.csv...")
    df_baseline = pd.read_csv(BASELINE_SUMMARY_CSV).set_index("donor")

    print("Pre-loading all donor series...")
    all_series = {iso: load_donor_series(iso) for iso in DONORS}

    results = []
    
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

        # ── Step 1: Base Data Preparation & Baseline Confounders ────────────
        print("  [1] Building training panel & adding confounders...")
        train_dfs = []
        for iso in DONORS:
            if iso == left_out:
                continue
            df_d = build_np_df(iso, all_series[iso])
            train_dfs.append(df_d)

        train_df = pd.concat(train_dfs, ignore_index=True)
        assert left_out not in train_df["ID"].values, f"{left_out} in training data!"

        train_df['eskom_2008'] = 0.0
        train_df['covid_pulse'] = 0.0
        train_df.loc[(train_df['ID'] == 'ZAF') & (train_df['ds'].dt.year == 2008), 'eskom_2008'] = 1.0
        train_df.loc[(train_df['ID'].isin(['EGY', 'CMR'])) & (train_df['ds'] >= '2020-04-01') & (train_df['ds'] <= '2020-06-01'), 'covid_pulse'] = 1.0
        train_df.loc[(train_df['ID'].isin(['EGY', 'CMR'])) & (train_df['ds'] == '2020-07-01'), 'covid_pulse'] = 0.5
        
        has_zaf = 'ZAF' in train_df['ID'].values
        has_covid = any(x in train_df['ID'].values for x in ['EGY', 'CMR'])

        # ── Step 2: Outer-Loop Optimization over Pulse Geometry ──────────────
        print(f"  [2] Grid searching pulse geometry (mu, sigma, reg) -> {len(MU_GRID)*len(SIGMA_GRID)*len(REG_SCALES)} fits...")
        best_loss = float('inf')
        best_params = None
        best_model = None

        total_fits = len(MU_GRID) * len(SIGMA_GRID) * len(REG_SCALES)
        fit_idx = 0
        
        # Pre-compute relative month distance for all rows to speed up grid search
        # Stored in a dict by ID
        t_arrays = {}
        for iso in train_df['ID'].unique():
            t_arrays[iso] = get_relative_month(train_df[train_df['ID'] == iso]['ds'], iso)
        
        for mu in MU_GRID:
            for sigma in SIGMA_GRID:
                # Pre-compute the pulse feature for this geometry
                pulse_feature = pd.Series(0.0, index=train_df.index)
                for iso in train_df['ID'].unique():
                    t_iso = t_arrays[iso]
                    pulse_feature[train_df['ID'] == iso] = np.exp(- (t_iso - mu)**2 / (2 * sigma**2))
                
                train_df['wc_pulse_feature'] = pulse_feature

                for reg_val in REG_SCALES:
                    fit_idx += 1
                    if fit_idx % 25 == 0:
                        print(f"      Running fit {fit_idx}/{total_fits}...")
                        
                    model = NeuralProphet(
                        growth='linear', n_forecasts=1, n_lags=0, n_changepoints=0,
                        yearly_seasonality=3, weekly_seasonality=False, daily_seasonality=False,
                        trend_global_local='local', season_global_local='local',
                        learning_rate=1e-3, epochs=300, batch_size=16, 
                        trend_reg=0.5, seasonality_reg=0.5
                    )
                    
                    model.add_future_regressor(name='wc_pulse_feature', regularization=reg_val, normalize='off')
                    if has_zaf:
                        model.add_future_regressor('eskom_2008', normalize='off')
                    if has_covid:
                        model.add_future_regressor('covid_pulse', normalize='off')
                        
                    metrics = model.fit(train_df, freq="MS", progress=None)  # Disable progress bar
                    final_loss = metrics['Loss'].iloc[-1]
                    
                    if final_loss < best_loss:
                        best_loss = final_loss
                        best_params = (mu, sigma, reg_val)
                        best_model = model

        mu_opt, sigma_opt, reg_opt = best_params
        print(f"    Optimal params: mu={mu_opt}, sigma={sigma_opt}, reg={reg_opt} | Loss={best_loss:.4f}")

        model_path = os.path.join(PROJECT_DIR, f"neuralprophet_parametric_fold_{left_out}.pkl")
        with open(model_path, "wb") as fh:
            pickle.dump(best_model, fh)

        # ── Step 3: Extract Reconstructed Smooth Percentage Lift ─────────────
        print("  [3] Extracting consensus pulse curve...")
        a = extract_amplitude_a(best_model, 'wc_pulse_feature')
        
        t_eval = np.arange(-5, 7)  # -5 to +6
        K_t = np.exp(- (t_eval - mu_opt)**2 / (2 * sigma_opt**2))
        consensus_pulse_pct = np.expm1(a * K_t)

        pulse_df = pd.DataFrame({
            "months_to_wc": t_eval,
            "K_t": K_t,
            "consensus_pulse_pct": consensus_pulse_pct,
        })
        pulse_df.to_csv(os.path.join(PROJECT_DIR, f"consensus_pulse_parametric_fold_{left_out}.csv"), index=False)
        print(f"    Amplitude 'a' = {a:.4f}")
        print(f"    Pulse (% lift) at t=0 = {consensus_pulse_pct[5] * 100:.2f}%")

        # ── Step 4: Fit Target Counterfactual Baseline ───────────────────────
        print("  [4] Fitting counterfactual baseline for left-out donor...")
        lo_series = all_series[left_out]
        train_lo_ser = lo_series[lo_series.index <= cutoff_date]

        df_cf = pd.DataFrame({
            "ds": train_lo_ser.index,
            "y":  np.log1p(train_lo_ser.values),
        })
        
        # Add confounders to left-out donor if applicable
        df_cf['eskom_2008'] = 0.0
        df_cf['covid_pulse'] = 0.0
        if left_out == 'ZAF':
            df_cf.loc[df_cf['ds'].dt.year == 2008, 'eskom_2008'] = 1.0
        if left_out in ['EGY', 'CMR']:
            df_cf.loc[(df_cf['ds'] >= '2020-04-01') & (df_cf['ds'] <= '2020-06-01'), 'covid_pulse'] = 1.0
            df_cf.loc[df_cf['ds'] == '2020-07-01', 'covid_pulse'] = 0.5

        baseline_model = NeuralProphet(
            growth='linear', n_forecasts=1, n_lags=0, n_changepoints=0,
            yearly_seasonality=3, weekly_seasonality=False, daily_seasonality=False,
            learning_rate=1e-3, epochs=300, batch_size=16
        )
        
        if left_out == 'ZAF':
            baseline_model.add_future_regressor('eskom_2008', normalize='off')
        if left_out in ['EGY', 'CMR']:
            baseline_model.add_future_regressor('covid_pulse', normalize='off')
            
        baseline_model.fit(df_cf, freq="MS", progress=None)

        # Forecast forward
        future_df = baseline_model.make_future_dataframe(df_cf, periods=12, n_historic_predictions=True)
        # Extend dummies accurately
        future_df['eskom_2008'] = 0.0
        future_df['covid_pulse'] = 0.0
        if left_out == 'ZAF':
            future_df.loc[future_df['ds'].dt.year == 2008, 'eskom_2008'] = 1.0
        if left_out in ['EGY', 'CMR']:
            future_df.loc[(future_df['ds'] >= '2020-04-01') & (future_df['ds'] <= '2020-06-01'), 'covid_pulse'] = 1.0
            future_df.loc[future_df['ds'] == '2020-07-01', 'covid_pulse'] = 0.5
            
        cf_forecast = baseline_model.predict(future_df)

        cf_forecast_ds = pd.to_datetime(cf_forecast["ds"])
        target_ds_series = pd.Series(eval_months)
        mask_cf = cf_forecast_ds.isin(target_ds_series)
        cf_log_vals = cf_forecast.loc[mask_cf, "yhat1"].values
        counterfactual_GWh = np.expm1(cf_log_vals)

        # ── Step 5: Multiplicative Pulse Injection & Evaluation ──────────────
        print("  [5] Multiplicative Pulse Injection & Evaluation...")
        predicted_GWh = counterfactual_GWh * (1.0 + consensus_pulse_pct)

        actual_all = lo_series[lo_series.index.isin(eval_months)]
        actual_GWh = actual_all.reindex(pd.DatetimeIndex(eval_months)).values

        valid_mask = ~np.isnan(actual_GWh)
        actual_GWh_v    = actual_GWh[valid_mask]
        predicted_GWh_v = predicted_GWh[valid_mask]
        cf_GWh_v        = counterfactual_GWh[valid_mask]
        pulse_pct_v     = consensus_pulse_pct[valid_mask]
        month_labels_v  = t_eval[valid_mask]

        mae  = float(np.mean(np.abs(actual_GWh_v - predicted_GWh_v)))
        rmse = float(np.sqrt(np.mean((actual_GWh_v - predicted_GWh_v) ** 2)))
        mape = float(np.mean(np.abs((actual_GWh_v - predicted_GWh_v) / actual_GWh_v)) * 100)

        sarima_rmse   = float(df_baseline.loc[left_out, "SARIMA_RMSE"])
        baseline_rmse = float(df_baseline.loc[left_out, "RMSE"])

        improvement_vs_sarima   = (sarima_rmse   - rmse) / sarima_rmse   * 100
        improvement_vs_baseline = (baseline_rmse - rmse) / baseline_rmse * 100

        actual_residual = actual_GWh_v - cf_GWh_v
        denom_actual = np.max(np.abs(actual_residual))
        denom_pulse  = np.max(np.abs(pulse_pct_v))

        if denom_actual > 0 and denom_pulse > 0:
            actual_res_norm = actual_residual / denom_actual
            pulse_norm      = pulse_pct_v      / denom_pulse
            shape_mse = float(np.mean((pulse_norm - actual_res_norm) ** 2))
        else:
            actual_res_norm = np.zeros_like(actual_residual)
            pulse_norm      = np.zeros_like(pulse_pct_v)
            shape_mse = np.nan

        print(f"    MAE={mae:.1f} GWh | RMSE={rmse:.1f} GWh | MAPE={mape:.2f}%")
        print(f"    vs Weighted Avg: {improvement_vs_baseline:+.2f}%")

        results.append({
            "donor": left_out,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "shape_MSE": shape_mse,
            "SARIMA_RMSE": sarima_rmse,
            "baseline_RMSE": baseline_rmse,
            "improvement_vs_sarima_pct": improvement_vs_sarima,
            "improvement_vs_baseline_pct": improvement_vs_baseline,
            "optimal_mu": mu_opt,
            "optimal_sigma": sigma_opt,
            "optimal_reg": reg_opt,
            "amplitude_a": a
        })

        # ── Plotting ──
        ax = axes_f[fold_idx]
        color = COLORS[fold_idx]
        ax.plot(month_labels_v, actual_GWh_v, "-k", lw=1.8, label="Actual")
        ax.plot(month_labels_v, predicted_GWh_v, "-r", lw=1.5, label="NP Parametric")
        ax.plot(month_labels_v, cf_GWh_v, "--b", lw=1.2, label="Counterfactual")
        ax.axvspan(-0.5, 1.5, color="tab:red", alpha=0.08, zorder=0)
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.set_xlim(-5, 6)
        ax.set_title(f"{left_out}\nRMSE={rmse:.0f} | MAPE={mape:.1f}%", fontsize=9)
        ax.set_xlabel("Months rel. to WC", fontsize=8)
        if fold_idx == 0:
            ax.set_ylabel("GWh", fontsize=8)
            ax.legend(fontsize=7)

        ax_s.plot(month_labels_v, actual_res_norm, "-", color=color, lw=1.5, label=f"{left_out} actual res")
        ax_s.plot(month_labels_v, pulse_norm, "--", color=color, lw=2.0, alpha=0.85, label=f"{left_out} Gaussian")

    fig_f.tight_layout(pad=1.5)
    fig_f.savefig(os.path.join(PROJECT_DIR, "loo_neuralprophet_parametric_forecast_plot.png"), dpi=150)
    plt.close(fig_f)

    ax_s.axvline(0, color="black", ls=":", lw=0.9)
    ax_s.set_xlim(-5, 6)
    ax_s.set_ylim(-1.2, 1.2)
    ax_s.set_xlabel("Months relative to WC start", fontsize=10)
    ax_s.set_ylabel("Normalized value", fontsize=10)
    ax_s.set_title("NeuralProphet Parametric LOO — Smooth Consensus vs Actual Residual", fontsize=10)
    ax_s.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    fig_s.tight_layout()
    fig_s.savefig(os.path.join(PROJECT_DIR, "loo_neuralprophet_parametric_pulse_shape.png"), dpi=150)
    plt.close(fig_s)

    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(PROJECT_DIR, "loo_neuralprophet_parametric_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("  GLOBAL SUMMARY (Parametric Gaussian Pulse)")
    print("=" * 60)
    n_beat_baseline = sum(1 for r in results if r["RMSE"] < r["baseline_RMSE"])
    mean_mape = df_out["MAPE"].mean()
    print(f"  Mean MAPE across 5 folds     : {mean_mape:.2f}%")
    print(f"  NP beats Weighted Average on : {n_beat_baseline}/5 donors")
    print("\nDone.")

if __name__ == "__main__":
    main()
