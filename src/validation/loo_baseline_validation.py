import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
from statsforecast.models import AutoARIMA, ARIMA

SARIMA_ORDERS = {
    "ZAF": {"order": (0, 1, 1), "seasonal_order": (1, 0, 1)},
    "CMR": {"order": (0, 1, 0), "seasonal_order": (1, 0, 2)},
    "RUS": {"order": (1, 0, 2), "seasonal_order": (1, 0, 1)},
    "QAT": {"order": (0, 1, 2), "seasonal_order": (1, 0, 1)},
    "EGY": {"order": (0, 0, 0), "seasonal_order": (1, 1, 0)},
}


np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\TS-project"
DATA_DIR = r"C:\Users\Hp\OneDrive\Desktop\Data\Data Time series project"

RESIDUALS_CSV = os.path.join(PROJECT_DIR, "donor_residuals_normalized.csv")

DONORS = ["QAT", "EGY", "RUS", "CMR", "ZAF"]

DONOR_WEIGHTS = {
    'QAT': 1.0,   
    'EGY': 0.7,   
    'CMR': 0.5,   
    'RUS': 0.3,   
    'ZAF': 0.6,   
}

DONOR_CONFIGS = {
    "ZAF": {"wc_start": pd.to_datetime("2010-06-01")},
    "CMR": {"wc_start": pd.to_datetime("2022-01-01")},
    "RUS": {"wc_start": pd.to_datetime("2018-06-01")},
    "QAT": {"wc_start": pd.to_datetime("2022-11-01")},
    "EGY": {"wc_start": pd.to_datetime("2019-06-01")},
}

def maybe_log(series: pd.Series):
    return np.log1p(series)

def load_donor_data(iso: str):
    if iso == "ZAF":
        path = os.path.join(DATA_DIR, "south_africa_donor_electricity_demand.csv")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], format="%Y-%B")
        df = df.set_index("date").sort_index()
        series = df["Consumption_GWh"].asfreq("MS").rename(iso)
    elif iso == "CMR":
        path = os.path.join(DATA_DIR, "cameroon_monthly_electricity _consumption.csv")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], format="%Y-%B")
        df = df.set_index("date").sort_index()
        series = df["Consumption_GWh"].asfreq("MS").rename(iso)
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
            iea_rus = iea[(iea["ISO 3 code"] == "RUS") & (iea["Variable"] == "Demand")].copy()
            iea_rus["date"] = pd.to_datetime(iea_rus["Date"])
            iea_rus = iea_rus.set_index("date").sort_index()
            series_iea = (iea_rus["Value"] * 1000).rename(iso)
        except:
            series_iea = pd.Series(dtype=float, name=iso)
            
        combined_idx = series_local.index.union(series_iea.index)
        series = pd.Series(index=combined_idx, dtype=float, name=iso)
        series.update(series_iea)
        series.update(series_local)
        series = series.asfreq("MS").rename(iso)
    elif iso == "QAT":
        path = os.path.join(DATA_DIR, "qatar_electricity_transmitted (1).csv")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], format="%Y-%B")
        df = df.set_index("date").sort_index()
        series_local = (df["Total_MWh"] / 1000).rename(iso)
        
        try:
            iea_path = os.path.join(DATA_DIR, "monthly_full_release_long_format.csv")
            iea = pd.read_csv(iea_path)
            iea_qat = iea[(iea["ISO 3 code"] == "QAT") & (iea["Variable"] == "Demand")].copy()
            iea_qat["date"] = pd.to_datetime(iea_qat["Date"])
            iea_qat = iea_qat.set_index("date").sort_index()
            series_iea = (iea_qat["Value"] * 1000).rename(iso)
        except:
            series_iea = pd.Series(dtype=float, name=iso)
            
        combined_idx = series_local.index.union(series_iea.index)
        series = pd.Series(index=combined_idx, dtype=float, name=iso)
        series.update(series_iea)
        series.update(series_local)
        series = series.asfreq("MS").rename(iso)
    elif iso == "EGY":
        iea_path = os.path.join(DATA_DIR, "monthly_full_release_long_format.csv")
        iea = pd.read_csv(iea_path)
        iea_egy = iea[(iea["ISO 3 code"] == "EGY") & (iea["Variable"] == "Demand")].copy()
        iea_egy["date"] = pd.to_datetime(iea_egy["Date"])
        iea_egy = iea_egy.set_index("date").sort_index()
        series = (iea_egy["Value"] * 1000).asfreq("MS").rename(iso)
    else:
        raise ValueError(f"Unknown donor {iso}")
        
    return series

def main():
    print("Loading normalized residual profiles...")
    df_res = pd.read_csv(RESIDUALS_CSV)
    df_res["unique_id"] = df_res["unique_id"].replace({"RSA": "ZAF"}) # Map RSA to ZAF
    
    results = []
    pulse_records = []
    
    fig_f, axes_f = plt.subplots(1, 5, figsize=(20, 4))
    fig_r, ax_r = plt.subplots(figsize=(10, 6))
    
    for i, left_out in enumerate(DONORS):
        print(f"\n{'='*50}\nFold {i+1}: Leaving out {left_out}\n{'='*50}")
        
        # ── Step 1: Compute weighted average pulse ──
        active_weights = {k: v for k, v in DONOR_WEIGHTS.items() if k != left_out}
        weight_sum = sum(active_weights.values())
        normalized_weights = {k: v / weight_sum for k, v in active_weights.items()}
        print(f"  Active weights: {normalized_weights}")
        
        # We need a consensus pulse for m in [-5, +6]
        # We'll compute it by doing a weighted sum per month
        consensus_pulse = []
        months_eval = np.arange(-5, 7)
        for m in months_eval:
            val_sum = 0.0
            for dnr, w in normalized_weights.items():
                # Get the value for donor dnr at month m
                row = df_res[(df_res["unique_id"] == dnr) & (df_res["months_to_wc"] == m)]
                if len(row) > 0:
                    val_sum += row["y"].values[0] * w
                else:
                    print(f"    WARNING: Missing residual for {dnr} at month {m}. Treating as 0.")
            consensus_pulse.append(val_sum)
        
        consensus_pulse = np.array(consensus_pulse)
        
        pulse_record = {"fold_left_out": left_out}
        for m, v in zip(months_eval, consensus_pulse):
            pulse_record[f"month_{m}"] = v
        pulse_records.append(pulse_record)
        
        # ── Step 2: SARIMA baseline ──
        print(f"  Step 2: SARIMA baseline for {left_out}...")
        series = load_donor_data(left_out)
        wc_start = DONOR_CONFIGS[left_out]["wc_start"]
        cutoff_date = wc_start - pd.DateOffset(months=6)
        train_sarima = series[series.index <= cutoff_date]
        
        train_log = maybe_log(train_sarima)
        orders = SARIMA_ORDERS[left_out]
        arima_model = ARIMA(
            order=orders["order"],
            seasonal_order=orders["seasonal_order"],
            season_length=12
        )
        arima_model.fit(train_log.values)

        
        fcst_sarima_log = arima_model.predict(h=12)["mean"]
        fcst_sarima_gwh = np.expm1(fcst_sarima_log)
        
        sarima_pred_dates = [cutoff_date + pd.DateOffset(months=m+6) for m in np.arange(-5, 7)]
        
        # Find t_peak in event window (0 to 6)
        # indices 5 to 11 correspond to months 0 to 6
        event_sarima_gwh = fcst_sarima_gwh[5:12]
        t_peak_val = np.max(event_sarima_gwh)
        
        # ── Step 3: Inject weighted average pulse ──
        print("  Step 3: Injecting weighted average pulse...")
        wc_lift_gwh = consensus_pulse * t_peak_val
        
        combined_fcst_gwh = fcst_sarima_gwh + wc_lift_gwh
        
        # ── Step 4: Evaluation ──
        print("  Step 4: Evaluation...")
        actual_gwh = series[series.index.isin(sarima_pred_dates)].values
        min_len = min(len(actual_gwh), len(combined_fcst_gwh))
        actual_gwh = actual_gwh[:min_len]
        combined_fcst_gwh = combined_fcst_gwh[:min_len]
        fcst_sarima_gwh = fcst_sarima_gwh[:min_len]
        consensus_pulse_trunc = consensus_pulse[:min_len]
        
        mae = np.mean(np.abs(actual_gwh - combined_fcst_gwh))
        rmse = np.sqrt(np.mean((actual_gwh - combined_fcst_gwh)**2))
        mape = np.mean(np.abs((actual_gwh - combined_fcst_gwh) / actual_gwh)) * 100
        
        rmse_sarima = np.sqrt(np.mean((actual_gwh - fcst_sarima_gwh)**2))
        
        actual_normalized = (actual_gwh - fcst_sarima_gwh) / t_peak_val
        shape_mse = np.mean((consensus_pulse_trunc - actual_normalized)**2)
        
        improvement_pct = ((rmse_sarima - rmse) / rmse_sarima) * 100
        
        results.append({
            "donor": left_out,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "shape_MSE": shape_mse,
            "SARIMA_RMSE": rmse_sarima,
            "improvement_pct": improvement_pct
        })
        
        # Plots
        ax = axes_f[i]
        ax.plot(months_eval[:min_len], fcst_sarima_gwh, "--", color="steelblue", label="SARIMA baseline")
        ax.plot(months_eval[:min_len], combined_fcst_gwh, "-", color="red", label="Combined Forecast")
        ax.plot(months_eval[:min_len], actual_gwh, "-", color="black", label="Actual")
        ax.axvspan(0, 1, color="red", alpha=0.1)
        ax.set_title(f"{left_out}\nRMSE={rmse:.0f} | MAPE={mape:.1f}%")
        ax.set_xlim([-5, 6])
        if i == 0:
            ax.legend(fontsize=8)
            
        color = plt.cm.tab10(i)
        ax_r.plot(months_eval[:min_len], actual_normalized, "-", color=color, label=f"{left_out} actual")
        ax_r.plot(months_eval[:min_len], consensus_pulse_trunc, "--", color=color, alpha=0.8, label=f"{left_out} weighted avg")

    fig_f.tight_layout()
    fig_f.savefig(os.path.join(PROJECT_DIR, "loo_baseline_forecast_plot.png"), dpi=150)
    plt.close(fig_f)
    
    ax_r.axvline(0, color="black", ls=":")
    ax_r.set_xlim([-5, 6])
    ax_r.set_ylabel("Normalized Value")
    ax_r.set_xlabel("Months relative to WC start")
    ax_r.legend(loc="upper left", bbox_to_anchor=(1,1))
    fig_r.tight_layout()
    fig_r.savefig(os.path.join(PROJECT_DIR, "loo_baseline_residual_overlay.png"), dpi=150)
    plt.close(fig_r)

    # Summaries
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(PROJECT_DIR, "loo_baseline_summary.csv"), index=False)
    
    df_pulse = pd.DataFrame(pulse_records)
    df_pulse.to_csv(os.path.join(PROJECT_DIR, "weighted_avg_pulse_per_fold.csv"), index=False)
    
    # Interpretation Block
    print("\n" + "="*50)
    print("Interpretation Block")
    print("="*50)
    print("Per donor:")
    for row in results:
        donor = row["donor"]
        rmse_comb = row["RMSE"]
        rmse_sar = row["SARIMA_RMSE"]
        imp_gwh = rmse_sar - rmse_comb
        imp_pct = row["improvement_pct"]
        improved = imp_gwh > 0
        
        print(f"[{donor}]")
        print(f"  1. Did weighted average improve over SARIMA-only? {'Yes' if improved else 'No'}")
        print(f"  2. GWh improvement: {imp_gwh:.1f}")
        print(f"  3. Percentage improvement: {imp_pct:.2f}%\n")
        
    print("="*50)
    print("Global Morocco Deployment Pulse (Full 5-donor weighted average):")
    weight_sum = sum(DONOR_WEIGHTS.values())
    global_weights = {k: v / weight_sum for k, v in DONOR_WEIGHTS.items()}
    global_pulse = []
    for m in months_eval:
        val_sum = 0.0
        for dnr, w in global_weights.items():
            row = df_res_full[(df_res_full["unique_id"] == dnr) & (df_res_full["months_to_wc"] == m)]
            if len(row) > 0:
                val_sum += row["y"].values[0] * w
        global_pulse.append(val_sum)
        
    for m, v in zip(months_eval, global_pulse):
        print(f"  Month {m:2d}: {v:.4f}")
        
    # We need df_res_full globally loaded for the last block:
    # Actually just reuse df_res from the top!
        
if __name__ == "__main__":
    df_res_full = pd.read_csv(RESIDUALS_CSV)
    df_res_full["unique_id"] = df_res_full["unique_id"].replace({"RSA": "ZAF"})
    main()
