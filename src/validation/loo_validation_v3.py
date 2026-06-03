import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from neuralforecast.models import NBEATSx
from neuralforecast import NeuralForecast
from statsforecast.models import AutoARIMA, ARIMA

SARIMA_ORDERS = {
    "ZAF": {"order": (0, 1, 1), "seasonal_order": (1, 0, 1)},
    "CMR": {"order": (0, 1, 0), "seasonal_order": (1, 0, 2)},
    "RUS": {"order": (1, 0, 2), "seasonal_order": (1, 0, 1)},
    "QAT": {"order": (0, 1, 2), "seasonal_order": (1, 0, 1)},
    "EGY": {"order": (0, 0, 0), "seasonal_order": (1, 1, 0)},
}


np.random.seed(42)
torch.manual_seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\TS-project"
DATA_DIR = r"C:\Users\Hp\OneDrive\Desktop\Data\Data Time series project"

RESIDUALS_AUG_CSV = os.path.join(PROJECT_DIR, "donor_residuals_augmented.csv")
CKPT_P1           = os.path.join(PROJECT_DIR, "pretrained_nbeatsx.ckpt")

DONORS = ["QAT", "EGY", "RUS", "CMR", "ZAF"]

# ── Architecture ─────────────────────────────────────────────────────────────
ARCH = dict(
    h=7,
    input_size=13, # Must match original pre-training
    stack_types=["trend", "seasonality", "identity"],
    n_harmonics=1,
    n_polynomials=2,
    dropout_prob_theta=0.5,
    futr_exog_list=["months_to_wc"],
)

PRE_EVENT_Y = 0.053
MIN_LEN = 20  # input_size (13) + h (7)

def pad_to_min_len(df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    parts = []
    for uid, grp in df.groupby("unique_id", sort=False):
        grp   = grp.sort_values("ds").reset_index(drop=True)
        n_pad = max(0, min_len - len(grp))
        if n_pad:
            first_ds  = grp["ds"].iloc[0]
            first_mtw = int(grp["months_to_wc"].iloc[0]) if "months_to_wc" in grp.columns else 0
            pad_ds    = [first_ds - pd.DateOffset(months=k) for k in range(n_pad, 0, -1)]
            pad_mtw   = [first_mtw - k                      for k in range(n_pad, 0, -1)]
            pad_df    = pd.DataFrame({"unique_id": uid, "ds": pad_ds, "y": PRE_EVENT_Y})
            if "months_to_wc" in grp.columns:
                pad_df["months_to_wc"] = pad_mtw
            for col in grp.columns:
                if col not in pad_df.columns:
                    pad_df[col] = 0.0
            grp = pd.concat([pad_df, grp], ignore_index=True)
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)

def maybe_log(series: pd.Series):
    return np.log1p(series)

def maybe_exp(series: pd.Series):
    return np.expm1(series)

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

DONOR_CONFIGS = {
    "ZAF": {"wc_start": pd.to_datetime("2010-06-01")},
    "CMR": {"wc_start": pd.to_datetime("2022-01-01")},
    "RUS": {"wc_start": pd.to_datetime("2018-06-01")},
    "QAT": {"wc_start": pd.to_datetime("2022-11-01")},
    "EGY": {"wc_start": pd.to_datetime("2019-06-01")},
}

def load_p1_weights(model: NBEATSx, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict", ckpt)
    net_state = {k.replace("model.", ""): v for k, v in state_dict.items() if "model." in k}
    missing, unexpected = model.load_state_dict(net_state, strict=False)
    print(f"    P1 weights loaded | missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")

def apply_freezing_v3(model: NBEATSx):
    num_blocks = len(model.blocks)
    for i, block in enumerate(model.blocks):
        if i < (num_blocks - 1):
            for param in block.parameters():
                param.requires_grad = False
        else:
            for param in block.parameters():
                param.requires_grad = True
    
    print("    Parameter requires_grad status:")
    for name, param in model.named_parameters():
        if "blocks" in name:
            print(f"      {name}: {param.requires_grad}")

def main():
    print("Loading augmented residual profiles...")
    df_aug = pd.read_csv(RESIDUALS_AUG_CSV)
    df_aug["ds"] = pd.to_datetime(df_aug["ds"])
    
    results = []
    
    fig_f, axes_f = plt.subplots(1, 5, figsize=(20, 4))
    
    fig_r, ax_r = plt.subplots(figsize=(10, 6))
    
    for i, left_out in enumerate(DONORS):
        print(f"\n{'='*50}\nFold {i+1}: Leaving out {left_out}\n{'='*50}")
        
        # ── Step 1: Prepare augmented training set ──
        df_train = df_aug[~df_aug["unique_id"].str.startswith(left_out)].copy()
        
        # Load forecast_residuals.csv to compute synthetic context
        df_raw_res = pd.read_csv(os.path.join(PROJECT_DIR, "forecast_residuals.csv"))
        df_raw_res["country"] = df_raw_res["country"].replace({"ZAF": "RSA"})
        
        # Compute ratios for each country
        ratios = {}
        for country in ["RSA", "CMR", "RUS", "QAT", "EGY"]:
            sub = df_raw_res[(df_raw_res["country"] == country) & (df_raw_res["months_to_wc"] >= -5) & (df_raw_res["months_to_wc"] <= -1)]
            sub = sub.sort_values("months_to_wc")
            val_minus_1 = sub[sub["months_to_wc"] == -1]["sarima_forecast"].values[0]
            sub["ratio_y"] = (sub["sarima_forecast"] / val_minus_1) * 0.053
            ratios[country] = dict(zip(sub["months_to_wc"], sub["ratio_y"]))
            
        # Replace y in df_train for months_to_wc in [-5, -1] with synthetic context
        df_train["base_donor"] = df_train["unique_id"].apply(lambda x: x.split("_")[0])
        for country in ["RSA", "CMR", "RUS", "QAT", "EGY"]:
            if country in ratios:
                for mtw, val in ratios[country].items():
                    mask = (df_train["base_donor"] == country) & (df_train["months_to_wc"] == mtw)
                    df_train.loc[mask, "y"] = val
        df_train = df_train.drop(columns=["base_donor"])
        
        n_profiles = df_train["unique_id"].nunique()
        print(f"  Step 1: Extracted {n_profiles} profiles for training (left-out donor excluded)")
        
        # ── Step 2: Fine-tune N-BEATSx ──
        print("  Step 2: Fine-tuning N-BEATSx...")
        arch_fold = ARCH.copy()
        arch_fold["early_stop_patience_steps"] = -1 # disabled to allow val_size=0
        
        model_fold = NBEATSx(
            **arch_fold,
            max_steps=100,
            learning_rate=1e-4,
            batch_size=8,
            random_seed=42,
            start_padding_enabled=True,   # allows training on series shorter than input_size
        )

        
        load_p1_weights(model_fold, CKPT_P1)
        apply_freezing_v3(model_fold)
        
        df_train_padded = pad_to_min_len(df_train, MIN_LEN)
        
        fcst = NeuralForecast(models=[model_fold], freq="MS")
        # Train
        fcst.fit(df=df_train_padded, val_size=0)
        
        ckpt_path = os.path.join(PROJECT_DIR, f"finetuned_fold_{left_out}.ckpt")
        torch.save(model_fold.state_dict(), ckpt_path)
        print(f"    Saved fold weights to {ckpt_path}")
        
        # ── Step 3: SARIMA Baseline ──
        print(f"  Step 3: SARIMA baseline for {left_out}...")
        series = load_donor_data(left_out)
        wc_start = DONOR_CONFIGS[left_out]["wc_start"]
        # Train up to month -6
        cutoff_date = wc_start - pd.DateOffset(months=6)
        train_sarima = series[series.index <= cutoff_date]
        
        # Fit auto_arima
        train_log = maybe_log(train_sarima)
        orders = SARIMA_ORDERS[left_out]
        arima_model = ARIMA(
            order=orders["order"],
            seasonal_order=orders["seasonal_order"],
            season_length=12
        )
        arima_model.fit(train_log.values)

        
        # Forecast 12 steps (months -5 to +6)
        fcst_sarima_log = arima_model.predict(h=12)["mean"]
        
        fcst_sarima_gwh = np.expm1(fcst_sarima_log)
        
        # Months array for these 12 predictions:
        sarima_pred_months = np.arange(-5, 7)
        sarima_pred_dates = [cutoff_date + pd.DateOffset(months=m+6) for m in sarima_pred_months]
        
        # We need the context values for NBEATSx input: months -13 to -1.
        # Months -13 to -6 are the last 8 training points.
        # Months -5 to -1 are the first 5 forecasted points.
        sarima_hist_gwh = train_sarima.values[-8:]
        sarima_context_gwh = np.concatenate([sarima_hist_gwh, fcst_sarima_gwh[0:5]])
        
        # Find t_peak in event window (0 to 6)
        # 0 to 6 correspond to indices 5 to 11 in fcst_sarima_gwh
        event_sarima_gwh = fcst_sarima_gwh[5:12]
        t_peak_val = np.max(event_sarima_gwh)
        
        # Normalize input context using the approved detrending + scaling mitigation
        # input_context(t) = [sarima_baseline(t) / sarima_baseline(t=-1)] * 0.053
        sarima_minus_1 = sarima_context_gwh[-1]
        input_context_norm = (sarima_context_gwh / sarima_minus_1) * 0.053
        
        # ── Step 4: N-BEATSx Pulse Injection ──
        print("  Step 4: N-BEATSx pulse injection...")
        context_ds = [cutoff_date + pd.DateOffset(months=m+6) for m in np.arange(-13, 0)]
        context_months = np.arange(-13, 0)
        
        hist_df = pd.DataFrame({
            "unique_id": f"{left_out}_inference",
            "ds": context_ds,
            "y": input_context_norm,
            "months_to_wc": context_months
        })
        
        futr_ds = [wc_start + pd.DateOffset(months=m) for m in range(0, 7)]
        futr_df = pd.DataFrame({
            "unique_id": f"{left_out}_inference",
            "ds": futr_ds,
            "months_to_wc": np.arange(0, 7)
        })
        
        nbeats_preds = fcst.predict(df=hist_df, futr_df=futr_df)
        nbeats_pulse_norm = nbeats_preds["NBEATSx"].values # Length 7 (months 0 to 6)
        
        # Calculate wc_lift_GWh(t)
        wc_lift_gwh = nbeats_pulse_norm * t_peak_val
        
        # Full combined forecast:
        # Months -5 to -1: sarima baseline + 0
        # Months 0 to 6: sarima baseline + wc_lift
        combined_fcst_gwh = np.copy(fcst_sarima_gwh)
        combined_fcst_gwh[5:12] += wc_lift_gwh
        
        # ── Step 5: Evaluation ──
        print("  Step 5: Evaluation...")
        actual_gwh = series[series.index.isin(sarima_pred_dates)].values
        
        # Truncate if actual data doesn't span the whole window
        min_len = min(len(actual_gwh), len(combined_fcst_gwh))
        actual_gwh = actual_gwh[:min_len]
        combined_fcst_gwh = combined_fcst_gwh[:min_len]
        fcst_sarima_gwh = fcst_sarima_gwh[:min_len]
        
        mae = np.mean(np.abs(actual_gwh - combined_fcst_gwh))
        rmse = np.sqrt(np.mean((actual_gwh - combined_fcst_gwh)**2))
        mape = np.mean(np.abs((actual_gwh - combined_fcst_gwh) / actual_gwh)) * 100
        
        # RMSE of SARIMA
        rmse_sarima = np.sqrt(np.mean((actual_gwh - fcst_sarima_gwh)**2))
        
        # Shape error for months 0 to 6 (where pulse exists)
        if min_len >= 12:
            actual_event_gwh = actual_gwh[5:12]
            actual_pulse_gwh = actual_event_gwh - event_sarima_gwh
            actual_pulse_norm = actual_pulse_gwh / t_peak_val
            shape_mse = np.mean((nbeats_pulse_norm - actual_pulse_norm)**2)
        else:
            shape_mse = np.nan
            
        results.append({
            "Left-out donor": left_out,
            "MAE (GWh)": mae,
            "RMSE (GWh)": rmse,
            "MAPE (%)": mape,
            "Shape MSE": shape_mse,
            "RMSE_SARIMA": rmse_sarima
        })
        print(f"    MAE={mae:.1f} | RMSE={rmse:.1f} | MAPE={mape:.2f}% | Shape MSE={shape_mse:.4f}")
        
        # Plot 1: Subplot for donor
        ax = axes_f[i]
        ax.plot(sarima_pred_months[:min_len], fcst_sarima_gwh, "--", color="steelblue", label="SARIMA baseline")
        ax.plot(sarima_pred_months[:min_len], combined_fcst_gwh, "-", color="red", label="Combined Forecast")
        ax.plot(sarima_pred_months[:min_len], actual_gwh, "-", color="black", label="Actual")
        ax.axvspan(0, 1, color="red", alpha=0.1)
        ax.set_title(f"{left_out} | RMSE={rmse:.0f} | MAPE={mape:.1f}%")
        ax.set_xlim([-5, 6])
        if i == 0:
            ax.legend(fontsize=8)
            
        # Plot 2: Overlay
        if min_len >= 12:
            cfg = next(v for k, v in DONOR_CONFIGS.items() if k == left_out)
            color = plt.cm.tab10(i)
            # Actual normalized over -5 to +6
            actual_res_norm = (actual_gwh - fcst_sarima_gwh) / t_peak_val
            ax_r.plot(sarima_pred_months, actual_res_norm, "-", color=color, label=f"{left_out} actual")
            
            # Predicted normalized pulse
            pred_pulse_norm_full = np.zeros(12)
            pred_pulse_norm_full[5:12] = nbeats_pulse_norm
            ax_r.plot(sarima_pred_months, pred_pulse_norm_full, "--", color=color, alpha=0.8, label=f"{left_out} predicted")

    fig_f.tight_layout()
    fig_f.savefig(os.path.join(PROJECT_DIR, "loo_forecast_plot.png"), dpi=150)
    plt.close(fig_f)
    
    ax_r.axvline(0, color="black", ls=":")
    ax_r.set_xlim([-5, 6])
    ax_r.set_ylabel("Normalized Value")
    ax_r.set_xlabel("Months relative to WC start")
    ax_r.legend(loc="upper left", bbox_to_anchor=(1,1))
    fig_r.tight_layout()
    fig_r.savefig(os.path.join(PROJECT_DIR, "loo_residual_overlay.png"), dpi=150)
    plt.close(fig_r)

    # Summary Table
    df_res = pd.DataFrame(results)
    
    mean_row = pd.DataFrame({
        "Left-out donor": ["Mean"],
        "MAE (GWh)": [df_res["MAE (GWh)"].mean()],
        "RMSE (GWh)": [df_res["RMSE (GWh)"].mean()],
        "MAPE (%)": [df_res["MAPE (%)"].mean()],
        "Shape MSE": [df_res["Shape MSE"].mean()]
    })
    
    df_save = pd.concat([df_res.drop(columns=["RMSE_SARIMA"]), mean_row], ignore_index=True)
    df_save.to_csv(os.path.join(PROJECT_DIR, "loo_summary.csv"), index=False)
    print("\nSummary Table:")
    print(df_save.to_string(index=False))
    
    # Interpretation
    print("\n" + "="*50)
    print("Interpretation Block")
    print("="*50)
    n_degraded = 0
    shape_rankings = []
    
    for row in results:
        donor = row["Left-out donor"]
        rmse_comb = row["RMSE (GWh)"]
        rmse_sar = row["RMSE_SARIMA"]
        shape = row["Shape MSE"]
        
        improved = rmse_comb < rmse_sar
        diff = rmse_sar - rmse_comb
        pct_imp = diff / rmse_sar * 100
        
        if not improved:
            n_degraded += 1
            
        shape_rankings.append((shape, donor))
        
        print(f"[{donor}] N-BEATSx improved over SARIMA? {'Yes' if improved else 'No'}")
        print(f"      GWh improvement: {diff:.1f}")
        print(f"      Percentage improvement: {pct_imp:.2f}%")
        
    shape_rankings.sort()
    print("\nShape MSE ranking (lowest is best):")
    for i, (val, donor) in enumerate(shape_rankings):
        print(f"  {i+1}. {donor} (MSE: {val:.4f})")
        
    if n_degraded > 2:
        print("\nWARNING: N-BEATSx correction is net-negative on majority of donors")
        print("Review residual normalization or revert to SARIMA-only baseline")
        
if __name__ == "__main__":
    main()
