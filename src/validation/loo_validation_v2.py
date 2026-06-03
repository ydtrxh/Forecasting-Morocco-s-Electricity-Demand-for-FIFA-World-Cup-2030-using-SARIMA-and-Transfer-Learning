"""
loo_validation_v2.py
WC 2030 Morocco Electricity Forecasting

Full LOO validation: SARIMA baseline + Multiplicative N-BEATSx lift
per held-out donor country.

Mirrors deployment methodology:
  1. Fit SARIMA on donor (cutoff at month -7).
  2. Fine-tune N-BEATSx on 4 remaining donors (augmented).
  3. Predict normalized pulse (-6 to +6).
  4. Compute multiplicative lift: predicted(t) = baseline(t) + pulse(t) * baseline(t).

Author: Younes, ENSAM Meknes (IATD)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsforecast.models import AutoARIMA

# Helpers from your local modules
from sarima_residual_extraction import DONORS, load_donor_data, maybe_log, maybe_exp
from augment_residuals import augment_donor, AUG_CONFIG
from nbeatsx_training import load_p1_weights, pad_to_min_len, make_futr_df, ARCH, PRE_EVENT_Y

np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\TS-project"
CKPT_P1     = os.path.join(PROJECT_DIR, "pretrained_nbeatsx.ckpt")

H = 13
INPUT_SIZE = 13
MIN_LEN = INPUT_SIZE + 2 * H

# ══════════════════════════════════════════════════════════════════════════════
#  SARIMA (Phase A)
# ══════════════════════════════════════════════════════════════════════════════

def build_covid_dummy(index: pd.DatetimeIndex) -> pd.Series:
    dummy = pd.Series(0.0, index=index, name="covid")
    for m, val in [("2020-04-01", 1.0), ("2020-05-01", 1.0),
                   ("2020-06-01", 1.0), ("2020-07-01", 0.5)]:
        ts = pd.Timestamp(m)
        if ts in dummy.index:
            dummy.loc[ts] = val
    return dummy

def fit_sarima_and_forecast(train_t: pd.Series, exog_train, iso: str, h: int, exog_fut):
    model = AutoARIMA(season_length=12, d=1, D=1, max_p=3, max_q=3, max_P=2, max_Q=2, trace=False)
    y_train = train_t.values
    X_train = exog_train.values if exog_train is not None else None
    X_fut   = exog_fut.values if exog_fut is not None else None
    
    print(f"    [{iso}] Running StatsForecast AutoARIMA...")
    res = model.forecast(y=y_train, X=X_train, h=h, X_future=X_fut)
    return res['mean']

def phase_a_sarima():
    print("\n" + "="*60)
    print("  PHASE A: SARIMA fits (all folds)")
    print("="*60)

    fold_data = {}
    for country, cfg in DONORS.items():
        iso = cfg["iso"]
        wc_start = cfg["wc_start"]
        
        # Override to strictly mirror deployment: Cutoff at -7, forecast -6 to +6
        cutoff = wc_start - pd.DateOffset(months=7)
        win_s  = wc_start - pd.DateOffset(months=6)
        win_e  = wc_start + pd.DateOffset(months=6)

        series = load_donor_data(country, cfg).asfreq("MS")
        train  = series[series.index <= cutoff]

        covid_dummy = build_covid_dummy(series.index)
        use_exog    = bool(covid_dummy.sum() > 0)
        exog_train  = covid_dummy[train.index].to_frame("covid") if use_exog else None

        train_t = maybe_log(train, cfg)

        # Forecast window
        n_fc   = H # exactly 13
        fc_idx = pd.date_range(cutoff + pd.DateOffset(months=1), periods=n_fc, freq="MS")
        exog_f = covid_dummy.reindex(fc_idx).to_frame("covid") if use_exog else None

        fc_vals = fit_sarima_and_forecast(train_t, exog_train, iso, h=n_fc, exog_fut=exog_f)
        fc_log  = pd.Series(fc_vals, index=fc_idx)
        fc_gwh  = maybe_exp(fc_log, cfg)

        actual_win = series.reindex(fc_idx)
        
        months_to_wc = np.arange(-6, 7) # -6 to +6

        fold_data[iso] = dict(
            actual_win   = actual_win,
            fc_win       = fc_gwh,
            months_to_wc = months_to_wc,
            ds           = list(fc_idx),
            cutoff       = cutoff
        )
    return fold_data


# ══════════════════════════════════════════════════════════════════════════════
#  N-BEATSx (Phase B & C)
# ══════════════════════════════════════════════════════════════════════════════

def apply_freezing_v2(model) -> None:
    """Freeze all NBEATS blocks except the LAST TWO."""
    blocks = model.blocks
    n = len(blocks)
    for i, block in enumerate(blocks):
        req = (i >= n - 2) # Last 2 blocks trainable
        for p in block.parameters():
            p.requires_grad = req

def loo_validation(sarima_data):
    print("\n" + "="*60)
    print("  PHASE B & C: N-BEATSx Fine-Tuning & Multiplicative Inference")
    print("="*60)

    import torch
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    import pytorch_lightning as pl
    pl.seed_everything(42)

    # ── Load pre-extracted residuals to build the augmented donor pool ──
    raw_df = pd.read_csv(os.path.join(PROJECT_DIR, "forecast_residuals.csv"))
    raw_df["ds"] = pd.to_datetime(raw_df["date"])
    raw_df["unique_id"] = raw_df["country"].replace({"ZAF": "RSA"})
    raw_df["y"] = raw_df["residual_norm"]
    
    # We only augment on the 13-month window for training
    raw_df = raw_df[(raw_df["months_to_wc"] >= -6) & (raw_df["months_to_wc"] <= 6)]

    loo_results = []
    loo_preds = {}

    for donor, s_data in sarima_data.items():
        iso_nn = "RSA" if donor == "ZAF" else donor
        print(f"\n--- Fold {donor} ({iso_nn}) ---")

        # ── Step 1: Augment the 4 remaining donors ──
        aug_list = []
        for d_id in AUG_CONFIG.keys():
            if d_id == iso_nn: continue
            d_df = raw_df[raw_df["unique_id"] == d_id].copy()
            # Generate 20 augmentations using original params but enforcing count
            # Our AUG_CONFIG has 10 scales * 3 shifts * 4 noises = 120.
            # To get 80 profiles total across 4 donors, we need 20 per donor.
            # We'll just generate the default and sample 20.
            dfs, _ = augment_donor(d_df, AUG_CONFIG[d_id], d_id)
            np.random.shuffle(dfs)
            aug_list.extend(dfs[:20]) # Take exactly 20 per donor
        
        df_train = pd.concat(aug_list, ignore_index=True)
        print(f"    Train: {df_train['unique_id'].nunique()} profiles (4 donors x 20)")
        df_train_padded = pad_to_min_len(df_train, MIN_LEN)

        # ── Step 2: Fine-tune N-BEATSx ──
        # val_size=0: include the entire WC pulse in training (not held out).
        # max_steps=200: give the last 2 blocks enough updates to adapt.
        arch_fold = ARCH.copy()
        arch_fold["early_stop_patience_steps"] = -1  # disable early stop
        model_fold = NBEATSx(
            **arch_fold,
            futr_exog_list = ["months_to_wc"],
            max_steps      = 200,
            learning_rate  = 1e-4,
            batch_size     = 8,
            random_seed    = 42,
        )
        load_p1_weights(model_fold, CKPT_P1)
        apply_freezing_v2(model_fold)

        fcst = NeuralForecast(models=[model_fold], freq="MS")
        fcst.fit(df=df_train_padded, val_size=0)  # all data in training

        # Save model
        ckpt_path = os.path.join(PROJECT_DIR, f"nbeats_finetuned_fold_{donor}2.ckpt")
        torch.save(fcst.models[0].state_dict(), ckpt_path)

        # ── Step 3: Multiplicative Inference ──
        # History = 13 months BEFORE the event (-19 to -7 relative to WC start).
        # Use PRE_EVENT_Y (not 0.0) so TemporalNorm scale ≈ 0.053, matching
        # the padded training rows which also use PRE_EVENT_Y.
        hist_months = np.arange(-19, -6)
        cutoff_date = s_data["cutoff"]
        pad_ds = [cutoff_date - pd.DateOffset(months=k) for k in range(INPUT_SIZE-1, -1, -1)]
        
        hist_df = pd.DataFrame({
            "unique_id": iso_nn, 
            "ds": pad_ds, 
            "y": PRE_EVENT_Y,          # non-zero → TemporalNorm scale = PRE_EVENT_Y
            "months_to_wc": hist_months
        })

        futr_df = make_futr_df(hist_df, H)

        try:
            preds = fcst.predict(df=hist_df, futr_df=futr_df)
            pred_col = [c for c in preds.columns if c not in ("unique_id", "ds")][0]
            pred_norm = preds[pred_col].values

            # MULTIPLICATIVE LIFT:
            sarima_gwh = s_data["fc_win"].values
            wc_lift_gwh = pred_norm * sarima_gwh
            final_pred_gwh = sarima_gwh + wc_lift_gwh
            actual_gwh = s_data["actual_win"].values

            # Shape error evaluation (on normalized space)
            # Calculate actual normalized residual for this 13-month window
            raw_resid_13 = actual_gwh - sarima_gwh
            actual_norm = raw_resid_13 / sarima_gwh

            shape_error = np.nanmean((pred_norm - actual_norm)**2)

            mae = np.nanmean(np.abs(final_pred_gwh - actual_gwh))
            rmse = np.sqrt(np.nanmean((final_pred_gwh - actual_gwh)**2))
            mape = np.nanmean(np.abs((final_pred_gwh - actual_gwh) / actual_gwh)) * 100

            loo_preds[donor] = {
                "months_to_wc": s_data["months_to_wc"],
                "actual_gwh": actual_gwh,
                "sarima_gwh": sarima_gwh,
                "pred_gwh": final_pred_gwh,
                "actual_norm": actual_norm,
                "pred_norm": pred_norm
            }

            print(f"    MAE={mae:.1f} | RMSE={rmse:.1f} | MAPE={mape:.2f}% | ShapeErr={shape_error:.3f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            mae, rmse, mape, shape_error = np.nan, np.nan, np.nan, np.nan
            loo_preds[donor] = None

        loo_results.append({
            "Left-out donor": donor,
            "MAE (GWh)": round(mae, 1),
            "RMSE (GWh)": round(rmse, 1),
            "MAPE (%)": round(mape, 2),
            "Shape error": round(shape_error, 4)
        })

    df_res = pd.DataFrame(loo_results)
    
    # Add Mean row
    mean_row = df_res.mean(numeric_only=True)
    mean_row["Left-out donor"] = "Mean"
    df_res = pd.concat([df_res, pd.DataFrame([mean_row])], ignore_index=True)
    
    df_res.to_csv(os.path.join(PROJECT_DIR, "loo_validation_metrics2.csv"), index=False)
    
    # Render table plot
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    table = ax.table(cellText=df_res.values, colLabels=df_res.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.savefig(os.path.join(PROJECT_DIR, "loo_summary_table2.png"), bbox_inches="tight", dpi=150)
    plt.close()

    return df_res, loo_preds


def plot_overlay(loo_preds):
    print("\n  Generating overlay plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors
    colors = {"QAT": "#9b59b6", "CMR": "#f1c40f", "ZAF": "#2ecc71", "EGY": "#e76f51", "RUS": "#3498db"}
    
    for donor, preds in loo_preds.items():
        if preds is None: continue
        mtw = preds["months_to_wc"]
        act = preds["actual_norm"]
        prd = preds["pred_norm"]
        c = colors.get(donor, "black")
        
        ax.plot(mtw, act, color=c, lw=2, alpha=0.7, label=f"{donor} (Actual)")
        ax.plot(mtw, prd, color=c, lw=2, ls="--", label=f"{donor} (Pred)")

    ax.axvspan(0, 1, color="red", alpha=0.1, label="Event Plateau (0 to +1)")
    ax.axhline(0, color="gray", lw=1)
    
    ax.set_title("N-BEATSx Normalized Pulse: Predicted vs Actual", fontweight="bold")
    ax.set_xlabel("Months to WC")
    ax.set_ylabel("Normalized Value")
    ax.set_ylim(-1, 1.2)
    
    # Legend deduplication
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.04, 1), loc="upper left")
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "loo_predicted_vs_actual2.png"), dpi=200, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    sarima_data = phase_a_sarima()
    df_res, loo_preds = loo_validation(sarima_data)
    plot_overlay(loo_preds)
    print("\n  Pipeline V2 Complete!")
