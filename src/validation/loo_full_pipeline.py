"""
loo_full_pipeline.py
WC 2030 Morocco Electricity Forecasting

Full LOO validation: SARIMA baseline + N-BEATSx residual correction
per held-out donor country.

Steps per fold:
  1. Fit SARIMA on held-out donor (with COVID pulse dummy)
  2. Re-fine-tune N-BEATSx on remaining augmented donors
  3. Predict normalized residual on held-out donor
  4. Reconstruct full forecast and compute metrics

Normalization note:
  The normalizer uses the event-window peak (months 0/+1) by design —
  identical to the augmented training set. Target is the pulse *shape*,
  not the absolute level. The scaler is a single scalar, not a model
  fitted on labels, so leakage risk is negligible and the choice ensures
  consistent scale between training and validation.

COVID dummy:
  Pulse specification: Apr 2020 -> 1.0, May 2020 -> 1.0,
  Jun 2020 -> 1.0, Jul 2020 -> 0.5, all other months -> 0.0.
  A flat annual dummy over-corrects Q1/Q4 and under-corrects Q2/Q3.

IMPORTANT — import order:
  torch / neuralforecast are imported LAZILY inside the N-BEATSx functions.
  On Windows, PyTorch ships its own MKL which conflicts with statsmodels'
  LAPACK routines if both are loaded in the same process before SARIMAX.fit()
  is called. By deferring the torch import until after all SARIMA folds are
  complete we avoid the C-level segfault.

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

# ── statsforecast only (no torch yet) ──────────────────────────────────────────
from statsforecast.models import AutoARIMA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── sarima_residual_extraction helpers (imports pmdarima + statsmodels only) ─
from sarima_residual_extraction import DONORS, load_donor_data, maybe_log, maybe_exp

np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\TS-project"
CKPT_P1     = os.path.join(PROJECT_DIR, "pretrained_nbeatsx.ckpt")
AUG_PATH    = os.path.join(PROJECT_DIR, "donor_residuals_augmented.csv")


# ══════════════════════════════════════════════════════════════════════════════
#  SARIMA helpers  (no torch dependency)
# ══════════════════════════════════════════════════════════════════════════════

def build_covid_dummy(index: pd.DatetimeIndex) -> pd.Series:
    """
    Pulse COVID dummy:
      Apr / May / Jun 2020 -> 1.0  (peak suppression)
      Jul 2020             -> 0.5  (partial recovery)
      all other months     -> 0.0
    """
    dummy = pd.Series(0.0, index=index, name="covid")
    for m, val in [("2020-04-01", 1.0), ("2020-05-01", 1.0),
                   ("2020-06-01", 1.0), ("2020-07-01", 0.5)]:
        ts = pd.Timestamp(m)
        if ts in dummy.index:
            dummy.loc[ts] = val
    return dummy


def fit_sarima_and_forecast(train_t: pd.Series, exog_train, iso: str, h: int, exog_fut):
    """
    Fit AutoARIMA and forecast h steps.
    Uses statsforecast (Numba) to bypass statsmodels C-extension segfaults.
    """
    model = AutoARIMA(season_length=12, d=1, D=1, max_p=3, max_q=3, max_P=2, max_Q=2, trace=False)
    y_train = train_t.values
    X_train = exog_train.values if exog_train is not None else None
    X_fut   = exog_fut.values if exog_fut is not None else None
    
    print(f"    [{iso}] Running StatsForecast AutoARIMA...")
    res = model.forecast(y=y_train, X=X_train, h=h, X_future=X_fut)
    
    # model.arima_res_ contains the order after fitting if needed
    order_dict = getattr(model, 'arima_res_', {}).get('arma', {})
    print(f"    [{iso}] AutoARIMA fit complete.")
    
    return res['mean']


def normalize_residual(raw_residual: pd.Series, wc_start: pd.Timestamp):
    """
    Normalize by event-window peak (months 0 and +1).
    Consistent with how augmented training profiles were scaled.
    """
    event_ts = [wc_start, wc_start + pd.DateOffset(months=1)]
    mask     = raw_residual.index.isin(event_ts)
    peak     = raw_residual[mask].abs().max() if mask.any() else np.nan
    if pd.isna(peak) or peak == 0:
        peak = raw_residual.abs().max()
    if pd.isna(peak) or peak == 0:
        peak = 1.0
    return raw_residual / peak, float(peak)


def sarima_all_folds(df_aug: pd.DataFrame) -> dict:
    """
    Phase A — run all 5 SARIMA folds BEFORE importing torch.
    Returns dict keyed by ISO with all data needed for the N-BEATSx phase.
    """
    print("\n" + "="*60)
    print("  PHASE A: SARIMA fits (all folds, no torch loaded)")
    print("="*60)

    fold_data = {}
    for country, cfg in DONORS.items():
        iso      = cfg["iso"]
        wc_start = cfg["wc_start"]
        cutoff   = cfg["training_cutoff"]
        win_s, win_e = cfg["residual_window"]

        print(f"\n--- Fold {iso} ({country.replace('_',' ').title()}) ---")

        series = load_donor_data(country, cfg).asfreq("MS")
        train  = series[series.index <= cutoff]

        covid_dummy = build_covid_dummy(series.index)
        use_exog    = bool(covid_dummy.sum() > 0)
        exog_train  = covid_dummy[train.index].to_frame("covid") if use_exog else None

        print(f"  Training on {len(train)} obs | exog={use_exog}")
        train_t = maybe_log(train, cfg)

        # Forecast through event window
        n_fc    = (win_e.year - cutoff.year) * 12 + (win_e.month - cutoff.month)
        fc_idx  = pd.date_range(cutoff + pd.DateOffset(months=1), periods=n_fc, freq="MS")
        exog_f  = covid_dummy.reindex(fc_idx).to_frame("covid") if use_exog else None

        fc_vals = fit_sarima_and_forecast(train_t, exog_train, iso, h=n_fc, exog_fut=exog_f)
        fc_log  = pd.Series(fc_vals, index=fc_idx)
        fc_gwh  = maybe_exp(fc_log, cfg)

        actual_win = series[(series.index >= win_s) & (series.index <= win_e)]
        fc_win     = fc_gwh.reindex(actual_win.index)
        raw_resid  = actual_win - fc_win

        resid_norm, peak_gwh = normalize_residual(raw_resid, wc_start)
        print(f"  Normalization peak: {peak_gwh:.2f} GWh (event-window design)")

        months_to_wc = [
            (d.year - wc_start.year)*12 + (d.month - wc_start.month)
            for d in actual_win.index
        ]

        fold_data[iso] = dict(
            actual_win   = actual_win,
            fc_win       = fc_win,
            raw_resid    = raw_resid,
            resid_norm   = resid_norm,
            peak_gwh     = peak_gwh,
            months_to_wc = months_to_wc,
            ds           = list(actual_win.index),
        )

    print("\n  All SARIMA folds complete.")
    return fold_data


# ══════════════════════════════════════════════════════════════════════════════
#  N-BEATSx helpers  (torch imported lazily here)
# ══════════════════════════════════════════════════════════════════════════════

def _import_nbeatsx_deps():
    """Lazy import of torch + neuralforecast to avoid MKL conflict with SARIMAX."""
    import torch
    torch.manual_seed(42)
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from nbeatsx_training import (
        ARCH, load_p1_weights, apply_freezing,
        pad_to_min_len, make_futr_df, H, INPUT_SIZE, MIN_LEN,
    )
    return torch, NeuralForecast, NBEATSx, ARCH, load_p1_weights, \
           apply_freezing, pad_to_min_len, make_futr_df, H, INPUT_SIZE, MIN_LEN


def build_input_history(df_resid: pd.DataFrame, input_size: int) -> pd.DataFrame:
    uid       = df_resid["unique_id"].iloc[0]
    first_ds  = df_resid["ds"].iloc[0]
    first_mtw = int(df_resid["months_to_wc"].iloc[0])
    pad_ds    = [first_ds - pd.DateOffset(months=k) for k in range(input_size, 0, -1)]
    pad_mtw   = [first_mtw - k                       for k in range(input_size, 0, -1)]
    pad_df    = pd.DataFrame({"unique_id": uid, "ds": pad_ds, "y": 0.0,
                               "months_to_wc": pad_mtw})
    return pd.concat([pad_df, df_resid], ignore_index=True)


def nbeatsx_all_folds(df_aug: pd.DataFrame, fold_data: dict) -> dict:
    """
    Phase B — fine-tune and predict for each fold.
    torch is imported here for the first time.
    """
    print("\n" + "="*60)
    print("  PHASE B: N-BEATSx fine-tuning + prediction (torch loaded now)")
    print("="*60)

    torch, NeuralForecast, NBEATSx, ARCH, load_p1_weights, \
        apply_freezing, pad_to_min_len, make_futr_df, H, INPUT_SIZE, MIN_LEN \
        = _import_nbeatsx_deps()

    nbeatsx_preds = {}

    for iso in fold_data:
        print(f"\n--- Fold {iso} ---")
        fd = fold_data[iso]

        # Training set: augmented profiles excluding this donor
        df_train = df_aug[
            df_aug["unique_id"].str.contains("_aug_") &
            ~df_aug["unique_id"].str.startswith(iso + "_")
        ].copy()
        df_padded = pad_to_min_len(df_train, MIN_LEN)
        print(f"  Training N-BEATSx on {df_train['unique_id'].nunique()} aug profiles")

        arch_fold = ARCH.copy()
        arch_fold["early_stop_patience_steps"] = 5
        model_fold = NBEATSx(
            **arch_fold,
            futr_exog_list = ["months_to_wc"],
            max_steps      = 100,
            learning_rate  = 1e-4,
            batch_size     = 8,
            random_seed    = 42,
        )
        load_p1_weights(model_fold, CKPT_P1)
        apply_freezing(model_fold)

        fcst = NeuralForecast(models=[model_fold], freq="MS")
        fcst.fit(df=df_padded, val_size=H)

        fold_ckpt = os.path.join(PROJECT_DIR, f"finetuned_fold_{iso}.ckpt")
        torch.save(fcst.models[0].state_dict(), fold_ckpt)
        print(f"  Fold checkpoint -> {fold_ckpt}")

        # Build normalized residual DataFrame for prediction evaluation
        df_resid_norm = pd.DataFrame({
            "unique_id"   : iso,
            "ds"          : fd["ds"],
            "y"           : fd["resid_norm"].values,
            "months_to_wc": fd["months_to_wc"],
        }).reset_index(drop=True)

        # The input history must be the `INPUT_SIZE` months strictly BEFORE the event window
        first_ds = pd.to_datetime(df_resid_norm["ds"].iloc[0])
        first_mtw = int(df_resid_norm["months_to_wc"].iloc[0])

        pad_ds    = [first_ds - pd.DateOffset(months=k) for k in range(INPUT_SIZE, 0, -1)]
        pad_mtw   = [first_mtw - k                       for k in range(INPUT_SIZE, 0, -1)]
        
        hist_df = pd.DataFrame({
            "unique_id": iso, 
            "ds": pad_ds, 
            "y": 0.0,
            "months_to_wc": pad_mtw
        })

        futr_df = make_futr_df(hist_df, H)

        try:
            preds    = fcst.predict(df=hist_df, futr_df=futr_df)
            pred_col = [c for c in preds.columns if c not in ("unique_id", "ds")][0]
            preds    = preds.rename(columns={pred_col: "y_pred"})
            futr_df2 = futr_df[["unique_id", "ds", "months_to_wc"]]
            preds    = preds.merge(futr_df2, on=["unique_id", "ds"], how="left")

            eval_df  = pd.merge(
                df_resid_norm.rename(columns={"y": "y_actual"}),
                preds[["months_to_wc", "y_pred"]],
                on="months_to_wc", how="inner",
            )
            act_norm  = eval_df["y_actual"].values
            pred_norm = eval_df["y_pred"].values

        except Exception as exc:
            print(f"  WARNING: prediction failed ({exc}) — using zeros")
            act_norm  = fd["resid_norm"].values
            pred_norm = np.zeros_like(act_norm)

        nbeatsx_preds[iso] = dict(act_norm=act_norm, pred_norm=pred_norm)

    return nbeatsx_preds


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics + plots  (no torch dependency)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics_and_plots(fold_data: dict, nbeatsx_preds: dict):
    def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b))**2)))
    def mae(a, b):  return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

    metrics   = []
    plot_data = {}
    COLORS    = {"QAT": "#9b59b6", "CMR": "#f1c40f", "RSA": "#2ecc71",
                 "EGY": "#e76f51", "RUS": "#3498db", "ZAF": "#1abc9c"}

    for iso, fd in fold_data.items():
        pn = nbeatsx_preds[iso]
        pred_resid_gwh = pn["pred_norm"] * fd["peak_gwh"]
        n_eval = len(pred_resid_gwh)

        actual_gwh = np.asarray(fd["actual_win"].values)[:n_eval]
        sarima_gwh = np.asarray(fd["fc_win"].values)[:n_eval]
        comb_gwh   = sarima_gwh + pred_resid_gwh

        rs = rmse(actual_gwh, sarima_gwh)
        rc = rmse(actual_gwh, comb_gwh)
        ms = mae(actual_gwh, sarima_gwh)
        mc = mae(actual_gwh, comb_gwh)
        rr = rmse(pn["act_norm"], pn["pred_norm"])

        print(f"  {iso}: RMSE S={rs:.2f}  C={rc:.2f} GWh | MAE S={ms:.2f} C={mc:.2f} | resid RMSE={rr:.4f}")

        metrics.append(dict(
            donor=iso,
            RMSE_sarima=round(rs, 2), RMSE_combined=round(rc, 2),
            MAE_sarima=round(ms, 2),  MAE_combined=round(mc, 2),
            residual_RMSE=round(rr, 4),
        ))
        plot_data[iso] = {
            "mtw": np.asarray(fd["months_to_wc"])[:n_eval],
            "actual": actual_gwh,
            "sarima": sarima_gwh,
            "combined": comb_gwh,
            "act_norm": pn["act_norm"],
            "pred_norm": pn["pred_norm"],
            "rs": rs, "rc": rc, "rr": rr, "color": COLORS.get(iso, "steelblue"),
        }

    # ── Metrics CSV ──────────────────────────────────────────────────────────
    df_m = pd.DataFrame(metrics)
    mp   = os.path.join(PROJECT_DIR, "loo_metrics.csv")
    df_m.to_csv(mp, index=False)
    print(f"\nMetrics saved -> {mp}")
    print(df_m.to_string(index=False))

    # ── Interpretation ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  INTERPRETATION FLAGS")
    print("="*60)
    degrades = 0
    for row in metrics:
        iso = row["donor"]
        rs  = row["RMSE_sarima"]
        rc  = row["RMSE_combined"]
        imp = rs - rc
        pct = (imp / rs) * 100 if rs > 0 else 0.0
        if rc > rs:
            degrades += 1
        flag = "IMPROVED" if rc <= rs else "DEGRADED"
        print(f"  {iso} [{flag}]: contrib={imp:+.2f} GWh ({pct:+.1f}%)")

    if degrades > 2:
        print("\n  WARNING: N-BEATSx correction is net-negative on majority of "
              "donors — review residual extraction or revert to SARIMA-only baseline.")
    else:
        print(f"\n  N-BEATSx degrades on {degrades}/5 donors — "
              "correction is net-positive overall.")

    # ── Forecast plot ─────────────────────────────────────────────────────────
    isos = list(plot_data.keys())
    fig, axes = plt.subplots(1, len(isos), figsize=(5*len(isos), 5))
    if len(isos) == 1:
        axes = [axes]
    fig.suptitle("LOO Full Forecast: SARIMA vs SARIMA + N-BEATSx",
                 fontsize=13, fontweight="bold")
    for ax, iso in zip(axes, isos):
        d = plot_data[iso]
        ax.plot(d["mtw"], d["actual"],   color="black",     lw=2.2, label="Actual")
        ax.plot(d["mtw"], d["sarima"],   color="royalblue", lw=1.6, ls="--", label="SARIMA only")
        ax.plot(d["mtw"], d["combined"], color="crimson",   lw=2.0, label="SARIMA + N-BEATSx")
        ax.axvline(0, color="gray", ls=":", lw=1.2)
        ax.set_title(iso, fontsize=11, fontweight="bold", color=d["color"])
        ax.set_xlabel("Months relative to WC start")
        ax.set_ylabel("Electricity demand (GWh)")
        ax.annotate(f"RMSE S: {d['rs']:.1f}\nRMSE C: {d['rc']:.1f}",
                    xy=(0.05, 0.84), xycoords="axes fraction", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fp = os.path.join(PROJECT_DIR, "loo_forecast_plot.png")
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nForecast plot -> {fp}")

    # ── Residual plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(isos), figsize=(5*len(isos), 5))
    if len(isos) == 1:
        axes = [axes]
    fig.suptitle("LOO Normalized Residuals: Actual vs N-BEATSx Prediction",
                 fontsize=13, fontweight="bold")
    for ax, iso in zip(axes, isos):
        d = plot_data[iso]
        ax.plot(d["mtw"], d["act_norm"],  color="black",   lw=2.2, label="Actual norm")
        ax.plot(d["mtw"], d["pred_norm"], color="crimson", lw=1.8, ls="--", label="N-BEATSx pred")
        ax.axvline(0, color="gray",      ls=":", lw=1.2)
        ax.axhline(0, color="lightgray", lw=0.6)
        ax.set_title(iso, fontsize=11, fontweight="bold", color=d["color"])
        ax.set_xlabel("Months relative to WC start")
        ax.set_ylabel("Normalised residual")
        ax.annotate(f"RMSE: {d['rr']:.4f}",
                    xy=(0.05, 0.90), xycoords="axes fraction", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    rp = os.path.join(PROJECT_DIR, "loo_residual_plot.png")
    plt.savefig(rp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Residual plot -> {rp}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_loo_pipeline():
    print("Loading augmented donor profiles...")
    df_aug = pd.read_csv(AUG_PATH, parse_dates=["ds"])
    print(f"  {df_aug['unique_id'].nunique()} profiles  |  {len(df_aug):,} rows\n")

    # Phase A: all SARIMA fits (torch NOT loaded yet)
    fold_data = sarima_all_folds(df_aug)

    # Phase B: lazy-import torch, fine-tune and predict
    nbeatsx_preds = nbeatsx_all_folds(df_aug, fold_data)

    # Phase C: metrics + plots (no torch needed)
    print("\n" + "="*60)
    print("  PHASE C: Metrics & Plots")
    print("="*60)
    compute_metrics_and_plots(fold_data, nbeatsx_preds)

    print("\n" + "="*60)
    print("  LOO PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_loo_pipeline()
