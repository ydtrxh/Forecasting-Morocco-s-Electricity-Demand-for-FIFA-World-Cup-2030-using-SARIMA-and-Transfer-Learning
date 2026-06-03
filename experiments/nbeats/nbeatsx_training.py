"""
nbeatsx_training.py
WC 2030 Morocco Electricity Forecasting
Two-Phase N-BEATSx Transfer Learning Pipeline

Phase 1: Pre-train on M4 Monthly    (learn general TS representations)
Phase 2: Fine-tune on donor profiles (learn WC electricity demand pulse)
LOO:     Leave-one-original-out validation for per-donor metrics

Author:  Younes, ENSAM Meknes (IATD)
Seed:    np.random.seed(42) + torch.manual_seed(42)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"

import numpy as np
import pandas as pd
import torch
import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)
torch.manual_seed(42)

# ── Imports ───────────────────────────────────────────────────────────────
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from datasetsforecast.m4 import M4
    print("neuralforecast and datasetsforecast imported OK.")
except ImportError as exc:
    sys.exit(f"Import error: {exc}\n"
             "Install with:  pip install neuralforecast datasetsforecast")

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\TS-project"
M4_CACHE    = os.path.join(PROJECT_DIR, "m4_cache")
os.makedirs(M4_CACHE, exist_ok=True)

CKPT_P1 = os.path.join(PROJECT_DIR, "pretrained_nbeatsx.ckpt")
CKPT_P2 = os.path.join(PROJECT_DIR, "finetuned_nbeatsx.ckpt")

# ── Architecture constants ─────────────────────────────────────────────────
H          = 13          # forecast horizon
INPUT_SIZE = 13          # lookback window
VAL_SIZE   = H           # validation set size for early stopping
MIN_LEN    = INPUT_SIZE + H + VAL_SIZE   # 39: minimum length for windowing

# Small non-zero y used when padding pre-event history.
# = mean residual_norm across all 5 donors in the -6 to +6 window.
# Ensures TemporalNorm scale ≈ 0.053 (not 0) so normalization is stable.
PRE_EVENT_Y = 0.053

DONORS = ["QAT", "CMR", "RSA", "EGY", "RUS"]
COLORS = {"QAT": "#9b59b6", "CMR": "#f1c40f", "RSA": "#2ecc71",
          "EGY": "#e76f51", "RUS": "#3498db"}

# Shared architecture — MUST be identical in both phases so weights load cleanly
# stack_types order: trend -> seasonality -> identity (generic, last block trainable)
ARCH = dict(
    h                         = H,
    input_size                = INPUT_SIZE,
    stack_types               = ["trend", "seasonality", "identity"],
    n_harmonics               = 1,
    n_polynomials             = 2,
    dropout_prob_theta        = 0.5,
    early_stop_patience_steps = 10,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Utility Functions
# ══════════════════════════════════════════════════════════════════════════════

def pad_to_min_len(df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    """
    Prepend zero-valued rows to series shorter than min_len so NeuralForecast
    can form at least one training window (requires input_size + h observations).
    months_to_wc is extended backwards from the first observed value.
    y = 0.0 for synthetic padding rows (neutral signal, far pre-event).
    """
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
            # carry through any extra columns as 0
            for col in grp.columns:
                if col not in pad_df.columns:
                    pad_df[col] = 0.0
            grp = pd.concat([pad_df, grp], ignore_index=True)
        parts.append(grp)
    return pd.concat(parts, ignore_index=True)


def make_futr_df(hist: pd.DataFrame, h: int) -> pd.DataFrame:
    """
    Build the future-exogenous DataFrame required by NeuralForecast.predict()
    when futr_exog_list is set.  One row per unique_id per forecast step.
    months_to_wc continues linearly from the last observed value.
    """
    rows = []
    for uid, grp in hist.groupby("unique_id", sort=False):
        grp      = grp.sort_values("ds")
        last_ds  = grp["ds"].iloc[-1]
        last_mtw = int(grp["months_to_wc"].iloc[-1])
        for step in range(1, h + 1):
            rows.append({
                "unique_id"   : uid,
                "ds"          : last_ds + pd.DateOffset(months=step),
                "months_to_wc": last_mtw + step,
            })
    return pd.DataFrame(rows)


def apply_freezing(model) -> None:
    """
    Freeze all NBEATS blocks except the last (identity/generic) block.
    Matches the freezing strategy specified in the task exactly:

        num_blocks = len(nbeatsx_module.blocks)
        for i, block in enumerate(nbeatsx_module.blocks):
            if i < (num_blocks - 1):  -> freeze
            else:                     -> trainable
    """
    blocks = model.blocks   # confirmed nn.ModuleList in NF 1.7.4
    n      = len(blocks)
    for i, block in enumerate(blocks):
        req = (i == n - 1)
        for p in block.parameters():
            p.requires_grad = req
    print(f"    Blocks total: {n}  |  frozen: {n - 1}  |  trainable (last): 1")


def print_param_status(model) -> None:
    """Print per-parameter requires_grad status and aggregate counts."""
    total     = 0
    trainable = 0
    print()
    for name, p in model.named_parameters():
        total     += p.numel()
        trainable += p.numel() if p.requires_grad else 0
        tag = "TRAINABLE" if p.requires_grad else "FROZEN   "
        print(f"      [{tag}] {name}")
    print(f"\n    Total parameters:     {total:,}")
    print(f"    Trainable parameters: {trainable:,}")
    print(f"    Frozen parameters:    {total - trainable:,}")


def load_p1_weights(model, path: str) -> None:
    """
    Load Phase-1 state dict into model with strict=False.
    strict=False allows the new exogenous input layer (months_to_wc) in Phase 2
    to be initialized randomly while all shared weights transfer cleanly.
    To prevent RuntimeError on shape mismatch for the input layers, we pop
    those keys from the loaded dict and let them initialize randomly.
    """
    sd       = torch.load(path, map_location="cpu")
    model_sd = model.state_dict()
    
    # Remove keys with shape mismatches so strict=False works safely
    mismatched = []
    for k, v in sd.items():
        if k in model_sd and v.shape != model_sd[k].shape:
            mismatched.append(k)
            
    for k in mismatched:
        sd.pop(k)

    info = model.load_state_dict(sd, strict=False)
    print(f"    P1 weights loaded | shape mismatched (randomized): {len(mismatched)} "
          f"| missing keys: {len(info.missing_keys)} "
          f"| unexpected keys: {len(info.unexpected_keys)}")
    if mismatched:
        print(f"      Mismatched (e.g. input layers): {mismatched[:3]}")
    if info.missing_keys:
        print(f"      Missing  (new in P2): {info.missing_keys[:5]}")
    if info.unexpected_keys:
        print(f"      Unexpected (P1 only): {info.unexpected_keys[:5]}")


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1 — Pre-training on M4 Monthly
# ══════════════════════════════════════════════════════════════════════════════

def phase1_pretrain() -> str:
    print("=" * 60)
    print("  PHASE 1 -- Pre-training on M4 Monthly")
    print("=" * 60)

    if os.path.exists(CKPT_P1):
        print(f"\n  Checkpoint found, skipping Phase 1:\n  {CKPT_P1}")
        return CKPT_P1

    # ── Load M4 Monthly ────────────────────────────────────────────────────
    print("\n  Loading M4 Monthly (will download on first run) ...")
    train_df, *_ = M4.load(directory=M4_CACHE, group="Monthly")
    train_df["ds"] = pd.to_datetime(train_df["ds"])
    print(f"  Loaded: {train_df['unique_id'].nunique():,} series  "
          f"| {len(train_df):,} observations")

    # Drop series too short for even one training window
    lens      = train_df.groupby("unique_id").size()
    valid_ids = lens[lens >= MIN_LEN].index
    train_df  = train_df[train_df["unique_id"].isin(valid_ids)].reset_index(drop=True)
    print(f"  After length filter (>= {MIN_LEN}): "
          f"{train_df['unique_id'].nunique():,} series")

    # ── Build model (no exogenous in Phase 1) ─────────────────────────────
    model_p1 = NBEATSx(
        **ARCH,
        futr_exog_list  = None,
        max_steps       = 1000,
        learning_rate   = 1e-3,
        batch_size      = 32,
        val_check_steps = 200,   # validation + logging every 200 steps
        random_seed     = 42,
    )

    n_params = sum(p.numel() for p in model_p1.parameters())
    print(f"\n  Total trainable parameters (Phase 1): {n_params:,}")

    # ── Train ──────────────────────────────────────────────────────────────
    fcst1 = NeuralForecast(models=[model_p1], freq="MS")
    print("\n  Training Phase 1 (max_steps=1000, val_check_steps=200) ...")
    fcst1.fit(df=train_df, val_size=H)

    # ── Save raw state dict ────────────────────────────────────────────────
    # Using torch.save(state_dict) rather than NF's save() so we can reload
    # with strict=False into Phase 2 model that has a different input layer.
    torch.save(fcst1.models[0].state_dict(), CKPT_P1)
    print(f"\n  Phase 1 weights saved -> {CKPT_P1}")
    return CKPT_P1


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — Fine-tuning on Augmented Donor Profiles
# ══════════════════════════════════════════════════════════════════════════════

def phase2_finetune(df_aug: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  PHASE 2 -- Fine-tuning on Augmented Donor Profiles")
    print("=" * 60)

    # Augmented profiles for training; original profiles for reference only
    df_train = df_aug[df_aug["unique_id"].str.contains("_aug_")].copy()
    df_ref   = df_aug[~df_aug["unique_id"].str.contains("_aug_")].copy()
    
    # Replace real pre-event residual context (months -5 to -1) with synthetic context
    df_raw_res = pd.read_csv(os.path.join(PROJECT_DIR, "forecast_residuals.csv"))
    df_raw_res["country"] = df_raw_res["country"].replace({"ZAF": "RSA"})
    
    ratios = {}
    for country in ["RSA", "CMR", "RUS", "QAT", "EGY"]:
        sub = df_raw_res[(df_raw_res["country"] == country) & (df_raw_res["months_to_wc"] >= -5) & (df_raw_res["months_to_wc"] <= -1)]
        sub = sub.sort_values("months_to_wc")
        val_minus_1 = sub[sub["months_to_wc"] == -1]["sarima_forecast"].values[0]
        sub["ratio_y"] = (sub["sarima_forecast"] / val_minus_1) * PRE_EVENT_Y
        ratios[country] = dict(zip(sub["months_to_wc"], sub["ratio_y"]))
        
    df_train["base_donor"] = df_train["unique_id"].apply(lambda x: x.split("_")[0])
    for country in ["RSA", "CMR", "RUS", "QAT", "EGY"]:
        if country in ratios:
            for mtw, val in ratios[country].items():
                mask = (df_train["base_donor"] == country) & (df_train["months_to_wc"] == mtw)
                df_train.loc[mask, "y"] = val
    df_train = df_train.drop(columns=["base_donor"])
    
    print(f"\n  Training : {df_train['unique_id'].nunique():,} augmented profiles")
    print(f"  Reference: {df_ref['unique_id'].nunique()} original profiles "
          f"(held out from training)")

    # Pad short series for NeuralForecast windowing
    df_padded = pad_to_min_len(df_train, MIN_LEN)
    min_len   = df_padded.groupby("unique_id").size().min()
    print(f"  After padding: min series length = {min_len}")

    # ── Build Phase 2 model ────────────────────────────────────────────────
    # Architecture is IDENTICAL to Phase 1, plus futr_exog_list=['months_to_wc']
    arch_p2 = ARCH.copy()
    arch_p2["early_stop_patience_steps"] = 5
    
    model_p2 = NBEATSx(
        **arch_p2,
        futr_exog_list            = ["months_to_wc"],
        max_steps                 = 100,
        learning_rate             = 1e-4,
        batch_size                = 8,
        random_seed               = 42,
    )

    # ── Transfer Phase 1 weights ───────────────────────────────────────────
    # strict=False: the new exogenous input layer in Phase 2 is not in the
    # Phase 1 checkpoint, so it retains its random initialization.
    print("\n  Loading Phase 1 weights:")
    load_p1_weights(model_p2, CKPT_P1)

    # ── Wrap and freeze BEFORE fit() ──────────────────────────────────────
    # Freezing must happen before NeuralForecast creates the optimizer so
    # that frozen parameters are excluded from gradient updates.
    fcst2       = NeuralForecast(models=[model_p2], freq="MS")
    nbeatsx_mod = fcst2.models[0]

    print("\n  Applying freezing strategy (freeze all blocks except last):")
    apply_freezing(nbeatsx_mod)

    print("\n  Detailed parameter status after freezing:")
    print_param_status(nbeatsx_mod)

    # ── Fine-tune ──────────────────────────────────────────────────────────
    print("\n  Training Phase 2 (max_steps=100) ...")
    fcst2.fit(df=df_padded, val_size=H)

    # ── Save ──────────────────────────────────────────────────────────────
    torch.save(fcst2.models[0].state_dict(), CKPT_P2)
    print(f"\n  Phase 2 weights saved -> {CKPT_P2}")
    return fcst2


# ══════════════════════════════════════════════════════════════════════════════
#  LOO Validation — Leave-One-Original-Out
# ══════════════════════════════════════════════════════════════════════════════

def loo_validation(df_aug: pd.DataFrame) -> tuple:
    print("\n" + "=" * 60)
    print("  LOO VALIDATION -- Leave-One-Original-Out")
    print("=" * 60)

    loo_results = []
    loo_preds   = {}

    for donor in DONORS:
        print(f"\n  --- Held-out: {donor} ---")

        # Training: all augmented profiles EXCLUDING this donor's variants
        df_train = df_aug[
            df_aug["unique_id"].str.contains("_aug_") &
            ~df_aug["unique_id"].str.startswith(donor + "_")
        ].copy()
        
        # Replace real pre-event residual context (months -5 to -1) with synthetic context
        df_raw_res = pd.read_csv(os.path.join(PROJECT_DIR, "forecast_residuals.csv"))
        df_raw_res["country"] = df_raw_res["country"].replace({"ZAF": "RSA"})
        
        ratios = {}
        for country in ["RSA", "CMR", "RUS", "QAT", "EGY"]:
            sub = df_raw_res[(df_raw_res["country"] == country) & (df_raw_res["months_to_wc"] >= -5) & (df_raw_res["months_to_wc"] <= -1)]
            sub = sub.sort_values("months_to_wc")
            val_minus_1 = sub[sub["months_to_wc"] == -1]["sarima_forecast"].values[0]
            sub["ratio_y"] = (sub["sarima_forecast"] / val_minus_1) * PRE_EVENT_Y
            ratios[country] = dict(zip(sub["months_to_wc"], sub["ratio_y"]))
            
        df_train["base_donor"] = df_train["unique_id"].apply(lambda x: x.split("_")[0])
        for country in ["RSA", "CMR", "RUS", "QAT", "EGY"]:
            if country in ratios:
                for mtw, val in ratios[country].items():
                    mask = (df_train["base_donor"] == country) & (df_train["months_to_wc"] == mtw)
                    df_train.loc[mask, "y"] = val
        df_train = df_train.drop(columns=["base_donor"])


        # Validation: the held-out donor's original profile
        df_val = (df_aug[df_aug["unique_id"] == donor]
                  .copy()
                  .sort_values("months_to_wc")
                  .reset_index(drop=True))

        print(f"    Train : {df_train['unique_id'].nunique()} aug profiles")
        print(f"    Val   : {len(df_val)} obs  "
              f"(months_to_wc [{df_val['months_to_wc'].min()}, "
              f"{df_val['months_to_wc'].max()}])")

        # Pad training series
        df_train_padded = pad_to_min_len(df_train, MIN_LEN)

        # ── Build LOO model ────────────────────────────────────────────────
        arch_loo = ARCH.copy()
        arch_loo["early_stop_patience_steps"] = 5
        
        loo_model = NBEATSx(
            **arch_loo,
            futr_exog_list            = ["months_to_wc"],
            max_steps                 = 100,
            learning_rate             = 1e-4,
            batch_size                = 8,
            random_seed               = 42,
        )

        load_p1_weights(loo_model, CKPT_P1)
        apply_freezing(loo_model)

        fcst_loo = NeuralForecast(models=[loo_model], freq="MS")
        fcst_loo.fit(df=df_train_padded, val_size=H)

        # ── Predict on held-out original ───────────────────────────────────
        # Provide full donor history (padded to INPUT_SIZE) as context.
        # NeuralForecast 1.7.4 predict(df=...) supports cold-start on
        # series not seen during training.
        hist_df = pad_to_min_len(df_val.copy(), INPUT_SIZE)
        futr_df = make_futr_df(hist_df, H)

        try:
            preds    = fcst_loo.predict(df=hist_df, futr_df=futr_df)
            pred_col = [c for c in preds.columns if c not in ("unique_id", "ds")][0]
            pred_y   = preds[pred_col].values
            pred_mtw = futr_df["months_to_wc"].values

            # Align predictions with actual observations by months_to_wc
            actual_sub = df_val[df_val["months_to_wc"].isin(pred_mtw)]
            if len(actual_sub) > 0:
                pred_s  = pd.Series(pred_y,  index=pred_mtw)
                act_s   = actual_sub.set_index("months_to_wc")["y"]
                common  = pred_s.index.intersection(act_s.index)
                p_arr   = pred_s.loc[common].values
                a_arr   = act_s.loc[common].values
            else:
                # fallback: compare by position on the last available actuals
                n     = min(H, len(df_val))
                a_arr = df_val["y"].values[-n:]
                p_arr = pred_y[:n]

            rmse = float(np.sqrt(np.mean((p_arr - a_arr) ** 2)))
            mae  = float(np.mean(np.abs(p_arr - a_arr)))
            print(f"    RMSE = {rmse:.4f}   MAE = {mae:.4f}   (n_compare={len(a_arr)})")

            loo_preds[donor] = {
                "actual_mtw": df_val["months_to_wc"].values,
                "actual_y"  : df_val["y"].values,
                "pred_mtw"  : pred_mtw,
                "pred_y"    : pred_y,
            }

        except Exception as exc:
            print(f"    ERROR during prediction: {exc}")
            rmse = mae = np.nan
            loo_preds[donor] = None

        loo_results.append({
            "donor": donor,
            "RMSE" : round(rmse, 4) if not np.isnan(rmse) else np.nan,
            "MAE"  : round(mae,  4) if not np.isnan(mae)  else np.nan,
        })

    # ── Save results ───────────────────────────────────────────────────────
    df_res   = pd.DataFrame(loo_results)
    res_path = os.path.join(PROJECT_DIR, "loo_validation_results.csv")
    df_res.to_csv(res_path, index=False)

    print("\n  LOO Summary:")
    print(df_res.to_string(index=False))
    print(f"\n  Results saved -> {res_path}")
    return df_res, loo_preds


# ══════════════════════════════════════════════════════════════════════════════
#  LOO Validation Plot
# ══════════════════════════════════════════════════════════════════════════════

def plot_loo(df_aug: pd.DataFrame,
             df_res: pd.DataFrame,
             loo_preds: dict) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(
        "N-BEATSx LOO Validation — Predicted vs Actual Normalised WC Demand Pulse",
        fontsize=12, fontweight="bold", y=1.02,
    )

    for ax, donor in zip(axes, DONORS):
        color = COLORS.get(donor, "steelblue")
        orig  = df_aug[df_aug["unique_id"] == donor].sort_values("months_to_wc")

        # Actual original profile — bold black
        ax.plot(orig["months_to_wc"], orig["y"],
                color="black", lw=2.5, zorder=5, label="Actual")

        # N-BEATSx prediction — red dashed
        info = loo_preds.get(donor)
        if info is not None:
            ax.plot(info["pred_mtw"], info["pred_y"],
                    color="red", lw=1.8, ls="--", zorder=6, label="N-BEATSx")

        # RMSE annotation
        row = df_res[df_res["donor"] == donor]
        if not row.empty:
            rmse_val = row["RMSE"].values[0]
            label    = f"RMSE = {rmse_val:.3f}" if not np.isnan(rmse_val) else "RMSE = N/A"
            ax.annotate(label, xy=(0.04, 0.93), xycoords="axes fraction",
                        fontsize=9, color="red",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="white", alpha=0.80))

        ax.axvline(0, color="crimson", ls=":", lw=1.2, alpha=0.65, label="WC start")
        ax.axhline(0, color="gray",    ls="-", lw=0.5)
        ax.set_title(donor, fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Months to WC start", fontsize=9)
        ax.set_ylabel("Normalised residual", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.30)

    plt.tight_layout()
    path = os.path.join(PROJECT_DIR, "loo_validation_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  LOO plot saved -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── Load augmented donor profiles ─────────────────────────────────────
    aug_path = os.path.join(PROJECT_DIR, "donor_residuals_augmented.csv")
    df_aug   = pd.read_csv(aug_path, parse_dates=["ds"])
    print(f"Loaded: {aug_path}")
    print(f"  {df_aug['unique_id'].nunique()} profiles  |  {len(df_aug):,} rows\n")

    # Confirm months_to_wc is registered as futr_exog (printed for verification)
    print(f"  futr_exog column: 'months_to_wc'  "
          f"range [{df_aug['months_to_wc'].min()}, {df_aug['months_to_wc'].max()}]")
    print(f"  y range (no abs applied): [{df_aug['y'].min():.4f}, {df_aug['y'].max():.4f}]\n")

    # ── Phase 1 ───────────────────────────────────────────────────────────
    phase1_pretrain()

    # ── Phase 2 ───────────────────────────────────────────────────────────
    phase2_finetune(df_aug)

    # ── LOO Validation ────────────────────────────────────────────────────
    df_res, loo_preds = loo_validation(df_aug)

    # ── LOO Plot ──────────────────────────────────────────────────────────
    plot_loo(df_aug, df_res, loo_preds)

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)
    total_profiles = df_aug["unique_id"].nunique()
    print(f"  Total profiles used in N-BEATSx training: {total_profiles}")
    print(f"\n  pretrained_nbeatsx.ckpt    -> {CKPT_P1}")
    print(f"  finetuned_nbeatsx.ckpt     -> {CKPT_P2}")
    print(f"  loo_validation_results.csv -> "
          f"{os.path.join(PROJECT_DIR, 'loo_validation_results.csv')}")
    print(f"  loo_validation_plot.png    -> "
          f"{os.path.join(PROJECT_DIR, 'loo_validation_plot.png')}")
    print("=" * 60)
