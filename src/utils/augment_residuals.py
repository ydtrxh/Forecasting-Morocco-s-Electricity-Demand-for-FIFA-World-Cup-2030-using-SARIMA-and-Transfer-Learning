"""
augment_residuals.py
WC 2030 Morocco Electricity Forecasting — Data Augmentation Pipeline

Augments normalized WC-event residual profiles from 5 donor countries
to produce a richer N-BEATSx training dataset.

Operations per donor:
  QAT, CMR, RSA  →  Magnitude scaling (0.85–1.15) × 10 variants
                     Phase shifting (-1, 0, +1 months), circular on y only
                     AR(1) correlated noise, 4 variants per scaled+shifted profile

  EGY, RUS       →  Magnitude scaling (0.90–1.10) × 8 variants
                     No phase shifting
                     AR(1) correlated noise, 4 variants per scaled profile

Author: Younes, ENSAM Meknès (IATD)
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import sys
import numpy as np
import pandas as pd
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\TS-project"

# ── Augmentation configuration per donor ─────────────────────────────────────
AUG_CONFIG = {
    "QAT": {"scale_range": (0.85, 1.15), "n_scale": 10, "phase_shift": True,  "n_noise": 4},
    "CMR": {"scale_range": (0.85, 1.15), "n_scale": 10, "phase_shift": True,  "n_noise": 4},
    "RSA": {"scale_range": (0.85, 1.15), "n_scale": 10, "phase_shift": True,  "n_noise": 4},
    "EGY": {"scale_range": (0.90, 1.10), "n_scale":  8, "phase_shift": False, "n_noise": 4},
    "RUS": {"scale_range": (0.90, 1.10), "n_scale":  8, "phase_shift": False, "n_noise": 4},
}

DONOR_COLORS = {
    "QAT": "#9b59b6",
    "CMR": "#f1c40f",
    "RSA": "#2ecc71",
    "EGY": "#e76f51",
    "RUS": "#3498db",
}


# ── 1. Load & Prepare Input Data ─────────────────────────────────────────────

def load_input_data() -> pd.DataFrame:
    """
    Load donor_residuals_normalized.csv — produced by fix_normalization.py.
    Normalization: y = (actual - sarima) / sarima_forecast(t_peak)
    This is the correct scale for the N-BEATSx pulse injection formula:
        wc_lift_GWh(t) = y(t) * t_peak_val

    NOTE: Do NOT read residual_norm from forecast_residuals.csv — that column
    uses residual / sarima_forecast(t), a per-timestep normalization that
    produces peaks far below the intended 5-15% range and destroys the
    physical meaning of the pulse magnitude.
    """
    norm_path = f"{PROJECT_DIR}\\donor_residuals_normalized.csv"
    df = pd.read_csv(norm_path)
    df["ds"] = pd.to_datetime(df["ds"])
    # Rename ZAF -> RSA if still present (fix_normalization may keep RSA already)
    df["unique_id"] = df["unique_id"].replace({"ZAF": "RSA"})
    df = df[["unique_id", "ds", "y", "months_to_wc"]].copy()
    df = df.sort_values(["unique_id", "months_to_wc"]).reset_index(drop=True)
    print(f"Canonical input loaded <- {norm_path}")
    return df


# ── 2. AR(1) Noise Generator ─────────────────────────────────────────────────

def generate_ar1_noise(y_vec: np.ndarray, rho: float, n_samples: int) -> list:
    """
    Generate n_samples AR(1) noise vectors of the same length as y_vec.

    Model:  ε_t = rho * ε_{t-1} + η_t,  η_t ~ N(0, σ * sqrt(1 - rho²))
    where   σ = 0.05 * (max(y) - min(y))

    This produces noise that is correlated at lag-1 with coefficient ρ
    and has marginal standard deviation σ, matching the series' amplitude.
    """
    T = len(y_vec)
    amplitude = float(np.max(y_vec) - np.min(y_vec))
    sigma = 0.05 * amplitude if amplitude > 1e-8 else 0.01

    noise_samples = []
    for _ in range(n_samples):
        eps = np.zeros(T)
        innov = np.random.normal(0, sigma * np.sqrt(max(1 - rho ** 2, 1e-6)), T)
        eps[0] = innov[0]
        for t in range(1, T):
            eps[t] = rho * eps[t - 1] + innov[t]
        noise_samples.append(eps)

    return noise_samples


# ── 3. Augment a Single Donor ─────────────────────────────────────────────────

def augment_donor(
    donor_df: pd.DataFrame,
    config: dict,
    donor_id: str,
) -> tuple[list, int]:
    """
    Produce augmented profiles for one donor country.

    Pipeline:
      1. Generate n_scale magnitude-scaled variants (uniform scalar on y).
      2. For each scaled variant, apply phase shifts (circular shift on y only;
         ds and months_to_wc are never modified).
      3. For each (scaled, shifted) pair, add n_noise AR(1) noise variants.

    Returns a list of DataFrames (one per augmented profile) and the total count.
    """
    # Sort by months_to_wc so positional indexing is consistent
    donor_df = donor_df.sort_values("months_to_wc").reset_index(drop=True)
    y_orig    = donor_df["y"].values.astype(float)
    ds_vals   = donor_df["ds"].values
    mtw_vals  = donor_df["months_to_wc"].values

    # Estimate AR(1) ρ from raw y
    if len(y_orig) >= 4:
        acf_vals = acf(y_orig, nlags=1, fft=False)
        rho = float(np.clip(acf_vals[1], -0.99, 0.99))
    else:
        rho = 0.0
    print(f"    [{donor_id}] estimated AR(1) rho = {rho:.3f}")

    n_scale = config["n_scale"]
    n_noise  = config["n_noise"]
    do_phase = config["phase_shift"]
    lo, hi   = config["scale_range"]

    # Phase shifts: circular shift on y positions only
    shifts = [-1, 0, 1] if do_phase else [0]

    augmented_dfs: list[pd.DataFrame] = []
    aug_i = 0

    # Draw all scale factors upfront (deterministic given seed)
    scale_factors = np.random.uniform(lo, hi, n_scale)

    for scale in scale_factors:
        y_scaled = y_orig * scale

        for shift in shifts:
            # Circular shift — preserves temporal structure without extrapolation
            y_shifted = np.roll(y_scaled, shift)

            # AR(1) noise variants added on top of scaled+shifted y
            noise_variants = generate_ar1_noise(y_shifted, rho, n_noise)

            for noise in noise_variants:
                y_aug = y_shifted + noise

                # ── Validation: y must not push far outside original bounds ──
                y_clip_min = y_orig.min() - 0.2
                y_clip_max = y_orig.max() + 0.2
                y_clipped = np.clip(y_aug, y_clip_min, y_clip_max)
                if not np.allclose(y_aug, y_clipped):
                    y_aug = y_clipped   # soft-clip instead of raising error

                # ── Validation: months_to_wc is never modified ───────────────
                # (ds and mtw_vals are shared from original — no mutation)

                aug_df = pd.DataFrame({
                    "unique_id":   f"{donor_id}_aug_{aug_i}",
                    "ds":          ds_vals,
                    "y":           y_aug,
                    "months_to_wc": mtw_vals,
                })
                augmented_dfs.append(aug_df)
                aug_i += 1

    return augmented_dfs, aug_i


# ── 4. Main Augmentation Pipeline ────────────────────────────────────────────

def run_augmentation_pipeline() -> pd.DataFrame:
    print("=" * 60)
    print("  WC Residual Augmentation Pipeline")
    print("=" * 60)

    df_input = load_input_data()

    all_profiles = [df_input]   # original profiles preserved first
    summary_rows = []

    for donor_id, config in AUG_CONFIG.items():
        print(f"\n-----------  {donor_id}  -----------")

        donor_df = df_input[df_input["unique_id"] == donor_id].copy()

        if donor_df.empty:
            print(f"  WARNING: No data found for {donor_id}, skipping.")
            continue

        orig_mtw_min = int(donor_df["months_to_wc"].min())
        orig_mtw_max = int(donor_df["months_to_wc"].max())
        print(f"  Original profile: {len(donor_df)} observations, "
              f"months_to_wc in [{orig_mtw_min}, {orig_mtw_max}]")

        aug_dfs, n_aug = augment_donor(donor_df, config, donor_id)
        all_profiles.extend(aug_dfs)

        n_variants_phase = len([-1, 0, 1]) if config["phase_shift"] else 1
        print(f"  Generated: {config['n_scale']} scales × "
              f"{n_variants_phase} shifts × {config['n_noise']} noise = {n_aug} profiles")

        summary_rows.append({
            "Donor":              donor_id,
            "Original profiles":  1,
            "Augmented profiles": n_aug,
            "Total":              1 + n_aug,
            "months_to_wc range": f"[{orig_mtw_min}, {orig_mtw_max}]",
        })

    # ── Combine ──────────────────────────────────────────────────────────────
    df_all = pd.concat(all_profiles, ignore_index=True)
    df_all["ds"] = pd.to_datetime(df_all["ds"])

    # ── Validation ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Validation Checks")
    print("=" * 60)
    all_ok = True
    for donor_id in AUG_CONFIG.keys():
        mask = df_all["unique_id"].str.startswith(donor_id)
        subset = df_all[mask]

        y_min = float(subset["y"].min())
        y_max = float(subset["y"].max())
        mtw_min = int(subset["months_to_wc"].min())
        mtw_max = int(subset["months_to_wc"].max())

        curr_donor_df = df_input[df_input["unique_id"] == donor_id]
        orig_min = float(curr_donor_df["y"].min())
        orig_max = float(curr_donor_df["y"].max())

        y_ok = (y_min >= orig_min - 0.25) and (y_max <= orig_max + 0.25)
        status = "[OK]" if y_ok else "[FAIL]"
        if not y_ok:
            all_ok = False

        print(f"  {status} {donor_id}: "
              f"months_to_wc in [{mtw_min:+d}, {mtw_max:+d}]  |  "
              f"y in [{y_min:.4f}, {y_max:.4f}]")

    if all_ok:
        print("\n  All validation checks passed [OK]")
    else:
        print("\n  WARNING: Some validation checks failed [FAIL]")

    # ── Summary Table ─────────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    total_profiles = int(df_all["unique_id"].nunique())
    print("\n" + "=" * 60)
    print("  Augmentation Summary")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\n  >> Total profiles going into N-BEATSx training: {total_profiles}")
    print("=" * 60)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = f"{PROJECT_DIR}\\donor_residuals_augmented.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\n  Augmented dataset saved -> {out_path}")
    print(f"  Rows: {len(df_all):,}   Unique profiles: {total_profiles}")

    return df_all


# ── 5. Visualisation ──────────────────────────────────────────────────────────

def plot_augmentation_overview(df_all: pd.DataFrame):
    """
    One subplot per donor.  Augmented profiles in thin translucent lines;
    original profile in bold black.  Saved to augmentation_overview.png.
    """
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(
        "Augmented WC Residual Profiles by Donor Country",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, donor_id in zip(axes, AUG_CONFIG.keys()):
        color = DONOR_COLORS[donor_id]

        # Plot augmented profiles (thin, coloured, semi-transparent)
        aug_mask = (df_all["unique_id"].str.startswith(f"{donor_id}_aug_"))
        for uid, grp in df_all[aug_mask].groupby("unique_id"):
            grp_sorted = grp.sort_values("months_to_wc")
            ax.plot(
                grp_sorted["months_to_wc"], grp_sorted["y"],
                color=color, alpha=0.10, lw=0.7,
            )

        # Plot original profile (bold black)
        orig = df_all[df_all["unique_id"] == donor_id].sort_values("months_to_wc")
        ax.plot(
            orig["months_to_wc"], orig["y"],
            color="black", lw=2.5, label="Original", zorder=10,
        )

        # Annotations
        ax.axvline(0, color="red", ls="--", lw=1.2, alpha=0.7, label="WC start")
        ax.axhline(0, color="gray", ls="-", lw=0.5)
        ax.set_title(donor_id, fontsize=13, fontweight="bold", color=color)
        ax.set_xlabel("Months to WC start", fontsize=9)
        ax.set_ylabel("Normalised residual", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = f"{PROJECT_DIR}\\augmentation_overview.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved -> {plot_path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_augmented = run_augmentation_pipeline()
    plot_augmentation_overview(df_augmented)
