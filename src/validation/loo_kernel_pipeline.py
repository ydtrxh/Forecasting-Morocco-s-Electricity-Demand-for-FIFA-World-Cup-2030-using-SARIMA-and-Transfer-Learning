"""
loo_kernel_pipeline.py
================================================================
Research-Grade LOO Electricity Demand Event Transfer Pipeline
================================================================
Architecture: Local SARIMA counterfactuals + Global NeuralEventKernel
Goal: Learn a transferable WC uplift pulse shape, validate LOO,
      select best config for Morocco 2030 deployment.

30 configurations = 3 models × 2 weighting schemes × 5 lambda_amp values
================================================================
"""

import os
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from statsforecast.models import ARIMA

warnings.filterwarnings("ignore")

# ── Reproducibility ─────────────────────────────────────────────────────────────
GLOBAL_SEED = 42

def set_all_seeds(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_all_seeds()

# ── Paths ────────────────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\TS-project"
DATA_DIR    = r"C:\Users\Hp\OneDrive\Desktop\Data\Data Time series project"
BASELINE_CSV = os.path.join(PROJECT_DIR, "loo_baseline_summary.csv")

QUICK_TEST = False   # Set True to run reduced epochs for debugging

# ── Donors / Event dates ─────────────────────────────────────────────────────────
DONORS = ["QAT", "EGY", "RUS", "CMR", "ZAF"]

EVENT_START_DATES = {
    "QAT": pd.Timestamp("2022-11-01"),
    "EGY": pd.Timestamp("2019-06-01"),
    "RUS": pd.Timestamp("2018-06-01"),
    "CMR": pd.Timestamp("2022-01-01"),
    "ZAF": pd.Timestamp("2010-06-01"),
}

# Proportional weights (e.g., GDP-scaled, prior domain knowledge)
DONOR_WEIGHTS_RAW = {
    "QAT": 0.35,
    "EGY": 0.20,
    "RUS": 0.25,
    "CMR": 0.10,
    "ZAF": 0.10,
}

DONOR_WEIGHTS_UNIFORM = {d: 1.0 / len(DONORS) for d in DONORS}

DONOR_INDEX = {d: i for i, d in enumerate(DONORS)}

EVENT_TYPE = {d: "FIFA_WC" for d in DONORS}

SARIMA_ORDERS = {
    "ZAF": {"order": (0, 1, 1), "seasonal_order": (1, 0, 1)},
    "CMR": {"order": (0, 1, 0), "seasonal_order": (1, 0, 2)},
    "RUS": {"order": (1, 0, 2), "seasonal_order": (1, 0, 1)},
    "QAT": {"order": (0, 1, 2), "seasonal_order": (1, 0, 1)},
    "EGY": {"order": (0, 0, 0), "seasonal_order": (1, 1, 0)},
}

# ── Experimental Grid ────────────────────────────────────────────────────────────
KERNEL_CONFIGS = {
    "Model_A": {"hidden_dim": 4,  "n_layers": 1},
    "Model_B": {"hidden_dim": 8,  "n_layers": 1},
    "Model_C": {"hidden_dim": 16, "n_layers": 2},
}

LAMBDA_AMP_GRID   = [0.001, 0.01, 0.1, 1.0, 10.0]
WEIGHTING_SCHEMES = ["weighted", "uniform"]

MAX_EPOCHS      = 50  if QUICK_TEST else 200
PATIENCE        = 10  if QUICK_TEST else 20
NORMALIZE_EVERY = 5
REL_MONTHS      = np.arange(-5, 7)          # Event window: -5 … +6
T_NORM          = REL_MONTHS / 6.0          # Normalized: exact division


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 0 — DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_donor_series(iso: str) -> pd.Series:
    """Return a monthly GWh pd.Series indexed by period-start dates."""
    if iso == "ZAF":
        path = os.path.join(DATA_DIR, "south_africa_donor_electricity_demand.csv")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], format="%Y-%B")
        return df.set_index("date").sort_index()["Consumption_GWh"].asfreq("MS")

    elif iso == "CMR":
        path = os.path.join(DATA_DIR, "cameroon_monthly_electricity _consumption.csv")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], format="%Y-%B")
        return df.set_index("date").sort_index()["Consumption_GWh"].asfreq("MS")

    elif iso == "RUS":
        path = os.path.join(DATA_DIR, "Russia_data.csv")
        df = pd.read_csv(path, sep=";")
        df.columns = ["date_str", "consumption_kwh_bn"]
        df["date"] = pd.to_datetime(df["date_str"].str.replace("'", "20", regex=False), format="%b %Y")
        series_local = (df.set_index("date").sort_index()["consumption_kwh_bn"] * 1000).rename(iso)
        try:
            iea = pd.read_csv(os.path.join(DATA_DIR, "monthly_full_release_long_format.csv"))
            sub = iea[(iea["ISO 3 code"] == "RUS") & (iea["Variable"] == "Demand")].copy()
            sub["date"] = pd.to_datetime(sub["Date"])
            series_iea = (sub.set_index("date").sort_index()["Value"] * 1000)
        except Exception:
            series_iea = pd.Series(dtype=float)
        idx = series_local.index.union(series_iea.index)
        s = pd.Series(index=idx, dtype=float)
        s.update(series_iea); s.update(series_local)
        return s.asfreq("MS").rename(iso)

    elif iso == "QAT":
        path = os.path.join(DATA_DIR, "qatar_electricity_transmitted (1).csv")
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], format="%Y-%B")
        series_local = (df.set_index("date").sort_index()["Total_MWh"] / 1000).rename(iso)
        try:
            iea = pd.read_csv(os.path.join(DATA_DIR, "monthly_full_release_long_format.csv"))
            sub = iea[(iea["ISO 3 code"] == "QAT") & (iea["Variable"] == "Demand")].copy()
            sub["date"] = pd.to_datetime(sub["Date"])
            series_iea = (sub.set_index("date").sort_index()["Value"] * 1000)
        except Exception:
            series_iea = pd.Series(dtype=float)
        idx = series_local.index.union(series_iea.index)
        s = pd.Series(index=idx, dtype=float)
        s.update(series_iea); s.update(series_local)
        return s.asfreq("MS").rename(iso)

    elif iso == "EGY":
        iea = pd.read_csv(os.path.join(DATA_DIR, "monthly_full_release_long_format.csv"))
        sub = iea[(iea["ISO 3 code"] == "EGY") & (iea["Variable"] == "Demand")].copy()
        sub["date"] = pd.to_datetime(sub["Date"])
        return (sub.set_index("date").sort_index()["Value"] * 1000).asfreq("MS").rename(iso)

    else:
        raise ValueError(f"Unknown donor: {iso}")


def normalize_weights(weights_raw: dict, active_donors: list) -> dict:
    """Re-normalize weights for active donors so they sum to 1.0."""
    sub = {d: weights_raw[d] for d in active_donors if d in weights_raw}
    total = sum(sub.values())
    return {d: v / total for d, v in sub.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — BUILD LOCAL SARIMA COUNTERFACTUALS
# ═══════════════════════════════════════════════════════════════════════════════

def build_sarima_counterfactual(iso: str, series: pd.Series) -> np.ndarray:
    """
    Fit SARIMA on log1p(series) up to month -6, forecast 12 months.
    Returns counterfactual_GWh for relative months [-5, +6].
    Hard stop: training cutoff = event_start - 6 months.
    """
    event_start = EVENT_START_DATES[iso]
    cutoff      = event_start - pd.DateOffset(months=6)

    train = series[series.index <= cutoff].dropna()
    train_log = np.log1p(train.values)

    orders = SARIMA_ORDERS[iso]
    model  = ARIMA(
        order=orders["order"],
        seasonal_order=orders["seasonal_order"],
        season_length=12,
    )
    model.fit(train_log)
    fcst_log = model.predict(h=12)["mean"]           # 12 steps forward
    return np.expm1(fcst_log)                        # shape (12,)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD UPLIFT TARGET TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def build_uplift_table(
    active_donors: list,
    all_series: dict,
    all_counterfactuals: dict,
) -> pd.DataFrame:
    """
    Pool uplift targets across active donors.
    uplift_target(t) = log1p(actual_GWh(t)) - log1p(counterfactual_GWh(t))
    No abs(). No sklearn scalers.
    """
    rows = []
    for iso in active_donors:
        series = all_series[iso]
        cf_gwh = all_counterfactuals[iso]   # shape (12,)
        event_start = EVENT_START_DATES[iso]

        eval_dates = [event_start + pd.DateOffset(months=int(m) - 5) for m in range(12)]
        actual_vals = series.reindex(pd.DatetimeIndex(eval_dates)).values

        for k, (rel_m, t_n) in enumerate(zip(REL_MONTHS, T_NORM)):
            act = actual_vals[k]
            cf  = cf_gwh[k]
            if np.isnan(act) or np.isnan(cf):
                continue
            uplift = np.log1p(act) - np.log1p(cf)
            rows.append({
                "country_id":    iso,
                "donor_idx":     DONOR_INDEX[iso],
                "rel_month":     int(rel_m),
                "t_normalized":  float(t_n),
                "uplift_target": float(uplift),
                "actual_GWh":    float(act),
                "cf_GWh":        float(cf),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — NEURAL EVENT KERNEL
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralEventKernel(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int, n_donors: int):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.shape_net = nn.Sequential(*layers)

        self.raw_amplitude = nn.Parameter(torch.zeros(n_donors))
        # Registered buffer: included in state_dict so save/restore preserves normalization invariance
        self.register_buffer("_shape_scale", torch.tensor(1.0, dtype=torch.float32))

    @property
    def amplitude(self):
        return torch.exp(self.raw_amplitude)

    def forward(self, t_normalized: torch.Tensor) -> torch.Tensor:
        return self.shape_net(t_normalized) / self._shape_scale.item()

    def normalize_shape_epoch(self):
        """
        Seamless normalization: preserves a_i * f_θ(t) exactly.
        _shape_scale is a registered buffer, so it is included in state_dict.
        adjustment = max_val / _shape_scale.item()
        raw_amplitude += log(adjustment) => exp(raw)*f/new_scale == exp(raw)*f/old_scale
        """
        with torch.no_grad():
            device = self.raw_amplitude.device
            t_seq  = torch.tensor(
                [t / 6.0 for t in range(-5, 7)],
                dtype=torch.float32,
                device=device
            ).unsqueeze(1)
            shape_vals = self.shape_net(t_seq)
            max_val    = shape_vals.abs().max().item()
            max_val    = np.clip(max_val, 1e-3, 100.0)

            if max_val > 1e-6:
                adjustment = max_val / self._shape_scale.item()
                self.raw_amplitude.data += torch.log(
                    torch.tensor(adjustment, dtype=torch.float32, device=device)
                )
                self._shape_scale.fill_(max_val)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — TRAINING OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════════════

def training_loss(
    kernel: NeuralEventKernel,
    t_norm: torch.Tensor,
    targets: torch.Tensor,
    donor_indices: torch.Tensor,
    weights: torch.Tensor,
    lambda_smooth: float = 0.01,
    lambda_amp:    float = 0.1,
) -> torch.Tensor:
    shape_output = kernel(t_norm).squeeze()
    amplitudes   = kernel.amplitude[donor_indices]
    predictions  = amplitudes * shape_output

    # Normalize by weight sum — keeps loss magnitude stable across folds/configs
    # (donor weights sum to 1.0 not N, so plain mean() would scale with sample count)
    mse_loss = torch.sum(weights * (predictions - targets) ** 2) / (torch.sum(weights) + 1e-8)

    # Smoothness regularizer on the full event grid
    t_seq    = torch.tensor(T_NORM, dtype=torch.float32, device=t_norm.device).unsqueeze(1)
    beta_seq = kernel(t_seq).squeeze()
    smoothness = lambda_smooth * torch.sum(
        (beta_seq[2:] - 2 * beta_seq[1:-1] + beta_seq[:-2]) ** 2
    )

    # Hybrid amplitude penalty: center around 1 + mild shrinkage
    amp_penalty = (
        lambda_amp * torch.mean(kernel.raw_amplitude ** 2)
        + 0.01 * torch.mean(kernel.amplitude ** 2)
    )

    return mse_loss + smoothness + amp_penalty


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train_kernel(
    kernel: NeuralEventKernel,
    t_norm: torch.Tensor,
    targets: torch.Tensor,
    donor_indices: torch.Tensor,
    weights: torch.Tensor,
    lambda_smooth:    float = 0.01,
    lambda_amp:       float = 0.1,
    max_epochs:       int   = MAX_EPOCHS,
    patience:         int   = PATIENCE,
    normalize_every:  int   = NORMALIZE_EVERY,
    seed:             int   = GLOBAL_SEED,
) -> NeuralEventKernel:
    set_all_seeds(seed)

    optimizer = torch.optim.Adam(kernel.parameters(), lr=1e-3, weight_decay=1e-3)

    best_loss       = float("inf")
    best_state      = None
    patience_counter = 0
    loss_history    = []

    for epoch in range(max_epochs):
        kernel.train()
        optimizer.zero_grad()

        loss = training_loss(
            kernel, t_norm, targets, donor_indices, weights,
            lambda_smooth, lambda_amp
        )
        loss.backward()

        # NaN/Inf safety guard — protects long grid searches from exploding losses
        if not torch.isfinite(loss):
            print(f"  WARNING: non-finite loss at epoch {epoch} — stopping early")
            break

        # Gradient clipping — prevents amplitude explosion in early training
        torch.nn.utils.clip_grad_norm_(kernel.parameters(), max_norm=5.0)

        optimizer.step()

        # Sparse seamless normalization
        if (epoch + 1) % normalize_every == 0 or epoch == max_epochs - 1:
            kernel.normalize_shape_epoch()

        lv = loss.item()
        loss_history.append(lv)

        # Smoothed early stopping
        check_loss = np.mean(loss_history[-normalize_every:]) if len(loss_history) >= normalize_every else lv

        if check_loss < best_loss - 1e-6:
            best_loss        = check_loss
            patience_counter = 0
            best_state       = {k: v.detach().clone() for k, v in kernel.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        kernel.load_state_dict(best_state)

    # Final normalization after restore — ensures max|f_θ| == 1.0 exactly
    kernel.normalize_shape_epoch()
    return kernel


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — TRANSFER AMPLITUDE & RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_transfer_amplitude(
    kernel: NeuralEventKernel,
    active_donors: list,
    weights_norm: dict,
) -> float:
    """Weighted average of learned amplitudes from training donors."""
    kernel.eval()
    a_transfer = 0.0
    with torch.no_grad():
        for d in active_donors:
            idx = DONOR_INDEX[d]
            a_d = kernel.amplitude[idx].item()
            a_transfer += weights_norm[d] * a_d
    return a_transfer


def reconstruct_forecast(
    counterfactual_GWh: np.ndarray,
    a_transfer: float,
    shape_output: np.ndarray,
) -> np.ndarray:
    """
    Correct reconstruction (exact formula, never simplified):
        predicted_GWh(t) = expm1(log1p(counterfactual_GWh(t)) + a_i * f_θ(t_norm))
    """
    log_cf = np.log1p(counterfactual_GWh)
    return np.expm1(log_cf + a_transfer * shape_output)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — QUERY KERNEL SHAPE
# ═══════════════════════════════════════════════════════════════════════════════

def query_kernel_shape(kernel: NeuralEventKernel) -> np.ndarray:
    """Evaluate f_θ(t_normalized) for t in [-5, +6]."""
    kernel.eval()
    with torch.no_grad():
        t_seq = torch.tensor(T_NORM, dtype=torch.float32).unsqueeze(1)
        return kernel(t_seq).squeeze().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(
    actual_GWh: np.ndarray,
    predicted_GWh: np.ndarray,
    counterfactual_GWh: np.ndarray,
    uplift_target: np.ndarray,
    shape_output: np.ndarray,
    sarima_rmse: float,
    baseline_rmse: float,
) -> dict:
    valid = ~np.isnan(actual_GWh)
    act   = actual_GWh[valid]
    pred  = predicted_GWh[valid]
    cf    = counterfactual_GWh[valid]

    rmse = float(np.sqrt(np.mean((act - pred) ** 2)))
    mae  = float(np.mean(np.abs(act - pred)))
    mape = float(np.mean(np.abs((act - pred) / act)) * 100)

    improvement_vs_sarima   = (sarima_rmse   - rmse) / sarima_rmse   * 100
    improvement_vs_baseline = (baseline_rmse - rmse) / baseline_rmse * 100

    # Shape MSE with epsilon guard
    denom_actual = max(float(np.max(np.abs(uplift_target[valid]))), 1e-6)
    denom_kernel = max(float(np.max(np.abs(shape_output[valid]))), 1e-6)
    actual_norm  = uplift_target[valid] / denom_actual
    kernel_norm  = shape_output[valid]  / denom_kernel
    shape_mse    = float(np.mean((kernel_norm - actual_norm) ** 2))

    return {
        "RMSE": rmse,
        "MAE":  mae,
        "MAPE": mape,
        "shape_MSE": shape_mse,
        "improvement_vs_sarima_pct":   improvement_vs_sarima,
        "improvement_vs_baseline_pct": improvement_vs_baseline,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOO PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Research-Grade LOO Kernel Pipeline — Morocco 2030 WC Forecasting")
    print("=" * 70)

    # Load baselines
    df_baseline = pd.read_csv(BASELINE_CSV).set_index("donor")

    # Pre-load all series
    print("\nLoading all donor series...")
    all_series = {iso: load_donor_series(iso) for iso in DONORS}

    # Pre-compute all SARIMA counterfactuals (once, used across all folds/configs)
    print("Computing SARIMA counterfactuals...")
    all_counterfactuals = {}
    all_sarima_rmse     = {}
    for iso in DONORS:
        cf = build_sarima_counterfactual(iso, all_series[iso])
        all_counterfactuals[iso] = cf

        # SARIMA RMSE over eval window
        event_start = EVENT_START_DATES[iso]
        eval_dates  = [event_start + pd.DateOffset(months=int(m) - 5) for m in range(12)]
        actual_vals = all_series[iso].reindex(pd.DatetimeIndex(eval_dates)).values
        valid       = ~np.isnan(actual_vals)
        sarima_r    = float(np.sqrt(np.mean((actual_vals[valid] - cf[valid]) ** 2)))
        all_sarima_rmse[iso] = sarima_r
        print(f"  {iso}: SARIMA_RMSE={sarima_r:.1f} GWh")

    # Containers for results
    fold_records     = []   # per-fold, per-config
    config_shapes    = {k: {} for k in KERNEL_CONFIGS}   # config → {donor: shape_array}

    # Build full config grid (30 configs)
    all_configs = []
    for model_name, kparams in KERNEL_CONFIGS.items():
        for scheme in WEIGHTING_SCHEMES:
            for lam in LAMBDA_AMP_GRID:
                cfg_id = f"{model_name}_{scheme}_lam{lam}"
                all_configs.append({
                    "config": cfg_id,
                    "model":  model_name,
                    "weighting": scheme,
                    "lambda_amp": lam,
                    **kparams,
                })

    print(f"\nTotal configurations: {len(all_configs)}")
    print(f"Total folds: {len(DONORS)}")
    print(f"Total training runs: {len(all_configs) * len(DONORS)}\n")

    # ── Main LOO loop ──────────────────────────────────────────────────────────
    for fold_idx, left_out in enumerate(DONORS):
        print(f"\n{'='*70}")
        print(f"  FOLD {fold_idx+1}/5 — Left-out: {left_out}")
        print(f"{'='*70}")

        active_donors = [d for d in DONORS if d != left_out]
        n_active      = len(active_donors)

        # Uplift table for this fold (training donors only)
        uplift_df = build_uplift_table(active_donors, all_series, all_counterfactuals)
        print(f"  Uplift table: {len(uplift_df)} rows from {n_active} donors")

        # Ground truth for left-out donor
        event_start_lo = EVENT_START_DATES[left_out]
        eval_dates_lo  = [event_start_lo + pd.DateOffset(months=int(m) - 5) for m in range(12)]
        actual_GWh_lo  = all_series[left_out].reindex(pd.DatetimeIndex(eval_dates_lo)).values
        cf_GWh_lo      = all_counterfactuals[left_out]
        sarima_r_lo    = all_sarima_rmse[left_out]
        baseline_r_lo  = float(df_baseline.loc[left_out, "RMSE"])

        # Uplift target for left-out (for shape MSE evaluation only)
        uplift_lo = np.array([
            np.log1p(a) - np.log1p(c) if not (np.isnan(a) or np.isnan(c)) else np.nan
            for a, c in zip(actual_GWh_lo, cf_GWh_lo)
        ])

        # ── Config loop ────────────────────────────────────────────────────────
        for cfg_idx, cfg in enumerate(all_configs):
            cfg_id      = cfg["config"]
            model_name  = cfg["model"]
            scheme      = cfg["weighting"]
            lam         = cfg["lambda_amp"]
            hidden_dim  = cfg["hidden_dim"]
            n_layers    = cfg["n_layers"]

            if cfg_idx % 10 == 0:
                print(f"  Config {cfg_idx+1}/{len(all_configs)}: {cfg_id}")

            # Weight scheme
            raw_weights = DONOR_WEIGHTS_RAW if scheme == "weighted" else DONOR_WEIGHTS_UNIFORM
            weights_norm = normalize_weights(raw_weights, active_donors)

            # Build tensors for training
            t_norm_arr  = uplift_df["t_normalized"].values.astype(np.float32)
            targets_arr = uplift_df["uplift_target"].values.astype(np.float32)
            didx_arr    = uplift_df["donor_idx"].values.astype(np.int64)
            weights_arr = np.array(
                [weights_norm.get(uplift_df["country_id"].iloc[k], 1.0 / n_active)
                 for k in range(len(uplift_df))],
                dtype=np.float32,
            )

            t_norm_t  = torch.tensor(t_norm_arr).unsqueeze(1)
            targets_t = torch.tensor(targets_arr)
            didx_t    = torch.tensor(didx_arr, dtype=torch.long)
            weights_t = torch.tensor(weights_arr)

            # Instantiate kernel (per-fold seeding for reproducibility)
            fold_seed = GLOBAL_SEED + fold_idx * 100 + cfg_idx
            set_all_seeds(fold_seed)

            kernel = NeuralEventKernel(
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_donors=len(DONORS),
            )

            # Train
            kernel = train_kernel(
                kernel, t_norm_t, targets_t, didx_t, weights_t,
                lambda_smooth=0.01,
                lambda_amp=lam,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                normalize_every=NORMALIZE_EVERY,
                seed=fold_seed,
            )

            # Transfer amplitude (weighted average of training donor amplitudes)
            a_transfer = compute_transfer_amplitude(kernel, active_donors, weights_norm)

            # Query pulse shape
            shape_output = query_kernel_shape(kernel)   # (12,)

            # Reconstruct forecast for left-out
            predicted_GWh_lo = reconstruct_forecast(cf_GWh_lo, a_transfer, shape_output)

            # Metrics
            metrics = compute_metrics(
                actual_GWh_lo, predicted_GWh_lo, cf_GWh_lo,
                uplift_lo, shape_output,
                sarima_r_lo, baseline_r_lo,
            )

            # Amplitude scalars (learned)
            amp_dict = {}
            kernel.eval()
            with torch.no_grad():
                for d in active_donors:
                    amp_dict[d] = float(kernel.amplitude[DONOR_INDEX[d]].item())
            log_amps = [kernel.raw_amplitude[DONOR_INDEX[d]].item() for d in active_donors]
            mean_abs_log_amp = float(np.mean(np.abs(log_amps)))
            # Identifiability diagnostic: large values signal shape collapse / amplitude explosion
            max_abs_log_amp  = float(np.max(np.abs(log_amps)))

            record = {
                "config":       cfg_id,
                "model":        model_name,
                "weighting":    scheme,
                "lambda_amp":   lam,
                "donor":        left_out,
                "RMSE":         metrics["RMSE"],
                "MAPE":         metrics["MAPE"],
                "shape_MSE":    metrics["shape_MSE"],
                "SARIMA_RMSE":  sarima_r_lo,
                "baseline_RMSE": baseline_r_lo,
                "improvement_vs_sarima_pct":   metrics["improvement_vs_sarima_pct"],
                "improvement_vs_baseline_pct": metrics["improvement_vs_baseline_pct"],
                "a_transfer":        a_transfer,
                "mean_abs_log_amplitude": mean_abs_log_amp,
                "max_abs_log_amplitude":  max_abs_log_amp,   # identifiability diagnostic
                **{f"a_{d}": amp_dict.get(d, np.nan) for d in DONORS},
            }
            fold_records.append(record)

            # Store shape for visualization
            if left_out not in config_shapes[model_name]:
                config_shapes[model_name][left_out] = {}
            config_shapes[model_name][left_out][cfg_id] = shape_output.copy()

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 10 — SUMMARY TABLES
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n\nBuilding summary tables...")
    df_fold = pd.DataFrame(fold_records)

    # Fold-level stability metrics
    fold_rmse_std = df_fold.groupby("config")["RMSE"].std().rename("fold_RMSE_std")
    donor_sens    = (df_fold.groupby("config")["RMSE"].max()
                    - df_fold.groupby("config")["RMSE"].min()).rename("donor_sensitivity")

    # Per-config pulse variance (across folds, across time)
    # For each config, collect all shape arrays per fold
    pulse_var_per_config = {}
    for cfg in all_configs:
        cfg_id  = cfg["config"]
        mname   = cfg["model"]
        shapes  = []
        for donor in DONORS:
            s = config_shapes[mname].get(donor, {}).get(cfg_id)
            if s is not None:
                shapes.append(s)
        if shapes:
            shapes_arr = np.array(shapes)    # (n_folds, 12)
            pulse_var_per_config[cfg_id] = float(np.mean(np.var(shapes_arr, axis=0)))
        else:
            pulse_var_per_config[cfg_id] = np.nan

    df_pv = pd.Series(pulse_var_per_config, name="mean_pulse_variance")

    # Aggregate table
    agg = df_fold.groupby(["config", "model", "weighting", "lambda_amp"]).agg(
        mean_RMSE=("RMSE", "mean"),
        mean_MAPE=("MAPE", "mean"),
        mean_shape_MSE=("shape_MSE", "mean"),
        mean_abs_log_amplitude=("mean_abs_log_amplitude", "mean"),
        max_abs_log_amplitude=("max_abs_log_amplitude", "mean"),
    ).reset_index()

    agg = (agg
           .join(fold_rmse_std, on="config")
           .join(donor_sens, on="config")
           .join(df_pv, on="config"))

    # Add fold_RMSE_std also to fold-level records
    df_fold = df_fold.join(fold_rmse_std, on="config")

    # Save
    df_fold_out = df_fold[[
        "config", "model", "weighting", "lambda_amp", "donor",
        "RMSE", "MAPE", "shape_MSE", "SARIMA_RMSE", "baseline_RMSE",
        "improvement_vs_sarima_pct", "improvement_vs_baseline_pct",
        "a_transfer", "fold_RMSE_std",
        "mean_abs_log_amplitude", "max_abs_log_amplitude",
    ]]
    df_fold_out.to_csv(os.path.join(PROJECT_DIR, "loo_kernel_summary.csv"), index=False)
    agg.to_csv(os.path.join(PROJECT_DIR, "loo_kernel_aggregate.csv"), index=False)
    print("Saved: loo_kernel_summary.csv, loo_kernel_aggregate.csv")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 11 — DECISION LOGIC
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  DECISION LOGIC")
    print("=" * 70)

    # Primary: best mean_RMSE
    best_idx    = agg["mean_RMSE"].idxmin()
    best_config = agg.loc[best_idx]
    print(f"\nBest by mean_RMSE: {best_config['config']}")
    print(f"  mean_RMSE={best_config['mean_RMSE']:.1f} | mean_MAPE={best_config['mean_MAPE']:.2f}%"
          f" | mean_pulse_variance={best_config['mean_pulse_variance']:.4f}")

    # Overfitting check: Model_C vs Model_A stability
    for lam in LAMBDA_AMP_GRID:
        for scheme in WEIGHTING_SCHEMES:
            r_a = agg.loc[(agg["model"] == "Model_A") & (agg["lambda_amp"] == lam) & (agg["weighting"] == scheme), "fold_RMSE_std"]
            r_c = agg.loc[(agg["model"] == "Model_C") & (agg["lambda_amp"] == lam) & (agg["weighting"] == scheme), "fold_RMSE_std"]
            if not r_a.empty and not r_c.empty:
                if r_c.values[0] > 2 * r_a.values[0]:
                    print(f"  WARNING: Model_C overfitting at lambda_amp={lam}, {scheme}")

    # Stability-adjusted selection (within 5% of best RMSE)
    best_rmse   = agg["mean_RMSE"].min()
    candidates  = agg[agg["mean_RMSE"] <= 1.05 * best_rmse].copy()
    stable_best = candidates.loc[candidates["mean_pulse_variance"].idxmin()]
    print(f"\nStability-adjusted best (within 5% RMSE window): {stable_best['config']}")
    print(f"  mean_RMSE={stable_best['mean_RMSE']:.1f} | mean_pulse_variance={stable_best['mean_pulse_variance']:.4f}")

    # Weighting insight
    w_agg = agg.groupby("weighting")["mean_RMSE"].mean()
    print(f"\nWeighting RMSE comparison:")
    for scheme, val in w_agg.items():
        print(f"  {scheme}: mean_RMSE={val:.1f}")

    # Architecture insight
    m_agg = agg.groupby("model")["mean_RMSE"].mean()
    print(f"\nArchitecture RMSE comparison:")
    for mname, val in m_agg.items():
        print(f"  {mname}: mean_RMSE={val:.1f}")

    print(f"\nFINAL RECOMMENDATION for Morocco 2030: {stable_best['config']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 9 — VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\nGenerating visualizations...")

    # ─ Plot 1: loo_forecast_grid.png ─────────────────────────────────────────
    # Top config per fold
    top_cfg = best_config["config"]
    df_top  = df_fold[df_fold["config"] == top_cfg]

    fig1, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=False)
    for ax, iso in zip(axes, DONORS):
        cf_gwh   = all_counterfactuals[iso]
        shape_k  = config_shapes[best_config["model"]].get(iso, {}).get(top_cfg, np.zeros(12))
        row      = df_top[df_top["donor"] == iso]
        a_tr     = row["a_transfer"].values[0] if len(row) else 0.0
        pred_gwh = reconstruct_forecast(cf_gwh, a_tr, shape_k)

        event_start = EVENT_START_DATES[iso]
        eval_dates  = [event_start + pd.DateOffset(months=int(m) - 5) for m in range(12)]
        actual_gwh  = all_series[iso].reindex(pd.DatetimeIndex(eval_dates)).values

        ax.plot(REL_MONTHS, actual_gwh, "-k",  lw=1.8, label="Actual")
        ax.plot(REL_MONTHS, cf_gwh,     "--b", lw=1.2, label="SARIMA CF")
        ax.plot(REL_MONTHS, pred_gwh,   "-r",  lw=1.5, label="Kernel Pred")
        ax.axvspan(-0.5, 1.5, color="tomato", alpha=0.10)
        ax.axvline(0, ls=":", lw=0.8, color="gray")
        ax.set_xlim(-5, 6)
        ax.set_xlabel("Months rel. WC", fontsize=8)
        if len(row):
            r  = row["RMSE"].values[0]
            mp = row["MAPE"].values[0]
            ax.set_title(f"{iso}\nRMSE={r:.0f} | MAPE={mp:.1f}%", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("GWh", fontsize=8)
            ax.legend(fontsize=7)

    fig1.suptitle(f"LOO Forecast Grid — {top_cfg}", fontsize=9)
    fig1.tight_layout(pad=1.5)
    fig1.savefig(os.path.join(PROJECT_DIR, "loo_forecast_grid.png"), dpi=150)
    plt.close(fig1)
    print("  Saved: loo_forecast_grid.png")

    # ─ Plot 2: kernel_pulse_overlay.png ──────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax2, mname in zip(axes2, KERNEL_CONFIGS.keys()):
        # Plot one representative config per model (lowest lambda_amp, weighted)
        rep_cfg = f"{mname}_weighted_lam{LAMBDA_AMP_GRID[0]}"
        colors  = plt.cm.tab10(np.linspace(0, 1, 5))
        all_shape_arr = []
        for k, iso in enumerate(DONORS):
            s = config_shapes[mname].get(iso, {}).get(rep_cfg)
            if s is not None:
                ax2.plot(REL_MONTHS, s, color=colors[k], lw=1.2, alpha=0.7, label=iso)
                all_shape_arr.append(s)
        if all_shape_arr:
            mean_shape = np.mean(all_shape_arr, axis=0)
            pv_row     = agg.loc[agg["config"] == rep_cfg, "mean_pulse_variance"]
            pv_val     = pv_row.values[0] if len(pv_row) else np.nan
            ax2.plot(REL_MONTHS, mean_shape, "-k", lw=2.5, label="Mean")
            ax2.set_title(f"{mname}\nmean_pulse_var={pv_val:.4f}", fontsize=9)
        ax2.axvline(0, ls=":", lw=0.9, color="gray")
        ax2.set_xlim(-5, 6)
        ax2.set_ylim(-1.2, 1.2)
        ax2.set_xlabel("Months rel. WC", fontsize=8)
        ax2.legend(fontsize=7)
    axes2[0].set_ylabel("f_θ (normalized)", fontsize=8)
    fig2.suptitle("Kernel Pulse Shape Overlay per Fold — Representative Configs", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(os.path.join(PROJECT_DIR, "kernel_pulse_overlay.png"), dpi=150)
    plt.close(fig2)
    print("  Saved: kernel_pulse_overlay.png")

    # ─ Plot 3: kernel_shape_vs_actual.png ────────────────────────────────────
    fig3, axes3 = plt.subplots(1, 5, figsize=(22, 4), sharey=False)
    for ax3, iso in zip(axes3, DONORS):
        event_start = EVENT_START_DATES[iso]
        eval_dates  = [event_start + pd.DateOffset(months=int(m) - 5) for m in range(12)]
        actual_gwh  = all_series[iso].reindex(pd.DatetimeIndex(eval_dates)).values
        cf_gwh      = all_counterfactuals[iso]
        uplift_act  = np.log1p(actual_gwh) - np.log1p(cf_gwh)
        denom       = max(np.nanmax(np.abs(uplift_act)), 1e-6)
        uplift_norm = uplift_act / denom

        rep_cfg = f"Model_A_weighted_lam{LAMBDA_AMP_GRID[0]}"
        shape_k = config_shapes["Model_A"].get(iso, {}).get(rep_cfg, np.zeros(12))
        denom_k = max(np.max(np.abs(shape_k)), 1e-6)
        shape_k_norm = shape_k / denom_k

        ax3.plot(REL_MONTHS, uplift_norm,  "-k",  lw=1.8, label="Actual uplift (norm)")
        ax3.plot(REL_MONTHS, shape_k_norm, "--r", lw=1.5, label="Kernel shape (norm)")
        ax3.axvline(0, ls=":", lw=0.8, color="gray")
        ax3.set_xlim(-5, 6); ax3.set_ylim(-1.4, 1.4)
        ax3.set_title(iso, fontsize=9)
        ax3.set_xlabel("Months rel. WC", fontsize=8)
        if ax3 is axes3[0]:
            ax3.set_ylabel("Normalized", fontsize=8)
            ax3.legend(fontsize=7)

    fig3.suptitle("Normalized Kernel Shape vs Actual Uplift per Fold", fontsize=9)
    fig3.tight_layout()
    fig3.savefig(os.path.join(PROJECT_DIR, "kernel_shape_vs_actual.png"), dpi=150)
    plt.close(fig3)
    print("  Saved: kernel_shape_vs_actual.png")

    # ─ Plot 4: amplitude_scalars.png ─────────────────────────────────────────
    rep_cfg  = f"Model_A_weighted_lam{LAMBDA_AMP_GRID[0]}"
    df_rep   = df_fold[df_fold["config"] == rep_cfg]
    a_cols   = [f"a_{d}" for d in DONORS if f"a_{d}" in df_fold.columns]
    if a_cols:
        fig4, axes4 = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
        for ax4, iso in zip(axes4, DONORS):
            row = df_rep[df_rep["donor"] != iso]   # training donors for this fold
            vals = [row[f"a_{d}"].mean() for d in DONORS if f"a_{d}" in row.columns]
            lbls = [d for d in DONORS]
            ax4.bar(lbls, [row[f"a_{d}"].mean() if f"a_{d}" in row.columns else 0 for d in DONORS],
                    color=plt.cm.tab10(np.linspace(0, 1, 5)))
            ax4.set_title(f"Left-out: {iso}", fontsize=8)
            ax4.set_ylabel("Amplitude a_i" if ax4 is axes4[0] else "")
            ax4.set_xticklabels(lbls, fontsize=7)
        fig4.suptitle(f"Learned Amplitude Scalars per Fold — {rep_cfg}", fontsize=9)
        fig4.tight_layout()
        fig4.savefig(os.path.join(PROJECT_DIR, "amplitude_scalars.png"), dpi=150)
        plt.close(fig4)
        print("  Saved: amplitude_scalars.png")

    # ─ Plot 5: lambda_amp_sensitivity.png ────────────────────────────────────
    fig5, ax5 = plt.subplots(figsize=(9, 6))
    for scheme in WEIGHTING_SCHEMES:
        for mname in KERNEL_CONFIGS:
            sub = agg[(agg["model"] == mname) & (agg["weighting"] == scheme)].sort_values("lambda_amp")
            ls = "-" if scheme == "weighted" else "--"
            mk = "o" if mname == "Model_A" else ("s" if mname == "Model_B" else "^")
            ax5.plot(
                sub["lambda_amp"], sub["mean_RMSE"],
                linestyle=ls, marker=mk, label=f"{mname}/{scheme}"
            )
    ax5.set_xscale("log")
    ax5.set_xlabel("lambda_amp", fontsize=10)
    ax5.set_ylabel("Mean RMSE (GWh)", fontsize=10)
    ax5.set_title("RMSE vs Lambda_amp Sensitivity", fontsize=10)
    ax5.legend(fontsize=8, loc="upper right")
    ax5.grid(True, alpha=0.3)
    fig5.tight_layout()
    fig5.savefig(os.path.join(PROJECT_DIR, "lambda_amp_sensitivity.png"), dpi=150)
    plt.close(fig5)
    print("  Saved: lambda_amp_sensitivity.png")

    print("\n" + "=" * 70)
    print("  Pipeline complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
