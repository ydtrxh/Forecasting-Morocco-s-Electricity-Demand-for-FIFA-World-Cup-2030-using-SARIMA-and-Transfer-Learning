"""
train_final_kernel.py
======================
Trains the final NeuralEventKernel (Model C: 1→16→16→1, Tanh) on ALL 5 donor
countries without any LOO holdout. Saves the state dict to kernel_final_modelC.pt
for use in the Morocco 2030 deployment pipeline.

Best config from grid search: lambda_amp=0.001, weighted scheme.
DONORS_ORDERED matches the deployment script exactly.
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

# ── Reproducibility ──────────────────────────────────────────────────────────
GLOBAL_SEED = 42

def set_all_seeds(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_all_seeds()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\TS-project"
DATA_DIR    = r"C:\Users\Hp\OneDrive\Desktop\Data\Data Time series project"

# ── MUST match deployment script order exactly ───────────────────────────────
DONORS_ORDERED = ["QAT", "EGY", "ZAF", "CMR", "RUS"]

DONOR_WEIGHTS_RAW = {
    "QAT": 1.0,
    "EGY": 0.7,
    "ZAF": 0.6,
    "CMR": 0.5,
    "RUS": 0.3,
}

EVENT_START_DATES = {
    "QAT": pd.Timestamp("2022-11-01"),
    "EGY": pd.Timestamp("2019-06-01"),
    "RUS": pd.Timestamp("2018-06-01"),
    "CMR": pd.Timestamp("2022-01-01"),
    "ZAF": pd.Timestamp("2010-06-01"),
}

SARIMA_ORDERS = {
    "ZAF": {"order": (0, 1, 1), "seasonal_order": (1, 0, 1)},
    "CMR": {"order": (0, 1, 0), "seasonal_order": (1, 0, 2)},
    "RUS": {"order": (1, 0, 2), "seasonal_order": (1, 0, 1)},
    "QAT": {"order": (0, 1, 2), "seasonal_order": (1, 0, 1)},
    "EGY": {"order": (0, 0, 0), "seasonal_order": (1, 1, 0)},
}

DONOR_INDEX   = {d: i for i, d in enumerate(DONORS_ORDERED)}
REL_MONTHS    = np.arange(-5, 7)
T_NORM        = REL_MONTHS / 6.0

MAX_EPOCHS      = 300   # Full training — no test holdout
PATIENCE        = 30
NORMALIZE_EVERY = 5

# Best config from LOO grid search
LAMBDA_AMP    = 0.001
LAMBDA_SMOOTH = 0.01
HIDDEN_DIM    = 16
N_LAYERS      = 2


# ── Data loading ─────────────────────────────────────────────────────────────
def load_donor_series(iso):
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
            idx = series_local.index.union(series_iea.index)
            s = pd.Series(index=idx, dtype=float)
            s.update(series_iea); s.update(series_local)
            return s.asfreq("MS").rename(iso)
        except Exception:
            return series_local.asfreq("MS")

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
            idx = series_local.index.union(series_iea.index)
            s = pd.Series(index=idx, dtype=float)
            s.update(series_iea); s.update(series_local)
            return s.asfreq("MS").rename(iso)
        except Exception:
            return series_local.asfreq("MS")

    elif iso == "EGY":
        iea = pd.read_csv(os.path.join(DATA_DIR, "monthly_full_release_long_format.csv"))
        sub = iea[(iea["ISO 3 code"] == "EGY") & (iea["Variable"] == "Demand")].copy()
        sub["date"] = pd.to_datetime(sub["Date"])
        return (sub.set_index("date").sort_index()["Value"] * 1000).asfreq("MS").rename(iso)

    raise ValueError(f"Unknown donor: {iso}")


def build_sarima_counterfactual(iso, series):
    event_start = EVENT_START_DATES[iso]
    cutoff = event_start - pd.DateOffset(months=6)
    train = series[series.index <= cutoff].dropna()
    train_log = np.log1p(train.values)
    orders = SARIMA_ORDERS[iso]
    model = ARIMA(order=orders["order"], seasonal_order=orders["seasonal_order"], season_length=12)
    model.fit(train_log)
    fcst_log = model.predict(h=12)["mean"]
    return np.expm1(fcst_log)


def build_uplift_table(all_series, all_counterfactuals):
    rows = []
    for iso in DONORS_ORDERED:
        series = all_series[iso]
        cf_gwh = all_counterfactuals[iso]
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
            })
    return pd.DataFrame(rows)


# ── NeuralEventKernel ─────────────────────────────────────────────────────────
class NeuralEventKernel(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_donors):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.shape_net = nn.Sequential(*layers)
        self.raw_amplitude = nn.Parameter(torch.zeros(n_donors))
        self.register_buffer("_shape_scale", torch.tensor(1.0, dtype=torch.float32))

    @property
    def amplitude(self):
        return torch.exp(self.raw_amplitude)

    def forward(self, t_normalized):
        return self.shape_net(t_normalized) / self._shape_scale.item()

    def normalize_shape_epoch(self):
        with torch.no_grad():
            t_seq = torch.tensor(T_NORM, dtype=torch.float32).unsqueeze(1)
            shape_vals = self.shape_net(t_seq)
            max_val = shape_vals.abs().max().item()
            max_val = float(np.clip(max_val, 1e-3, 100.0))
            if max_val > 1e-6:
                adjustment = max_val / self._shape_scale.item()
                self.raw_amplitude.data += torch.log(
                    torch.tensor(adjustment, dtype=torch.float32)
                )
                self._shape_scale.fill_(max_val)


def training_loss(kernel, t_norm, targets, donor_indices, weights,
                  lambda_smooth=LAMBDA_SMOOTH, lambda_amp=LAMBDA_AMP):
    shape_output = kernel(t_norm).squeeze()
    amplitudes   = kernel.amplitude[donor_indices]
    predictions  = amplitudes * shape_output
    mse_loss = torch.sum(weights * (predictions - targets) ** 2) / (torch.sum(weights) + 1e-8)

    t_seq    = torch.tensor(T_NORM, dtype=torch.float32).unsqueeze(1)
    beta_seq = kernel(t_seq).squeeze()
    smoothness = lambda_smooth * torch.sum(
        (beta_seq[2:] - 2 * beta_seq[1:-1] + beta_seq[:-2]) ** 2
    )
    amp_penalty = (
        lambda_amp * torch.mean(kernel.raw_amplitude ** 2)
        + 0.01 * torch.mean(kernel.amplitude ** 2)
    )
    return mse_loss + smoothness + amp_penalty


def train_kernel(kernel, t_norm, targets, donor_indices, weights):
    set_all_seeds()
    optimizer = torch.optim.Adam(kernel.parameters(), lr=1e-3, weight_decay=1e-3)
    best_loss = float("inf")
    best_state = None
    patience_counter = 0
    loss_history = []

    for epoch in range(MAX_EPOCHS):
        kernel.train()
        optimizer.zero_grad()
        loss = training_loss(kernel, t_norm, targets, donor_indices, weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(kernel.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % NORMALIZE_EVERY == 0:
            kernel.normalize_shape_epoch()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in kernel.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch+1} (best loss={best_loss:.6f})")
                break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:4d} | loss={loss_val:.6f}", flush=True)

    if best_state is not None:
        kernel.load_state_dict(best_state)
    kernel.eval()
    return kernel, loss_history


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("FINAL KERNEL TRAINING — Model C (all 5 donors)", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Load all donor series
    print("\n[1/4] Loading donor series...", flush=True)
    all_series = {}
    for iso in DONORS_ORDERED:
        s = load_donor_series(iso)
        all_series[iso] = s
        print(f"  {iso}: {len(s)} months  {s.index[0].strftime('%Y-%m')} → {s.index[-1].strftime('%Y-%m')}", flush=True)

    # Step 2: Build SARIMA counterfactuals
    print("\n[2/4] Building SARIMA counterfactuals...", flush=True)
    all_cf = {}
    for iso in DONORS_ORDERED:
        cf = build_sarima_counterfactual(iso, all_series[iso])
        all_cf[iso] = cf
        print(f"  {iso}: counterfactual OK (12 months)", flush=True)

    # Step 3: Build uplift table
    print("\n[3/4] Building uplift table...", flush=True)
    df_uplift = build_uplift_table(all_series, all_cf)
    print(f"  {len(df_uplift)} data points across {df_uplift['country_id'].nunique()} donors", flush=True)

    # Prepare tensors
    t_norm       = torch.tensor(df_uplift["t_normalized"].values, dtype=torch.float32).unsqueeze(1)
    targets      = torch.tensor(df_uplift["uplift_target"].values, dtype=torch.float32)
    donor_idxs   = torch.tensor(df_uplift["donor_idx"].values, dtype=torch.long)

    # Compute training weights (normalized raw weights per row)
    total_w = sum(DONOR_WEIGHTS_RAW.values())
    w_arr   = np.array([DONOR_WEIGHTS_RAW[iso] / total_w for iso in df_uplift["country_id"]])
    weights  = torch.tensor(w_arr, dtype=torch.float32)

    # Step 4: Train kernel
    print("\n[4/4] Training NeuralEventKernel (Model C: hidden=16, layers=2)...", flush=True)
    kernel = NeuralEventKernel(hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS, n_donors=len(DONORS_ORDERED))
    kernel, loss_history = train_kernel(kernel, t_norm, targets, donor_idxs, weights)

    # Save kernel weights
    save_path = os.path.join(PROJECT_DIR, "kernel_final_modelC.pt")
    torch.save(kernel.state_dict(), save_path)
    print(f"\n  Kernel saved → {save_path}", flush=True)

    # Print per-donor amplitudes
    print("\nPer-donor learned amplitudes:", flush=True)
    amps = kernel.amplitude.detach().numpy()
    for i, iso in enumerate(DONORS_ORDERED):
        print(f"  {iso}: a_i = {amps[i]:.4f}  (weight = {DONOR_WEIGHTS_RAW[iso]/total_w:.3f})", flush=True)

    # Print pulse shape
    print("\nNormalized pulse shape f_θ(t):", flush=True)
    with torch.no_grad():
        t_seq = torch.tensor(T_NORM, dtype=torch.float32).unsqueeze(1)
        shape = kernel(t_seq).squeeze().numpy()
    for t, v in zip(REL_MONTHS, shape):
        print(f"  t={t:+d}: f(t) = {v:.4f}", flush=True)

    # Plot loss curve
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, color='steelblue', linewidth=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Final Kernel Training Loss (Model C — All 5 Donors)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "final_kernel_training_loss.png"), dpi=150)
    plt.close()

    # Plot learned pulse
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(REL_MONTHS, shape, color='steelblue', edgecolor='black', width=0.6)
    axes[0].axvline(0, color='darkorange', linestyle='--', linewidth=1.5, label='WC start')
    axes[0].set_xlabel("Months relative to WC start")
    axes[0].set_ylabel("Normalized shape f_θ(t)")
    axes[0].set_title("Learned WC Uplift Pulse Shape")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Weighted transfer amplitude
    total_w = sum(DONOR_WEIGHTS_RAW.values())
    nw = {k: v/total_w for k, v in DONOR_WEIGHTS_RAW.items()}
    a_morocco = sum(nw[d] * amps[i] for i, d in enumerate(DONORS_ORDERED))
    log_uplift = a_morocco * shape
    pct_uplift = (np.exp(log_uplift) - 1) * 100

    axes[1].bar(REL_MONTHS, pct_uplift, color='crimson', edgecolor='black', width=0.6, alpha=0.8)
    axes[1].axvline(0, color='darkorange', linestyle='--', linewidth=1.5, label='WC start')
    axes[1].axhline(0, color='black', linewidth=0.7)
    axes[1].set_xlabel("Months relative to WC start")
    axes[1].set_ylabel("Estimated % uplift for Morocco")
    axes[1].set_title(f"Morocco WC Uplift (a_morocco = {a_morocco:.4f})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, "final_kernel_pulse_shape.png"), dpi=150)
    plt.close()
    print("\nPlots saved.", flush=True)
    print("=" * 60, flush=True)
    print("FINAL KERNEL TRAINING COMPLETE", flush=True)
    print("=" * 60, flush=True)
