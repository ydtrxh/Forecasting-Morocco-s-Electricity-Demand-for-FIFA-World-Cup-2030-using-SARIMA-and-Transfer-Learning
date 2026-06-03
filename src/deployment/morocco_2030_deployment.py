"""
morocco_2030_deployment.py
===========================
Deploy the final SARIMA + NeuralEventKernel pipeline to forecast Morocco's
monthly electricity demand from the current date through December 2030,
including the FIFA World Cup uplift starting June 2030.

HARD CONSTRAINTS (from project spec):
  - Reconstruction: (baseline + 1) × exp(a_morocco × f_θ(t)) − 1
  - t_normalized = t / 6.0 exactly
  - Pulse applied only where −5 ≤ months_to_wc ≤ 6
  - Both interval bounds back-transformed individually with expm1
  - SARIMA fitted on log1p(y) — all forecasts back-transformed with expm1
  - DONORS_ORDERED matches training order exactly
  - kernel.eval() called before inference, torch.no_grad() wraps all inference
  - np.random.seed(42) at top
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Must be before pyplot import — prevents GUI segfaults on Windows
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR = r"C:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\TS-project"
DATA_FILE   = r"C:\Users\Hp\OneDrive\Desktop\Data\Data Time series project\consommation_electrique_maroc_2016_2025_final.csv"
KERNEL_PATH = os.path.join(PROJECT_DIR, "kernel_final_modelC.pt")

# ── Event parameters ──────────────────────────────────────────────────────────
WC_START_DATE    = pd.Timestamp("2030-06-01")
WC_EVENT_WINDOW  = range(-5, 7)               # months -5 to +6
FORECAST_END     = pd.Timestamp("2030-12-01") # inclusive

# Deployment weights (from project spec)
DONORS_ORDERED = ["QAT", "EGY", "ZAF", "CMR", "RUS"]  # MUST match training order
DONOR_WEIGHTS_RAW = {
    "QAT": 1.0,
    "EGY": 0.7,
    "ZAF": 0.6,
    "CMR": 0.5,
    "RUS": 0.3,
}

print("=" * 65, flush=True)
print("MOROCCO 2030 WC FORECAST — DEPLOYMENT PIPELINE", flush=True)
print("=" * 65, flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA AND TRAIN FINAL SARIMA ON FULL MOROCCO HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/8] Loading Morocco data and fitting final SARIMA...", flush=True)

df = pd.read_csv(DATA_FILE)
df = df.rename(columns={"Date": "ds", "Energie_GWh": "y"})
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)

# Hard constraint: do NOT include any months from June 2030 onward
df = df[df["ds"] < pd.Timestamp("2030-06-01")].copy()

y_log = np.log1p(df["y"].values)

from statsforecast.models import AutoARIMA  # lazy import after numpy/pandas to avoid MKL conflicts

sarima_final = AutoARIMA(
    season_length=12,
    d=1, D=1,
    max_p=3, max_q=3,
    max_P=2, max_Q=2,
)
sarima_final.fit(y_log)

mod_dict       = sarima_final.model_
arma           = mod_dict.get("arma", [0]*6)
final_order    = (arma[0], 1, arma[1])
final_seasonal = (arma[2], 1, arma[3], 12)
final_aic      = mod_dict.get("aic", np.nan)
final_bic      = mod_dict.get("bic", np.nan)

print(f"  Final SARIMA order:          {final_order}", flush=True)
print(f"  Final SARIMA seasonal order: {final_seasonal}", flush=True)
print(f"  AIC: {final_aic:.2f} | BIC: {final_bic:.2f}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — GENERATE SARIMA BASELINE FORECAST WITH NATIVE PREDICTION INTERVALS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/8] Generating SARIMA baseline forecast with 85% prediction intervals...", flush=True)

last_observed = df["ds"].max()

# Number of months from last_observed+1 to 2030-12 inclusive
first_forecast = last_observed + pd.DateOffset(months=1)
n_forecast_months = (
    (FORECAST_END.year - first_forecast.year) * 12
    + (FORECAST_END.month - first_forecast.month)
    + 1
)
print(f"  Forecast horizon: {n_forecast_months} months "
      f"({first_forecast.strftime('%Y-%m')} → {FORECAST_END.strftime('%Y-%m')})", flush=True)

# statsforecast returns 'lo-85' and 'hi-85' with level=[85]
forecast_result = sarima_final.predict(h=n_forecast_months, level=[85])
forecast_log      = forecast_result["mean"]
lower_log         = forecast_result.get("lo-85",    forecast_result.get("lower-85",   forecast_log * 0.85))
upper_log         = forecast_result.get("hi-85",    forecast_result.get("upper-85",   forecast_log * 1.15))

# Build forecast datetime index
forecast_index = pd.date_range(
    start=first_forecast,
    periods=n_forecast_months,
    freq="MS",
)

# Back-transform to GWh — both bounds individually with expm1 (never transform midpoint and offset)
baseline_GWh       = np.expm1(forecast_log)
baseline_lower_GWh = np.expm1(lower_log)
baseline_upper_GWh = np.expm1(upper_log)

print(f"  Baseline at WC start (Jun-2030): "
      f"{baseline_GWh[list(forecast_index).index(WC_START_DATE)]:.1f} GWh", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — LOAD FINAL NEURAL KERNEL AND COMPUTE TRANSFER AMPLITUDE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/8] Loading final neural kernel...", flush=True)

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

kernel = NeuralEventKernel(hidden_dim=16, n_layers=2, n_donors=5)
kernel.load_state_dict(torch.load(KERNEL_PATH, map_location="cpu"))
kernel.eval()   # No dropout active

# Compute weighted transfer amplitude (Morocco has no observed event)
total_w = sum(DONOR_WEIGHTS_RAW.values())
normalized_weights = {k: v / total_w for k, v in DONOR_WEIGHTS_RAW.items()}

amplitudes = kernel.amplitude.detach().numpy()
a_morocco = sum(
    normalized_weights[d] * amplitudes[i]
    for i, d in enumerate(DONORS_ORDERED)
)

print(f"  Weighted transfer amplitude a_morocco: {a_morocco:.4f}", flush=True)
print("  Per-donor amplitudes:", flush=True)
for i, d in enumerate(DONORS_ORDERED):
    print(f"    {d}: a_i = {amplitudes[i]:.4f}  (weight = {normalized_weights[d]:.3f})", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — EVALUATE KERNEL PULSE OVER EVENT WINDOW
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/8] Evaluating WC pulse over event window [-5, +6]...", flush=True)

t_seq = torch.tensor(
    [t / 6.0 for t in range(-5, 7)], dtype=torch.float32
).unsqueeze(1)

with torch.no_grad():
    shape_output = kernel(t_seq).squeeze().numpy()   # shape: (12,)

# Full log-space uplift per event window month
log_uplift = a_morocco * shape_output   # shape: (12,)

print("  WC pulse — log-space uplift per month:", flush=True)
for t, val in zip(range(-5, 7), log_uplift):
    pct = (np.exp(val) - 1) * 100
    print(f"    Month {t:+d}: log_uplift = {val:.4f}  →  {pct:+.2f}% lift", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — INJECT WC PULSE INTO FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/8] Injecting WC pulse into forecast...", flush=True)

predicted_GWh   = baseline_GWh.copy()
predicted_lower = baseline_lower_GWh.copy()
predicted_upper = baseline_upper_GWh.copy()

for i, date in enumerate(forecast_index):
    months_to_wc = (date.year - 2030) * 12 + (date.month - 6)

    if -5 <= months_to_wc <= 6:
        t_idx  = months_to_wc + 5   # maps [-5,+6] → [0,11]
        uplift = log_uplift[t_idx]

        # Multiplicative injection — correct log1p inversion
        # Hard constraint: (baseline + 1) × exp(a_morocco × f_θ(t)) − 1
        predicted_GWh[i]   = (baseline_GWh[i]       + 1) * np.exp(uplift) - 1
        predicted_lower[i] = (baseline_lower_GWh[i] + 1) * np.exp(uplift) - 1
        predicted_upper[i] = (baseline_upper_GWh[i] + 1) * np.exp(uplift) - 1


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — ASSEMBLE FULL RESULTS DATAFRAME
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/8] Assembling results DataFrame...", flush=True)

results_df = pd.DataFrame({
    "ds":               forecast_index,
    "baseline_GWh":     baseline_GWh,
    "predicted_GWh":    predicted_GWh,
    "lower_85_GWh":     predicted_lower,
    "upper_85_GWh":     predicted_upper,
    "months_to_wc":     [(d.year - 2030)*12 + (d.month - 6) for d in forecast_index],
    "wc_lift_GWh":      predicted_GWh - baseline_GWh,
    "wc_lift_pct":      (predicted_GWh - baseline_GWh) / baseline_GWh * 100,
    "in_event_window":  [(-5 <= (d.year-2030)*12+(d.month-6) <= 6) for d in forecast_index],
})

csv_path = os.path.join(PROJECT_DIR, "morocco_2030_forecast.csv")
results_df.to_csv(csv_path, index=False)
print(f"  Saved → {csv_path}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7/8] Generating publication-quality visualization...", flush=True)

fig, ax = plt.subplots(figsize=(16, 7))

# Historical observed demand — solid black
ax.plot(df["ds"], df["y"], color="black", linewidth=1.5,
        label="Historical demand (observed)")

# SARIMA counterfactual baseline — blue dashed
ax.plot(results_df["ds"], results_df["baseline_GWh"],
        color="steelblue", linestyle="--", linewidth=1.5,
        label="SARIMA baseline (no WC)")

# Combined forecast with WC lift — red solid
ax.plot(results_df["ds"], results_df["predicted_GWh"],
        color="crimson", linewidth=2.0,
        label="SARIMA + WC neural kernel")

# 85% prediction interval — shaded
ax.fill_between(
    results_df["ds"],
    results_df["lower_85_GWh"],
    results_df["upper_85_GWh"],
    alpha=0.15, color="crimson",
    label="85% prediction interval",
)

# WC event window shading — light orange
wc_window_start = pd.Timestamp("2030-01-01")   # month -5
wc_window_end   = pd.Timestamp("2030-12-01")   # month +6
ax.axvspan(wc_window_start, wc_window_end,
           alpha=0.08, color="orange", label="WC event window (−5 to +6)")

# WC start vertical line — June 2030
ax.axvline(pd.Timestamp("2030-06-01"),
           color="darkorange", linestyle=":", linewidth=1.5,
           label="WC start (June 2030)")

# Peak lift annotation
peak_row = results_df.loc[results_df["wc_lift_GWh"].idxmax()]
ax.annotate(
    f"Peak WC lift\n+{peak_row['wc_lift_pct']:.1f}%\n({peak_row['wc_lift_GWh']:.0f} GWh)",
    xy=(peak_row["ds"], peak_row["predicted_GWh"]),
    xytext=(peak_row["ds"] + pd.DateOffset(months=3),
            peak_row["predicted_GWh"] * 1.02),
    arrowprops=dict(arrowstyle="->", color="black"),
    fontsize=10, color="crimson",
)

ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Electricity Demand (GWh)", fontsize=12)
ax.set_title(
    "Morocco Electricity Demand Forecast — FIFA World Cup 2030\n"
    "SARIMA Baseline + Neural Event Kernel Uplift | 85% Prediction Interval",
    fontsize=13,
)
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plot_path = os.path.join(PROJECT_DIR, "morocco_2030_forecast.png")
plt.savefig(plot_path, dpi=150)
plt.close()
print(f"  Saved → {plot_path}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — CONSOLE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65, flush=True)
print("MOROCCO 2030 WC FORECAST — DEPLOYMENT SUMMARY", flush=True)
print("=" * 65, flush=True)
print(f"SARIMA order:            {final_order}")
print(f"SARIMA seasonal order:   {final_seasonal}")
print(f"Forecast horizon:        {forecast_index[0].strftime('%Y-%m')} → {forecast_index[-1].strftime('%Y-%m')}")
print(f"Transfer amplitude:      a_morocco = {a_morocco:.4f}")
print()
print("WC event window impact:")
event_rows = results_df[results_df["in_event_window"]]
peak_idx   = event_rows["wc_lift_GWh"].idxmax()
print(f"  Peak lift month:       {event_rows.loc[peak_idx, 'ds'].strftime('%Y-%m')}")
print(f"  Peak lift (GWh):       +{event_rows['wc_lift_GWh'].max():.1f} GWh")
print(f"  Peak lift (%):         +{event_rows['wc_lift_pct'].max():.2f}%")
print(f"  Total cumulative lift: +{event_rows['wc_lift_GWh'].sum():.1f} GWh")
print()
print("Prediction interval width at peak month:")
peak = event_rows.loc[peak_idx]
print(f"  Lower 85%:             {peak['lower_85_GWh']:.1f} GWh")
print(f"  Point forecast:        {peak['predicted_GWh']:.1f} GWh")
print(f"  Upper 85%:             {peak['upper_85_GWh']:.1f} GWh")
print("=" * 65)
