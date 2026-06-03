import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend — must be before any pyplot import
import matplotlib.pyplot as plt
import seaborn as sns
from statsforecast.models import AutoARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera, norm
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_FILE = r"C:\Users\Hp\OneDrive\Desktop\Data\Data Time series project\consommation_electrique_maroc_2016_2025_final.csv"
PROJECT_DIR = r"c:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\TS-project"

print("=" * 60, flush=True)
print("MOROCCO SARIMA BASELINE PIPELINE (LAYER 1)", flush=True)
print("=" * 60, flush=True)

# =====================================================================
# PHASE 1: ALL COMPUTATION (no plotting)
# =====================================================================

# --- Step 1: Load ---
print("\n[1/9] Loading data...", flush=True)
df = pd.read_csv(DATA_FILE)
df = df.rename(columns={'Date': 'ds', 'Energie_GWh': 'y'})
df['ds'] = pd.to_datetime(df['ds'])
df = df.sort_values('ds').set_index('ds')
y_log = np.log1p(df['y'])
rolling_mean = y_log.rolling(12).mean()
rolling_std  = y_log.rolling(12).std()
print("  OK", flush=True)

# --- Step 2: Stationarity ---
print("\n[2/9] Stationarity tests...", flush=True)
adf_result  = adfuller(y_log, autolag='AIC')
kpss_result = kpss(y_log, regression='ct', nlags='auto')
print(f"  ADF p={adf_result[1]:.4f}  KPSS p={kpss_result[1]:.4f}", flush=True)

# --- Step 3: AutoARIMA global fit ---
print("\n[3/9] Global AutoARIMA order search...", flush=True)
t0 = time.time()
sarima_model = AutoARIMA(season_length=12, d=1, D=1, max_p=2, max_q=2, max_P=1, max_Q=1)
sarima_model.fit(y_log.values)
mod_dict       = sarima_model.model_
arma           = mod_dict.get('arma', [0]*6)
order          = (arma[0], 1, arma[1])
seasonal_order = (arma[2], 1, arma[3], 12)
aic_global     = mod_dict.get('aic', np.nan)
bic_global     = mod_dict.get('bic', np.nan)
print(f"  Took {time.time()-t0:.1f}s  =>  SARIMA{order}x{seasonal_order}  AIC={aic_global:.2f}", flush=True)

# Differenced series for ACF/PACF (computed here, plotted later)
y_diff = y_log.diff(1).diff(12).dropna()

# --- Step 4: Residual diagnostics ---
print("\n[4/9] Residual diagnostics...", flush=True)
fitted_vals    = sarima_model.predict_in_sample()['fitted']
residuals_full = y_log.values - fitted_vals
resid_mask     = ~np.isnan(residuals_full)
residuals      = residuals_full[resid_mask]
resid_index    = y_log.index[resid_mask]

lb_result  = acorr_ljungbox(residuals, lags=[6, 12, 24], return_df=True)
jb_stat, jb_p = jarque_bera(residuals)
arch_test  = het_arch(residuals, nlags=12)
arch_p     = arch_test[1]
roll_resid_std = pd.Series(residuals, index=resid_index).rolling(12).std()
print(f"  JB p={jb_p:.4f}  ARCH p={arch_p:.4f}", flush=True)
print(f"  Ljung-Box:\n{lb_result.to_string()}", flush=True)

# Save order file
with open(os.path.join(PROJECT_DIR, 'morocco_sarima_order.txt'), 'w') as f:
    f.write(f"Order: {order}\nSeasonal Order: {seasonal_order}\nAIC: {aic_global}\nBIC: {bic_global}\n")

# --- Step 5: Rolling-origin CV ---
print("\n[5/9] Rolling-origin expanding-window CV (step=3)...", flush=True)

def walk_forward_cv(y_log, initial_train_months=60, horizon=12, step=3):
    results = []
    n = len(y_log)
    for cutoff in range(initial_train_months, n - horizon + 1, step):
        train       = y_log[:cutoff]
        actual_log  = y_log[cutoff:cutoff + horizon]
        actual_gwh  = np.expm1(actual_log).values
        cutoff_date = y_log.index[cutoff - 1]
        print(f"  -> Cutoff {cutoff_date.date()} | train={len(train)}", flush=True)
        t_fold = time.time()

        order_used          = (np.nan,)*3
        seasonal_order_used = (np.nan, np.nan, np.nan, 12)
        aic_fold = bic_fold = np.nan
        forecast_gwh = np.full(horizon, np.nan)

        try:
            fm = AutoARIMA(season_length=12, d=1, D=1, max_p=2, max_q=2, max_P=1, max_Q=1)
            fm.fit(train.values)
            md = fm.model_
            ar = md.get('arma', [0]*6)
            order_used          = (ar[0], 1, ar[1])
            seasonal_order_used = (ar[2], 1, ar[3], 12)
            aic_fold = md.get('aic', np.nan)
            bic_fold = md.get('bic', np.nan)
            forecast_gwh = np.expm1(fm.predict(h=horizon)['mean'])
        except Exception as e:
            print(f"     [!] Fold failed: {e}", flush=True)

        fit_time = time.time() - t_fold

        # Seasonal naive
        snaive_gwh = np.expm1(train.iloc[-12:].values)
        # Drift
        avg_trend  = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
        drift_gwh  = np.expm1(train.iloc[-1] + np.arange(1, horizon+1)*avg_trend)

        eps = 1e-6
        def metrics(f):
            if np.isnan(f).any(): return np.nan, np.nan, np.nan
            rmse = np.sqrt(np.mean((actual_gwh - f)**2))
            mape = np.mean(np.abs((actual_gwh - f)/np.maximum(actual_gwh, eps)))*100
            bias = np.mean(f - actual_gwh)
            return rmse, mape, bias

        rmse,    mape,    bias    = metrics(forecast_gwh)
        sn_rmse, sn_mape, sn_bias = metrics(snaive_gwh)
        dr_rmse, dr_mape, dr_bias = metrics(drift_gwh)

        results.append({
            'cutoff': cutoff_date, 'horizon': horizon,
            'fit_time_seconds': fit_time,
            'order': str(order_used), 'seasonal_order': str(seasonal_order_used),
            'aic': aic_fold, 'bic': bic_fold,
            'rmse': rmse, 'mape': mape, 'bias': bias,
            'snaive_rmse': sn_rmse, 'snaive_mape': sn_mape, 'snaive_bias': sn_bias,
            'drift_rmse':  dr_rmse,  'drift_mape':  dr_mape,  'drift_bias':  dr_bias,
            'actual_gwh': actual_gwh, 'forecast_gwh': forecast_gwh,
        })
    return pd.DataFrame(results)

cv_results = walk_forward_cv(y_log, initial_train_months=60, horizon=12, step=3)
cv_results.drop(columns=['actual_gwh', 'forecast_gwh']).to_csv(
    os.path.join(PROJECT_DIR, 'morocco_cv_results.csv'), index=False)
print(f"  {len(cv_results)} folds complete.", flush=True)

# --- Step 6: Seasonal stability ---
print("\n[6/9] Seasonal stability...", flush=True)
monthly_apes = []
for h in range(1, 13):
    apes = [
        abs(row['actual_gwh'][h-1] - row['forecast_gwh'][h-1]) /
        max(row['actual_gwh'][h-1], 1e-6) * 100
        for _, row in cv_results.iterrows() if not np.isnan(row['rmse'])
    ]
    monthly_apes.append(np.mean(apes))

# =====================================================================
# PHASE 2: ALL PLOTTING (after all computation is done)
# =====================================================================
print("\n[7/9] Generating plots...", flush=True)

# EDA
print("  Plotting EDA...", flush=True)
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
axes[0].plot(df.index, df['y'], color='steelblue')
axes[0].set_title('Raw Morocco Electricity Demand (GWh)')
axes[0].set_ylabel('GWh')
axes[1].plot(y_log.index, y_log, color='seagreen')
axes[1].set_title('Log-Transformed Series (log1p)')
axes[1].set_ylabel('log(GWh)')
axes[2].plot(rolling_mean.index, rolling_mean, label='Rolling 12M Mean', color='darkorange')
ax2_twin = axes[2].twinx()
ax2_twin.plot(rolling_std.index, rolling_std, label='Rolling 12M Std', color='crimson', alpha=0.6)
axes[2].set_title('Rolling Mean & Std — Variance Stability Check')
for ax in axes:
    ax.axvspan('2020-01-01', '2020-12-31', alpha=0.15, color='red', label='COVID')
    ax.axvspan('2024-01-01', '2025-12-31', alpha=0.15, color='purple', label='AFCON prep')
    ax.legend(loc='upper left', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, 'morocco_eda.png'), dpi=150)
plt.close('all')
print("  EDA plot saved.", flush=True)

# ACF/PACF — manual bar chart (avoids LAPACK segfaults from statsmodels plot functions)
print("  Plotting ACF/PACF...", flush=True)
def plot_acf_manual(ax, series, lags=36, title='ACF'):
    acf_vals = acf(series, nlags=lags, fft=True)
    ci = 1.96 / np.sqrt(len(series))
    ax.bar(range(len(acf_vals)), acf_vals, color='steelblue', width=0.3)
    ax.axhline(ci,  color='red', linestyle='--', linewidth=0.8)
    ax.axhline(-ci, color='red', linestyle='--', linewidth=0.8)
    ax.axhline(0,   color='black', linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel('Lag')

def plot_pacf_manual(ax, series, lags=36, title='PACF'):
    pacf_vals = pacf(series, nlags=lags, method='ywm')
    ci = 1.96 / np.sqrt(len(series))
    ax.bar(range(len(pacf_vals)), pacf_vals, color='darkorange', width=0.3)
    ax.axhline(ci,  color='red', linestyle='--', linewidth=0.8)
    ax.axhline(-ci, color='red', linestyle='--', linewidth=0.8)
    ax.axhline(0,   color='black', linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel('Lag')

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
plot_acf_manual(axes[0], y_diff, lags=36, title='ACF — Differenced log series (d=1, D=1)')
plot_pacf_manual(axes[1], y_diff, lags=36, title='PACF — Differenced log series')
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, 'morocco_acf_pacf.png'), dpi=150)
plt.close('all')
print("  ACF/PACF plot saved.", flush=True)

# 5-Panel Residual Diagnostics
print("  Plotting Residual Diagnostics...", flush=True)
fig = plt.figure(figsize=(15, 12))
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((3, 2), (1, 0))
ax3 = plt.subplot2grid((3, 2), (1, 1))
ax4 = plt.subplot2grid((3, 2), (2, 0))
ax5 = plt.subplot2grid((3, 2), (2, 1))
ax1.plot(resid_index, residuals, color='darkblue', linewidth=0.8)
ax1.axvspan('2020-01-01', '2020-12-31', alpha=0.2, color='red', label='COVID')
ax1.set_title('Residuals over Time')
ax1.legend()
plot_acf_manual(ax2,  residuals, lags=24, title='Residual ACF')
plot_pacf_manual(ax3, residuals, lags=24, title='Residual PACF')
ax4.hist(residuals, bins=15, density=True, color='steelblue', alpha=0.7)
x = np.linspace(residuals.min(), residuals.max(), 100)
ax4.plot(x, norm.pdf(x, residuals.mean(), residuals.std()), 'k', lw=2, label='Normal')
ax4.set_title('Residual Histogram')
ax4.legend()
ax5.plot(roll_resid_std.index, roll_resid_std, color='crimson')
ax5.set_title('Rolling 12-Month Residual Std')
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, 'morocco_residual_diagnostics.png'), dpi=150)
plt.close('all')
print("  Residual Diagnostics plot saved.", flush=True)

# RMSE over time
print("  Plotting RMSE over time...", flush=True)
plt.figure(figsize=(12, 5))
plt.plot(cv_results['cutoff'], cv_results['rmse'],       marker='o', label='SARIMA',         lw=2)
plt.plot(cv_results['cutoff'], cv_results['snaive_rmse'], marker='x', linestyle='--', label='Seasonal Naive', alpha=0.7)
plt.plot(cv_results['cutoff'], cv_results['drift_rmse'],  marker='^', linestyle=':',  label='Drift',          alpha=0.7)
plt.axvspan('2020-01-01', '2020-12-31', alpha=0.15, color='red',    label='COVID')
plt.axvspan('2024-01-01', '2025-12-31', alpha=0.15, color='orange', label='AFCON prep')
plt.xlabel('Training Cutoff Date')
plt.ylabel('12-month Forecast RMSE (GWh)')
plt.title('Rolling-Origin RMSE over Time — Structural Break Detection')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, 'morocco_cv_rmse_over_time.png'), dpi=150)
plt.close('all')
print("  RMSE over time plot saved.", flush=True)

# Seasonal stability
print("  Plotting Seasonal stability...", flush=True)
plt.figure(figsize=(10, 5))
plt.bar(range(1, 13), monthly_apes, color='steelblue', edgecolor='black')
plt.xlabel('Forecast Horizon (Month)')
plt.ylabel('Mean APE (%)')
plt.title('SARIMA Seasonal Stability (MAPE by Forecast Horizon)')
plt.xticks(range(1, 13))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, 'morocco_seasonal_stability.png'), dpi=150)
plt.close('all')

print("  All 5 plots saved.", flush=True)

# =====================================================================
# STEP 8 & 9: DIAGNOSTIC SUMMARY
# =====================================================================
print("\n" + "=" * 60, flush=True)
print("MOROCCO SARIMA BASELINE DIAGNOSTIC SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"Selected order (Global): SARIMA{order}x{seasonal_order}")
print(f"AIC:                     {aic_global:.2f}")
print(f"BIC:                     {bic_global:.2f}")
print()
print("Stationarity (log series):")
print(f"  ADF p-value:           {adf_result[1]:.4f} ({'stationary' if adf_result[1]<0.05 else 'non-stationary'})")
print(f"  KPSS p-value:          {kpss_result[1]:.4f} ({'stationary' if kpss_result[1]>0.05 else 'non-stationary'})")
print()
print("Formal Residual Tests:")
for _, row in lb_result.iterrows():
    flag = "✓ white noise" if row['lb_pvalue'] > 0.05 else "✗ autocorrelation remains"
    print(f"  Ljung-Box (Lag {int(row.name):2d}): p={row['lb_pvalue']:.4f}  {flag}")
print(f"  Jarque-Bera (Normal): p={jb_p:.4f}       {'✓ normal' if jb_p>0.05 else '✗ heavy tails'}")
print(f"  ARCH (Homosked):      p={arch_p:.4f}     {'✓ homoskedastic' if arch_p>0.05 else '✗ volatility clustering'}")
print()
print("Rolling-Origin Evaluation (12-month horizon, step=3):")
print(f"  Folds evaluated:      {len(cv_results)}")
print(f"  Mean SARIMA RMSE:     {cv_results['rmse'].mean():.1f} GWh")
print(f"  Std  SARIMA RMSE:     {cv_results['rmse'].std():.1f} GWh")
print(f"  Mean SARIMA MAPE:     {cv_results['mape'].mean():.2f}%")
print(f"  Mean SARIMA Bias:     {cv_results['bias'].mean():.1f} GWh")
print(f"  Avg Fit Time / Fold:  {cv_results['fit_time_seconds'].mean():.2f} sec")
print()
print("Baseline Comparisons:")
print(f"  S-Naive RMSE:         {cv_results['snaive_rmse'].mean():.1f} GWh  (SARIMA {'beats' if cv_results['rmse'].mean()<cv_results['snaive_rmse'].mean() else 'loses to'} naive)")
print(f"  Drift   RMSE:         {cv_results['drift_rmse'].mean():.1f} GWh")
print()

lb_pass      = all(lb_result['lb_pvalue'] > 0.05)
cv_mape_pass = cv_results['mape'].mean() < 10.0

print("Readiness for SARIMAX:")
print(f"  Ljung-Box pass:       {'✓' if lb_pass else '✗ residual autocorrelation remains'}")
print(f"  CV MAPE < 10%:        {'✓' if cv_mape_pass else '✗ investigate before adding regressors'}")
if lb_pass and cv_mape_pass:
    print("\n  ✓ SARIMA baseline credible — proceed to SARIMAX with exogenous variables")
else:
    print("\n  ✗ Address diagnostics before adding exogenous variables")
print("=" * 60)
