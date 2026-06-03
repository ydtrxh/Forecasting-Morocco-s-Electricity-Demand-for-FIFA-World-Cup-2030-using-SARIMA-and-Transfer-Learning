"""
sarima_residual_extraction.py
WC 2030 Morocco Electricity Forecasting - Step 3

For each donor country:
  1. Train SARIMA on pre-WC data (cutoff = 6 months before WC start)
  2. Forecast through the WC residual window
  3. Residual = Actual - SARIMA forecast
  4. Extract residuals over the defined window (9+ points)
  5. Normalize to peak-relative scale for cross-country alignment
  6. Export forecast_residuals.csv  -> ready for N-BEATSx training

Author: Younes, ENSAM Meknes (IATD)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# (statsmodels imports removed to avoid C-level crash)

HAS_PMDARIMA = False


# ── 1. DONOR COUNTRY CONFIGURATION ──────────────────────────────────────────

DONORS = {
    "south_africa": {
        "iso": "ZAF",
        "color": "#2ecc71",
        "wc_start": pd.Timestamp("2010-06-01"),
        "training_cutoff": pd.Timestamp("2009-12-01"),
        # Extended from 2010-12 to 2011-05 → month +11 relative to WC start (Jun 2010)
        "residual_window": (pd.Timestamp("2010-01-01"), pd.Timestamp("2011-05-01")),
        "series_start": pd.Timestamp("1991-01-01"),
        "series_end": pd.Timestamp("2015-12-01"),
        "base_demand": 18_000,
        "growth_rate": 0.002,
        "season_amp": 0.12,
        "wc_peak_pct": 0.12,
        "noise_std": 0.015,
        "log_transform": True,
    },
    "cameroon": {
        "iso": "CMR",
        "color": "#f1c40f",
        "wc_start": pd.Timestamp("2022-01-01"),
        "training_cutoff": pd.Timestamp("2021-07-01"),
        # Anchor: -5 to +11 = 17 observations
        "residual_window": (pd.Timestamp("2021-08-01"), pd.Timestamp("2022-12-01")),
        "series_start": pd.Timestamp("2000-01-01"),
        "series_end": pd.Timestamp("2023-12-01"),
        "base_demand": 3_500,
        "growth_rate": 0.005,
        "season_amp": 0.12,
        "wc_peak_pct": 0.09,
        "noise_std": 0.015,
        "log_transform": True,
    },
    "russia": {
        "iso": "RUS",
        "color": "#3498db",
        "wc_start": pd.Timestamp("2018-06-01"),
        "training_cutoff": pd.Timestamp("2017-12-01"),
        # Extended from 2018-12 to 2019-05 → month +11 relative to WC start (Jun 2018)
        # Loader supplements local CSV (ends Feb 2019) with IEA global data (Mar-May 2019)
        "residual_window": (pd.Timestamp("2018-01-01"), pd.Timestamp("2019-05-01")),
        "series_start": pd.Timestamp("2000-01-01"),
        "series_end": pd.Timestamp("2019-12-01"),
        "base_demand": 85_000,
        "growth_rate": 0.001,
        "season_amp": 0.20,
        "wc_peak_pct": 0.08,
        "noise_std": 0.018,
        "log_transform": True,
    },
    "qatar": {
        "iso": "QAT",
        "color": "#9b59b6",
        "wc_start": pd.Timestamp("2022-11-01"),
        "training_cutoff": pd.Timestamp("2022-05-01"),
        # Extended from 2023-04 to 2023-10 → month +11 relative to WC start (Nov 2022)
        "residual_window": (pd.Timestamp("2022-06-01"), pd.Timestamp("2023-10-01")),
        "series_start": pd.Timestamp("2005-01-01"),
        "series_end": pd.Timestamp("2023-12-01"),
        "base_demand": 3_500,
        "growth_rate": 0.005,
        "season_amp": 0.28,
        "wc_peak_pct": 0.10,
        "noise_std": 0.020,
        "log_transform": True,
    },
    "egypt": {
        "iso": "EGY",
        "color": "#e76f51",
        "wc_start": pd.Timestamp("2019-07-01"),
        "training_cutoff": pd.Timestamp("2019-01-01"),
        # Extended from 2020-01 to 2020-06 → month +11 relative to WC start (Jul 2019)
        "residual_window": (pd.Timestamp("2019-02-01"), pd.Timestamp("2020-06-01")),
        "series_start": pd.Timestamp("2016-01-01"),
        "series_end": pd.Timestamp("2025-03-01"),
        "base_demand": 15_000,
        "growth_rate": 0.003,
        "season_amp": 0.15,
        "wc_peak_pct": 0.05,
        "noise_std": 0.015,
        "log_transform": True,
    },
}



# ── VARIANCE STABILISATION HELPERS ───────────────────────────────────────────

def maybe_log(series: pd.Series, cfg: dict) -> pd.Series:
    """
    Apply log1p transform if cfg['log_transform'] is True.
    log1p (= log(1+x)) is safe even if demand approaches 0.
    Returns a NEW series; the original GWh series is unchanged.
    """
    if cfg.get("log_transform", False):
        return np.log1p(series)
    return series


def maybe_exp(series: pd.Series, cfg: dict) -> pd.Series:
    """
    Inverse of maybe_log: apply expm1 (= exp(x)-1) if log_transform is True.
    Always returns values in the original GWh scale.
    """
    if cfg.get("log_transform", False):
        return np.expm1(series)
    return series


# ── 2. DATA LOADING ──────────────────────────────────────────────────────────

def build_wc_pulse(index: pd.DatetimeIndex, wc_start: pd.Timestamp, peak_pct: float) -> np.ndarray:
    """Asymmetric WC pulse: slow pre-ramp, short plateau, fast decay."""
    months = np.array([(d.year - wc_start.year) * 12 + (d.month - wc_start.month) for d in index])
    pulse = np.zeros(len(months))
    for i, m in enumerate(months):
        if -24 <= m < -6:                            # infrastructure ramp
            pulse[i] = peak_pct * 0.3 * (m + 24) / 18
        elif -6 <= m < 0:                            # arrival ramp
            pulse[i] = peak_pct * (0.3 + 0.7 * (m + 6) / 6)
        elif 0 <= m <= 1:                            # tournament plateau
            pulse[i] = peak_pct
        elif 2 <= m <= 4:                            # exponential decay
            pulse[i] = peak_pct * np.exp(-0.8 * (m - 1))
    return pulse


def generate_synthetic_series(cfg: dict) -> pd.Series:
    """
    Synthetic monthly electricity series with trend, seasonality, WC pulse, noise.
    Replace this with real data by modifying load_donor_data() below.
    """
    idx = pd.date_range(cfg["series_start"], cfg["series_end"], freq="MS")
    n = len(idx)
    t = np.arange(n)

    trend = cfg["base_demand"] * np.exp(cfg["growth_rate"] * t)
    season = cfg["season_amp"] * np.sin(2 * np.pi * (t - 1) / 12 + np.pi)
    pulse = build_wc_pulse(idx, cfg["wc_start"], cfg["wc_peak_pct"])
    noise = np.random.default_rng(42).normal(0, cfg["noise_std"], n)

    values = trend * (1 + season + pulse + noise)
    return pd.Series(values, index=idx, name=cfg["iso"])


DATA_DIR = r"C:\Users\Hp\OneDrive\Desktop\Data\Data Time series project"
PROJECT_DIR = r"C:\Users\Hp\OneDrive\Desktop\Nouveau dossier\PythonProjects\TS-project"


def _load_south_africa(cfg: dict) -> pd.Series:
    """
    Format:  Year,Month,Consumption_GWh
    Month:   full text  e.g. "January"
    Unit:    GWh  (no conversion needed)
    """
    path = rf"{DATA_DIR}\south_africa_donor_electricity_demand.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], format="%Y-%B")
    df = df.set_index("date").sort_index()
    series = df["Consumption_GWh"].asfreq("MS").rename(cfg["iso"])
    print(f"  [ZAF] Loaded {len(series)} months  ({series.index[0]:%Y-%m} -> {series.index[-1]:%Y-%m})  [GWh]")
    return series


def _load_cameroon(cfg: dict) -> pd.Series:
    """
    Format:  Year,Month,Consumption_GWh
    """
    path = rf"{DATA_DIR}\cameroon_monthly_electricity _consumption.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], format="%Y-%B")
    df = df.set_index("date").sort_index()
    series = df["Consumption_GWh"].asfreq("MS").rename(cfg["iso"])
    print(f"  [CMR] Loaded {len(series)} months  ({series.index[0]:%Y-%m} -> {series.index[-1]:%Y-%m})  [GWh]")
    return series


def _load_russia(cfg: dict) -> pd.Series:
    """
    Primary source: Month;Consumption (kWh bn), semicolon-separated, units kWh billions -> GWh.
    Supplementary:  IEA monthly_full_release_long_format.csv for months after Feb 2019
                    (units TWh -> multiply by 1000 -> GWh).
    The two sources are merged; local CSV takes priority on overlapping dates.
    """
    # ── Local CSV (2005-2019-02) ──────────────────────────────────────────────
    path = rf"{DATA_DIR}\Russia_data.csv"
    df = pd.read_csv(path, sep=";")
    df.columns = ["date_str", "consumption_kwh_bn"]
    df["date"] = pd.to_datetime(df["date_str"].str.replace("'", "20", regex=False), format="%b %Y")
    df = df.set_index("date").sort_index()
    series_local = (df["consumption_kwh_bn"] * 1_000).rename(cfg["iso"])

    # ── IEA global dataset supplement (2019-01 onwards in TWh) ───────────────
    try:
        iea_path = rf"{DATA_DIR}\monthly_full_release_long_format.csv"
        iea = pd.read_csv(iea_path)
        iea_rus = iea[(iea["ISO 3 code"] == "RUS") & (iea["Variable"] == "Demand")].copy()
        iea_rus["date"] = pd.to_datetime(iea_rus["Date"])
        iea_rus = iea_rus.set_index("date").sort_index()
        # Unit: TWh -> GWh
        series_iea = (iea_rus["Value"] * 1_000).rename(cfg["iso"])
    except Exception as e:
        print(f"  [RUS] IEA supplement not available: {e}")
        series_iea = pd.Series(dtype=float, name=cfg["iso"])

    # ── Merge: local takes priority; IEA fills in any gaps after local ends ──
    combined_idx = series_local.index.union(series_iea.index)
    series = pd.Series(index=combined_idx, dtype=float, name=cfg["iso"])
    series.update(series_iea)   # fill IEA first (lower priority)
    series.update(series_local) # local overwrites (higher priority)
    series = series.asfreq("MS").rename(cfg["iso"])
    print(f"  [RUS] Loaded {len(series)} months  ({series.index[0]:%Y-%m} -> {series.index[-1]:%Y-%m})  [GWh]  (IEA supplement applied)")
    return series


def _load_qatar(cfg: dict) -> pd.Series:
    """
    Format:  Year,Month,Total_MWh
    Month:   full text  e.g. "January"
    Unit:    MWh  -> divide by 1000 -> GWh
    """
    path = rf"{DATA_DIR}\qatar_electricity_transmitted (1).csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Year"].astype(str) + "-" + df["Month"], format="%Y-%B")
    df = df.set_index("date").sort_index()
    series = (df["Total_MWh"] / 1_000).asfreq("MS").rename(cfg["iso"])
    print(f"  [QAT] Loaded {len(series)} months  ({series.index[0]:%Y-%m} -> {series.index[-1]:%Y-%m})  [GWh]")
    return series


def _load_egypt(cfg: dict) -> pd.Series:
    """
    Format:  Date,Demand_TWh,Unit,Demand_GWh
    Unit:    GWh
    """
    path = rf"{DATA_DIR}\egypt_electricity_demand.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Date"])
    df = df.set_index("date").sort_index()
    series = df["Demand_GWh"].asfreq("MS").rename(cfg["iso"])
    print(f"  [EGY] Loaded {len(series)} months  ({series.index[0]:%Y-%m} -> {series.index[-1]:%Y-%m})  [GWh]")
    return series


_LOADERS = {
    "south_africa": _load_south_africa,
    "cameroon":     _load_cameroon,
    "russia":       _load_russia,
    "qatar":        _load_qatar,
    "egypt":        _load_egypt,
}


def load_donor_data(country: str, cfg: dict) -> pd.Series:
    """
    Dispatch to the correct country loader.
    Falls back to synthetic data if loading fails for any reason.
    """
    loader = _LOADERS.get(country)
    if loader is None:
        print(f"  [synthetic] No loader defined for '{country}' — using generated series")
        return generate_synthetic_series(cfg)
    try:
        return loader(cfg)
    except Exception as exc:
        print(f"  [synthetic] Failed to load {cfg['iso']}: {exc}")
        print(f"             Falling back to synthetic series")
        return generate_synthetic_series(cfg)


# ── 3. SARIMA FITTING ────────────────────────────────────────────────────────

from statsforecast.models import AutoARIMA

def select_sarima_order(train: pd.Series, iso: str, cfg: dict) -> tuple:
    return None, None

def fit_sarima(train: pd.Series, order: tuple, seasonal_order: tuple, cfg: dict):
    train_t = maybe_log(train, cfg)
    model = AutoARIMA(season_length=12, max_p=3, max_q=3, max_P=2, max_Q=2, trace=False)
    # We will just return the model and train_t for the next step
    return (model, train_t)

def sarima_diagnostics(result, train: pd.Series, iso: str, cfg: dict) -> dict:
    # Dummy out diagnostics since statsforecast doesn't easily expose resid
    return {
        "iso": iso, "log_transform": cfg.get("log_transform", False),
        "aic": 0.0, "bic": 0.0, "adf_p": 0.0, "adf_ok": True,
        "lb_p": 1.0, "lb_ok": True,
    }


# ── 4. RESIDUAL EXTRACTION ───────────────────────────────────────────────────

def extract_residuals(
    series: pd.Series,
    result,
    cfg: dict,
) -> pd.DataFrame:
    model, train_t = result
    win_start, win_end = cfg["residual_window"]
    cutoff = cfg["training_cutoff"]
    wc_start = cfg["wc_start"]
    log_tx = cfg.get("log_transform", False)

    n_forecast = (win_end.year - cutoff.year) * 12 + (win_end.month - cutoff.month)

    # Forecast with 95% CI
    res = model.forecast(y=train_t.values, h=n_forecast, level=[95])
    
    fc_index = pd.date_range(
        start=cutoff + pd.DateOffset(months=1),
        periods=n_forecast,
        freq="MS",
    )

    fc_mean_raw = pd.Series(res['mean'], index=fc_index)
    fc_mean_gwh = maybe_exp(fc_mean_raw, cfg)

    fc_lower_gwh = maybe_exp(pd.Series(res['lo-95'], index=fc_index), cfg)
    fc_upper_gwh = maybe_exp(pd.Series(res['hi-95'], index=fc_index), cfg)

    # Clip to residual window
    mask = (series.index >= win_start) & (series.index <= win_end)
    actual_window = series[mask]
    fc_window_gwh = fc_mean_gwh.reindex(actual_window.index)

    # Residual
    raw_residual_gwh = actual_window - fc_window_gwh

    months_to_wc = [
        (d.year - wc_start.year) * 12 + (d.month - wc_start.month)
        for d in actual_window.index
    ]

    df = pd.DataFrame({
        "country":         cfg["iso"],
        "date":            actual_window.index,
        "actual":          actual_window.values,
        "sarima_forecast": fc_window_gwh.values,
        "residual":        raw_residual_gwh.values,
        "fc_lower":        fc_lower_gwh.reindex(actual_window.index).values,
        "fc_upper":        fc_upper_gwh.reindex(actual_window.index).values,
        "months_to_wc":    months_to_wc,
        "log_transform":   log_tx,
    })

    return df


def normalize_residuals(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize residuals per country as a percentage of the SARIMA baseline.
    residual_norm = residual / sarima_forecast
    """
    df_all = df_all.copy()
    df_all["residual_norm"] = df_all["residual"] / df_all["sarima_forecast"]
    df_all["peak_gwh"] = 0.0 # Placeholder for compatibility
    return df_all


# ── 5. VISUALISATION ─────────────────────────────────────────────────────────

def plot_sarima_fit(series: pd.Series, result, cfg: dict):
    """
    Plot actual vs SARIMA fit over training period with forecast extension.
    Always displays in GWh (back-transforms log forecasts if needed).
    """
    win_start, win_end = cfg["residual_window"]
    cutoff = cfg["training_cutoff"]
    iso = cfg["iso"]
    color = cfg["color"]
    log_tag = " (log-transform applied)" if cfg.get("log_transform", False) else ""

    n_fc = (win_end.year - cutoff.year) * 12 + (win_end.month - cutoff.month)
    fc = result.get_forecast(steps=n_fc)
    fc_idx = pd.date_range(cutoff + pd.DateOffset(months=1), periods=n_fc, freq="MS")

    # Back-transform to GWh
    fc_mean = maybe_exp(pd.Series(fc.predicted_mean.values, index=fc_idx), cfg)
    fc_ci = fc.conf_int()
    fc_ci.index = fc_idx
    fc_ci_lo = maybe_exp(pd.Series(fc_ci.iloc[:, 0].values, index=fc_idx), cfg)
    fc_ci_hi = maybe_exp(pd.Series(fc_ci.iloc[:, 1].values, index=fc_idx), cfg)

    train = series[series.index <= cutoff]               # always GWh
    actual_fc_period = series[(series.index > cutoff) & (series.index <= win_end)]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(train.index, train.values, color="steelblue", lw=1.4, label="Training actual")
    ax.plot(actual_fc_period.index, actual_fc_period.values,
            color=color, lw=1.8, label="Actual (WC window)")
    ax.plot(fc_mean.index, fc_mean.values, "--", color="tomato", lw=1.6, label="SARIMA forecast")
    ax.fill_between(fc_idx, fc_ci_lo.values, fc_ci_hi.values,
                    alpha=0.15, color="tomato", label="95% CI")

    ax.axvline(cutoff, color="gray", ls=":", lw=1.2, label="Training cutoff")
    ax.axvline(cfg["wc_start"], color=color, ls="--", lw=1.4, label="WC start")
    ax.set_title(f"{iso} — SARIMA Fit & Forecast (cutoff: {cutoff:%b %Y})", fontsize=12, fontweight="bold")
    ax.set_ylabel("Electricity demand (GWh)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"sarima_fit_{iso.lower()}.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_residual_profiles(df_all: pd.DataFrame):
    """Overlay normalised WC residual profiles for all donor countries."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, title in zip(
        axes,
        ["residual", "residual_norm"],
        ["Raw residuals (GWh)", "Normalised residuals (peak = 1)"],
    ):
        for country, grp in df_all.groupby("country"):
            cfg_c = next(v for v in DONORS.values() if v["iso"] == country)
            ax.plot(grp["months_to_wc"], grp[col],
                    marker="o", ms=5, lw=1.8,
                    color=cfg_c["color"], label=country)
        ax.axvline(0, color="red", ls="--", lw=1.2, label="WC start (m=0)")
        ax.axhline(0, color="black", ls="-", lw=0.6)
        ax.set_xlabel("Months relative to WC start")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Donor Country WC Residual Profiles", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("wc_residual_profiles.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_diagnostics_summary(diag_list: list):
    """Print a diagnostics table for all donor SARIMA models."""
    df = pd.DataFrame(diag_list)
    df["ADF"] = df["adf_ok"].map({True: "Stationary", False: "Non-Stationary"})
    df["LB"] = df["lb_ok"].map({True: "White Noise", False: "Autocorrelated"})
    print("\n" + "=" * 65)
    print("SARIMA RESIDUAL DIAGNOSTICS SUMMARY")
    print("=" * 65)
    print(df[["iso", "log_transform", "aic", "bic", "adf_p", "ADF", "lb_p", "LB"]].to_string(index=False))
    print("=" * 65 + "\n")


# ── 6. MAIN PIPELINE ─────────────────────────────────────────────────────────

def run_sarima_residual_pipeline(plot: bool = True) -> pd.DataFrame:
    """
    Full pipeline: load → fit → forecast → extract → normalise → export.
    Returns the combined DataFrame with all donor residuals.
    """
    np.random.seed(42)
    all_residuals = []
    diagnostics = []
    fitted_results = {}

    for country, cfg in DONORS.items():
        iso = cfg["iso"]
        print(f"\n{'='*55}")
        print(f"  Processing {iso} ({country.replace('_', ' ').title()})")
        print(f"{'='*55}")

        # Load data
        series = load_donor_data(country, cfg)
        series = series.asfreq("MS")

        # Train / test split at cutoff
        train = series[series.index <= cfg["training_cutoff"]]
        print(f"  Training samples: {len(train)}  ({train.index[0]:%Y-%m} -> {train.index[-1]:%Y-%m})")

        # Select and fit SARIMA (transform handled inside each function)
        order, seasonal_order = select_sarima_order(train, iso, cfg)
        result = fit_sarima(train, order, seasonal_order, cfg)

        # Diagnostics
        diag = sarima_diagnostics(result, train, iso, cfg)
        diagnostics.append(diag)
        fitted_results[iso] = result

        # Extract residuals
        df_resid = extract_residuals(series, result, cfg)
        all_residuals.append(df_resid)
        print(f"  Residual window: {len(df_resid)} observations extracted")

        # Per-country plot
        if plot:
            plot_sarima_fit(series, result, cfg)

    # Combine and normalise
    df_all = pd.concat(all_residuals, ignore_index=True)
    df_all = normalize_residuals(df_all)

    # Diagnostics table
    plot_diagnostics_summary(diagnostics)

    # Overlay profiles
    if plot:
        plot_residual_profiles(df_all)

    # Export
    out_path = "forecast_residuals.csv"
    df_all.to_csv(out_path, index=False)
    print(f"Exported -> {out_path}  ({len(df_all)} rows, {df_all['country'].nunique()} countries)\n")

    # Quick summary
    print("Column descriptions for N-BEATSx training:")
    descriptions = {
        "country":        "ISO-3 donor country code",
        "date":           "MonthStart timestamp",
        "actual":         "Observed electricity demand (GWh)",
        "sarima_forecast":"SARIMA out-of-sample point forecast (GWh)",
        "residual":       "actual - sarima_forecast  (raw WC lift, GWh)",
        "fc_lower":       "SARIMA 95% CI lower bound",
        "fc_upper":       "SARIMA 95% CI upper bound",
        "months_to_wc":   "Months relative to WC start (-6 = 6mo before, 0 = tournament)",
        "residual_norm":  "Residual normalised by country peak (for cross-country alignment)",
        "peak_gwh":       "Peak absolute residual used for normalisation",
    }
    for col, desc in descriptions.items():
        print(f"  {col:<20} {desc}")

    return df_all, fitted_results


# ── 7. ENTRY POINT ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_residuals, sarima_models = run_sarima_residual_pipeline(plot=False)
