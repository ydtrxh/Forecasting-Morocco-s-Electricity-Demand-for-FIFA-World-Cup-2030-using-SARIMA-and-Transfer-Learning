import json
import os

def new_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def new_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# --- 01 - Project Overview ---
notebook["cells"].append(new_markdown_cell("""<div align="center">

# ⚡ Forecasting Morocco's Electricity Demand for FIFA World Cup 2030
### *A Hybrid SARIMA & Transfer-Learned Neural Event Kernel Approach*

</div>

---

## 01. Project Overview

Morocco is co-hosting the **2030 FIFA World Cup**. This mega-event presents an unprecedented demand shock to the Moroccan electricity grid (ONEE). Standard forecasting frameworks (like ARIMA or Prophet) fail to capture this because **Morocco has never hosted a World Cup**, meaning there is no historical event signature to learn from.

### The Solution
This project proposes a **two-layer hybrid architecture**:
1. **Layer 1 (SARIMA)**: Captures the historical counterfactual trend and seasonality of Morocco's grid.
2. **Layer 2 (Neural Event Kernel)**: A compact neural network trained on *donor countries* (Qatar, Russia, South Africa, Egypt, Cameroon) that have hosted similar events. It learns a normalized "event pulse shape", which is then transferred and dynamically scaled for Morocco."""))

notebook["cells"].append(new_code_cell("""import sys
import os
import pandas as pd
from IPython.display import Image, display

# Ensure we can import from the src directory
sys.path.append(os.path.abspath('../src'))

# Optional: Set pandas display options for cleaner tables
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)"""))

# --- 02 - Data Exploration ---
notebook["cells"].append(new_markdown_cell("""---
## 02. Data Exploration

To understand the baseline trajectory, we first ingest and visualize Morocco's historical electricity demand (2016–2025). We also visualize the **donor residuals**—the log-space demand shocks extracted from previous host countries."""))

notebook["cells"].append(new_code_cell("""# Load Morocco Historical Data
df_morocco = pd.read_csv('../data/raw/consommation_electrique_maroc_2016_2025_final.csv', index_col='Date', parse_dates=True)
display(df_morocco.tail())

# Display Exploratory Data Analysis (EDA) plot for Morocco
display(Image(filename='../figures/eda/morocco_eda.png', width=900))"""))

notebook["cells"].append(new_code_cell("""# Display extracted donor World Cup/AFCON event pulses
display(Image(filename='../figures/eda/wc_residual_profiles.png', width=900))"""))

# --- 03 - Stationarity Analysis ---
notebook["cells"].append(new_markdown_cell("""---
## 03. Stationarity Analysis

Before modeling the SARIMA baseline, we must ensure the series is stationary. The ACF and PACF plots demonstrate strict 12-month seasonality and non-stationarity, justifying our $d=1$ and $D=1$ differencing parameters."""))

notebook["cells"].append(new_code_cell("""# Display ACF/PACF Analysis
display(Image(filename='../figures/eda/morocco_acf_pacf.png', width=900))"""))

# --- 04 - SARIMA Baseline ---
notebook["cells"].append(new_markdown_cell("""---
## 04. SARIMA Baseline Development

Using the rigorous Box-Jenkins methodology and `StatsForecast AutoARIMA`, we fitted a $SARIMA(1,1,1)\\times(0,1,2)_{12}$ model on the log-transformed data. This acts as our **Counterfactual**—what demand would look like *without* the World Cup."""))

notebook["cells"].append(new_code_cell("""# Display SARIMA Residual Diagnostics
display(Image(filename='../figures/sarima/morocco_residual_diagnostics.png', width=900))"""))

notebook["cells"].append(new_code_cell("""# Display Model Stability Check (Rolling Origin CV)
display(Image(filename='../figures/validation/morocco_cv_rmse_over_time.png', width=800))"""))

# --- 05 - Neural Event Kernel ---
notebook["cells"].append(new_markdown_cell("""---
## 05. Neural Event Kernel Methodology

The core innovation is the **Neural Event Kernel** $f_\\theta(t_{norm})$. 
Instead of predicting absolute GWh, the network predicts a dimensionless, normalized pulse shape $\\in [-1, 1]$. 

The full reconstruction formula per event-window month $t \\in [-5, +6]$ is:
$$ \\hat{y}(t) = \\exp\\Bigl(\\log(1 + y_{SARIMA}(t)) + a_i \\cdot f_\\theta(t_{norm})\\Bigr) - 1 $$
where $a_i$ is a learned per-donor amplitude scalar."""))

notebook["cells"].append(new_code_cell("""# Display the learned kernel pulse shape and normalized donor actuals
display(Image(filename='../figures/kernel/final_kernel_pulse_shape.png', width=800))"""))

notebook["cells"].append(new_code_cell("""# Display training loss convergence
display(Image(filename='../figures/kernel/final_kernel_training_loss.png', width=600))"""))

# --- 06 - Leave-One-Out Validation ---
notebook["cells"].append(new_markdown_cell("""---
## 06. Leave-One-Out Validation Results

To prove that the learned kernel generalizes to an unseen grid (simulating Morocco), we performed a **150-run Grid Search** using a Leave-One-Out (LOO) cross-validation scheme across all 5 donors. In each fold, a donor was held out, the model trained on the remaining 4, and the shape was transferred to the held-out target."""))

notebook["cells"].append(new_code_cell("""# Display the LOO Forecast Grid across all 5 hold-out scenarios
display(Image(filename='../figures/validation/loo_forecast_grid.png', width=1000))"""))

notebook["cells"].append(new_code_cell("""# Display Kernel Shape vs Actual Uplift during LOO transfer
display(Image(filename='../figures/kernel/kernel_shape_vs_actual.png', width=900))"""))

# --- 07 - Model Selection ---
notebook["cells"].append(new_markdown_cell("""---
## 07. Model Selection

The grid search tested:
- **Architectures**: Model A (Dim=4), Model B (Dim=8), Model C (Dim=16, 2 Layers)
- **Transfer Weights**: Uniform vs. Proportional (based on grid similarity)
- **Regularization ($\\lambda_a$)**: 0.001 to 10.0

**Model C with proportional weighting and weak regularization ($\\lambda=0.001$)** heavily outperformed the others, achieving a mean out-of-sample RMSE of **687 GWh**."""))

notebook["cells"].append(new_code_cell("""# Load and display the aggregated LOO metrics
df_loo_agg = pd.read_csv('../outputs/tables/loo_kernel_aggregate.csv')

# Sort to show the best models at the top
df_loo_best = df_loo_agg.sort_values('Mean_RMSE').head(10)
display(df_loo_best)"""))

notebook["cells"].append(new_code_cell("""# Display Regularization Sensitivity
display(Image(filename='../figures/kernel/lambda_amp_sensitivity.png', width=700))"""))

# --- 08 - Morocco 2030 Forecast ---
notebook["cells"].append(new_markdown_cell("""---
## 08. Morocco 2030 Forecast Deployment

Using the optimal architecture, we trained the final kernel on **all 5 donors**. We transferred the pulse to Morocco using a weighted amplitude scalar:
$$ a_{Morocco} = \\sum w_i \\cdot a_i = 0.0599 $$
which equates to a peak uplift of approximately **+6.14%**."""))

notebook["cells"].append(new_code_cell("""# Load the final deployment forecast table
df_forecast = pd.read_csv('../outputs/forecasts/morocco_2030_forecast.csv')

# Highlight the event window (June 2030 to December 2030)
display(df_forecast.tail(12))"""))

notebook["cells"].append(new_code_cell("""# Display the final, capstone Morocco 2030 trajectory
display(Image(filename='../figures/deployment/morocco_2030_forecast.png', width=1000))"""))

# --- 09 - Conclusions ---
notebook["cells"].append(new_markdown_cell("""---
## 09. Conclusions

1. **Successful Transfer:** The log-space multiplicative residual approach successfully bridged massive grid-scale differences (from Cameroon's 600 GWh to Russia's 85,000 GWh).
2. **Deep Architecture Needed:** The complex dynamics of a mega-event (slow ramp-up, peak, rapid decay) require non-linear representational capacity (Model C > Model A).
3. **Morocco Projections:** The model projects a peak demand lift of **+322.8 GWh** in October 2030 (+6.14% above baseline), representing significant, critical pressure on ONEE's infrastructure planning.
4. **Reproducibility:** The full training pipeline, SARIMA baseline scripts, and environment dependencies are fully open-sourced in this repository.

*End of Showcase Notebook.*"""))

# Write to file
with open('notebooks/Time_series_project.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully at notebooks/Time_series_project.ipynb")
