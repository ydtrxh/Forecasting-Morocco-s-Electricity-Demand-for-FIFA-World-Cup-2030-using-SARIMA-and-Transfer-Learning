# Reproducing the Paper Results

To ensure total transparency and reproducibility, the entire research pipeline from raw data to final deployment is orchestrated via a sequence of Python scripts in the `src/` directory.

Follow these steps in order to reproduce all intermediate tables, trained models, and final figures.

---

## Step 1: Morocco SARIMA Baseline Validation

This script fits the `AutoARIMA` model on Morocco's historical data, verifies the $(1, 1, 1) \times (0, 1, 2)_{12}$ specification, and calculates the baseline performance metrics.

```bash
python src/morocco_sarima_baseline.py
```
*Expected output: AIC/BIC scores, chosen orders, and cross-validation RMSE.*

## Step 2: Donor Residual Extraction

This script iterates over the five donor countries (Qatar, Egypt, Russia, South Africa, Cameroon). For each, it trains a local SARIMA model up to 6 months prior to their respective tournament, generates a counterfactual, and extracts the log-space uplift residual.

```bash
python src/sarima_residual_extraction.py
```
*Expected output: `data/processed/donor_residuals_normalized.csv`*

## Step 3: LOO Kernel Grid Search

This is the most computationally intensive step. It executes the 150-configuration Leave-One-Out (LOO) cross-validation grid search to identify the optimal architecture and regularization parameters.

```bash
python src/loo_kernel_pipeline.py
```
*Expected runtime: 20–40 minutes on a standard CPU.*
*Expected output: `outputs/loo_kernel_summary.csv` and `outputs/loo_kernel_aggregate.csv`*

## Step 4: Final Kernel Training

Using the optimal configuration identified in Step 3 (`Model_C`, `lambda_amp=0.001`, `proportional weights`), this script trains the final Neural Event Kernel jointly on all five donors simultaneously.

```bash
python src/train_final_kernel.py
```
*Expected output: The PyTorch weight file `models/kernel_final_modelC.pt`*

## Step 5: Morocco 2030 Deployment

The final script stitches the pipeline together. It fits the deployment SARIMA on Morocco's full history, extracts the pulse shape and transferred amplitude from the final kernel, injects the uplift, and plots the final result.

```bash
python src/morocco_2030_deployment.py
```
*Expected output: `outputs/morocco_2030_forecast.csv` and `figures/morocco_2030_forecast.png`*
