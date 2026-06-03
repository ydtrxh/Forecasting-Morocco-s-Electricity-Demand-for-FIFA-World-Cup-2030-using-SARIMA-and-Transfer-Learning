# Repository Structure

The project is organized into clear directories separating raw data, Jupyter notebooks for exploration, production source code, trained models, and generated outputs.

```text
wc2030-morocco-electricity-forecast/
│
├── 📄 README.md                        ← Project overview
├── 📄 LICENSE
├── 📄 requirements.txt                 ← Python dependencies
├── 📄 .gitignore
│
├── 📁 data/
│   ├── raw/                            ← Original, immutable source files
│   │   ├── consommation_electrique_maroc_2016_2025_final.csv
│   │   ├── qatar_electricity_transmitted.csv
│   │   ├── south_africa_donor_electricity_demand.csv
│   │   ├── cameroon_monthly_electricity_consumption.csv
│   │   ├── Russia_data.csv
│   │   └── monthly_full_release_long_format.csv  ← IEA dataset (EGY, RUS, QAT)
│   └── processed/
│       ├── donor_residuals_normalized.csv        ← Log-space uplift targets
│       ├── donor_residuals_augmented.csv         ← Augmented profiles
│       └── morocco_cv_results.csv                ← Rolling-origin CV metrics
│
├── 📁 notebooks/
│   ├── 01_eda_morocco.ipynb            ← EDA: Morocco series
│   ├── 02_donor_residual_extraction.ipynb  ← SARIMA counterfactual extraction
│   ├── 03_loo_kernel_validation.ipynb  ← LOO grid search results analysis
│   └── 04_morocco_2030_deployment.ipynb ← Final forecast visualization
│
├── 📁 src/
│   ├── morocco_sarima_baseline.py      ← Layer 1 baseline pipeline
│   ├── sarima_residual_extraction.py   ← Donor counterfactual extraction
│   ├── loo_kernel_pipeline.py          ← 150-run LOO grid search script
│   ├── train_final_kernel.py           ← Final Neural Kernel training
│   └── morocco_2030_deployment.py      ← Full deployment pipeline
│
├── 📁 models/
│   └── kernel_final_modelC.pt          ← Final PyTorch weights
│
├── 📁 outputs/
│   ├── morocco_2030_forecast.csv       ← Final monthly forecast (2026–2030)
│   ├── loo_kernel_aggregate.csv        ← Aggregated LOO metrics (30 configs)
│   └── loo_kernel_summary.csv          ← Fold-level metrics (150 runs)
│
└── 📁 figures/
    ├── morocco_eda.png                 ← Raw series, rolling stats
    ├── morocco_acf_pacf.png            ← ACF/PACF diagnostics
    ├── final_kernel_pulse_shape.png    ← Learned Model C pulse shape
    └── morocco_2030_forecast.png       ← Final deployment visualization
```
