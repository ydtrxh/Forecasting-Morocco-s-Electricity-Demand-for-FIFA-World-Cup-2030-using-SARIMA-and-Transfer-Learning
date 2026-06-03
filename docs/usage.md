# Usage Guide

This page explains how to use the pre-trained models and generated data for your own analysis without re-running the entire training pipeline.

## Accessing the Forecast Data

The final output of the project is the deployment forecast for Morocco (January 2026 – December 2030). This data is provided in a clean CSV format.

**File Location:** `outputs/morocco_2030_forecast.csv`

### Loading with Pandas

```python
import pandas as pd

# Load the forecast
forecast_df = pd.read_csv("outputs/morocco_2030_forecast.csv")
forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

# Display the World Cup event window
event_window = forecast_df[forecast_df["in_event_window"] == True]
print(event_window[["ds", "predicted_GWh", "wc_lift_GWh", "wc_lift_pct"]])
```

## Loading the Pre-Trained Neural Kernel

If you want to extract the learned pulse shape $f_\theta(t_{\text{norm}})$ directly for visualization or transfer to another country, you can load the final PyTorch weights.

**File Location:** `models/kernel_final_modelC.pt`

### Inference Code

```python
import torch
import torch.nn as nn
import numpy as np

# 1. Define the architecture (Model C)
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

# 2. Instantiate and load weights
kernel = NeuralEventKernel(hidden_dim=16, n_layers=2, n_donors=5)
kernel.load_state_dict(torch.load("models/kernel_final_modelC.pt", map_location="cpu"))
kernel.eval()

# 3. Generate the normalized pulse shape
t_seq = torch.tensor([t / 6.0 for t in range(-5, 7)], dtype=torch.float32).unsqueeze(1)
with torch.no_grad():
    pulse_shape = kernel(t_seq).squeeze().numpy()

print(pulse_shape)
```

## Exploring the Notebooks

The `notebooks/` directory contains interactive Jupyter notebooks that break down the methodology. They are highly recommended for understanding the underlying data logic.

- **`01_eda_morocco.ipynb`**: Walkthrough of stationarity testing and SARIMA parameter selection.
- **`03_loo_kernel_validation.ipynb`**: Detailed analysis of the 150-run grid search, including architecture comparisons and stability checks.
- **`04_morocco_2030_deployment.ipynb`**: Step-by-step reconstruction of the final forecast chart.
