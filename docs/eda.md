# Exploratory Data Analysis

Exploratory Data Analysis (EDA) on the Moroccan electricity demand series establishes the foundational characteristics that inform the Layer 1 SARIMA modeling strategy.

## Morocco Demand Characteristics

Morocco's monthly electricity demand (2016–2025) exhibits three dominant structural components:

1.  **Strong Multiplicative Seasonality:** A pronounced 12-month cycle with a significant peak in the summer months (July/August), driven by intense cooling demand. The amplitude of this seasonal peak grows proportionally with the baseline level of demand, indicating a multiplicative relationship.
2.  **Long-Term Upward Trend:** A consistent, albeit occasionally perturbed, secular growth trend reflecting macroeconomic expansion, population growth, and increasing electrification rates.
3.  **Variance Instability:** As the absolute level of demand increases, the variance of the seasonal fluctuations also increases.

## Transformation Strategy

To address the multiplicative seasonality and variance instability, the raw demand series $y_t$ (in GWh) is transformed using the natural logarithm:

$$y_t' = \log(1 + y_t)$$

This log transformation converts the multiplicative seasonality into additive seasonality, making the series amenable to linear modeling frameworks like ARIMA, and stabilizes the variance across the observation period.

## Visual Diagnostics

*(Refer to `figures/morocco_eda.png` and `figures/morocco_acf_pacf.png` in the repository for detailed visual diagnostics.)*

*   **Time Series Plot (Log Scale):** Clearly reveals the stabilized variance and linearized trend post-transformation.
*   **Autocorrelation Function (ACF):** Strong periodic spikes at lags 12, 24, 36... confirm the deterministic 12-month seasonality. Slow decay at early lags indicates non-stationarity in the mean (presence of trend).
*   **Partial Autocorrelation Function (PACF):** Significant spikes at early lags (1, 2) and seasonal lags (12) suggest the need for both non-seasonal and seasonal autoregressive terms in the final model specification.

These EDA findings directly guide the formal stationarity testing and the specification of the SARIMA integration parameters ($d$ and $D$).
