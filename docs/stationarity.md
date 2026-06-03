# Stationarity Analysis

Following the Exploratory Data Analysis, rigorous statistical testing is required to determine the integration order ($d, D$) for the SARIMA baseline model. The log-transformed Moroccan electricity demand series must be rendered stationary before autoregressive and moving average parameters can be estimated.

## Statistical Tests

We apply two complementary statistical tests to assess stationarity:

1.  **Augmented Dickey-Fuller (ADF) Test:** Tests the null hypothesis that a unit root is present (series is non-stationary).
2.  **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test:** Tests the null hypothesis that the series is trend-stationary.

## Differencing Strategy

Given the strong trend and 12-month seasonality identified in the EDA, we evaluate the series at different levels of differencing:

1.  **Level (No Differencing):** The raw log-transformed series fails both ADF (cannot reject unit root) and KPSS (rejects stationarity) tests, confirming non-stationarity.
2.  **First Difference ($d=1$):** Removes the long-term trend, but seasonal non-stationarity remains.
3.  **Seasonal Difference ($D=1$):** Removes the seasonal pattern, but a stochastic trend may persist.
4.  **First + Seasonal Difference ($d=1, D=1$):** Applying both operations:
    $$\Delta \Delta_{12} \log(1 + y_t) = (\log(1 + y_t) - \log(1 + y_{t-1})) - (\log(1 + y_{t-12}) - \log(1 + y_{t-13}))$$

    This combined differencing successfully passes both the ADF and KPSS tests, indicating a stationary series suitable for ARMA modeling.

## Conclusion

The stationarity analysis unequivocally prescribes the integration parameters for the SARIMA model:

*   Non-seasonal differencing order: **$d = 1$**
*   Seasonal differencing order: **$D = 1$** (with period $s = 12$)

This structural choice ensures that the Layer 1 baseline model correctly accounts for Morocco's underlying growth trajectory and seasonal cycles before attempting to inject the World Cup uplift.
