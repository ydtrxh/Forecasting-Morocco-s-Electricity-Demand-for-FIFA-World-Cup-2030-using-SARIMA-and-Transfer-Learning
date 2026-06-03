# Future Work

The current forecasting pipeline provides a robust, statistically defensible estimate of the World Cup electricity demand shock. However, several methodological extensions could further refine the precision and reliability of the forecast.

## 1. Exogenous Regressors (SARIMAX)

The current Layer 1 baseline relies purely on endogenous historical demand data. Integrating exogenous macroeconomic and demographic regressors could improve the stability of the 5-year trend extrapolation (2026–2030):

- **GDP Growth Projections:** Real GDP growth is a primary driver of baseline electricity demand.
- **Population Growth:** Structural demographic shifts affect base load.
- **Electrification Policy Targets:** Government targets for rural electrification and industrial zone development.

## 2. Weather De-weathering (Climate Reanalysis)

Morocco's grid is highly sensitive to summer temperatures (cooling demand). The current model uses deterministic seasonality to capture this effect. 

A significant improvement would involve **de-weathering** the historical series using ERA5 climate reanalysis data (e.g., Cooling Degree Days and 2m temperature profiles). This would decompose the seasonal variance into a pure structural component and a stochastic weather-driven component, preventing extreme historical heatwaves from artificially inflating the baseline forecast.

## 3. Conformal Prediction Intervals

Currently, the 85% prediction intervals are generated analytically under Gaussian assumptions by the SARIMA model. While computationally efficient, these intervals may not perfectly cover complex, non-normal real-world variance.

Implementing **Split Conformal Prediction** on top of the point forecasts would provide distribution-free, mathematically guaranteed coverage calibration for the uncertainty bounds.

## 4. Probabilistic Amplitude Transfer

The transfer amplitude $a_{\text{Morocco}}$ is currently a deterministic point estimate (weighted average). 

A Bayesian approach could model the donor amplitudes as samples from a hierarchical prior distribution. The resulting posterior over $a_{\text{Morocco}}$ would allow the transfer uncertainty to be rigorously propagated into the final prediction intervals, widening them to reflect the inherent epistemic uncertainty of the transfer.

## 5. Post-Event Infrastructure Legacy

The current Neural Event Kernel decays back to the baseline by Month +6. In reality, mega-events leave a permanent infrastructure footprint (new stadiums, expanded transit networks, upgraded hotels) that permanently shifts the baseline demand upwards.

Future work could fit a parametric **level-shift function** (e.g., a Gompertz curve) calibrated on historical analogs (like Barcelona 1992 or South Korea 2002) to model this permanent legacy effect from 2031 onward.
