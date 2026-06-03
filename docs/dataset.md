# Dataset Description

## Data Sources

This project relies on historical monthly electricity consumption data for Morocco and five donor countries. 

1. **Morocco:** Historical electricity consumption (Jan 2016 – Dec 2025) provided by the **Office National de l'Électricité et de l'Eau Potable (ONEE)**.
2. **Donor Countries (Egypt, Russia, Qatar):** Sourced from the **International Energy Agency (IEA) Monthly Electricity Statistics**.
3. **Donor Countries (Cameroon, South Africa):** Sourced from national grid operators / local statistical agencies, processed into monthly aggregates.

All datasets represent **net electricity demand/consumption** measured in Gigawatt-hours (GWh).

---

## Donor Events Summary

The core dataset contains event signatures from the following historical FIFA World Cup (WC) and African Cup of Nations (AFCON) tournaments:

| Country | Tournament | Kickoff Date | Grid Scale (GWh/month) | Event Signal Quality |
|:--------|:-----------|:-------------|:----------------------:|:--------------------:|
| **Qatar (QAT)** | FIFA World Cup 2022 | November 2022 | ~3,500–5,000 | ✅ Strong |
| **Russia (RUS)** | FIFA World Cup 2018 | June 2018 | ~85,000–95,000 | ⚠️ Weak |
| **South Africa (ZAF)** | FIFA World Cup 2010 | June 2010 | ~18,000–22,000 | ⚠️ Moderate |
| **Egypt (EGY)** | AFCON 2019 | June 2019 | ~14,000–18,000 | ✅ Moderate |
| **Cameroon (CMR)** | AFCON 2022 | January 2022 | ~600–800 | ✅ Moderate |

> **Grid Scale Threshold Rule:** Morocco's current grid scale (~3,500–5,500 GWh/month) most closely matches **Qatar 2022**. This structural similarity makes Qatar the most relevant analog for establishing baseline demand uplift proportions, justifying its highest weighting in the amplitude transfer model.

---

## Data Preprocessing

To ensure comparability and model stability, the following preprocessing steps are uniformly applied:

1. **Temporal Alignment:** All series are resampled to a strict Start-of-Month (`MS`) frequency. Missing values within the historical window are imputed using seasonal interpolation.
2. **Log Transformation:** Demand series are converted to log-space using `np.log1p()`. This serves two critical functions:
    * Stabilizes the variance of multiplicative seasonal patterns.
    * Converts absolute GWh differences into scale-invariant percentage uplifts, enabling direct comparison between countries with vastly different grid sizes (e.g., Cameroon vs. Russia).
3. **Event Window Alignment:** For the neural event kernel, chronological dates are mapped to a relative time grid $t \in [-5, +6]$ centered on the tournament kickoff month ($t=0$).

The preprocessed data forms the foundation for both the Layer 1 SARIMA baseline modeling and the Layer 2 Neural Event Kernel extraction.
