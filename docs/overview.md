# Project Overview

## Motivation

Morocco is co-hosting the **2030 FIFA World Cup** alongside Spain and Portugal, with its venues set to host a substantial share of the matches. This represents an unprecedented electricity demand planning challenge: Morocco's national grid operator (ONEE) must reliably supply power to stadiums, hotels, transportation networks, fan zones, and media infrastructure—many of which are entirely new constructions—over a twelve-month period of elevated, event-driven demand.

Standard forecasting frameworks, including SARIMAX, are designed to extrapolate historical growth trends and seasonal patterns. They have **no mechanism to model the demand shock** that accompanies a mega-event—a systematic, time-localized upward perturbation that begins months before the tournament (infrastructure commissioning, logistics mobilization), peaks during match play, and decays in the weeks following the final.

---

## The Fundamental Challenge

The core difficulty is one of **data scarcity combined with structural novelty**:

- Morocco has **never hosted a FIFA World Cup** — its electricity demand series contains no such event signature.
- The demand lift produced by a World Cup is a **rare, high-impact shock** occurring at most once per decade at the national level.
- Classical models (ARIMA, exponential smoothing) cannot extrapolate to event conditions they have never observed.
- Deep learning models trained solely on Morocco's ~108-month history cannot learn a transferable event representation from so few observations.

This is fundamentally a **zero-shot transfer learning problem**: we must infer an event signature that has never been seen in the target domain.

---

## Why Standard Models Fail

Consider a naive SARIMAX approach:

1. Fit SARIMA on Morocco's 2016–2025 monthly consumption.
2. Project forward through 2030.
3. Add a dummy variable for the World Cup months.

The coefficient on the dummy variable **cannot be estimated** — there are zero historical observations of a World Cup in Morocco. The model would either refuse to estimate the parameter, revert to zero, or require an arbitrary prior. Any output would be statistically meaningless.

---

## Proposed Hybrid Methodology

This project resolves the zero-shot problem with a **two-layer hybrid architecture** that separates the forecasting problem into two tractable sub-problems:

### Layer 1 — SARIMA Counterfactual Baseline

A locally fitted Seasonal ARIMA model captures Morocco's historical growth trend and multiplicative seasonality, producing a statistically rigorous **counterfactual forecast**: what demand would be in the absence of any World Cup effect.

$$y_t = \text{SARIMA}(1,1,1) \times (0,1,2)_{12} \quad \text{on} \quad \log_{1+}(\text{consumption}_{\text{GWh}})$$

Key design decisions:
- **Log-transformation** (`log1p`) stabilizes multiplicative seasonal variance.
- **`d=1, D=1`** differencing imposed from stationarity tests.
- Model order selected by minimizing AIC on the full 108-month training window.
- 85% prediction intervals computed analytically.

### Layer 2 — Transfer-Learned Neural Event Kernel

A compact neural network $f_\theta(t_{\text{norm}})$ is trained on **five donor countries** that have previously hosted the World Cup or AFCON tournaments. The network learns a **globally shared, normalized event pulse shape** that is structurally invariant across donor countries when expressed in log-space.

This shape is then transferred to Morocco via a **weighted amplitude scalar** $a_{\text{Morocco}} = \sum_i w_i a_i$, and injected into the SARIMA baseline multiplicatively in log-space.

The combination produces a forecast that is both:
- **Statistically credible** — anchored by SARIMA, with proper prediction intervals.
- **Event-aware** — informed by cross-country transfer learning from five historical mega-events.

---

## Research Significance

This methodology addresses a class of problem that is increasingly common in energy economics and infrastructure planning: **forecasting the effect of unprecedented events on complex systems**. The donor-country transfer learning framework is generalizable to:

- Olympic Games hosting
- Major international conferences (e.g., COP climate summits)
- Expo and World Fair hosting
- Pandemic demand shocks (where donor countries offer early-wave data)

The project demonstrates that **structural invariance in log-space**, combined with **amplitude calibration via grid-scale similarity**, can enable robust transfer learning even when the magnitude difference between donor and target systems spans two orders of magnitude.
