# Research Question

## Primary Research Question

> **Can electricity demand shocks observed during previous FIFA World Cup and AFCON tournaments in donor countries be reliably transferred to estimate Morocco's electricity demand during the 2030 FIFA World Cup, and how should the transferred amplitude be calibrated?**

---

## Sub-Questions

### Q1 — Shape Invariance

*Is the event uplift pulse shape structurally invariant across donor countries with different grid scales when expressed in log-space?*

This question motivates the core architectural choice. If the log-space uplift shape $f_\theta(t_{\text{norm}})$ is country-agnostic, then a single shared neural network can represent all donors and transfer its learned geometry to Morocco. If it is not invariant, the transfer learning approach would be invalidated.

**Finding:** The analysis shows strong shape invariance across Qatar, Egypt, South Africa, and Cameroon. Russia exhibits partial deviation, attributed to its summer-peaking grid profile and the atypically weak 2018 WC signal relative to Russia's large baseline demand.

---

### Q2 — Architecture Selection

*Which neural kernel architecture and regularization setting best recovers the held-out donor uplift in a Leave-One-Out (LOO) cross-validation framework?*

Three architectures are tested:

| Architecture | Hidden Dim | Layers | Parameters |
|---|---|---|---|
| Model A | 4 | 1 | ~25 |
| Model B | 8 | 1 | ~57 |
| **Model C** | **16** | **2** | **~305** |

Five amplitude regularization strengths $\lambda_a \in \{0.001, 0.01, 0.1, 1.0, 10.0\}$ and two weighting schemes (uniform, proportional) are evaluated across all architectures, giving **30 configurations × 5 folds = 150 training runs**.

**Finding:** Model C with proportional weighting and $\lambda_a = 0.001$ achieves the best LOO RMSE (687.2 GWh, MAPE 2.83%).

---

### Q3 — Prediction Interval Range

*What is the statistically defensible range for Morocco's grid demand during the 12-month event window?*

The 85% prediction interval is computed analytically by `statsforecast` on the SARIMA log-space forecast, then back-transformed individually using `expm1` on both bounds. The WC uplift is injected multiplicatively, propagating the interval through:

$$\hat{y}^{(85\%\,\text{lower})}(t) = \bigl(y^{\text{lo-85}}_{\text{SARIMA}}(t) + 1\bigr)\cdot\exp(a_{\text{Morocco}} \cdot f_\theta(t_{\text{norm}})) - 1$$

**Finding:** Peak month (October 2030) interval: **4,913.6 — 6,343.0 GWh** (point forecast: 5,582.8 GWh).

---

### Q4 — Weighting Scheme

*Does a proportional donor weighting scheme (based on grid scale similarity to Morocco) outperform a uniform transfer?*

Donor relevance weights are assigned based on structural proximity to Morocco's grid scale (~3,500–5,500 GWh/month):

| Donor | Raw Weight $r_i$ | Normalized $w_i$ | Rationale |
|---|---|---|---|
| QAT | 1.0 | 0.323 | Closest grid scale to Morocco |
| EGY | 0.7 | 0.226 | Similar MENA context |
| ZAF | 0.6 | 0.194 | Sub-Saharan Africa reference |
| CMR | 0.5 | 0.161 | African tournament experience |
| RUS | 0.3 | 0.097 | Grid 20× larger, weak signal |

**Finding:** Proportional weighted transfer achieves mean RMSE of 4,518.6 GWh vs. 4,557.6 GWh for uniform, confirming that domain-knowledge-based weighting provides consistent improvement. The margin is modest because the pulse shapes are similar across donors; the main gain comes from correct amplitude calibration via Qatar dominance.
