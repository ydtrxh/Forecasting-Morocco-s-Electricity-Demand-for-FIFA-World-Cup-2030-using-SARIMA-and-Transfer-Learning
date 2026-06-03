# Experimental Results

This section details the empirical findings from the 150-configuration Leave-One-Out (LOO) grid search.

## Best Configuration

The rigorous cross-validation identified the following configuration as the optimal model for transferring event demand shocks:

- **Architecture:** `Model C` (2 layers, 16 hidden units)
- **Weighting Scheme:** Proportional weighted transfer
- **Amplitude Regularization ($\lambda_a$):** 0.001 (Weak)
- **Configuration ID:** `Model_C_weighted_lam0.001`

### Performance Metrics (Averaged across 5 LOO folds)

| Metric | Value |
|:-------|:-----:|
| **Mean LOO RMSE** | **687.2 GWh** |
| **Mean LOO MAPE** | **2.83%** |
| **Mean Shape MSE** | **0.204** |
| **Mean Pulse Variance** | 0.045 |

---

## Architectural Comparison

The depth of the neural event kernel proved critical for capturing the complex dynamics of a mega-event.

| Architecture | Mean RMSE (GWh) | Improvement vs Model A |
|:-------------|:---------------:|:----------------------:|
| Model A (4-dim, 1 layer) | 6,282.3 | Baseline |
| Model B (8-dim, 1 layer) | 4,751.4 | +24.4% |
| **Model C (16-dim, 2 layers)** | **2,580.6** | **+58.9%** |

*Note: Averages taken across all regularization and weighting schemes for each architecture.*

**Conclusion:** Shallow networks (Model A) suffer from high structural bias and cannot model the non-linear phase transitions of a World Cup (slow ramp-up, extended plateau, rapid decay). Model C's extra capacity allows it to learn this canonical pulse geometry.

---

## Weighting Scheme Comparison

We tested whether transferring amplitude via a simple uniform average of donors ($w_i = 0.2$) was inferior to a proportional weighting scheme based on grid-scale similarity to the target (Morocco).

| Scheme | Mean RMSE (GWh) |
|:-------|:---------------:|
| Uniform Transfer | 4,557.6 |
| **Proportional Weighted** | **4,518.6** |

**Conclusion:** Domain-knowledge-based proportional weighting yields a consistent, albeit modest, improvement. The similarity of the underlying pulse shapes across donors means the primary benefit of weighting lies in anchoring the transferred amplitude heavily to the most structurally similar analog (Qatar).

---

## Regularization Sensitivity

The amplitude regularization parameter $\lambda_a$ controls how aggressively the learned amplitudes are shrunk toward the mean. 

**Finding:** Weak regularization ($\lambda_a = 0.001$) is optimal. Stronger penalties ($\lambda_a \geq 1.0$) force amplitudes toward zero to minimize the penalty term, which destroys the magnitude information necessary for accurate GWh reconstruction, resulting in severe under-forecasting.

---

## Key Insights

1. **Log-Space Structural Invariance is Valid:** The success of the LOO reconstruction validates the core hypothesis: the relative (log-space) demand pulse of a mega-event is highly consistent across nations, bridging grid scales from Cameroon (~600 GWh) to Russia (~85,000 GWh).
2. **Egypt is a Structural Outlier:** Analysis of fold-level metrics reveals that reconstructing Egypt's AFCON 2019 signal is the most difficult task. This is consistent with its highly unusual summer-peaking grid profile combined with the atypically extreme heat that dominated demand during that specific tournament.
