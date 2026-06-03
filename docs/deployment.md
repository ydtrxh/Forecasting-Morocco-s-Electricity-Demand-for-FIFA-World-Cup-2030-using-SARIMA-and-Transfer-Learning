# Morocco 2030 Deployment

With the optimal hyperparameter configuration (`Model_C_weighted_lam0.001`) identified via LOO cross-validation, the final deployment pipeline generates the 60-month operational forecast for Morocco (January 2026 – December 2030).

## 1. Final Model Training

Instead of holding out a donor, the final Neural Event Kernel is trained jointly on **all five donor countries** simultaneously. This leverages the maximum available historical information to learn the most robust event pulse shape possible.

The resulting transferred amplitude for Morocco is a weighted average of all 5 learned donor amplitudes:

$$a_{\text{Morocco}} = \sum_i w_i \cdot a_i = 0.0599$$

This amplitude implies a peak unscaled uplift of approximately **+6.17%** relative to the counterfactual baseline.

## 2. Final SARIMA Baseline

The SARIMA baseline model is fitted on Morocco's complete historical record up to December 2025. It projects the underlying growth trend and standard seasonal cycling forward for 60 months. 

Crucially, **85% prediction intervals** are generated along with the point forecast to quantify the inherent uncertainty in the 5-year macro projection.

## 3. The Injection Phase

The World Cup uplift is injected exclusively during the defined 12-month event window (January 2030 to December 2030, corresponding to $t \in [-5, +6]$ relative to the June 2030 kickoff).

The multiplicative injection is applied independently to the point forecast and the upper/lower bounds:

$$\hat{y}(t) = \bigl(y_{\text{SARIMA}}(t) + 1\bigr) \cdot \exp\!\bigl(0.0599 \cdot f_{\theta,\text{final}}(t_{\text{norm}})\bigr) - 1$$

## 4. Final Forecast Trajectory

| Month | Baseline (GWh) | Predicted (GWh) | Lower 85% | Upper 85% | Net Lift (GWh) | Lift (%) |
|:------|:--------------:|:---------------:|:---------:|:---------:|:--------------:|:--------:|
| Jan 2030 | 4,778.5 | 4,942.0 | 4,405.4 | 5,543.8 | +163.4 | +3.42% |
| Feb 2030 | 4,405.1 | 4,554.0 | 4,051.0 | 5,119.5 | +148.9 | +3.38% |
| Mar 2030 | 4,764.0 | 4,928.2 | 4,376.8 | 5,549.0 | +164.2 | +3.45% |
| Apr 2030 | 4,672.7 | 4,843.1 | 4,295.2 | 5,460.9 | +170.5 | +3.65% |
| May 2030 | 5,099.7 | 5,303.3 | 4,697.1 | 5,987.7 | +203.6 | +3.99% |
| **Jun 2030** | **5,184.4** | **5,415.3** | **4,790.1** | **6,122.1** | **+230.9** | **+4.45%** |
| Jul 2030 | 5,866.0 | 6,158.1 | 5,440.1 | 6,970.8 | +292.1 | +4.98% |
| Aug 2030 | 5,869.7 | 6,191.8 | 5,463.0 | 7,017.7 | +322.1 | +5.49% |
| Sep 2030 | 5,334.5 | 5,648.9 | 4,977.9 | 6,410.4 | +314.5 | +5.90% |
| **Oct 2030** | **5,260.0** | **5,582.8** | **4,913.6** | **6,343.0** | **+322.8** | **+6.14%** |
| Nov 2030 | 4,843.4 | 5,142.5 | 4,520.6 | 5,849.8 | +299.1 | +6.18% |
| Dec 2030 | 4,941.9 | 5,239.1 | 4,600.1 | 5,967.0 | +297.2 | +6.01% |

### Strategic Planning Implications

1. **Peak Demand Shock:** The maximum deviation from the baseline occurs in **October 2030 (+322.8 GWh)**, not during the tournament itself (June/July). This delayed peak reflects post-tournament structural drag, sustained tourism, and ongoing infrastructure operation.
2. **Total Event Footprint:** The cumulative additional demand generated across the 12-month window is **+2,929.1 GWh**, representing a substantial energy procurement challenge for the national operator.
3. **Summer Capacity Test:** While October shows the highest *relative* lift, **August 2030** features the highest *absolute* predicted demand (6,191.8 GWh) due to the compounding effect of the World Cup pulse on top of Morocco's intense baseline summer cooling peak.
