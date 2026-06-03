# Leave-One-Out Validation

Validating a transfer learning approach for rare events is challenging due to the small sample size (five donor countries). Standard train/test splits would leave too few examples for either phase. To rigorously evaluate the Neural Event Kernel's ability to generalize to an unseen country (like Morocco), we employ **Leave-One-Out (LOO) Cross-Validation**.

## The LOO Framework

The LOO procedure iterates through the five donor countries (Qatar, Egypt, Russia, South Africa, Cameroon). In each of the 5 folds:

1. **Hold Out**: One country is selected as the "target" (simulating Morocco).
2. **Train**: The Neural Event Kernel is trained from scratch on the remaining 4 donor countries. This learns a shape $f_\theta(t)$ and 4 donor amplitudes $a_i$.
3. **Transfer**: An amplitude for the held-out country is estimated using the weighted average of the 4 training donors' amplitudes.
4. **Reconstruct**: The predicted uplift is generated for the held-out country: $a_{\text{held-out}} \cdot f_\theta(t_{\text{norm}})$.
5. **Evaluate**: The reconstructed total demand is compared against the actual historical demand observed in the held-out country during its event window.

## Evaluation Metrics

For each fold, we compute several key metrics on the held-out country's event window (months -5 to +6):

- **RMSE (Root Mean Square Error):** Measures absolute reconstruction error in GWh.
- **MAPE (Mean Absolute Percentage Error):** Measures relative error, providing a scale-independent assessment.
- **Shape MSE:** Evaluates how well the normalized geometry of the predicted pulse matches the normalized actual log-space uplift, isolating shape accuracy from magnitude accuracy.
- **Pulse Variance:** A diagnostic metric tracking the variance of the kernel output. Very low variance indicates "shape collapse" (the network outputs a flat line), typically caused by excessive regularization.

## Grid Search Methodology

To determine the optimal model architecture and hyperparameters, the entire 5-fold LOO process is wrapped inside a **150-configuration grid search**:

- **Architectures:** Model A (shallow), Model B (medium), Model C (deep).
- **Weighting Schemes:** Uniform weights vs. Proportional weights (based on grid-scale relevance).
- **Amplitude Regularization ($\lambda_a$):** 5 values from 0.001 to 10.0.

Total training runs: 3 models × 2 schemes × 5 lambdas × 5 folds = **150 independent model trainings**.

The configuration that yields the lowest **Mean LOO RMSE** across all 5 folds, while maintaining stability (avoiding shape collapse and extreme inter-fold variance), is selected as the final architecture for the Morocco 2030 deployment.
