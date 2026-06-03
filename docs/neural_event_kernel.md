# Neural Event Kernel

The core innovation of this project is the **Transfer-Learned Neural Event Kernel**, which models the transient electricity demand shock induced by a mega-event.

## Concept: Scale-Invariant Pulse Shape

Classical forecasting fails for unprecedented events because it cannot extrapolate a shock it hasn't seen. We solve this by learning from *donor countries* that have hosted similar events.

The fundamental assumption is **structural invariance in log-space**: the *shape* and *relative magnitude* (percentage lift) of a World Cup demand pulse is consistent across different host nations, even if their absolute grid scales differ massively.

To operationalize this, we define the uplift target as the log-space residual between actual observed demand during a donor's historical event and its respective SARIMA counterfactual:

$$u_i(t) = \log(1 + y_i^{\text{actual}}(t)) - \log(1 + y_i^{\text{counterfactual}}(t))$$

## Mathematical Formulation

The full reconstruction formula for any country's event-window demand is:

$$\hat{y}_i(t) = \exp\!\Bigl(\log(1 + y_i^{\text{counterfactual}}(t)) + a_i \cdot f_\theta(t_{\text{norm}})\Bigr) - 1$$

Where:
- $y_i^{\text{counterfactual}}(t)$: The SARIMA baseline prediction.
- $a_i$: A country-specific **amplitude scalar** representing the peak magnitude of the shock.
- $f_\theta(t_{\text{norm}})$: The **Neural Event Kernel**, a continuous function mapping normalized time to a normalized pulse shape.
- $t_{\text{norm}} = t / 6.0$, where $t \in [-5, +6]$ months relative to the tournament kickoff ($t=0$).

## Network Architecture (Model C)

The kernel $f_\theta$ is parameterized as a compact feed-forward PyTorch neural network:

```
Input:   t_norm ∈ ℝ¹   (scalar normalized time)
         ↓
Linear(1 → 16) + Tanh
         ↓
Linear(16 → 16) + Tanh
         ↓
Linear(16 → 1)
         ↓
Output:  f_θ(t_norm)   (normalized pulse shape)
```

**Seamless Normalization**: During training, a normalization procedure is applied every 5 epochs to enforce $\max_t |f_\theta(t)| = 1$. Simultaneously, the learned amplitude parameters $a_i$ are inversely scaled to preserve the product $a_i \cdot f_\theta(t)$. This ensures $a_i$ carries the full magnitude information, forcing the network to learn a pure, canonical geometric shape.

## Shared Loss Objective

The kernel is trained jointly across all donor countries using a weighted MSE loss, plus regularization terms:

$$\mathcal{L} = \underbrace{\frac{\sum_i w_i (a_i \cdot f_\theta(t_i) - u_i)^2}{\sum_i w_i}}_{\text{Weighted MSE}} + \underbrace{\lambda_s \sum_{t} (\Delta^2 \beta)^2_t}_{\text{Smoothness}} + \underbrace{\lambda_a \cdot \frac{1}{N}\sum_i \tilde{a}_i^2}_{\text{Amplitude Penalty}}$$

Where:
- $w_i$: Donor relevance weights.
- $\Delta^2$: Second-order finite difference operator (penalizes jaggedness in the pulse).
- $a_i = \exp(\tilde{a}_i)$: Parameterized in log-space to guarantee positive amplitudes.

## Transferring to Morocco

Morocco has no historical event data, so its amplitude $a_{\text{Morocco}}$ cannot be learned directly. Instead, it is estimated via a weighted average of the donor amplitudes learned during training:

$$a_{\text{Morocco}} = \sum_{i \in \text{donors}} w_i \cdot a_i$$

This derived amplitude is then multiplied by the learned shape $f_\theta(t)$ to project the shock onto Morocco's 2030 grid.
