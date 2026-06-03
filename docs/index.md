# Morocco 2030 WC Demand Forecast — Documentation

```{toctree}
:maxdepth: 1
:hidden:

overview
research_question
dataset
eda
stationarity
sarima_baseline
neural_event_kernel
loo_validation
results
deployment
repository_structure
installation
usage
reproducing_results
future_work
api/index
references
```

````{grid} 1 1 2 2
:gutter: 3

```{grid-item-card} ⚡ Project Overview
:link: overview
:link-type: doc

Understand the motivation, challenge, and hybrid methodology for forecasting Morocco's electricity demand uplift during the 2030 FIFA World Cup.
```

```{grid-item-card} 🧠 Neural Event Kernel
:link: neural_event_kernel
:link-type: doc

Explore the compact PyTorch neural network that learns a transferable World Cup demand pulse from five donor countries.
```

```{grid-item-card} 📊 Results
:link: results
:link-type: doc

150-configuration LOO grid search results — best configuration, architecture comparison, and key findings.
```

```{grid-item-card} 🇲🇦 Morocco 2030 Deployment
:link: deployment
:link-type: doc

The final deployment pipeline: SARIMA baseline + neural uplift injection, producing a 60-month forecast through December 2030.
```

````

---

## Morocco 2030 WC Demand Forecast

> *Forecasting Morocco's Electricity Demand for FIFA World Cup 2030 using SARIMA and Transfer-Learned Neural Event Kernels*

**Institution:** ENSAM Meknès — Filière IATD  
**Author:** Younes Benchikhi  
**Status:** Research · Active  

---

### Project at a Glance

Morocco co-hosts the **2030 FIFA World Cup** alongside Spain and Portugal. This project estimates the resulting electricity demand uplift on Morocco's national grid using a **two-layer hybrid architecture**:

1. **SARIMA Counterfactual Baseline** — captures Morocco's historical seasonality and trend, providing a *what-would-have-happened-without-the-World-Cup* forecast.
2. **Transfer-Learned Neural Event Kernel** — a compact PyTorch network trained on five donor countries (Qatar, Egypt, Russia, South Africa, Cameroon) that learns a globally shared, normalized event pulse shape, then transfers it to Morocco.

---

### Key Performance Metrics

| Metric | Value |
|--------|-------|
| Best Configuration | `Model_C_weighted_lam0.001` |
| Mean LOO RMSE | **687.2 GWh** |
| Mean LOO MAPE | **2.83%** |
| Mean Shape MSE | 0.204 |
| Peak WC Lift (Oct 2030) | **+322.8 GWh (+6.14%)** |
| Total Cumulative Lift | **+2,929.1 GWh** |
| Transfer Amplitude $a_{\text{Morocco}}$ | **0.0599** |

---

### Final Reconstruction Formula

$$\hat{y}_{\text{Morocco}}(t) = \exp\!\Bigl(\log(1 + y_{\text{SARIMA}}(t)) + a_{\text{Morocco}} \cdot f_\theta(t_{\text{norm}})\Bigr) - 1$$

where $t_{\text{norm}} = t / 6.0$, $t \in [-5, +6]$ months relative to June 2030 kickoff.

---

### Architecture Diagram

```
                        ┌────────────────────────────────────────┐
                        │          Morocco Historical Data        │
                        │     (Jan 2016 – Dec 2025, GWh)         │
                        └──────────────┬─────────────────────────┘
                                       │ log1p transform
                                       ▼
                        ┌────────────────────────────────────────┐
                        │  SARIMA(1,1,1)×(0,1,2)₁₂ Baseline     │
                        │  60-month forecast + 85% intervals     │
                        └──────────────┬─────────────────────────┘
                                       │ Layer 1 counterfactual
                                       ▼
              ┌────────────────────────────────────────────────────┐
              │          Transfer-Learned Neural Event Kernel       │
              │                                                    │
              │  Donor Countries ──► NeuralEventKernel             │
              │  QAT · EGY · RUS     1 → 16 → 16 → 1 (Tanh)       │
              │  CMR · ZAF           max|f_θ| = 1 (normalized)     │
              │                                                    │
              │  a_Morocco = Σ wᵢ · aᵢ = 0.0599                   │
              └──────────────┬─────────────────────────────────────┘
                             │ Layer 2 uplift injection
                             ▼
              ┌────────────────────────────────────────────────────┐
              │      Final Morocco 2030 Forecast (Jan 2026 – Dec 2030)   │
              │      Peak uplift: +6.14% in October 2030           │
              └────────────────────────────────────────────────────┘
```

---

### Quick Start

```bash
git clone https://github.com/your-username/wc2030-morocco-electricity-forecast.git
cd wc2030-morocco-electricity-forecast
conda create -n wc2030 python=3.10
conda activate wc2030
pip install -r requirements.txt
python src/morocco_2030_deployment.py
```

See the {doc}`installation` and {doc}`reproducing_results` pages for the full pipeline walkthrough.

---

### Citation

```bibtex
@misc{benchikhi2026morocco,
  title        = {Forecasting Morocco's Electricity Demand for FIFA World Cup 2030
                  using SARIMA and Transfer-Learned Neural Event Kernels},
  author       = {Benchikhi, Younes},
  year         = {2026},
  institution  = {ENSAM Meknès},
  howpublished = {\url{https://github.com/your-username/wc2030-morocco-electricity-forecast}},
  note         = {Undergraduate Research Project, Filière IATD}
}
```

---

*ENSAM Meknès · Filière IATD · 2025–2026*
