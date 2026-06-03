# Installation Guide

This guide provides instructions for setting up the environment required to run the forecasting models and neural event kernel.

## Prerequisites

- **Python:** Version 3.10 or higher.
- **Git:** For cloning the repository.
- **Conda (Recommended) or pip:** For managing Python dependencies.

## 1. Clone the Repository

Clone the project repository to your local machine:

```bash
git clone https://github.com/your-username/wc2030-morocco-electricity-forecast.git
cd wc2030-morocco-electricity-forecast
```

## 2. Set Up the Environment

It is highly recommended to use a virtual environment to avoid dependency conflicts, particularly with scientific libraries like `statsforecast` and `torch`.

### Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n wc2030 python=3.10 -y

# Activate the environment
conda activate wc2030

# Install the required packages
pip install -r requirements.txt
```

### Using venv

```bash
# Create a new virtual environment
python -m venv venv

# Activate the environment (Windows)
venv\Scripts\activate

# Activate the environment (Linux/macOS)
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

## 3. Verify Installation

To verify that the installation was successful and all dependencies are correctly loaded, you can run a quick check on the primary deployment script (it should execute the first step without errors):

```bash
python src/morocco_sarima_baseline.py
```

If the script runs and outputs the SARIMA order without throwing an `ImportError`, the environment is ready.

## 4. Hardware Acceleration

The `NeuralEventKernel` is implemented in PyTorch. Due to the very small size of the network (16 hidden units, 2 layers) and the small dataset size (5 donors × 12 months), training on a standard CPU is exceptionally fast (less than 20 minutes for the entire 150-run LOO grid search).

GPU acceleration (CUDA/MPS) is supported by the code but is not strictly necessary for this project. If you wish to use a GPU, ensure you have the appropriate PyTorch build installed for your system.
