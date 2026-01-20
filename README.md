# FlareTorch

**FlareTorch** is a basic machine learning module designed for solar flare forecasting.

## Description

This repository contains the source code and scripts for training and evaluating models for solar flare prediction. It utilizes PyTorch/PyTorch Lightning for building and training neural networks.

## Ready Dataset


## Project Structure

The repository is organized as follows:

- **`FlareTorch/`**: The main Python package containing the core model architectures, dataloaders, and utility functions.
- **`scripts/`**: Executable scripts for training, testing, and data processing.
- **`pyproject.toml`**: Project configuration and dependency definitions.
- **`uv.lock`**: Lockfile ensuring reproducible environments (managed by `uv`).

## Installation

### Prerequisites

- Python (Version specified in `.python-version`, e.g., 3.10+)
- [uv](https://github.com/astral-sh/uv) (Recommended for dependency management)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/JinsuHongg/FlareTorch.git
   cd FlareTorch
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

## Usage

### Training a Model

   ```bash
      python scripts/train.py --config configs/default.yaml
   ```

### Inference / Evaluation

   ```bash
      python scripts/evaluate.py --model_path checkpoints/best_model.ckpt
   ```

## Contact
### Authors
 - Jinsu Hong, jhong36@gsu.edu
 - Berkay Aydin, baydin2@gsu.edu


