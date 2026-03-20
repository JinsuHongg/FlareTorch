# FlareTorch

[![PyPI version](https://badge.fury.io/py/flaretorch.svg)](https://badge.fury.io/py/flaretorch)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**FlareTorch** is a machine learning library for solar flare forecasting with uncertainty quantification.

## Features

- Deep learning models (ResNet variants with MCDropout, Quantile Regression)
- Uncertainty quantification methods (Conformal Prediction, CQR, Laplace Approximation)
- PyTorch Lightning integration for easy training and evaluation
- Hydra-based configuration management
- Weights & Biases integration for experiment tracking

## Installation

### From PyPI (when released)

```bash
pip install flaretorch
```

### From Source

```bash
git clone https://github.com/JinsuHongg/FlareTorch.git
cd FlareTorch
uv sync
```

### Development Installation

```bash
git clone https://github.com/JinsuHongg/FlareTorch.git
cd FlareTorch
uv sync --all-extras
```

## Quick Start

```python
import flaretorch
from flaretorch.models import ResNetMCD

# Access version
print(flaretorch.__version__)

# Use models
model = ResNetMCD(...)
```

## Project Structure

```
flaretorch/
├── src/flaretorch/      # Main package
│   ├── models/           # Neural network architectures
│   ├── datasets/         # Dataset implementations
│   ├── datamodules/       # PyTorch Lightning datamodules
│   ├── metrics/           # Evaluation metrics
│   ├── explainability/    # Uncertainty quantification
│   ├── tasks/             # Training and calibration tasks
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── docs/                  # Documentation
├── configs/               # Hydra configuration files
└── scripts/               # CLI entry points
```

## Usage

### Training a Model

```bash
flare-train --config configs/default.yaml
```

### Evaluation

```bash
flare-eval --model_path checkpoints/best_model.ckpt
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use FlareTorch in your research, please cite:

```bibtex
@software{flaretorch2024,
  title = {FlareTorch: Machine Learning for Solar Flare Forecasting},
  authors = {Jinsu Hong and Berkay Aydin},
  year = {2024},
  url = {https://github.com/JinsuHongg/FlareTorch}
}
```

## Contact

- **Jinsu Hong** - jinsuhong.knight@gmail.com / jhong36@gsu.edu
- **Berkay Aydin** - baydin2@gsu.edu
