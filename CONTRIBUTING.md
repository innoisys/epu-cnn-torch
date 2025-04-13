# Contributing to EPU-CNN-Torch

Thank you for your interest in contributing to EPU-CNN-Torch! This document provides guidelines and instructions for contributing to the project.

## Project Overview

EPU-CNN-Torch is a PyTorch implementation of "E Pluribus Unum Interpretatble Convolutional Neural Network" (EPU). The project focuses on inherently interpretable image classification and includes support for yaml-based custom model configurations.

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- CUDA (for GPU support)
- Other dependencies listed in requirements.txt

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/epu-cnn-torch.git
cd epu-cnn-torch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `scripts/`: Contains training and evaluation scripts
- `model/`: Core model implementation
- `utils/`: Utility functions and custom transforms
- `configs/`: YAML configuration files for model and training parameters
- `data/`: Directory for datasets (not included in repository)
- `logs/`: Training logs and TensorBoard files
- `checkpoints/`: Model checkpoints
- `models/`: Saved model files

## Contributing Guidelines

### Code Style

- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Include docstrings for all functions and classes
- Keep functions focused and single-purpose

### Pull Request Process

1. Fork the repository
2. Create a new branch for your feature/fix:
```bash
git checkout -b feature/your-feature-name
```
3. Make your changes
4. Add tests if applicable
5. Update documentation if needed
6. Submit a pull request with a clear description of changes

### Configuration Changes

When modifying model configurations:
- Update the relevant YAML files in `configs/`
- Document any new parameters in this file
- Test the configuration with different datasets


### Bug Reports

When reporting bugs, please include:
- Description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (Python version, PyTorch version, etc.)

## Model Training

To train a model:
1. Prepare your dataset in the appropriate format
2. Update the configuration in `configs/`
3. Run the training script:
```bash
python scripts/train.py
```

## License

By contributing to this project, you agree that your contributions will be licensed under the project's license.

## Questions?

If you have any questions about contributing, please open an issue in the repository. 