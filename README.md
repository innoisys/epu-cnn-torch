# EPU-CNN PyTorch Implementation

This is a PyTorch implementation of "E pluribus unum interpretable convolutional neural networks" (EPU-CNN). The original TensorFlow implementation can be found at [innoisys/EPU-CNN](https://github.com/innoisys/EPU-CNN).

## Overview

EPU-CNN is a framework for creating inherently interpretable CNN models based on Generalized Additive Models. It consists of multiple CNN sub-networks, each processing a different perceptual feature representation of the input image. The model provides both classification predictions and human-interpretable explanations of its decision-making process.

## Features

- PyTorch implementation with modern best practices
- Configurable architecture through YAML configuration
- Comprehensive visualization tools for model interpretations
- Support for both binary and multi-class classification
- Memory-efficient implementation with feature map tracking
- Built-in early stopping and model checkpointing
- TensorBoard integration for training monitoring

## Installation

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

```
epu-cnn-torch/
├── configs/                 # YAML configuration files
├── data/                   # Dataset directory
├── logs/                   # TensorBoard logs
├── models/                 # Saved model checkpoints
├── model/                  # Model implementation
│   ├── epu.py             # Main EPU model
│   ├── layers.py          # Custom layers
│   └── subnetworks.py     # Subnetwork implementations
├── scripts/               # Training and inference scripts
│   ├── train.py          # Training script
│   ├── eval.py           # Evaluation script
│   └── inference.py      # Inference and visualization script
└── utils/                # Utility functions
    ├── epu_utils.py      # EPU-specific utilities
    └── custom_transforms.py  # Custom image transforms
```

## Usage

### 1. Configuration

Create a YAML configuration file in `configs/` with the following structure:

```yaml
epu:
  n_classes: 1  # Number of output classes
  n_subnetworks: 4  # Number of perceptual feature maps
  subnetwork: "subnetavg"  # Subnetwork architecture
  subnetwork_architecture:
    n_blocks: 3  # Number of blocks in subnetwork
    block_1:
      in_channels: 1
      out_channels: 64
      n_conv_layers: 2
      # ... other block configurations
  epu_activation: "sigmoid"
  categorical_input_features: ["a", "b", "sobel", "gauss"]  # Feature names

train_parameters:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
  input_size: 96
  # ... other training parameters
```

### 2. Training

Train the model using the provided script:

```bash
python scripts/train.py
```

The script will:
- Load the configuration
- Initialize the model
- Train with early stopping
- Save checkpoints and TensorBoard logs
- Save the final model

### 3. Evaluation

Evaluate the model on a test set:

```bash
python scripts/eval.py
```

This will:
- Load the trained model
- Evaluate on test data
- Print metrics (accuracy, AUC, precision, recall, F1)

### 4. Inference and Visualization

Generate interpretations for test images:

```bash
python scripts/inference.py
```

This will:
- Process each test image
- Generate visualizations:
  - Combined interpretation plots
  - Feature contribution bar plots
  - Individual feature heatmaps
- Save results in the `interpretations/` directory

## Visualization Examples

The inference script generates three types of visualizations for each image:

1. **Combined Interpretation Plot** (`{image_name}_interpretations.png`):
   - Shows input features and their interpretations
   - Helps understand how different features contribute to the prediction

2. **Feature Contributions** (`{image_name}_contributions.png`):
   - Bar plot showing relative importance of each feature
   - Helps identify which features are most influential

3. **Feature Heatmaps** (`{image_name}_{feature_name}_heatmap.png`):
   - Detailed heatmaps for each feature
   - Shows spatial distribution of feature importance

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{dimas2023pluribus,
  title = {E pluribus unum interpretable convolutional neural networks},
  author = {Dimas, George and Cholopoulou, Eirini and Iakovidis, Dimitris K},
  journal = {Scientific Reports},
  volume = {13},
  number = {1},
  pages = {11421},
  year = {2023},
  publisher = {Nature Publishing Group UK London}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original EPU-CNN implementation by [innoisys/EPU-CNN](https://github.com/innoisys/EPU-CNN)
- PyTorch team for the excellent deep learning framework
- All contributors and users of this implementation 