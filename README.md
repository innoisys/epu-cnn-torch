# EPU-CNN PyTorch Implementation

[![GitHub stars](https://img.shields.io/github/stars/innoisys/epu-cnn-torch.svg?style=flat&label=Star)](https://github.com/innoisys/epu-cnn-torch/)
[![Readme](https://img.shields.io/badge/README-green.svg)](README.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a PyTorch implementation of "E pluribus unum interpretable convolutional neural networks" (EPU-CNN). The original TensorFlow implementation can be found at [innoisys/EPU-CNN](https://github.com/innoisys/EPU-CNN).

## Overview

EPU-CNN is a framework for creating inherently interpretable CNN models based on Generalized Additive Models. It consists of multiple CNN sub-networks, each processing a different perceptual feature representation of the input image. The model provides both classification predictions and human-interpretable explanations of its decision-making process.

## Features

- PyTorch implementation with modern best practices
- Configurable architecture through YAML configuration
- Comprehensive visualization tools for model interpretations
- Support for binary image classification (multiclass to be added)
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
  model_name: "epu_banapple"  # Name of your model
  n_classes: 1  # Number of output classes (1 for binary classification)
  n_subnetworks: 4  # Number of perceptual feature maps
  subnetwork: "subnetavg"  # Subnetwork architecture
  subnetwork_architecture:
    n_blocks: 3  # Number of blocks in subnetwork
    block_1:
      in_channels: 1
      out_channels: 32
      n_conv_layers: 2
      # ... other block configurations
  epu_activation: "sigmoid"
  categorical_input_features: ["red-green", "blue-yellow", "high-frequencies", "low-frequencies"]

train_parameters:
  epochs: 10
  learning_rate: 0.001
  batch_size: 32
  input_size: 128
  dataset_path: "path/to/your/dataset"
  label_mapping:
    class1: 1
    class2: 0
  # ... other training parameters
```

### 2. Training

Train the model using the provided script:

```bash
python scripts/train.py --config_path configs/your_config.yaml
```

Required Arguments:
- `--config_path`: Path to your YAML configuration file

The script will:
- Load the configuration from the specified YAML file
- Initialize the model with the specified architecture
- Create necessary directories for logs and checkpoints
- Train the model with early stopping
- Save checkpoints and TensorBoard logs in the `logs/` directory
- Save the final model and configurations in the `checkpoints/` directory

Example output structure:
```
checkpoints/
└── epu_banapple_10epochs_0/
    ├── epu_banapple_10epochs_0.pt        # Best model checkpoint
    ├── epu_banapple_10epochs_0_final.pt  # Final model
    ├── epu.config                         # Model configuration
    └── train.config                       # Training configuration

logs/
└── epu_banapple_10epochs_0/              # TensorBoard logs
```

### 3. Evaluation

Evaluate the model on a test set:

```bash
python scripts/eval.py --model_path checkpoints/epu_banapple_10epochs_0 --test_data path/to/test/data --batch_size 32 --confidence 0.5
```

Required Arguments:
- `--model_path`: Path to the directory containing the saved model and configurations
- `--test_data`: Path to the test data directory

Optional Arguments:
- `--batch_size`: Batch size for evaluation (default: 32)
- `--output_dir`: Directory to save evaluation results (default: 'eval_results')
- `--confidence`: Classifier confidence threshold (default: 0.5)

The script will:
- Load the trained model and its configurations
- Evaluate on the test data
- Generate and save evaluation metrics including:
  - Overall metrics (accuracy, loss)
  - For binary classification:
    - Confusion matrix
    - Classification report (precision, recall, F1-score)
- Save results in JSON format in the specified output directory

Example output structure:
```
checkpoints/epu_banapple_10epochs_0/
└── eval_results/
    ├── eval_results_20240315_123456.json  # Evaluation metrics
    ├── confusion_matrix_20240315_123456.txt  # Confusion matrix
    └── classification_report_20240315_123456.json  # Classification report
```

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

## Known Issues

### Windows Path Handling
- The current implementation may have issues with path handling in Windows environments
- When providing paths in arguments or configuration files:
  - Use forward slashes (`/`) instead of backslashes (`\`)
  - For absolute paths, make sure they are properly formatted (e.g., `X:\\path\\to\\data\\folder` instead of `X:\path\to\data` or `X:/path/to/data`)
  - <b>Relative paths from the project root are recommended when possible</b>
- If you encounter path-related errors:
  - Double-check path separators in your configuration files
  - Ensure paths in command-line arguments use forward slashes
  - Consider using relative paths instead of absolute paths
- Examples:
  ```bash
  # Good path examples
  python scripts/train.py --config_path configs/model_config.yaml
  python scripts/eval.py --model_path checkpoints/my_model --test_data data/test
  
  # Bad path examples (Windows style, may cause issues)
  python scripts\train.py --config_path configs\model_config.yaml
  python scripts/eval.py --model_path X:\path\checkpoints\my_model --test_data X:\data\test
  ```

## TODO
- [X] Refine README.md
- [X] Implement interpretation visualizations in a nice format
- [ ] Add Wavelet PFM extraction
- [ ] Add Multiclass Training and Evaluation code
- [X] Refine YAML-based EPU-CNN configuration
- [ ] Fix path handling for Windows

## Acknowledgments

- Original EPU-CNN implementation by [innoisys/EPU-CNN](https://github.com/innoisys/EPU-CNN)