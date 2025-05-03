# EPU-CNN PyTorch Implementation

[![GitHub stars](https://img.shields.io/github/stars/innoisys/epu-cnn-torch.svg?style=flat&label=Star)](https://github.com/innoisys/epu-cnn-torch/)
[![Readme](https://img.shields.io/badge/README-green.svg)](README.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is a PyTorch implementation of "E pluribus unum interpretable convolutional neural networks" (EPU-CNN). The original TensorFlow implementation can be found at [innoisys/EPU-CNN](https://github.com/innoisys/EPU-CNN).

## Overview

EPU-CNN is a framework for creating inherently interpretable CNN models based on Generalized Additive Models. It consists of multiple CNN sub-networks, each processing a different perceptual feature representation of the input image. The model provides both classification predictions and human-interpretable explanations of its decision-making process.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Configuration](#1-configuration)
  - [2. Supported Dataset Structures and Training](#2-supported-dataset-structures-and-training)
    - [2.1 Filename-based Structure (Binary Classification)](#21-filename-based-structure-binary-classification)
    - [2.2 Folder-based Structure (Multiclass Classification)](#22-folder-based-structure-multiclass-classification)
  - [3. Evaluation](#3-evaluation)
  - [4. Inference and Visualization](#4-inference-and-visualization)
- [Visualization Examples](#visualization-examples)
- [Citation](#citation)
- [License](#license)
- [Known Issues](#known-issues)
  - [Windows Path Handling](#windows-path-handling)
- [TODO](#todo)
- [Acknowledgments](#acknowledgments)

## Features

- PyTorch implementation with modern best practices
- Configurable architecture through YAML configuration
- Comprehensive visualization tools for model interpretations
- Support for both binary and multiclass image classification
- Built-in early stopping and model checkpointing
- TensorBoard integration for training monitoring
- Comprehensive evaluation metrics and reporting

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
├── checkpoints/            # Saved model checkpoints and configurations
├── model/                  # Model implementation
│   ├── epu.py             # Main EPU model
│   ├── layers.py          # Custom layers
│   └── subnetworks.py     # Subnetwork implementations
├── scripts/               # Training and inference scripts
│   ├── train.py          # Binary classification training script
│   ├── multiclass_train.py # Multiclass classification training script
│   ├── eval.py           # Evaluation script
│   └── inference.py      # Inference and visualization script
└── utils/                # Utility functions
    ├── epu_utils.py      # EPU-specific utilities
    ├── data_utils.py     # Dataset handling utilities
    └── custom_transforms.py  # Custom image transforms
```

## Usage

### 1. Configuration

Create a YAML configuration file in `configs/` with the following structure:

```yaml
epu:
    model_name: "epu_banapple"
    n_classes: 1  # Number of output classes (1 for binary, >1 for multiclass)
    n_subnetworks: 4
    subnetwork: "subnetavg"
    subnetwork_architecture:
        n_classes: 1
        n_blocks: 3
        has_pooling: true
        pooling_type: "globalaveragepooling"
        pooling_kernel_size: [1, 1]
        pooling_stride: [1, 1]
        has_contribution_head: true
        block_1:
            in_channels: 1
            out_channels: 32
            n_conv_layers: 2
            kernel_size: [3, 3]
            stride: [1, 1]
            padding: 1
            activation: "relu"
            has_norm: true
            norm_type: "batchnorm2d"
            has_pooling: false
            pooling_type: None
            pooling_kernel_size: [2, 2]
            pooling_stride: [1, 1]
        # ... other block configurations
    epu_activation: "sigmoid"
    categorical_input_features: ["red-green", "blue-yellow", "high-frequencies", "low-frequencies"]

train_parameters:
    mode: "binary"  # or "multiclass"
    dataset_parser: "filename_parser"  # or "folder_parser"
    loss: "binary_cross_entropy"  # or "categorical_cross_entropy"
    epochs: 10
    learning_rate: 0.001
    image_extension: "jpg"
    batch_size: 32
    shuffle: true
    num_workers: 0
    pin_memory: false
    input_size: 128
    persistent_workers: false
    early_stopping_patience: 25
    dataset_path: "./data/banapple"
    label_mapping:
        apple: 1
        banana: 0
```

### 2. Supported Dataset Structures and Training

EPU-CNN-Torch supports two dataset organization patterns. Here are complete examples of both structures:

#### 2.1 Filename-based Structure (Binary Classification)

```
dataset/
├── train/
│   ├── apple_001.jpg
│   ├── apple_002.jpg
│   ├── apple_003.jpg
│   ├── banana_001.jpg
│   ├── banana_002.jpg
│   └── banana_003.jpg
├── validation/
│   ├── apple_004.jpg
│   ├── apple_005.jpg
│   ├── banana_004.jpg
│   └── banana_005.jpg
└── test/
    ├── apple_006.jpg
    ├── apple_007.jpg
    ├── banana_006.jpg
    └── banana_007.jpg
```

**Configuration Example:**
```yaml
train_parameters:
    mode: "binary"
    dataset_parser: "filename_parser"
    dataset_path: "dataset"
    label_mapping:
        apple: 1
        banana: 0
    image_extension: "jpg"
```

**Usage with train.py:**
```bash
# Basic training
python scripts/train.py --config_path configs/binary_config.yaml

# Training with TensorBoard monitoring
python scripts/train.py --config_path configs/binary_config.yaml --tensorboard
```

When using the `--tensorboard` flag, the script automatically:
- Launches TensorBoard as a subprocess
- Sets up monitoring on the `logs` directory
- Makes TensorBoard accessible at `http://localhost:6006`
- Enables real-time monitoring of training metrics, model graphs, and histograms

#### 2.2 Folder-based Structure (Multiclass Classification)

```
dataset/
├── train/
│   ├── apple/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── image_003.jpg
│   ├── banana/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── image_003.jpg
│   └── orange/
│       ├── image_001.jpg
│       ├── image_002.jpg
│       └── image_003.jpg
├── validation/
│   ├── apple/
│   │   ├── image_004.jpg
│   │   └── image_005.jpg
│   ├── banana/
│   │   ├── image_004.jpg
│   │   └── image_005.jpg
│   └── orange/
│       ├── image_004.jpg
│       └── image_005.jpg
└── test/
    ├── apple/
    │   ├── image_006.jpg
    │   └── image_007.jpg
    ├── banana/
    │   ├── image_006.jpg
    │   └── image_007.jpg
    └── orange/
        ├── image_006.jpg
        └── image_007.jpg
```

**Configuration Example:**
```yaml
train_parameters:
    mode: "multiclass"
    dataset_parser: "folder_parser"
    dataset_path: "dataset"
    label_mapping:
        apple: 0
        banana: 1
        orange: 2
    image_extension: "jpg"
```

**Usage with multiclass_train.py:**
```bash
# Basic training
python scripts/multiclass_train.py --config_path configs/multiclass_config.yaml

# Training with TensorBoard monitoring
python scripts/multiclass_train.py --config_path configs/multiclass_config.yaml --tensorboard
```

The `--tensorboard` flag provides the same monitoring capabilities for multiclass training:
- Automatic TensorBoard launch at `http://localhost:6006`

**Key Points:**
- Both structures require a consistent organization across train/validation/test splits
- Image names can be arbitrary but must be unique within their directories
- Supported image formats include jpg, jpeg, png, etc.
- The validation folder name must be exactly "validation" (not "val" or other variants)
- For filename-based structure, class names must be present in the filenames
- For folder-based structure, folder names must match the class names in label_mapping
- Binary and Multiclass classification can use both filename-based and folder-based structure

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
  - For multiclass classification:
    - Confusion matrix
    - Classification report
    - Per-class metrics (precision, recall, F1-score, support)
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

The inference script generates two types of visualizations of the inference intepretations for each image:

1. **Relative Similarity Scores (RSS)** (`{image_name}_rss.png`):
   - Shows the contribution of each perceptual feature to a repsective class
   - Interprets how different perceptual features contribute to the prediction

2. **Perceptual Relevance Maps** (`{image_name}_all_prm.png`):
   - A Heatmap indicating the region that is found relevant for the assessment of the RSS score
   - Indicates the spatial regions that were considered for the assessment of the image per perceptual feature

### Example Outputs From [CIFAKE Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

Below are example visualizations automatically generated by the EPU-CNN framework when running [`inference.py`](scripts/inference.py):

#### Relative Similarity Scores (RSS) per Perceptual Feature

![RSS Example](./assets/459%20(7).jpg_rss.png)

*Figure: Relative similarity scores (RSS) for each perceptual feature, showing how much each feature contributed to the model's decision for the input image.*

#### Perceptual Relevance Maps (PRM) per Perceptual Feature

![PRM Example](./assets/459%20(7).jpg_all_prm.png)

*Figure: Perceptual relevance maps (PRM) for each perceptual feature, visualizing the spatial importance of each feature in the input image.*

> These interpretation visualizations are automatically generated when you run:
> ```bash
> python scripts/inference.py --model_path <your_model_path> --image_path <your_image_path>
> ```
> The results are saved in the `interpretations/` directory.

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
- [X] Add Multiclass Training and Evaluation code
- [X] Refine YAML-based EPU-CNN configuration
- [ ] Fix path handling for Windows
- [ ] Add Dataset-wide Interpretations
- [X] Provide support for either data structure on both mutliclass and binary classification training
- [ ] Support for Contribution Auxilary loss
- [ ] Add setup
- [ ] Add visualization in README.md
- [ ] Add utility for deciding augmentations from config file
 
## Acknowledgments

- Original EPU-CNN implementation by [innoisys/EPU-CNN](https://github.com/innoisys/EPU-CNN)