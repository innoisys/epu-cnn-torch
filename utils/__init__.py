"""
EPU-CNN Utilities Package

This package contains utility functions and classes for training, evaluation,
data loading, and configuration management for EPU-CNN models.
"""

# Configuration classes
from utils.epu_utils import (
    EPUConfig,
    BlockConfig,
    SubnetworkConfig,
    ClassificationHeadConfig,
)

# Training utilities
from utils.epu_utils import (
    trainer,
    validate,
    calculate_metrics,
    TensorboardLogger,
    EarlyStopping,
    TensorboardLoggerCallback,
    EarlyStoppingCallback,
)

# Model utilities
from utils.epu_utils import (
    load_model,
    module_mapping,
    preprocess_image,
    preprocess_images,
    estimate_average_rss,
    plot_average_rss,
)

# Data utilities
from utils.data_utils import (
    EPUDataset,
    DatasetParser,
    FolderDatasetParser,
    FilenameDatasetParser,
)

# Custom transforms
from utils.custom_transforms import (
    ImageToPFM,
    PFMToTensor,
)

# Path utilities
from utils.path_utils import (
    load_model_configs,
    get_checkpoint_path,
)

# Custom mappings
from utils.mappings import custom_module_mapping

__all__ = [
    # Configuration
    "EPUConfig",
    "BlockConfig",
    "SubnetworkConfig",
    "ClassificationHeadConfig",
    # Training
    "trainer",
    "validate",
    "calculate_metrics",
    "TensorboardLogger",
    "EarlyStopping",
    "TensorboardLoggerCallback",
    "EarlyStoppingCallback",
    # Model
    "load_model",
    "module_mapping",
    "preprocess_image",
    "preprocess_images",
    "estimate_average_rss",
    "plot_average_rss",
    # Data
    "EPUDataset",
    "DatasetParser",
    "FolderDatasetParser",
    "FilenameDatasetParser",
    # Transforms
    "ImageToPFM",
    "PFMToTensor",
    # Paths
    "load_model_configs",
    "get_checkpoint_path",
    # Custom
    "custom_module_mapping",
]
