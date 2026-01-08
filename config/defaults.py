"""
Default configuration constants for EPU-CNN.

This module contains default values for various configuration parameters
used throughout the codebase.
"""

# Cache sizes for dataset loading
DEFAULT_CACHE_SIZE = 10000
DEFAULT_VAL_CACHE_SIZE = 10000

# Training defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 100

# Model defaults
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_EPU_ACTIVATION = "sigmoid"

# Logging defaults
DEFAULT_LOG_DIR = "logs/runs/experiment"
DEFAULT_CHECKPOINT_DIR = "checkpoints"

# Evaluation defaults
DEFAULT_METRICS = ["accuracy", "precision", "recall", "f1", "auc"]
