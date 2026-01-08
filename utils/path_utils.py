"""
Path utilities for cross-platform path handling.
"""
import os
from pathlib import Path
from typing import Tuple


def load_model_configs(model_path: str) -> Tuple[str, str, str]:
    """
    Load configuration and checkpoint paths from model directory.

    This function handles cross-platform path conversion properly,
    avoiding platform-specific separators.

    Args:
        model_path: Relative or absolute path to model directory.
                   Can use forward slashes regardless of platform.

    Returns:
        Tuple of (epu_config_path, train_config_path, checkpoint_dir)

    Example:
        >>> epu_cfg, train_cfg, ckpt_dir = load_model_configs("checkpoints/my_model")
        >>> # Works on both Unix and Windows
    """
    # Convert string path to Path object (handles any separator)
    base_path = Path(model_path)

    # Make absolute if relative
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path

    # Build config paths
    epu_config_path = str(base_path / "epu.config")
    train_config_path = str(base_path / "train.config")

    return epu_config_path, train_config_path, str(base_path)


def get_checkpoint_path(model_path: str, experiment_name: str) -> str:
    """
    Get the checkpoint file path for a given model and experiment.

    Args:
        model_path: Path to model directory
        experiment_name: Name of the experiment (without .pt extension)

    Returns:
        Full path to checkpoint file
    """
    base_path = Path(model_path)
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path

    return str(base_path / f"{experiment_name}.pt")
