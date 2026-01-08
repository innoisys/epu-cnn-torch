"""
Pytest configuration and fixtures for EPU-CNN tests.
"""
import pytest
import torch
import numpy as np
from pathlib import Path


@pytest.fixture
def device():
    """Fixture for torch device (CPU for testing)."""
    return torch.device("cpu")


@pytest.fixture
def sample_image():
    """Fixture for a sample image tensor."""
    # 3 channels (RGB), 224x224
    return torch.randn(3, 1, 224, 224)


@pytest.fixture
def sample_label():
    """Fixture for a sample label."""
    return torch.tensor([1])


@pytest.fixture
def temp_config_dir(tmp_path):
    """Fixture for temporary configuration directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sample_label_mapping():
    """Fixture for a sample label mapping."""
    return {"class_0": 0, "class_1": 1}
