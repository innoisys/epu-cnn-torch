"""
Tests for path utilities (cross-platform path handling).
"""
import pytest
from pathlib import Path
from utils.path_utils import load_model_configs, get_checkpoint_path


def test_load_model_configs_relative_path(tmp_path):
    """Test loading model configs with relative path."""
    # Create temporary model directory
    model_dir = tmp_path / "checkpoints" / "test_model"
    model_dir.mkdir(parents=True)

    # Create dummy config files
    (model_dir / "epu.config").touch()
    (model_dir / "train.config").touch()

    # Test with relative path (using forward slashes)
    rel_path = f"{tmp_path}/checkpoints/test_model"
    epu_cfg, train_cfg, ckpt_dir = load_model_configs(rel_path)

    assert Path(epu_cfg).name == "epu.config"
    assert Path(train_cfg).name == "train.config"
    assert Path(ckpt_dir).is_absolute()


def test_load_model_configs_absolute_path(tmp_path):
    """Test loading model configs with absolute path."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    epu_cfg, train_cfg, ckpt_dir = load_model_configs(str(model_dir))

    assert Path(epu_cfg).parent == model_dir
    assert Path(train_cfg).parent == model_dir
    assert Path(ckpt_dir) == model_dir


def test_get_checkpoint_path(tmp_path):
    """Test getting checkpoint path."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    ckpt_path = get_checkpoint_path(str(model_dir), "experiment_001")

    assert Path(ckpt_path).name == "experiment_001.pt"
    assert Path(ckpt_path).parent == model_dir


def test_get_checkpoint_path_with_forward_slashes(tmp_path):
    """Test checkpoint path works with forward slashes on all platforms."""
    # Use forward slashes (Unix-style) - should work on Windows too
    model_path = f"{tmp_path}/checkpoints/model"

    ckpt_path = get_checkpoint_path(model_path, "test")

    # Verify path is valid and uses correct separators
    path_obj = Path(ckpt_path)
    assert path_obj.name == "test.pt"
    assert path_obj.is_absolute()
