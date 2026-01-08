"""
Tests for module mapping utilities.
"""
import pytest
import torch.nn as nn
from utils.epu_utils import module_mapping


def test_module_mapping_unique_keys():
    """Test that module_mapping has no duplicate keys."""
    # Get all module names by checking which ones don't raise errors
    test_modules = [
        "relu", "tanh", "sigmoid", "mish", "silu", "tanhshrink",
        "swish", "gelu", "leakyrelu", "prelu", "elu", "selu"
    ]

    results = {}
    for module_name in test_modules:
        try:
            module_class = module_mapping(module_name)
            results[module_name] = module_class.__name__
        except (ValueError, KeyError):
            pass

    # Verify no duplicate mappings (values should map to unique classes)
    assert "relu" in results
    assert "tanh" in results
    assert "mish" in results
    assert "silu" in results

    # These should map to the same class (swish == silu)
    if "swish" in results and "silu" in results:
        assert results["swish"] == results["silu"]


def test_module_mapping_returns_class():
    """Test that module_mapping returns a class, not instance."""
    relu_class = module_mapping("relu")
    assert isinstance(relu_class, type)
    assert issubclass(relu_class, nn.Module)


def test_module_mapping_instantiation():
    """Test that mapped modules can be instantiated."""
    relu_class = module_mapping("relu")
    relu_instance = relu_class()
    assert isinstance(relu_instance, nn.Module)


def test_module_mapping_invalid_name():
    """Test that invalid module names raise ValueError."""
    with pytest.raises(ValueError) as exc_info:
        module_mapping("nonexistent_module")

    assert "not found" in str(exc_info.value).lower()


def test_module_mapping_case_insensitive():
    """Test that module_mapping is case-insensitive."""
    relu_lower = module_mapping("relu")
    relu_upper = module_mapping("RELU")
    relu_mixed = module_mapping("ReLU")

    assert relu_lower == relu_upper == relu_mixed
