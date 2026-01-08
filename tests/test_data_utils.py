"""
Tests for data utilities.
"""
import pytest
from utils.data_utils import LRUCache


def test_lru_cache_basic_operations():
    """Test basic LRU cache operations."""
    cache = LRUCache(capacity=2)

    # Test put and get
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.hits == 1
    assert cache.misses == 0

    # Test miss
    assert cache.get("nonexistent") is None
    assert cache.misses == 1


def test_lru_cache_capacity_eviction():
    """Test that LRU cache evicts least recently used items."""
    cache = LRUCache(capacity=2)

    cache.put("key1", "value1")
    cache.put("key2", "value2")
    cache.put("key3", "value3")  # Should evict key1

    assert cache.get("key1") is None  # Evicted
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"


def test_lru_cache_update():
    """Test updating existing key."""
    cache = LRUCache(capacity=2)

    cache.put("key1", "value1")
    cache.put("key1", "value1_updated")

    assert cache.get("key1") == "value1_updated"


def test_lru_cache_stats():
    """Test cache statistics."""
    cache = LRUCache(capacity=3)

    cache.put("k1", "v1")
    cache.put("k2", "v2")

    cache.get("k1")  # hit
    cache.get("k2")  # hit
    cache.get("k3")  # miss

    stats = cache.get_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 2/3
    assert stats["size"] == 2


def test_lru_cache_clear():
    """Test cache clearing."""
    cache = LRUCache(capacity=2)

    cache.put("key1", "value1")
    cache.put("key2", "value2")

    cache.clear()

    assert cache.get("key1") is None
    assert cache.get("key2") is None
    stats = cache.get_stats()
    assert stats["hits"] == 0
    assert stats["misses"] == 2
