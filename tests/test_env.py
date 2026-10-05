"""
Unit tests for WebPrefetchEnv environment.
"""

import pytest
import numpy as np
from src.environment.web_env import WebPrefetchEnv, LRUCache


def test_lru_cache():
    cache = LRUCache(max_capacity_mb=3.0)
    assert cache.occupancy_ratio == 0.0

    cache.put(1, 1.0)
    assert cache.contains(1)
    assert cache.current_size_mb == 1.0

    cache.put(2, 1.5)
    assert cache.contains(2)
    assert cache.current_size_mb == 2.5

    # Eviction test: adding item 3 (1.0MB) exceeds 3.0MB -> evicts item 1 (LRU)
    cache.put(3, 1.0)
    assert not cache.contains(1)
    assert cache.contains(2) and cache.contains(3)
    assert cache.current_size_mb == 2.5


def test_environment_reset_and_step():
    env = WebPrefetchEnv(dataset_path="data/sample/sample_data.json", config_path="configs/config.yaml")
    obs, info = env.reset(seed=42)

    assert obs.shape == (31,)
    assert info["step_idx"] == 0
    assert "current_page" in info

    # Step action 0 (Don't prefetch)
    next_obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float)
    assert next_obs.shape == (31,) or terminated
    assert info["action"] == 0


def test_environment_prefetch_action():
    env = WebPrefetchEnv(dataset_path="data/sample/sample_data.json", config_path="configs/config.yaml")
    obs, info = env.reset(seed=123)

    # Step action 1 (Prefetch)
    next_obs, reward, terminated, truncated, info = env.step(1)
    assert info["action"] == 1
    assert "perceived_latency_ms" in info
    assert "cache_hit" in info
