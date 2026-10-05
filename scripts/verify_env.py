"""
Verification script for WebPrefetchEnv Gymnasium environment.
"""

import sys
import numpy as np
from src.environment.web_env import WebPrefetchEnv

def test_environment():
    print("[Env Test] Initializing WebPrefetchEnv...")
    env = WebPrefetchEnv(dataset_path="data/processed/msnbc_processed.json", config_path="configs/config.yaml")

    obs, info = env.reset(seed=42)
    print(f"[Env Test] Reset successful! Observation shape: {obs.shape}, State dim: {env.state_dim}")
    assert obs.shape[0] == env.state_dim, f"Expected shape ({env.state_dim},), got {obs.shape}"

    step_count = 0
    total_reward = 0.0

    print("[Env Test] Running test episode steps...")
    done = False
    while not done:
        # Sample random action: 0 (Don't Prefetch) or 1 (Prefetch)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        total_reward += reward

        print(f" Step {step_count}: Action={action}, Reward={reward:.4f}, Latency={info['perceived_latency_ms']:.1f}ms, Cache Hit={info['cache_hit']}")

    print(f"[Env Test] Episode completed in {step_count} steps. Total Reward: {total_reward:.4f}")
    print("[Env Test] Environment verification PASSED!")

if __name__ == "__main__":
    test_environment()
