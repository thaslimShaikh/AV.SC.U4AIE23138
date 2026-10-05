"""
Unit tests for Baseline Policies and DQN Agent.
"""

import pytest
import numpy as np
from src.environment.web_env import WebPrefetchEnv
from src.baselines.heuristics import NoPrefetchPolicy, AlwaysPrefetchPolicy, RuleBasedPolicy
from src.agents.dqn_agent import DQNAgent


def test_baseline_policies():
    no_pref = NoPrefetchPolicy()
    always_pref = AlwaysPrefetchPolicy()
    rule_pref = RuleBasedPolicy(threshold=0.35)

    dummy_obs = np.zeros(31)
    dummy_info = {"prediction_confidence": 0.50}

    assert no_pref.select_action(dummy_obs, dummy_info) == 0
    assert always_pref.select_action(dummy_obs, dummy_info) == 1
    assert rule_pref.select_action(dummy_obs, dummy_info) == 1

    dummy_info_low = {"prediction_confidence": 0.20}
    assert rule_pref.select_action(dummy_obs, dummy_info_low) == 0


def test_dqn_agent():
    agent = DQNAgent(state_dim=31, action_dim=2, hidden_dims=[32, 16], batch_size=8)
    dummy_state = np.random.randn(31).astype(np.float32)

    action = agent.select_action(dummy_state)
    assert action in [0, 1]

    q_vals = agent.get_q_values(dummy_state)
    assert len(q_vals) == 2

    # Push experiences and update
    for _ in range(10):
        s = np.random.randn(31).astype(np.float32)
        ns = np.random.randn(31).astype(np.float32)
        agent.memory.push(s, 1, 1.0, ns, False)

    loss = agent.update()
    assert loss is not None and loss >= 0.0
