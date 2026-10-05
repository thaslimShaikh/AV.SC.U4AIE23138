"""
Verification script for DQN agent and baseline policies.
"""

import sys
import numpy as np
from src.environment.web_env import WebPrefetchEnv
from src.baselines.heuristics import NoPrefetchPolicy, AlwaysPrefetchPolicy, RuleBasedPolicy
from src.agents.dqn_agent import DQNAgent

def test_agent_and_baselines():
    print("[Agent Test] Initializing environment...")
    env = WebPrefetchEnv(dataset_path="data/processed/msnbc_processed.json", config_path="configs/config.yaml")

    # Baselines test
    p_no = NoPrefetchPolicy()
    p_always = AlwaysPrefetchPolicy()
    p_rule = RuleBasedPolicy(threshold=0.35)

    obs, info = env.reset(seed=42)
    a_no = p_no.select_action(obs, info)
    a_always = p_always.select_action(obs, info)
    a_rule = p_rule.select_action(obs, info)

    print(f"[Baselines Test] NoPrefetch: {a_no}, AlwaysPrefetch: {a_always}, RuleBased: {a_rule}")
    assert a_no == 0 and a_always == 1, "Baseline action output invalid!"

    # DQN Agent test
    print("[Agent Test] Initializing DQNAgent...")
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_space.n)

    q_vals = agent.get_q_values(obs)
    print(f"[Agent Test] Initial Q-values: {q_vals}")
    assert len(q_vals) == 2, f"Expected 2 Q-values, got {len(q_vals)}"

    # Dry run 1 episode + update
    print("[Agent Test] Executing dry run episode with agent updates...")
    obs, info = env.reset(seed=42)
    done = False
    step_count = 0
    losses = []

    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.memory.push(obs, action, reward, next_obs, done)
        loss = agent.update()
        if loss is not None:
            losses.append(loss)

        obs = next_obs
        step_count += 1

    agent.decay_epsilon()
    print(f"[Agent Test] Episode completed in {step_count} steps. Memory size: {len(agent.memory)}")
    print("[Agent Test] Agent and Baselines verification PASSED!")

if __name__ == "__main__":
    test_agent_and_baselines()
