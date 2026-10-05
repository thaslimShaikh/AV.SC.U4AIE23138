"""
Comprehensive Evaluation Pipeline.
Evaluates No Prefetch, Always Prefetch, Rule-Based, and DQN policies across multiple scenarios
and seeds, generating benchmark metrics, standard deviations, and tradeoff matrices.
"""

import os
import json
import yaml
import numpy as np
import torch

from src.environment.web_env import WebPrefetchEnv
from src.baselines.heuristics import NoPrefetchPolicy, AlwaysPrefetchPolicy, RuleBasedPolicy
from src.agents.dqn_agent import DQNAgent


def evaluate_policy_on_env(policy, env, num_episodes=50, is_dqn=False):
    """Evaluates a policy on a given Gym environment across num_episodes."""
    ep_rewards = []
    all_latencies = []
    all_hits = []
    all_prefetches = []
    all_requests = []
    all_unnecessary = []
    all_bytes = []
    all_cache_occ = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=1000 + ep)
        done = False
        ep_reward = 0.0

        while not done:
            if is_dqn:
                action = policy.select_action(obs, evaluate=True)
            else:
                action = policy.select_action(obs, info)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            all_latencies.append(info["perceived_latency_ms"])
            all_hits.append(1 if info["cache_hit"] else 0)
            all_requests.append(1)
            all_bytes.append(info["bytes_transferred_mb"])
            all_cache_occ.append(info["cache_occupancy_ratio"])

            if info["action"] == 1:
                all_prefetches.append(1)
                all_unnecessary.append(1 if info["unnecessary_prefetch"] else 0)

            obs = next_obs

        ep_rewards.append(ep_reward)

    total_requests = max(1, sum(all_requests))
    total_prefetches = max(1, sum(all_prefetches))

    avg_lat = float(np.mean(all_latencies))
    med_lat = float(np.median(all_latencies))
    p95_lat = float(np.percentile(all_latencies, 95))
    hit_rate = float(sum(all_hits) / total_requests)
    unnecessary_rate = float(sum(all_unnecessary) / total_prefetches) if sum(all_prefetches) > 0 else 0.0
    avg_bytes = float(np.mean(all_bytes))
    avg_cache_occ = float(np.mean(all_cache_occ))
    avg_reward = float(np.mean(ep_rewards))

    return {
        "avg_latency": avg_lat,
        "median_latency": med_lat,
        "p95_latency": p95_lat,
        "prefetch_hit_rate": hit_rate,
        "unnecessary_prefetch_rate": unnecessary_rate,
        "bandwidth_usage_mb_per_step": avg_bytes,
        "cache_occupancy_ratio": avg_cache_occ,
        "avg_reward": avg_reward
    }


def run_full_evaluation(config_path="configs/config.yaml", num_episodes=100):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset_path = "data/processed/msnbc_processed.json"
    scenarios = config.get("network_scenarios", {})

    # Load DQN agents (Latency-Aware and Simplified)
    dqn_latency = DQNAgent(state_dim=31, action_dim=2)
    model_lat_path = os.path.join(config["logging"]["model_dir"], "dqn_latency_aware.pt")
    if os.path.exists(model_lat_path):
        dqn_latency.load(model_lat_path)
    else:
        print(f"[Warning] Model {model_lat_path} not found! Using initialized agent.")

    dqn_simplified = DQNAgent(state_dim=31, action_dim=2)
    model_simp_path = os.path.join(config["logging"]["model_dir"], "dqn_simplified_reward.pt")
    if os.path.exists(model_simp_path):
        dqn_simplified.load(model_simp_path)

    policies = {
        "No Prefetch": (NoPrefetchPolicy(), False),
        "Always Prefetch": (AlwaysPrefetchPolicy(), False),
        "Rule-Based": (RuleBasedPolicy(threshold=config["baselines"]["rule_based_threshold"]), False),
        "DQN (Simplified)": (dqn_simplified, True),
        "DQN (Latency-Aware)": (dqn_latency, True)
    }

    scenarios_to_run = {
        "Standard": {},
        "Fast Network": scenarios.get("fast", {}),
        "Slow Network": scenarios.get("slow", {}),
        "High Latency": scenarios.get("high_latency", {}),
        "Limited Cache": scenarios.get("limited_cache", {})
    }

    full_results = {}

    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE MULTI-SCENARIO EVALUATION")
    print("=" * 60)

    for sc_name, sc_params in scenarios_to_run.items():
        print(f"\n--- Scenario: {sc_name} ---")
        env = WebPrefetchEnv(dataset_path=dataset_path, config_path=config_path, mode="test", scenario_override=sc_params)
        full_results[sc_name] = {}

        # Reference No Prefetch latency for latency reduction calculation
        no_prefetch_res = evaluate_policy_on_env(policies["No Prefetch"][0], env, num_episodes=num_episodes, is_dqn=False)
        base_lat = no_prefetch_res["avg_latency"]

        for pol_name, (pol, is_dqn) in policies.items():
            metrics = evaluate_policy_on_env(pol, env, num_episodes=num_episodes, is_dqn=is_dqn)

            # Compute Latency Reduction Percentage
            lat_red_pct = ((base_lat - metrics["avg_latency"]) / max(1e-5, base_lat)) * 100.0
            metrics["latency_reduction_pct"] = float(lat_red_pct)

            full_results[sc_name][pol_name] = metrics

            print(f"[{pol_name:20s}] Latency: {metrics['avg_latency']:6.1f}ms | Reduction: {lat_red_pct:5.1f}% | Hit Rate: {metrics['prefetch_hit_rate']*100:5.1f}% | Unnecessary: {metrics['unnecessary_prefetch_rate']*100:5.1f}% | BW: {metrics['bandwidth_usage_mb_per_step']:.2f}MB | Reward: {metrics['avg_reward']:.2f}")

    # Multi-seed Statistical Run for Standard Scenario
    print("\n--- Running Multi-Seed Validation (3 seeds) for Standard Scenario ---")
    multi_seed_results = {pol_name: [] for pol_name in policies.keys()}
    seeds = [42, 123, 999]

    for seed in seeds:
        np.random.seed(seed)
        env = WebPrefetchEnv(dataset_path=dataset_path, config_path=config_path, mode="test")
        no_pref_seed = evaluate_policy_on_env(policies["No Prefetch"][0], env, num_episodes=50, is_dqn=False)
        base_lat_seed = no_pref_seed["avg_latency"]

        for pol_name, (pol, is_dqn) in policies.items():
            m = evaluate_policy_on_env(pol, env, num_episodes=50, is_dqn=is_dqn)
            m["latency_reduction_pct"] = ((base_lat_seed - m["avg_latency"]) / max(1e-5, base_lat_seed)) * 100.0
            multi_seed_results[pol_name].append(m)

    # Compute Means and Std Devs
    statistical_summary = {}
    for pol_name, runs in multi_seed_results.items():
        keys = runs[0].keys()
        statistical_summary[pol_name] = {}
        for k in keys:
            vals = [r[k] for r in runs]
            statistical_summary[pol_name][f"{k}_mean"] = float(np.mean(vals))
            statistical_summary[pol_name][f"{k}_std"] = float(np.std(vals))

    eval_payload = {
        "scenarios": full_results,
        "statistical_summary": statistical_summary
    }

    results_dir = config["logging"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    eval_file = os.path.join(results_dir, "evaluation_results.json")
    with open(eval_file, "w") as f:
        json.dump(eval_payload, f, indent=2)

    print(f"\n[Evaluation] Comprehensive evaluation finished! Saved to {eval_file}")
    return eval_payload


if __name__ == "__main__":
    run_full_evaluation()
