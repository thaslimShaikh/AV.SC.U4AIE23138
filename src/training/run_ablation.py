"""
Ablation Study Training Runner.
Trains and compares Model A (DQN with Simplified Reward) vs Model B (DQN with Latency-Aware Reward).
"""

import os
import json
import yaml
from src.training.train import train_dqn

def run_ablation_study(config_path="configs/config.yaml", num_episodes=300):
    print("=" * 60)
    print("STARTING ABLATION STUDY: Latency-Aware Reward vs Simplified Reward")
    print("=" * 60)

    # 1. Train Model A: Simplified Reward
    print("\n--- Training Model A: Simplified Reward ---")
    agent_simp, history_simp = train_dqn(
        config_path=config_path,
        num_episodes=num_episodes,
        reward_type="simplified",
        seed=42,
        save_name="dqn_simplified_reward.pt"
    )

    # 2. Train Model B: Latency-Aware Reward
    print("\n--- Training Model B: Latency-Aware Reward ---")
    agent_lat, history_lat = train_dqn(
        config_path=config_path,
        num_episodes=num_episodes,
        reward_type="latency_aware",
        seed=42,
        save_name="dqn_latency_aware.pt"
    )

    ablation_summary = {
        "simplified_reward": {
            "final_train_reward": float(history_simp["episode_rewards"][-1]),
            "best_val_reward": max([v["avg_reward"] for v in history_simp["val_evaluations"]]),
            "best_val_latency": min([v["avg_latency"] for v in history_simp["val_evaluations"]]),
            "history": history_simp
        },
        "latency_aware_reward": {
            "final_train_reward": float(history_lat["episode_rewards"][-1]),
            "best_val_reward": max([v["avg_reward"] for v in history_lat["val_evaluations"]]),
            "best_val_latency": min([v["avg_latency"] for v in history_lat["val_evaluations"]]),
            "history": history_lat
        }
    }

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    ablation_file = os.path.join(results_dir, "ablation_results.json")
    with open(ablation_file, "w") as f:
        json.dump(ablation_summary, f, indent=2)

    print(f"\n[Ablation] Ablation study completed successfully! Results saved to {ablation_file}")

if __name__ == "__main__":
    run_ablation_study()
