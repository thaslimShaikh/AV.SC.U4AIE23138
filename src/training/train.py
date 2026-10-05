"""
Training Pipeline for DQN Web Prefetching Agent.
Supports multi-seed training, validation evaluation, checkpoint saving, and metric logging.
"""

import os
import json
import yaml
import numpy as np
import torch
from src.environment.web_env import WebPrefetchEnv
from src.agents.dqn_agent import DQNAgent


def evaluate_agent(agent, env, num_episodes=50):
    """Evaluates agent performance on validation/test environment."""
    rewards = []
    latencies = []
    hits = []
    prefetches = []
    requests = []
    unnecessary = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep + 1000)
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.select_action(obs, evaluate=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            latencies.append(info["perceived_latency_ms"])
            if info["cache_hit"]:
                hits.append(1)
            else:
                hits.append(0)
            if info["action"] == 1:
                prefetches.append(1)
                if info["unnecessary_prefetch"]:
                    unnecessary.append(1)
                else:
                    unnecessary.append(0)
            requests.append(1)

            obs = next_obs

        rewards.append(ep_reward)

    total_reqs = max(1, sum(requests))
    total_pref = max(1, sum(prefetches))

    metrics = {
        "avg_reward": float(np.mean(rewards)),
        "avg_latency": float(np.mean(latencies)),
        "median_latency": float(np.median(latencies)),
        "p95_latency": float(np.percentile(latencies, 95)),
        "prefetch_hit_rate": float(sum(hits) / total_reqs),
        "unnecessary_prefetch_rate": float(sum(unnecessary) / total_pref) if sum(prefetches) > 0 else 0.0,
        "prefetch_count": sum(prefetches)
    }
    return metrics


def train_dqn(config_path="configs/config.yaml", num_episodes=None, reward_type="latency_aware", seed=42, save_name="dqn_model.pt"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set random seeds
    random_seed = seed or config["seed"]
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Load environment
    dataset_path = "data/processed/msnbc_processed.json"
    train_env = WebPrefetchEnv(dataset_path=dataset_path, config_path=config_path, mode="train", reward_type=reward_type)
    val_env = WebPrefetchEnv(dataset_path=dataset_path, config_path=config_path, mode="val", reward_type=reward_type)

    dqn_cfg = config["dqn"]
    agent = DQNAgent(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_space.n,
        hidden_dims=dqn_cfg["hidden_dims"],
        lr=dqn_cfg["learning_rate"],
        gamma=dqn_cfg["gamma"],
        buffer_capacity=dqn_cfg["buffer_capacity"],
        batch_size=dqn_cfg["batch_size"],
        target_update_freq=dqn_cfg["target_update_freq"],
        epsilon_start=dqn_cfg["epsilon_start"],
        epsilon_end=dqn_cfg["epsilon_end"],
        epsilon_decay=dqn_cfg["epsilon_decay"],
        double_dqn=dqn_cfg.get("double_dqn", True)
    )

    max_episodes = num_episodes or config["training"]["episodes_demo"]
    eval_freq = config["training"]["eval_freq"]

    history = {
        "episode_rewards": [],
        "losses": [],
        "epsilons": [],
        "val_evaluations": []
    }

    best_val_reward = -float("inf")
    model_dir = config["logging"]["model_dir"]
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, save_name)

    print(f"[Training] Starting DQN training ({reward_type}) for {max_episodes} episodes, seed={random_seed}...")

    for ep in range(1, max_episodes + 1):
        obs, info = train_env.reset()
        done = False
        ep_reward = 0.0
        ep_losses = []

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, info = train_env.step(action)
            done = terminated or truncated

            agent.memory.push(obs, action, reward, next_obs, done)
            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)

            ep_reward += reward
            obs = next_obs

        agent.decay_epsilon()

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        history["episode_rewards"].append(ep_reward)
        history["losses"].append(avg_loss)
        history["epsilons"].append(agent.epsilon)

        if ep % eval_freq == 0 or ep == max_episodes:
            val_metrics = evaluate_agent(agent, val_env, num_episodes=30)
            val_metrics["episode"] = ep
            history["val_evaluations"].append(val_metrics)

            print(f"[Ep {ep}/{max_episodes}] Reward: {ep_reward:.2f} | Val Reward: {val_metrics['avg_reward']:.2f} | Val Latency: {val_metrics['avg_latency']:.1f}ms | Epsilon: {agent.epsilon:.3f}")

            if val_metrics["avg_reward"] > best_val_reward:
                best_val_reward = val_metrics["avg_reward"]
                agent.save(best_model_path)

    # Save final history
    results_dir = config["logging"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    history_file = os.path.join(results_dir, f"training_history_{reward_type}_seed{random_seed}.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"[Training] Finished training ({reward_type})! Best model saved to {best_model_path}")
    return agent, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--reward_type", type=str, default="latency_aware")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_name", type=str, default="dqn_latency_aware.pt")
    args = parser.parse_args()

    train_dqn(num_episodes=args.episodes, reward_type=args.reward_type, seed=args.seed, save_name=args.save_name)
