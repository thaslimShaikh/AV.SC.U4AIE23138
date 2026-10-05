"""
Visualization Suite for Latency-Aware Web Prefetching RL Project.
Generates publication-quality charts for training curves, policy comparisons,
scenarios, trade-offs, and ablation studies.
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style for clean, publication-ready figures
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def plot_training_curves(history_lat_path, history_simp_path, output_dir="figures"):
    """Plots Training Rewards, Losses, and Epsilon Decay over episodes."""
    os.makedirs(output_dir, exist_ok=True)

    with open(history_lat_path, "r") as f:
        hist_lat = json.load(f)
    with open(history_simp_path, "r") as f:
        hist_simp = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Episode Rewards
    episodes_lat = list(range(1, len(hist_lat["episode_rewards"]) + 1))
    axes[0].plot(episodes_lat, hist_lat["episode_rewards"], alpha=0.3, color="navy", label="Raw Latency-Aware")
    # Smooth moving average
    window = 15
    if len(hist_lat["episode_rewards"]) >= window:
        smooth_lat = np.convolve(hist_lat["episode_rewards"], np.ones(window)/window, mode='valid')
        axes[0].plot(range(window, len(hist_lat["episode_rewards"]) + 1), smooth_lat, color="navy", linewidth=2, label="Smoothed Latency-Aware")

    axes[0].set_title("Episode Reward vs Training Episode", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Total Reward")
    axes[0].legend()

    # 2. Training Loss
    axes[1].plot(episodes_lat, hist_lat["losses"], color="darkred", alpha=0.8, linewidth=1.5)
    axes[1].set_title("DQN Loss vs Episode", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("MSE Loss")

    # 3. Epsilon Decay
    axes[2].plot(episodes_lat, hist_lat["epsilons"], color="darkgreen", linewidth=2)
    axes[2].set_title("Epsilon Decay Curve", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Epsilon ($\epsilon$)")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "01_training_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved training curves to {out_path}")


def plot_policy_comparison(eval_path, output_dir="figures"):
    """Plots Latency, Hit Rate, and Bandwidth comparisons across policies for Standard scenario."""
    os.makedirs(output_dir, exist_ok=True)

    with open(eval_path, "r") as f:
        eval_data = json.load(f)

    std_data = eval_data["scenarios"]["Standard"]
    policies = list(std_data.keys())
    latencies = [std_data[p]["avg_latency"] for p in policies]
    p95_latencies = [std_data[p]["p95_latency"] for p in policies]
    hit_rates = [std_data[p]["prefetch_hit_rate"] * 100 for p in policies]
    unnecessary = [std_data[p]["unnecessary_prefetch_rate"] * 100 for p in policies]
    bandwidth = [std_data[p]["bandwidth_usage_mb_per_step"] for p in policies]

    colors = ["#708090", "#d9534f", "#f0ad4e", "#5bc0de", "#2e6da4"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Bar plot 1: Average & P95 Latency
    x = np.arange(len(policies))
    width = 0.35
    axes[0, 0].bar(x - width/2, latencies, width, label="Avg Latency", color="#2e6da4")
    axes[0, 0].bar(x + width/2, p95_latencies, width, label="P95 Latency", color="#d9534f", alpha=0.8)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(policies, rotation=15, ha="right")
    axes[0, 0].set_ylabel("Latency (ms)")
    axes[0, 0].set_title("Resource Loading Latency (Lower is Better)", fontweight="bold")
    axes[0, 0].legend()

    # Bar plot 2: Prefetch Hit Rate
    axes[0, 1].bar(policies, hit_rates, color="#5cb85c")
    axes[0, 1].set_xticklabels(policies, rotation=15, ha="right")
    axes[0, 1].set_ylabel("Hit Rate (%)")
    axes[0, 1].set_title("Prefetch Hit Rate (Higher is Better)", fontweight="bold")
    axes[0, 1].set_ylim(0, 100)

    # Bar plot 3: Unnecessary Prefetch Rate
    axes[1, 0].bar(policies, unnecessary, color="#d9534f")
    axes[1, 0].set_xticklabels(policies, rotation=15, ha="right")
    axes[1, 0].set_ylabel("Unnecessary Prefetch (%)")
    axes[1, 0].set_title("Unnecessary Prefetch Rate (Lower is Better)", fontweight="bold")
    axes[1, 0].set_ylim(0, 100)

    # Bar plot 4: Bandwidth Consumption
    axes[1, 1].bar(policies, bandwidth, color="#f0ad4e")
    axes[1, 1].set_xticklabels(policies, rotation=15, ha="right")
    axes[1, 1].set_ylabel("MB / Step")
    axes[1, 1].set_title("Bandwidth Consumption per Request Step", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "02_policy_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved policy comparison to {out_path}")


def plot_latency_bandwidth_tradeoff(eval_path, output_dir="figures"):
    """Scatter plot showing the Latency vs Bandwidth tradeoff."""
    os.makedirs(output_dir, exist_ok=True)

    with open(eval_path, "r") as f:
        eval_data = json.load(f)

    std_data = eval_data["scenarios"]["Standard"]

    plt.figure(figsize=(9, 6))
    colors_map = {
        "No Prefetch": "gray",
        "Always Prefetch": "red",
        "Rule-Based": "orange",
        "DQN (Simplified)": "purple",
        "DQN (Latency-Aware)": "blue"
    }

    for pol, metrics in std_data.items():
        bw = metrics["bandwidth_usage_mb_per_step"]
        lat = metrics["avg_latency"]
        c = colors_map.get(pol, "black")
        plt.scatter(bw, lat, color=c, s=180, zorder=5, label=pol)
        plt.annotate(pol, (bw + 0.005, lat + 5), fontsize=11, fontweight="bold")

    plt.xlabel("Bandwidth Usage (MB / Step)", fontsize=12, fontweight="bold")
    plt.ylabel("Average Latency (ms)", fontsize=12, fontweight="bold")
    plt.title("Latency vs Bandwidth Trade-off Spectrum", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "03_latency_bandwidth_tradeoff.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved tradeoff plot to {out_path}")


def plot_scenario_performance(eval_path, output_dir="figures"):
    """Plots policy average latency across network scenarios."""
    os.makedirs(output_dir, exist_ok=True)

    with open(eval_path, "r") as f:
        eval_data = json.load(f)

    scenarios = list(eval_data["scenarios"].keys())
    policies = ["No Prefetch", "Always Prefetch", "Rule-Based", "DQN (Latency-Aware)"]

    matrix = []
    for sc in scenarios:
        row = [eval_data["scenarios"][sc][p]["avg_latency"] for p in policies]
        matrix.append(row)

    matrix = np.array(matrix)
    x = np.arange(len(scenarios))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#708090", "#d9534f", "#f0ad4e", "#2e6da4"]

    for i, p in enumerate(policies):
        ax.bar(x + (i - 1.5) * width, matrix[:, i], width, label=p, color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Latency (ms)", fontsize=12)
    ax.set_title("Performance Across Different Network & Cache Scenarios", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_yscale("log")  # Log scale due to slow network high latency values

    plt.tight_layout()
    out_path = os.path.join(output_dir, "04_scenario_performance.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved scenario performance to {out_path}")


def generate_all_plots():
    history_lat = "results/training_history_latency_aware_seed42.json"
    history_simp = "results/training_history_simplified_seed42.json"
    eval_file = "results/evaluation_results.json"

    if os.path.exists(history_lat) and os.path.exists(history_simp):
        plot_training_curves(history_lat, history_simp)

    if os.path.exists(eval_file):
        plot_policy_comparison(eval_file)
        plot_latency_bandwidth_tradeoff(eval_file)
        plot_scenario_performance(eval_file)

if __name__ == "__main__":
    generate_all_plots()
