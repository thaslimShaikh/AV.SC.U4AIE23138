"""
Dataset Preprocessing Module.
Parses MSNBC raw sequence data, builds transition matrix & resource profiles,
and splits sessions into train/val/test sets without data leakage.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from src.data.download_msnbc import download_msnbc_data, MSNBC_CATEGORIES

# Estimated resource size distributions (in MB) for web categories
# e.g., video/audio rich categories (on-air, news) vs light categories (summary, weather)
CATEGORY_RESOURCE_PROFILES = {
    1: {"name": "frontpage", "avg_size_mb": 1.2, "std_size_mb": 0.3},
    2: {"name": "news", "avg_size_mb": 1.5, "std_size_mb": 0.4},
    3: {"name": "tech", "avg_size_mb": 2.0, "std_size_mb": 0.5},
    4: {"name": "local", "avg_size_mb": 1.0, "std_size_mb": 0.2},
    5: {"name": "opinion", "avg_size_mb": 0.8, "std_size_mb": 0.2},
    6: {"name": "on-air", "avg_size_mb": 3.5, "std_size_mb": 1.0},
    7: {"name": "misc", "avg_size_mb": 1.1, "std_size_mb": 0.3},
    8: {"name": "weather", "avg_size_mb": 0.6, "std_size_mb": 0.1},
    9: {"name": "msn-news", "avg_size_mb": 1.4, "std_size_mb": 0.3},
    10: {"name": "health", "avg_size_mb": 1.2, "std_size_mb": 0.3},
    11: {"name": "living", "avg_size_mb": 1.3, "std_size_mb": 0.3},
    12: {"name": "business", "avg_size_mb": 1.6, "std_size_mb": 0.4},
    13: {"name": "msn-sports", "avg_size_mb": 2.2, "std_size_mb": 0.6},
    14: {"name": "sports", "avg_size_mb": 2.5, "std_size_mb": 0.7},
    15: {"name": "summary", "avg_size_mb": 0.5, "std_size_mb": 0.1},
    16: {"name": "bbs", "avg_size_mb": 0.9, "std_size_mb": 0.2},
    17: {"name": "travel", "avg_size_mb": 1.8, "std_size_mb": 0.5}
}

def parse_msnbc_file(raw_path, min_len=3, max_len=50):
    """Reads MSNBC text file and parses into lists of sequences (category indices 0..16)."""
    sessions = []
    with open(raw_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [int(p) for p in line.split() if p.isdigit()]
            # Filter parts to valid 1..17 categories
            parts = [p for p in parts if 1 <= p <= 17]
            if min_len <= len(parts) <= max_len:
                # Convert to 0-indexed categories (0..16)
                sessions.append([p - 1 for p in parts])
    return sessions

def compute_transition_matrix(sessions, num_categories=17, laplace_smoothing=1.0):
    """Computes first-order Markov transition probability matrix P[src][dst]."""
    counts = np.ones((num_categories, num_categories)) * laplace_smoothing
    for session in sessions:
        for src, dst in zip(session[:-1], session[1:]):
            counts[src, dst] += 1.0
    transition_probs = counts / counts.sum(axis=1, keepdims=True)
    return transition_probs

def preprocess_and_save(config_path="configs/config.yaml", seed=42):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    np.random.seed(seed)
    raw_path = config["dataset"]["raw_path"]

    if not os.path.exists(raw_path):
        raw_path = download_msnbc_data(config_path)

    min_len = config["dataset"]["min_session_length"]
    max_len = config["dataset"]["max_session_length"]
    num_cats = config["dataset"]["num_categories"]

    print(f"[Preprocess] Loading sessions from {raw_path}...")
    sessions = parse_msnbc_file(raw_path, min_len=min_len, max_len=max_len)
    print(f"[Preprocess] Parsed {len(sessions)} valid sessions.")

    # Shuffle sessions deterministically
    indices = np.arange(len(sessions))
    np.random.shuffle(indices)
    shuffled_sessions = [sessions[i] for i in indices]

    # Split ratios
    train_ratio = config["dataset"]["train_ratio"]
    val_ratio = config["dataset"]["val_ratio"]

    n_train = int(len(shuffled_sessions) * train_ratio)
    n_val = int(len(shuffled_sessions) * val_ratio)

    train_sessions = shuffled_sessions[:n_train]
    val_sessions = shuffled_sessions[n_train:n_train + n_val]
    test_sessions = shuffled_sessions[n_train + n_val:]

    print(f"[Preprocess] Train: {len(train_sessions)}, Val: {len(val_sessions)}, Test: {len(test_sessions)}")

    # Transition matrix computed strictly from train_sessions to avoid data leakage
    transition_matrix = compute_transition_matrix(train_sessions, num_categories=num_cats)

    processed_dir = config["dataset"]["processed_dir"]
    sample_dir = config["dataset"]["sample_dir"]
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Save processed data
    dataset_payload = {
        "num_categories": num_cats,
        "categories_map": MSNBC_CATEGORIES,
        "resource_profiles": CATEGORY_RESOURCE_PROFILES,
        "transition_matrix": transition_matrix.tolist(),
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "test_sessions": test_sessions
    }

    processed_file = os.path.join(processed_dir, "msnbc_processed.json")
    with open(processed_file, "w") as f:
        json.dump(dataset_payload, f)

    # Save a small sample dataset payload in data/sample/ for quick testing/fallback
    sample_payload = {
        "num_categories": num_cats,
        "categories_map": MSNBC_CATEGORIES,
        "resource_profiles": CATEGORY_RESOURCE_PROFILES,
        "transition_matrix": transition_matrix.tolist(),
        "train_sessions": train_sessions[:500],
        "val_sessions": val_sessions[:100],
        "test_sessions": test_sessions[:100]
    }
    sample_file = os.path.join(sample_dir, "sample_data.json")
    with open(sample_file, "w") as f:
        json.dump(sample_payload, f)

    print(f"[Preprocess] Saved full processed data to {processed_file}")
    print(f"[Preprocess] Saved sample dataset to {sample_file}")

    return processed_file

if __name__ == "__main__":
    preprocess_and_save()
