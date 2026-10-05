"""
Data Acquisition Script for MSNBC Web Navigation Dataset.
Downloads dataset from UCI ML Repository or generates fallback synthetic sequence data.
"""

import os
import gzip
import urllib.request
import numpy as np
import yaml

# MSNBC 17 Page Categories Mapping
MSNBC_CATEGORIES = {
    1: "frontpage",
    2: "news",
    3: "tech",
    4: "local",
    5: "opinion",
    6: "on-air",
    7: "misc",
    8: "weather",
    9: "msn-news",
    10: "health",
    11: "living",
    12: "business",
    13: "msn-sports",
    14: "sports",
    15: "summary",
    16: "bbs",
    17: "travel"
}

def download_msnbc_data(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    raw_path = config["dataset"]["raw_path"]
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    if os.path.exists(raw_path) and os.path.getsize(raw_path) > 1000:
        print(f"[Data] Raw MSNBC data already exists at {raw_path}")
        return raw_path

    url = config["dataset"]["raw_url"]
    gz_path = raw_path + ".gz"

    print(f"[Data] Attempting to download MSNBC dataset from {url}...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response, open(gz_path, 'wb') as out_file:
            out_file.write(response.read())

        print(f"[Data] Decompressing {gz_path}...")
        with gzip.open(gz_path, 'rb') as f_in, open(raw_path, 'wb') as f_out:
            f_out.write(f_in.read())

        if os.path.exists(gz_path):
            os.remove(gz_path)
        print(f"[Data] Successfully downloaded and decompressed to {raw_path}")
        return raw_path

    except Exception as e:
        print(f"[Data] Warning: Download failed ({e}). Generating realistic fallback dataset...")
        return generate_fallback_data(raw_path)

def generate_fallback_data(raw_path, num_sessions=5000, seed=42):
    np.random.seed(seed)
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    # Realistic transition weights between categories
    # E.g. frontpage (1) transitions to news (2), sports (14), weather (8), etc.
    lines = []
    header_comment = "# MSNBC web navigation sequence fallback dataset\n"
    lines.append(header_comment)
    header_comment_2 = "# Category mapping: 1:frontpage, 2:news, 3:tech, 4:local, 5:opinion, 6:on-air, 7:misc, 8:weather, 9:msn-news, 10:health, 11:living, 12:business, 13:msn-sports, 14:sports, 15:summary, 16:bbs, 17:travel\n"
    lines.append(header_comment_2)

    # Simple Markov transition simulation for synthetic fallback
    num_cats = 17
    transition_probs = np.ones((num_cats, num_cats)) * 0.02
    # Add high-probability transitions
    transitions = [
        (1, [2, 8, 12, 14, 3]),  # frontpage -> news, weather, business, sports, tech
        (2, [1, 4, 5, 15]),      # news -> frontpage, local, opinion, summary
        (3, [1, 12, 16]),        # tech -> frontpage, business, bbs
        (8, [1, 4]),             # weather -> frontpage, local
        (12, [1, 2, 3]),         # business -> frontpage, news, tech
        (14, [1, 13]),           # sports -> frontpage, msn-sports
        (11, [1, 10, 17]),       # living -> frontpage, health, travel
    ]
    for src, dsts in transitions:
        transition_probs[src - 1, [d - 1 for d in dsts]] += 0.2

    # Normalize rows
    transition_probs = transition_probs / transition_probs.sum(axis=1, keepdims=True)

    with open(raw_path, "w") as f:
        f.write(header_comment)
        f.write(header_comment_2)
        for _ in range(num_sessions):
            session_length = np.random.randint(3, 25)
            curr = np.random.choice([1, 2, 3, 8, 12, 14], p=[0.4, 0.2, 0.1, 0.1, 0.1, 0.1])
            seq = [curr]
            for _ in range(session_length - 1):
                next_cat = np.random.choice(range(1, num_cats + 1), p=transition_probs[curr - 1])
                seq.append(next_cat)
                curr = next_cat
            f.write(" ".join(map(str, seq)) + "\n")

    print(f"[Data] Generated synthetic fallback dataset with {num_sessions} sessions at {raw_path}")
    return raw_path

if __name__ == "__main__":
    download_msnbc_data()
