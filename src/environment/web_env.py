"""
Custom Gymnasium Environment for Latency-Aware Web Resource Prefetching.
Models web navigation sessions, LRU cache behavior, dynamic network bandwidth/latency,
and prefetching decisions (Prefetch vs Don't Prefetch).
"""

import json
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LRUCache:
    """Least Recently Used (LRU) Cache simulation for web resources."""

    def __init__(self, max_capacity_mb):
        self.max_capacity_mb = max_capacity_mb
        self.current_size_mb = 0.0
        self.cache = {}  # key: category_idx, val: size_mb
        self.usage_order = []  # LRU list (head = oldest, tail = most recently used)

    def contains(self, item_id):
        return item_id in self.cache

    def get(self, item_id):
        if item_id in self.cache:
            self.usage_order.remove(item_id)
            self.usage_order.append(item_id)
            return True
        return False

    def put(self, item_id, size_mb):
        if item_id in self.cache:
            # Update size and move to end
            self.current_size_mb -= self.cache[item_id]
            self.cache[item_id] = size_mb
            self.current_size_mb += size_mb
            self.usage_order.remove(item_id)
            self.usage_order.append(item_id)
            return

        # Evict LRU items if needed
        while self.current_size_mb + size_mb > self.max_capacity_mb and len(self.usage_order) > 0:
            evict_id = self.usage_order.pop(0)
            evict_size = self.cache.pop(evict_id)
            self.current_size_mb -= evict_size

        if self.current_size_mb + size_mb <= self.max_capacity_mb:
            self.cache[item_id] = size_mb
            self.current_size_mb += size_mb
            self.usage_order.append(item_id)

    def clear(self):
        self.cache.clear()
        self.usage_order.clear()
        self.current_size_mb = 0.0

    @property
    def occupancy_ratio(self):
        if self.max_capacity_mb <= 0:
            return 0.0
        return min(1.0, self.current_size_mb / self.max_capacity_mb)


class WebPrefetchEnv(gym.Env):
    """
    Gymnasium Environment for Latency-Aware Web Resource Prefetching.

    State Representation (31 features):
    - Current Page Category (One-Hot, 17 dims)
    - Next Page Transition Probabilities from Markov model (17 dims, max normalized/top prob used)
    - Predicted Next Page Top Category Probability (1 dim)
    - Is Predicted Next Page Cached (1 dim: 0 or 1)
    - Current Network Bandwidth in Mbps (normalized, 1 dim)
    - Base Network RTT Latency in ms (normalized, 1 dim)
    - Predicted Resource Size in MB (normalized, 1 dim)
    - Cache Occupancy Ratio (0.0 to 1.0, 1 dim)
    - Previous Action Taken (0 or 1, 1 dim)
    - Recent Cache Hit Rate in current session (1 dim)
    - Step Index in Session (normalized, 1 dim)

    Action Space:
    - 0: DON'T PREFETCH
    - 1: PREFETCH PREDICTED RESOURCE
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, dataset_path="data/processed/msnbc_processed.json", config_path="configs/config.yaml", mode="train", reward_type="latency_aware", scenario_override=None):
        super(WebPrefetchEnv, self).__init__()

        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        env_cfg = self.config["environment"]
        self.reward_type = reward_type or env_cfg.get("reward_type", "latency_aware")

        # Load processed dataset
        with open(dataset_path, "r") as f:
            self.data = json.load(f)

        self.num_categories = self.data["num_categories"]
        self.transition_matrix = np.array(self.data["transition_matrix"])
        self.resource_profiles = self.data["resource_profiles"]

        if mode == "train":
            self.sessions = self.data["train_sessions"]
        elif mode == "val":
            self.sessions = self.data["val_sessions"]
        else:
            self.sessions = self.data["test_sessions"]

        # Network parameters
        scenario = scenario_override or {}
        self.bandwidth_mbps = scenario.get("bandwidth_mbps", env_cfg["bandwidth_mbps"])
        self.base_latency_ms = scenario.get("latency_ms", env_cfg["base_latency_ms"])
        self.max_cache_size_mb = scenario.get("max_cache_size_mb", env_cfg["max_cache_size_mb"])

        # Reward weights
        self.latency_weight = env_cfg["latency_weight"]
        self.bandwidth_penalty_weight = env_cfg["bandwidth_penalty_weight"]
        self.cache_occupancy_weight = env_cfg["cache_occupancy_weight"]

        # Cache simulation
        self.cache = LRUCache(max_capacity_mb=self.max_cache_size_mb)

        # Action Space: 0 = Don't Prefetch, 1 = Prefetch
        self.action_space = spaces.Discrete(2)

        # State Space Vector Dimension = 17 (onehot) + 1 (prob) + 1 (is_cached) + 1 (bw) + 1 (lat) + 1 (size) + 1 (occupancy) + 1 (prev_act) + 1 (hit_rate) + 1 (step_norm) = 25 dims
        # Or if top 5 transition probabilities: 17 + 5 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 = 31 dims
        self.state_dim = 31
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )

        # Episode state variables
        self.current_session_idx = 0
        self.current_session = []
        self.step_idx = 0
        self.prev_action = 0
        self.session_hits = 0
        self.session_prefetches = 0
        self.session_requests = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Clear cache and pick session
        self.cache.clear()
        session_id = np.random.randint(0, len(self.sessions))
        self.current_session = self.sessions[session_id]
        self.step_idx = 0
        self.prev_action = 0
        self.session_hits = 0
        self.session_prefetches = 0
        self.session_requests = 0

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _get_observation(self):
        curr_page = self.current_session[self.step_idx]

        # One-hot encoding of current page (17 dims)
        one_hot_page = np.zeros(self.num_categories, dtype=np.float32)
        one_hot_page[curr_page] = 1.0

        # Transition probabilities from current page (17 dims) -> select top 5 probs (5 dims)
        trans_probs = self.transition_matrix[curr_page]
        top_5_indices = np.argsort(trans_probs)[::-1][:5]
        top_5_probs = trans_probs[top_5_indices]

        # Best predicted next resource category
        predicted_next = top_5_indices[0]
        top_prob = top_5_probs[0]

        # Check if predicted next is cached
        is_cached = 1.0 if self.cache.contains(predicted_next) else 0.0

        # Estimate size of predicted resource
        profile = self.resource_profiles.get(str(predicted_next + 1), {"avg_size_mb": 1.5})
        pred_size_mb = profile["avg_size_mb"]

        # Feature normalization
        norm_bw = min(1.0, self.bandwidth_mbps / 100.0)
        norm_lat = min(1.0, self.base_latency_ms / 500.0)
        norm_size = min(1.0, pred_size_mb / 5.0)
        cache_occ = self.cache.occupancy_ratio
        prev_act = float(self.prev_action)
        hit_rate = (self.session_hits / max(1, self.session_requests))
        step_norm = float(self.step_idx) / max(1.0, float(len(self.current_session)))

        # Combine into state vector (31 dimensions)
        # 17 (one_hot) + 5 (top_5_probs) + 1 (top_prob) + 1 (is_cached) + 1 (norm_bw) + 1 (norm_lat) + 1 (norm_size) + 1 (cache_occ) + 1 (prev_act) + 1 (hit_rate) + 1 (step_norm) = 31
        obs = np.concatenate([
            one_hot_page,
            top_5_probs,
            [top_prob],
            [is_cached],
            [norm_bw],
            [norm_lat],
            [norm_size],
            [cache_occ],
            [prev_act],
            [hit_rate],
            [step_norm]
        ], axis=0).astype(np.float32)

        return obs

    def step(self, action):
        """
        Executes prefetch decision action for current step:
        Action 0: Don't Prefetch
        Action 1: Prefetch Top Predicted Resource

        Then user advances to next page request in sequence.
        """
        curr_page = self.current_session[self.step_idx]
        trans_probs = self.transition_matrix[curr_page]
        predicted_next = np.argmax(trans_probs)

        # Actual next user request
        self.step_idx += 1
        actual_next = self.current_session[self.step_idx]
        actual_profile = self.resource_profiles.get(str(actual_next + 1), {"avg_size_mb": 1.5})
        actual_size_mb = actual_profile["avg_size_mb"]

        pred_profile = self.resource_profiles.get(str(predicted_next + 1), {"avg_size_mb": 1.5})
        pred_size_mb = pred_profile["avg_size_mb"]

        # Track prefetch execution
        prefetches_performed = 0
        unnecessary_prefetch = False
        bytes_transferred_mb = 0.0

        if action == 1:
            self.session_prefetches += 1
            prefetches_performed = 1
            # If not in cache, fetch it now (costs bandwidth)
            if not self.cache.contains(predicted_next):
                self.cache.put(predicted_next, pred_size_mb)
                bytes_transferred_mb += pred_size_mb

            # Check if prefetch was helpful or unnecessary
            if predicted_next != actual_next:
                unnecessary_prefetch = True

        # User actual request occurs
        self.session_requests += 1
        cache_hit = self.cache.contains(actual_next)

        if cache_hit:
            self.session_hits += 1
            # Cache HIT: minimal latency (local retrieval from memory ~2ms)
            perceived_latency_ms = 2.0
            self.cache.get(actual_next)  # Update LRU order
        else:
            # Cache MISS: download actual resource from network
            # Latency = Base RTT + Transfer Time (Size in Mbits / Bandwidth in Mbps * 1000)
            transfer_time_ms = (actual_size_mb * 8.0 / max(0.1, self.bandwidth_mbps)) * 1000.0
            perceived_latency_ms = self.base_latency_ms + transfer_time_ms
            bytes_transferred_mb += actual_size_mb
            # Place requested resource in cache
            self.cache.put(actual_next, actual_size_mb)

        # Calculate Reward
        reward = self._calculate_reward(
            perceived_latency_ms=perceived_latency_ms,
            cache_hit=cache_hit,
            action=action,
            unnecessary_prefetch=unnecessary_prefetch,
            bytes_transferred_mb=bytes_transferred_mb
        )

        self.prev_action = action
        terminated = (self.step_idx >= len(self.current_session) - 1)
        truncated = False

        obs = self._get_observation() if not terminated else np.zeros(self.state_dim, dtype=np.float32)
        info = self._get_info(
            perceived_latency_ms=perceived_latency_ms,
            cache_hit=cache_hit,
            unnecessary_prefetch=unnecessary_prefetch,
            bytes_transferred_mb=bytes_transferred_mb,
            action=action
        )

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, perceived_latency_ms, cache_hit, action, unnecessary_prefetch, bytes_transferred_mb):
        """
        Calculates normalized reward for RL agent.
        """
        if self.reward_type == "simplified":
            # Simplified reward (for ablation study): purely hit rate focused
            if cache_hit and action == 1:
                return 1.0
            elif unnecessary_prefetch:
                return -0.5
            else:
                return 0.0

        # Latency-Aware Reward Function
        # 1. Normalized Latency Penalty (Scale: 0 to -2.0)
        # Normalizing base latency around ~200ms
        latency_penalty = - (perceived_latency_ms / 200.0) * self.latency_weight

        # 2. Bandwidth Cost Penalty (Scale: -0.1 to -0.5 for unnecessary transfers)
        bandwidth_penalty = 0.0
        if action == 1 and unnecessary_prefetch:
            bandwidth_penalty = - (bytes_transferred_mb / 2.0) * self.bandwidth_penalty_weight

        # 3. Successful Prefetch Bonus
        hit_bonus = 1.0 if (cache_hit and action == 1) else 0.0

        # 4. Cache Footprint Penalty
        cache_penalty = - self.cache.occupancy_ratio * self.cache_occupancy_weight

        total_reward = hit_bonus + latency_penalty + bandwidth_penalty + cache_penalty
        return float(total_reward)

    def _get_info(self, perceived_latency_ms=0.0, cache_hit=False, unnecessary_prefetch=False, bytes_transferred_mb=0.0, action=0):
        curr_page = self.current_session[self.step_idx] if self.step_idx < len(self.current_session) else self.current_session[-1]
        trans_probs = self.transition_matrix[curr_page]
        predicted_next = int(np.argmax(trans_probs))
        top_prob = float(trans_probs[predicted_next])

        return {
            "step_idx": self.step_idx,
            "session_length": len(self.current_session),
            "current_page": curr_page,
            "predicted_next": predicted_next,
            "prediction_confidence": top_prob,
            "action": action,
            "perceived_latency_ms": perceived_latency_ms,
            "cache_hit": cache_hit,
            "unnecessary_prefetch": unnecessary_prefetch,
            "bytes_transferred_mb": bytes_transferred_mb,
            "cache_occupancy_mb": self.cache.current_size_mb,
            "cache_occupancy_ratio": self.cache.occupancy_ratio,
            "session_hits": self.session_hits,
            "session_prefetches": self.session_prefetches,
            "session_requests": self.session_requests,
            "bandwidth_mbps": self.bandwidth_mbps,
            "base_latency_ms": self.base_latency_ms
        }
