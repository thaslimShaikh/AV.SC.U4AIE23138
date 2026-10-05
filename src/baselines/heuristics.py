"""
Baseline Heuristic Policies for Web Resource Prefetching.
Implementations:
1. NoPrefetchPolicy: Always returns 0 (DON'T PREFETCH)
2. AlwaysPrefetchPolicy: Always returns 1 (PREFETCH)
3. RuleBasedPolicy: Prefetches if transition probability exceeds threshold
"""

import numpy as np

class NoPrefetchPolicy:
    """Baseline 1: Conservative strategy - never prefetch."""
    def select_action(self, obs, info=None):
        return 0

class AlwaysPrefetchPolicy:
    """Baseline 2: Aggressive strategy - always prefetch top predicted resource."""
    def select_action(self, obs, info=None):
        return 1

class RuleBasedPolicy:
    """
    Baseline 3: Markov transition probability threshold strategy.
    Prefetches if predicted next resource probability >= threshold.
    """
    def __init__(self, threshold=0.35):
        self.threshold = threshold

    def select_action(self, obs, info=None):
        if info is not None and "prediction_confidence" in info:
            prob = info["prediction_confidence"]
        else:
            # Transition top probability is at index 22 in state vector
            prob = float(obs[22])
        return 1 if prob >= self.threshold else 0
