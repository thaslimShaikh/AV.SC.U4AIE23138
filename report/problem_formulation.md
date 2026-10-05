# Academic Project Report: Latency-Aware Web Resource Prefetching Using Reinforcement Learning

## 1. Problem Formulation & Objectives

### 1.1 Problem Context
Modern web applications consist of rich media resources, including high-resolution imagery, JavaScript bundles, stylesheets, and streaming media. As network bandwidth increases, perceived page loading latency remains dominated by round-trip network transmission delays (RTT) and connection handshake overheads.

Predictive web resource prefetching aims to reduce user-perceived loading latency by retrieving candidate resources into a local browser cache prior to explicit user navigation requests.

### 1.2 The Trade-Off Dilemma
Naive prefetching strategies present a fundamental systems dilemma:
1. **Under-prefetching (No Prefetch)**: Safe from resource waste, but fails to reduce loading latency on highly predictable transition paths.
2. **Over-prefetching (Always Prefetch)**: Maximizes potential prefetch hit rate, but incurs massive bandwidth bloat, cache pollution (evicting valuable existing assets), and power drain on mobile devices.

### 1.3 Core Objectives
- Formulate web prefetching as a sequential decision-making Markov Decision Process (MDP).
- Construct a custom Gymnasium simulation environment (`WebPrefetchEnv`) reflecting user navigation transitions, LRU cache behavior, resource sizes, and dynamic network conditions.
- Design a principled **Latency-Aware Reward Function** that balances latency reduction against bandwidth overheads and cache footprint penalties.
- Implement a **Deep Q-Network (DQN)** agent with Double Q-learning and experience replay to learn optimal prefetching policies.
- Conduct rigorous comparative evaluations against established baselines across variable network scenarios.
