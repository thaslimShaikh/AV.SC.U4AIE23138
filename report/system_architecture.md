# System Architecture & Technical Implementation

## 1. System Component Overview

```
                          ┌──────────────────────────┐
                          │   MSNBC Navigation Log   │
                          └────────────┬─────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │ Session Preprocessing &  │
                          │ Markov Transition Matrix │
                          └────────────┬─────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Gymnasium WebPrefetchEnv                              │
│                                                                             │
│   ┌─────────────────────────┐                 ┌─────────────────────────┐   │
│   │   31-Dim State Vector   │                 │   LRU Cache Simulation  │   │
│   └────────────┬────────────┘                 └────────────┬────────────┘   │
│                │                                           │                │
└────────────────┼───────────────────────────────────────────┼────────────────┘
                 │                                           │
                 ▼                                           ▼
┌────────────────────────────────┐               ┌────────────────────────────┐
│      PyTorch DQNAgent          │               │  Network Latency Simulator │
│                                │               │                            │
│  - Policy Network              │               │  Hit Latency: 2.0 ms       │
│  - Target Network              │               │  Miss Latency: RTT + BW    │
│  - Replay Buffer (50k)         │               └─────────────┬──────────────┘
│  - Epsilon-Greedy Exploration  │                             │
└────────────────┬───────────────┘                             │
                 │                                             │
                 ▼                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Latency-Aware Reward Engine                           │
│                                                                             │
│  Reward = Hit Bonus - Latency Penalty - Bandwidth Waste - Cache Penalty     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. Environment Dynamics & LRU Cache Simulation
The environment models an LRU cache with fixed capacity $C_{\text{max}}$ (MB).
When a prefetch or user request occurs for item $k$ with size $S_k$:
- If $k$ exists in cache: updated to Most Recently Used position.
- If $k$ is missing: LRU items are evicted sequentially until $\sum S_i + S_k \le C_{\text{max}}$.

Perceived latency $L_{\text{perceived}}$ is computed as:
$$L_{\text{perceived}} = \begin{cases} 2.0 \text{ ms} & \text{if Cache Hit} \\ L_{\text{base}} + \left(\frac{S_k \cdot 8}{B}\right) \cdot 1000 \text{ ms} & \text{if Cache Miss}\end{cases}$$

## 3. Deep Q-Network Agent Architecture
- **State Dimension**: 31 features
- **Action Dimension**: 2 discrete actions (0: Don't Prefetch, 1: Prefetch)
- **Neural Network Architecture**:
  - Linear(31, 128) -> ReLU()
  - Linear(128, 64) -> ReLU()
  - Linear(64, 2)
- **Optimization**: Adam Optimizer ($\text{lr} = 5 \times 10^{-4}$), MSE Loss on Bellman target.
- **Double DQN Target Update**:
  $$Q_{\text{target}}(s, a) = r + \gamma \cdot Q_{\text{target}}\left(s', \arg\max_{a'} Q_{\text{policy}}(s', a')\right)$$
