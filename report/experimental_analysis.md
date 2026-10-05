# Experimental Analysis & Results Interpretation

## 1. Multi-Scenario Benchmark Results Analysis

Evaluations were performed across 5 distinct network scenarios:

1. **Standard Scenario (10 Mbps, 50ms RTT)**:
   - **No Prefetch**: Avg Latency = 825.2ms, Hit Rate = 31.1%, Bandwidth = 0.99 MB/step.
   - **Always Prefetch**: Avg Latency = 638.3ms, Hit Rate = 47.9%, Bandwidth = 1.22 MB/step (85.9% unnecessary prefetches).
   - **DQN (Latency-Aware)**: Avg Latency = 661.7ms, Hit Rate = 45.1%, Bandwidth = 1.20 MB/step.
   - *Interpretation*: DQN achieves a 19.8% latency reduction while reducing unnecessary network bandwidth usage compared to Always Prefetch.

2. **Slow Network Scenario (2 Mbps, 150ms RTT)**:
   - **No Prefetch**: Avg Latency = 4054.7ms.
   - **DQN (Latency-Aware)**: Avg Latency = 3252.2ms (19.8% Latency Reduction).
   - *Interpretation*: Under constrained bandwidth, latency reduction via prefetching is significantly magnified in absolute time saved (~800ms per step).

3. **Limited Cache Scenario (5 MB Cache)**:
   - **No Prefetch**: Avg Latency = 1029.0ms, Hit Rate = 14.1%.
   - **Always Prefetch**: Avg Latency = 932.6ms, Bandwidth = 2.10 MB/step (high cache eviction rate).
   - **DQN (Latency-Aware)**: Avg Latency = 933.3ms, Bandwidth = 2.04 MB/step.
   - *Interpretation*: Constrained cache causes increased thrashing under Always Prefetching, which DQN learns to mitigate by selective prefetching.

## 2. Ablation Study Insights
Comparing Model A (Simplified Hit-Rate Focused Reward) vs Model B (Latency-Aware Reward):
- Model A prioritizes pure prefetch hit rate regardless of resource size or network speed.
- Model B incorporates latency penalties and bandwidth costs, resulting in disciplined action Q-values that adapt under constrained network environments.
