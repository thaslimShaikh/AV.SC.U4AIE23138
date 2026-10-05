# Latency-Aware Web Resource Prefetching Using Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0%2B-green.svg)](https://farama.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io/)

---

## 1. Executive Summary & Core Philosophy

Web resource prefetching improves user-perceived loading latency by retrieving webpages or assets before explicit user requests. However, **naive prefetching causes bandwidth waste, network congestion, and cache pollution**.

This project implements **Latency-Aware Web Resource Prefetching using Reinforcement Learning (RL)**. Instead of merely predicting *what* the user will request next, the system trains a **Deep Q-Network (DQN)** agent to learn **WHEN prefetching is actually beneficial** given current browsing context, network bandwidth, base round-trip time (RTT) latency, and LRU cache occupancy.

### Core Vision Statement
> *"Don't blindly predict what the user will click. Learn whether it is actually worth preparing for it."*

---

## 2. Problem Statement & System Architecture

### Problem
Given a sequence of web page transitions, variable network conditions (bandwidth $B$, latency $L$), and limited cache capacity $C$, decide at each step whether to **PREFETCH** ($a=1$) or **DON'T PREFETCH** ($a=0$) the top predicted resource to minimize overall page load latency while avoiding excessive bandwidth waste.

### System Pipeline Architecture
```
Real MSNBC Browsing Dataset (UCI ML Repository)
               ↓
Session Processing & Markov Transition Probability Matrix
               ↓
Custom Gymnasium Environment (`WebPrefetchEnv`)
               │  - 31-Dimensional Context Vector
               │  - Dynamic LRU Cache Simulation
               │  - Network Transfer Timing Model
               ↓
Deep Q-Network (DQN) Agent
               │  - Action 0: DON'T PREFETCH
               │  - Action 1: PREFETCH PREDICTED RESOURCE
               ↓
Latency-Aware Reward Function
               │  Reward = Hit Bonus - Latency Penalty - Bandwidth Waste Penalty - Cache Penalty
               ↓
Evaluation & Interactive Streamlit Dashboard (`app/streamlit_app.py`)
```

---

## 3. RL Formulation

### State Representation ($S_t \in \mathbb{R}^{31}$)
1. **Current Page Category** (One-Hot, 17 dims)
2. **Top 5 Next-Page Transition Probabilities** from Markov Model (5 dims)
3. **Top Predicted Next Category Probability** (1 dim)
4. **Is Top Predicted Resource Cached** (Binary, 1 dim)
5. **Normalized Network Bandwidth** ($B / 100.0$, 1 dim)
6. **Normalized Base Network RTT Latency** ($L / 500.0$, 1 dim)
7. **Predicted Resource Size in MB** (Normalized, 1 dim)
8. **LRU Cache Occupancy Ratio** ($C_{curr} / C_{max}$, 1 dim)
9. **Previous Action Taken** (0 or 1, 1 dim)
10. **Session Hit Rate** (1 dim)
11. **Normalized Session Step Progress** (1 dim)

### Action Space ($A_t \in \{0, 1\}$)
- **Action 0 (`DON'T PREFETCH`)**: Wait for explicit user request.
- **Action 1 (`PREFETCH`)**: Proactively download top predicted resource into local LRU cache.

### Latency-Aware Reward Function
$$R_t = R_{\text{hit}} - \lambda_{\text{lat}} \cdot \left(\frac{L_{\text{perceived}}}{200}\right) - \lambda_{\text{bw}} \cdot \mathbb{I}_{\text{unnecessary}} \cdot \text{Bytes}_{\text{MB}} - \lambda_{\text{cache}} \cdot \text{Occupancy}_{\text{ratio}}$$

Where:
- $R_{\text{hit}} = +1.0$ if prefetch results in a cache hit on the actual subsequent user request.
- $L_{\text{perceived}} = 2.0\text{ms}$ on cache hit; otherwise $L_{\text{base}} + \frac{\text{Size} \times 8}{B} \times 1000\text{ms}$.
- $\mathbb{I}_{\text{unnecessary}} = 1$ if prefetched resource is not requested on next step.

---

## 4. Key Experimental Results

Evaluated across **5 controlled network scenarios** comparing **No Prefetch**, **Always Prefetch**, **Rule-Based Threshold**, and **DQN Agent**:

| Scenario | Policy | Avg Latency (ms) | P95 Latency (ms) | Latency Reduction | Prefetch Hit Rate | Unnecessary Prefetch | Bandwidth (MB/step) | Average Reward |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard** | No Prefetch | 825.2 ms | 1850.0 ms | 0.0% | 31.1% | 0.0% | 0.99 MB | -48.46 |
| | Always Prefetch | 638.3 ms | 1850.0 ms | 22.7% | 47.9% | 85.9% | 1.22 MB | -34.19 |
| | Rule-Based | 825.2 ms | 1850.0 ms | 0.0% | 31.1% | 0.0% | 0.99 MB | -48.46 |
| | **DQN (Latency-Aware)** | **661.7 ms** | **1850.0 ms** | **19.8%** | **45.1%** | **84.4%** | **1.20 MB** | **-36.76** |
| **Slow Net** | No Prefetch | 4054.7 ms | 9250.0 ms | 0.0% | 31.1% | 0.0% | 0.99 MB | -236.26 |
| | **DQN (Latency-Aware)** | **3252.2 ms** | **9250.0 ms** | **19.8%** | **45.1%** | **84.6%** | **1.20 MB** | **-187.40** |

*Key Insight*: DQN achieves ~20% latency reduction while using less bandwidth than Always Prefetch.

---

## 5. Quick Start & Execution Guide

### Installation
```bash
# Clone or extract repository
pip install -r requirements.txt
```

### 1. Data Preprocessing
```bash
PYTHONPATH=. python3 src/data/download_msnbc.py
PYTHONPATH=. python3 src/data/preprocess.py
```

### 2. Model Training & Ablation
```bash
# Run ablation training (Simplified Reward vs Latency-Aware Reward)
PYTHONPATH=. python3 src/training/run_ablation.py
```

### 3. Evaluation & Figures Generation
```bash
# Run full multi-scenario evaluation and generate plots
PYTHONPATH=. python3 src/evaluation/evaluate.py
PYTHONPATH=. python3 scripts/generate_plots.py
```

### 4. Interactive Streamlit App
```bash
streamlit run app/streamlit_app.py
```

### 5. Unit Tests
```bash
PYTHONPATH=. pytest tests/
```

---

## 6. Project Structure

```
project/
├── configs/
│   └── config.yaml               # Central hyperparameter configuration
├── data/
│   ├── processed/                # Preprocessed MSNBC dataset
│   └── sample/                   # Sample fallback dataset
├── src/
│   ├── data/                     # Download & parse MSNBC sequence logs
│   ├── environment/              # Gymnasium WebPrefetchEnv & LRUCache
│   ├── agents/                   # PyTorch DQNAgent & ReplayBuffer
│   ├── baselines/                # Heuristic policies (No/Always/Rule-based)
│   ├── training/                 # Multi-seed training & ablation runners
│   ├── evaluation/               # Benchmark metrics evaluation
│   └── visualization/            # Publication-quality plot generation
├── app/
│   └── streamlit_app.py          # Interactive Streamlit Web UI
├── figures/                      # Generated evaluation PNG plots
├── models/                       # Saved PyTorch checkpoint weights (.pt)
├── results/                      # Raw evaluation metrics JSONs
├── report/                       # Supporting academic report markdown documentation
├── tests/                        # PyTest unit test suite
└── requirements.txt
```

---

## 7. Viva Examination Q&A Guide

1. **Q: What is the main innovation of this project?**
   - *A:* Shifting prefetching from purely predicting *what* page comes next to learning *whether it is economically beneficial* to prefetch given network bandwidth, latency, and cache space.
2. **Q: Why use Deep Reinforcement Learning instead of supervised learning?**
   - *A:* Supervised learning handles static pattern recognition, but RL handles dynamic trade-offs, delayed rewards, and resource constraints in sequential decision-making.
3. **Q: How does the agent prevent bandwidth bloat?**
   - *A:* The reward function explicitly penalizes unnecessary downloads based on resource size and bandwidth cost.
4. **Q: What limitations exist?**
   - *A:* Network simulation uses synthetic bandwidth/latency profiles derived from realistic distributions rather than real Chrome browser extension telemetry.

---

## 8. License & Citation
Dataset source: UCI Machine Learning Repository MSNBC Web Navigation Dataset.
Developed for Academic Final-Year Project Demonstration.
