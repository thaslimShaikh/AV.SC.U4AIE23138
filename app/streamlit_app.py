"""
Streamlit Web Application: Latency-Aware Web Resource Prefetching Using Reinforcement Learning
Features:
1. Interactive Live Browsing Simulation (Step-by-step prefetch decision, cache hit/miss latency visualization)
2. Scenario Explorer (Fast, Slow, High Latency, Limited Cache)
3. Benchmark & Baseline Model Comparison
4. Training Curves & Ablation Study Insights
5. Architecture & Viva Preparation Q&A
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Import project modules
from src.environment.web_env import WebPrefetchEnv
from src.agents.dqn_agent import DQNAgent
from src.baselines.heuristics import NoPrefetchPolicy, AlwaysPrefetchPolicy, RuleBasedPolicy

# Page Configuration
st.set_page_config(
    page_title="PreFetch RL - Latency-Aware Prefetching",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-hit {
        color: #059669;
        font-weight: bold;
    }
    .status-miss {
        color: #DC2626;
        font-weight: bold;
    }
    .decision-badge-prefetch {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 6px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .decision-badge-noprefetch {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_env():
    config_path = "configs/config.yaml"
    dataset_path = "data/processed/msnbc_processed.json"

    if not os.path.exists(dataset_path):
        dataset_path = "data/sample/sample_data.json"

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize DQN Agent
    dqn_agent = DQNAgent(state_dim=31, action_dim=2)
    model_path = os.path.join(config["logging"]["model_dir"], "dqn_latency_aware.pt")
    if os.path.exists(model_path):
        dqn_agent.load(model_path)

    # Baselines
    no_prefetch = NoPrefetchPolicy()
    always_prefetch = AlwaysPrefetchPolicy()
    rule_based = RuleBasedPolicy(threshold=config["baselines"]["rule_based_threshold"])

    return config, dataset_path, dqn_agent, no_prefetch, always_prefetch, rule_based


config, dataset_path, dqn_agent, no_prefetch, always_prefetch, rule_based = load_models_and_env()

# Sidebar Navigation
st.sidebar.title("⚡ PreFetch RL")
st.sidebar.caption("Latency-Aware Resource Prefetching")
app_mode = st.sidebar.radio("Navigation", [
    "🌐 Live Interactive Simulation",
    "📊 Policy Benchmark Comparison",
    "🔬 Ablation Study & Training",
    "🏗️ System Architecture & Viva Q&A"
])

# -----------------------------------------------------------------------------
# MODE 1: LIVE INTERACTIVE SIMULATION
# -----------------------------------------------------------------------------
if app_mode == "🌐 Live Interactive Simulation":
    st.markdown('<p class="main-header">Live Browsing Simulation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Demonstration of DQN prefetching decisions vs baseline strategies in real-time user sessions.</p>', unsafe_allow_html=True)

    # Scenario Selection
    col_sc1, col_sc2 = st.columns([1, 2])
    with col_sc1:
        scenario_choice = st.selectbox("Select Network Scenario:", [
            "Standard (10 Mbps, 50ms)",
            "Fast Network (50 Mbps, 20ms)",
            "Slow Network (2 Mbps, 150ms)",
            "High Latency (10 Mbps, 300ms)",
            "Limited Cache (5MB Cache)"
        ])

    scenario_params = {}
    if "Fast" in scenario_choice:
        scenario_params = config["network_scenarios"]["fast"]
    elif "Slow" in scenario_choice:
        scenario_params = config["network_scenarios"]["slow"]
    elif "High" in scenario_choice:
        scenario_params = config["network_scenarios"]["high_latency"]
    elif "Limited" in scenario_choice:
        scenario_params = config["network_scenarios"]["limited_cache"]

    # Session State Initialization
    if "sim_env" not in st.session_state or st.sidebar.button("🔄 Reset / New Session"):
        st.session_state.sim_env = WebPrefetchEnv(
            dataset_path=dataset_path,
            config_path="configs/config.yaml",
            mode="test",
            scenario_override=scenario_params
        )
        st.session_state.obs, st.session_state.info = st.session_state.sim_env.reset()
        st.session_state.step_history = []

    env = st.session_state.sim_env
    info = st.session_state.info
    obs = st.session_state.obs

    # Current State Cards
    c1, c2, c3, c4 = st.columns(4)
    cat_map = env.data["categories_map"]
    curr_cat_name = cat_map.get(str(info["current_page"] + 1), "Unknown")
    pred_cat_name = cat_map.get(str(info["predicted_next"] + 1), "Unknown")

    with c1:
        st.metric("Current Page", f"{curr_cat_name.upper()}")
    with c2:
        st.metric("Predicted Next Resource", f"{pred_cat_name.upper()}", f"{info['prediction_confidence']*100:.1f}% Prob")
    with c3:
        st.metric("Network Condition", f"{info['bandwidth_mbps']} Mbps", f"{info['base_latency_ms']} ms RTT")
    with c4:
        st.metric("Cache Occupancy", f"{info['cache_occupancy_mb']:.1f} MB", f"{info['cache_occupancy_ratio']*100:.1f}% Capacity")

    st.divider()

    # Agent Q-values and Decision Breakdown
    st.subheader("🤖 RL Agent Decision Analysis")
    q_vals = dqn_agent.get_q_values(obs)
    dqn_action = dqn_agent.select_action(obs, evaluate=True)

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.write("**DQN Q-Values for Current Context:**")
        q_df = pd.DataFrame({
            "Action": ["0: DON'T PREFETCH", "1: PREFETCH"],
            "Q-Value": [q_vals[0], q_vals[1]]
        })
        st.bar_chart(q_df.set_index("Action"))

    with col_q2:
        st.write("**Action Comparison Across Policies:**")
        a_no = no_prefetch.select_action(obs, info)
        a_always = always_prefetch.select_action(obs, info)
        a_rule = rule_based.select_action(obs, info)

        p_comp_df = pd.DataFrame([
            {"Policy": "No Prefetch", "Decision": "DON'T PREFETCH" if a_no == 0 else "PREFETCH"},
            {"Policy": "Always Prefetch", "Decision": "DON'T PREFETCH" if a_always == 0 else "PREFETCH"},
            {"Policy": "Rule-Based", "Decision": "DON'T PREFETCH" if a_rule == 0 else "PREFETCH"},
            {"Policy": "DQN (Latency-Aware)", "Decision": "DON'T PREFETCH" if dqn_action == 0 else "PREFETCH"}
        ])
        st.dataframe(p_comp_df, hide_index=True, use_container_width=True)

    # User Request Execution Button
    if st.button("▶️ Step: User Requests Next Page", type="primary"):
        next_obs, reward, terminated, truncated, step_info = env.step(dqn_action)
        st.session_state.step_history.append(step_info)
        st.session_state.obs = next_obs
        st.session_state.info = step_info

        if terminated:
            st.success("🎉 User Session Completed!")

        st.rerun()

    # Live Session Outcomes
    if st.session_state.step_history:
        st.subheader("📋 Session Request Log")
        log_data = []
        for h in st.session_state.step_history:
            log_data.append({
                "Step": h["step_idx"],
                "Action Taken": "PREFETCH" if h["action"] == 1 else "NO PREFETCH",
                "Perceived Latency (ms)": f"{h['perceived_latency_ms']:.1f}",
                "Cache Outcome": "HIT 🎯" if h["cache_hit"] else "MISS ❌",
                "Unnecessary Prefetch?": "YES ⚠️" if h["unnecessary_prefetch"] else "NO",
                "Bandwidth Used (MB)": f"{h['bytes_transferred_mb']:.2f}"
            })
        st.dataframe(pd.DataFrame(log_data), use_container_width=True)

# -----------------------------------------------------------------------------
# MODE 2: POLICY BENCHMARK COMPARISON
# -----------------------------------------------------------------------------
elif app_mode == "📊 Policy Benchmark Comparison":
    st.markdown('<p class="main-header">Policy Benchmark Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive evaluation metrics comparing DQN against baselines across multiple network scenarios.</p>', unsafe_allow_html=True)

    eval_path = "results/evaluation_results.json"
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            eval_data = json.load(f)

        scenario_sel = st.selectbox("Select Scenario View:", list(eval_data["scenarios"].keys()))
        sc_data = eval_data["scenarios"][scenario_sel]

        # Summary Table
        table_rows = []
        for pol_name, metrics in sc_data.items():
            table_rows.append({
                "Policy": pol_name,
                "Avg Latency (ms)": f"{metrics['avg_latency']:.1f}",
                "Median Latency (ms)": f"{metrics['median_latency']:.1f}",
                "P95 Latency (ms)": f"{metrics['p95_latency']:.1f}",
                "Latency Reduction (%)": f"{metrics.get('latency_reduction_pct', 0.0):.1f}%",
                "Prefetch Hit Rate (%)": f"{metrics['prefetch_hit_rate']*100:.1f}%",
                "Unnecessary Prefetch (%)": f"{metrics['unnecessary_prefetch_rate']*100:.1f}%",
                "Bandwidth (MB/Step)": f"{metrics['bandwidth_usage_mb_per_step']:.2f}",
                "Avg Reward": f"{metrics['avg_reward']:.2f}"
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🖼️ Publication Benchmark Figures")

        f1, f2 = st.columns(2)
        if os.path.exists("figures/02_policy_comparison.png"):
            with f1:
                st.image("figures/02_policy_comparison.png", caption="Policy Metrics Comparison", use_column_width=True)
        if os.path.exists("figures/03_latency_bandwidth_tradeoff.png"):
            with f2:
                st.image("figures/03_latency_bandwidth_tradeoff.png", caption="Latency vs Bandwidth Spectrum", use_column_width=True)

        if os.path.exists("figures/04_scenario_performance.png"):
            st.image("figures/04_scenario_performance.png", caption="Performance Across Scenarios", use_column_width=True)

    else:
        st.warning("Evaluation results file not found. Run evaluation script first.")

# -----------------------------------------------------------------------------
# MODE 3: ABLATION STUDY & TRAINING ANALYTICS
# -----------------------------------------------------------------------------
elif app_mode == "🔬 Ablation Study & Training":
    st.markdown('<p class="main-header">Ablation Study & Training Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluating the necessity of the Latency-Aware reward function vs simplified reward.</p>', unsafe_allow_html=True)

    ablation_path = "results/ablation_results.json"
    if os.path.exists(ablation_path):
        with open(ablation_path, "r") as f:
            ablation_data = json.load(f)

        m_a = ablation_data["simplified_reward"]
        m_b = ablation_data["latency_aware_reward"]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Model A (Simplified Reward) Best Val Reward", f"{m_a['best_val_reward']:.2f}")
        with c2:
            st.metric("Model B (Latency-Aware Reward) Best Val Reward", f"{m_b['best_val_reward']:.2f}")

        st.divider()

        if os.path.exists("figures/01_training_curves.png"):
            st.image("figures/01_training_curves.png", caption="Training Rewards, Loss, and Epsilon Decay", use_column_width=True)
    else:
        st.warning("Ablation results file not found.")

# -----------------------------------------------------------------------------
# MODE 4: SYSTEM ARCHITECTURE & VIVA Q&A
# -----------------------------------------------------------------------------
elif app_mode == "🏗️ System Architecture & Viva Q&A":
    st.markdown('<p class="main-header">System Architecture & Viva Q&A Guide</p>', unsafe_allow_html=True)

    st.subheader("📐 System Architecture")
    st.code("""
Real MSNBC Browsing Data
       ↓
Markov Transition Matrix & Category Profiles
       ↓
Custom Gymnasium Environment (WebPrefetchEnv)
       │  - State Vector (31 dims: current page, predictions, network bandwidth, latency, cache)
       │  - LRU Cache Simulator
       ↓
Deep Q-Network (DQN) Agent
       │  - Action 0: DON'T PREFETCH
       │  - Action 1: PREFETCH
       ↓
Latency-Aware Reward Function
       │  Reward = Hit Bonus - Latency Penalty - Bandwidth Waste Penalty - Cache Occupancy Penalty
       ↓
Evaluation & Streamlit Interactive Demo
    """, language="text")

    st.divider()
    st.subheader("🎓 Viva Examination Quick Reference Q&A")

    qas = [
        ("1. What is the central research problem?", "Web prefetching improves user perceived loading times, but naive prefetching wastes network bandwidth, memory, and cache. The problem is learning WHEN prefetching is actually beneficial given network conditions, cache states, and prediction confidence."),
        ("2. Why Reinforcement Learning instead of supervised prediction?", "Supervised models only predict WHAT page the user clicks next, but cannot decide WHETHER IT IS WORTH PREFETCHING under dynamic, variable network and cache resource constraints. RL models sequential decisions and cost-benefit trade-offs."),
        ("3. How is the RL State represented?", "31-dimensional vector containing one-hot encoded page category, top transition probabilities from Markov model, network bandwidth (Mbps), base RTT latency (ms), resource size, LRU cache occupancy ratio, previous action, and session hit rate."),
        ("4. What are the RL Actions?", "Discrete binary action space: Action 0 = Don't Prefetch, Action 1 = Prefetch top predicted resource."),
        ("5. How does the Latency-Aware Reward function work?", "Combines latency reduction bonus/penalty, bandwidth waste penalty for unnecessary prefetches, and cache footprint penalty. This prevents the agent from blindly prefetching everything."),
        ("6. What baseline algorithms were compared?", "1) No Prefetching, 2) Always Prefetching, 3) Rule-Based Probability Threshold prefetching, and 4) DQN Agents.")
    ]

    for q, a in qas:
        with st.expander(q):
            st.write(a)
