"""
AdaptGreen: Interactive Web & Research Dashboard
A Streamlit dashboard for real-time mobile context monitoring,
multi-tier inference dispatch, feasibility filtering, offline Q-learning,
and academic paper recommendation.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from adaptgreen.config import SimulationConfig, CONFIG, DEVICE_PRESETS
from adaptgreen.cme.simulator import ContextMonitoringEngine, DeviceState
from adaptgreen.pdu.policy import PolicyDecisionUnit
from adaptgreen.pdu.feasibility import FeasibilityFilter
from adaptgreen.pdu.q_learning import QLearningAgent
from adaptgreen.evaluation.simulator import SimulationRunner
from adaptgreen.evaluation.metrics import (
    compute_top_k_indices,
    compute_top_k_overlap,
    compute_mrr,
    compute_ndcg_at_k,
    evaluate_recommendation_accuracy,
    compute_energy_saving_pct,
)
from adaptgreen.runtime.executor import RuntimeInferenceExecutor
from adaptgreen.runtime.hardware_profiler import HardwareProfiler
from adaptgreen.runtime.adb_device_runner import ADBDeviceRunner

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AdaptGreen Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished publication styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b7837;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .tier-badge-t1 { background-color: #2166ac; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .tier-badge-t2 { background-color: #e08214; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .tier-badge-t3 { background-color: #b2182b; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .tier-badge-t4 { background-color: #35978f; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    .tier-badge-t5 { background-color: #762a83; color: white; padding: 4px 10px; border-radius: 6px; font-weight: bold; }
    /* Hide Streamlit Deploy button and dev toolbar */
    .stDeployButton {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        visibility: hidden !important;
        display: none !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }
    footer {
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Resource Singletons
# -----------------------------------------------------------------------------
@st.cache_resource
def get_inference_executor() -> RuntimeInferenceExecutor:
    executor = RuntimeInferenceExecutor(CONFIG)
    executor.warmup()
    # Pre-populate sample cache
    executor.pre_populate_cache([
        "Navigate to the nearest public electric vehicle charging station.",
        "Summarize unread text messages from emergency contacts.",
        "What is the traffic outlook for the evening commute?",
        "Set a reminder for the project team sync tomorrow morning.",
    ])
    return executor

@st.cache_resource
def get_pdu_instance() -> PolicyDecisionUnit:
    return PolicyDecisionUnit(CONFIG)

@st.cache_resource
def get_paper_corpus() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    p_path = CONFIG.papers_file
    emb_path = CONFIG.base_dir / "data" / "corpus_embeddings.npy"

    papers = []
    if p_path.exists():
        with open(p_path, "r", encoding="utf-8") as f:
            papers = json.load(f)

    if emb_path.exists():
        embs = np.load(emb_path)
    else:
        # Fallback dummy embeddings if not pre-computed
        embs = np.zeros((len(papers), 384), dtype=np.float32)

    return papers, embs


# -----------------------------------------------------------------------------
# Application Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🌿 AdaptGreen: Context-Adaptive Mobile AI Inference</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Research Dashboard & Live Runtime Emulator (ICCA 2026)</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎛️ Real-Time Context & PDU Engine",
    "⚡ Live Inference & Paper Recommender",
    "📊 Full Scenario Simulation & Visualizer",
    "📱 Multi-Device Profiles & Architecture",
])


# =============================================================================
# TAB 1: Real-Time Context Engine & PDU Visualizer
# =============================================================================
with tab1:
    st.subheader("1. Environmental Context Sensing & PDU Decision Engine")
    st.markdown("Configure the 6-dimensional mobile telemetry vector to inspect Stage 1 constraint feasibility and Stage 2 offline Q-learning policy selection.")

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Mobile Hardware Telemetry")

        device_preset = st.selectbox(
            "Device Class Preset",
            [
                "Mid-Range (Default)",
                "Flagship",
                "Budget",
                "Samsung Galaxy S23 Ultra (Snapdragon 8 Gen 2)",
                "Google Pixel 8 Pro (Tensor G3)",
                "Redmi Note 12 / Helio G99",
                "Custom"
            ],
            index=0,
            help="Selects hardware memory headroom floor, thermal thresholds, and SoC characteristics."
        )

        preset_key_map = {
            "Mid-Range (Default)": "MID_RANGE",
            "Flagship": "FLAGSHIP",
            "Budget": "BUDGET",
            "Samsung Galaxy S23 Ultra (Snapdragon 8 Gen 2)": "GALAXY_S23",
            "Google Pixel 8 Pro (Tensor G3)": "PIXEL_8",
            "Redmi Note 12 / Helio G99": "HELIO_G99",
        }
        active_preset_key = preset_key_map.get(device_preset, "MID_RANGE")
        if active_preset_key in DEVICE_PRESETS:
            preset_cfg = DEVICE_PRESETS[active_preset_key]
        else:
            # Dynamic reload or fallback to MID_RANGE
            try:
                import importlib, adaptgreen.config
                importlib.reload(adaptgreen.config)
                preset_cfg = adaptgreen.config.DEVICE_PRESETS.get(active_preset_key, DEVICE_PRESETS["MID_RANGE"])
            except Exception:
                preset_cfg = DEVICE_PRESETS["MID_RANGE"]

        st.markdown("---")
        st.subheader("Scenario Quick-Load")
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)

        scenario_clicked = None
        if c1.button("Foreground Active"):
            scenario_clicked = "FOREGROUND_ACTIVE"
        if c2.button("Background Batch"):
            scenario_clicked = "BACKGROUND_BATCH"
        if c3.button("Thermal Stress"):
            scenario_clicked = "THERMAL_STRESS"
        if c4.button("Battery Critical"):
            scenario_clicked = "BATTERY_CRITICAL"
        if st.button("🌐 Network Offload (T3 → T4 → T5)", use_container_width=True):
            scenario_clicked = "NETWORK_ASSISTED_OFFLOAD"

        cme = ContextMonitoringEngine(42)
        if scenario_clicked:
            demo_state = cme.generate_state(scenario_clicked, epoch=45, total_epochs=60)
            default_batt = float(demo_state.battery)
            default_temp = float(demo_state.temperature)
            default_rssi = float(demo_state.rssi)
            default_user = demo_state.user_activity
            default_task = demo_state.task_type
            default_ram = float(demo_state.available_ram)
        else:
            default_batt = 45.0
            default_temp = 68.0
            default_rssi = -65.0
            default_user = 1
            default_task = 1
            default_ram = 1400.0

        st.markdown("---")
        st.subheader("Telemetry Sliders (6D State)")

        battery = st.slider("Battery Level (%)", min_value=5.0, max_value=100.0, value=default_batt, step=1.0)
        temp = st.slider("SoC Temperature (°C)", min_value=25.0, max_value=95.0, value=default_temp, step=0.5)
        available_ram = st.slider("Available RAM (MB)", min_value=300.0, max_value=4000.0, value=default_ram, step=50.0)
        rssi = st.slider("Network Signal RSSI (dBm)", min_value=-120.0, max_value=0.0, value=default_rssi, step=1.0)

        user_activity = st.radio("User Interactivity (Display)", [1, 0], format_func=lambda x: "Active Foreground (U=1)" if x == 1 else "Background Sync (U=0)", index=0 if default_user == 1 else 1)
        task_type = st.selectbox("Task Workload Classification", [1, 2, 3], format_func=lambda x: {1: "1: Interactive Query", 2: "2: Periodic Background Sync", 3: "3: Analytical / Batch"}[x], index=default_task - 1)

    # Active State
    state = DeviceState(
        battery=battery,
        temperature=temp,
        rssi=rssi,
        user_activity=user_activity,
        task_type=task_type,
        available_ram=available_ram,
    )

    pdu = get_pdu_instance()
    # Configure PDU feasibility filter constraints according to device preset
    from adaptgreen.config import FeasibilityConstraints
    device_constraints = FeasibilityConstraints(
        thermal_max=preset_cfg.thermal_max,
        battery_min=preset_cfg.battery_min,
        ram_min=preset_cfg.ram_min,
    )
    pdu.feasibility_filter = FeasibilityFilter(device_constraints)

    feasible_tiers, violations = pdu.feasibility_filter.evaluate(state)
    state_key = pdu.q_agent.discretize_state(state)
    q_values = pdu.q_agent.get_q_values(state_key)
    decision = pdu.decide(state, epoch=1)

    # Top Row: Metrics & Active Tier Badge
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("State Hash Key", state_key)
    with col_b:
        badge_cls = f"tier-badge-{decision.selected_tier.lower()}"
        tier_names = {"T1": "T1 (FP32)", "T2": "T2 (Pruned)", "T3": "T3 (INT8)", "T4": "T4 (Cloud)", "T5": "T5 (Cache)"}
        st.markdown(f"**PDU Selected Tier**<br><span class='{badge_cls}' style='font-size:1.2rem;'>{tier_names.get(decision.selected_tier, decision.selected_tier)}</span>", unsafe_allow_html=True)
    with col_c:
        st.metric("Hysteresis Cooldown", f"{decision.cooldown_remaining} epochs remaining")
    with col_d:
        st.metric("Catastrophic Emergency", "Active (Override)" if decision.is_emergency else "Normal")

    st.markdown("---")

    # Middle Row: Feasibility Matrix & Q-Learning Inspector
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Stage 1: Feasibility Filter ($C_1 - C_6$)")
        st.caption("Filters out candidate tiers that violate current environmental hardware limits.")

        feasibility_rows = []
        for tier in ["T1", "T2", "T3", "T4", "T5"]:
            is_feas = tier in feasible_tiers
            viols = violations.get(tier, [])
            status_symbol = "✅ Feasible" if is_feas else "❌ Infeasible"
            reason = "; ".join(viols) if viols else "All constraints satisfied"
            feasibility_rows.append({
                "Tier": tier,
                "Model": tier_names[tier],
                "Status": status_symbol,
                "Violation Details": reason
            })

        df_feas = pd.DataFrame(feasibility_rows)
        st.dataframe(df_feas, use_container_width=True, hide_index=True)

        st.info(f"**Feasible Set:** {', '.join(feasible_tiers)}")

    with col_right:
        st.subheader("Stage 2: Tabular Q-Values & Policy Selection")
        st.caption(f"Q-values $Q(s, a)$ computed for discretized state bin: `{state_key}`")

        # Plotly Bar Chart of Q-Values
        tier_colors = {
            "T1": "#2166ac",
            "T2": "#e08214",
            "T3": "#b2182b",
            "T4": "#35978f",
            "T5": "#762a83"
        }
        df_q = pd.DataFrame([
            {"Tier": t, "Q-Value": q_values.get(t, 0.0), "Feasible": t in feasible_tiers}
            for t in ["T1", "T2", "T3", "T4", "T5"]
        ])

        fig_q = go.Figure()
        fig_q.add_trace(go.Bar(
            x=df_q["Tier"],
            y=df_q["Q-Value"],
            marker_color=[tier_colors[t] if df_q.loc[df_q["Tier"] == t, "Feasible"].values[0] else "#cbd5e0" for t in df_q["Tier"]],
            text=[f"{v:.2f}" for v in df_q["Q-Value"]],
            textposition="auto",
        ))
        fig_q.update_layout(
            title="Q-Values Across Actions (Grey = Infeasible)",
            xaxis_title="Action Tier",
            yaxis_title="Learned Q-Value",
            height=320,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_q, use_container_width=True)


# =============================================================================
# TAB 2: Live Inference & Academic Paper Recommender
# =============================================================================
with tab2:
    st.subheader("2. Live PyTorch Inference & Academic Paper Recommender")
    st.markdown("Execute text queries through the active tier selected by the PDU, measure CPU latency, and perform nearest-neighbor semantic search over 200 academic papers.")

    executor = get_inference_executor()
    papers, corpus_embs = get_paper_corpus()

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        sample_queries = [
            "Context-Adaptive Runtime Systems for Mobile Neural Networks",
            "Energy-Efficient Transformer Architectures with Linear Complexity",
            "Dynamic Voltage and Frequency Scaling for Neural Processing Units",
            "Miniaturized Sentence Transformers for On-Device Semantic Indexing",
            "Post-Training Quantization of Large Language Models via Outlier Smoothing",
            "Carbon-Aware Scheduling of Machine Learning Inference in Data Centers",
        ]
        query_choice = st.selectbox("Select Sample Query or Type Below:", ["(Custom Query)"] + sample_queries)
        if query_choice == "(Custom Query)":
            user_query = st.text_input("Query String:", value="Adaptive neural network quantization on edge mobile devices.")
        else:
            user_query = st.text_input("Query String:", value=query_choice)

    with col_q2:
        tier_override = st.selectbox(
            "Execute Under Tier:",
            ["(Auto: PDU Decision)", "T1 (FP32)", "T2 (Pruned)", "T3 (INT8)", "T4 (Cloud)", "T5 (Cache)"],
            index=0
        )
        if tier_override == "(Auto: PDU Decision)":
            target_tier = decision.selected_tier
        else:
            target_tier = tier_override[:2]

    run_btn = st.button("🚀 Execute Live Inference & Retrieve Papers", type="primary")

    if run_btn:
        with st.spinner(f"Executing forward pass on CPU using {tier_names.get(target_tier, target_tier)}..."):
            # Execute evaluated tier
            t0 = time.perf_counter()
            ev_result = executor.execute(user_query, tier=target_tier, rssi=rssi)
            ev_lat = ev_result.latency_ms

            # Execute T1 reference for golden baseline comparison
            t1_result = executor.execute(user_query, tier="T1")

            # Execute T3 reference
            t3_result = executor.execute(user_query, tier="T3")

            # Semantic Paper Retrieval (Top-5)
            ref_top5 = compute_top_k_indices(t1_result.embedding, corpus_embs, k=5)
            ev_top5 = compute_top_k_indices(ev_result.embedding, corpus_embs, k=5)
            top5_overlap = compute_top_k_overlap(ev_top5, ref_top5, k=5)
            ndcg_5 = compute_ndcg_at_k(ev_top5, ref_top5, k=5)
            mrr_val = compute_mrr(ev_top5, ref_top5, k=5)

            # Cosine similarity vs T1 reference
            dot = np.dot(ev_result.embedding, t1_result.embedding)
            sim = dot / (np.linalg.norm(ev_result.embedding) * np.linalg.norm(t1_result.embedding))

        # Metrics display
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Execution Tier", ev_result.tier)
        m2.metric("Wall-Clock Latency", f"{ev_result.latency_ms:.1f} ms")
        m3.metric("Simulated Energy", f"{ev_result.energy_mj:.1f} mJ", delta=f"{compute_energy_saving_pct(45.0, ev_result.energy_mj):.1f}% vs T1")
        m4.metric("Cosine Fidelity", f"{sim:.4f}")
        m5.metric("Top-5 Overlap", f"{top5_overlap * 100.0:.1f}%")
        m6.metric("NDCG@5 / MRR", f"{ndcg_5:.3f} / {mrr_val:.2f}")

        st.markdown("---")
        st.subheader("📚 Top-5 Academic Paper Recommendations (200 Paper Corpus)")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.markdown(f"**Retrieved Under {tier_names.get(target_tier, target_tier)} (Evaluated Tier):**")
            for rank, idx in enumerate(ev_top5, start=1):
                paper = papers[idx] if idx < len(papers) else {"title": f"Paper #{idx}", "category": "AI", "abstract": "N/A"}
                in_ref = "✅ [In T1 Top-5]" if idx in ref_top5 else "⚠️ [Differs from T1]"
                st.markdown(f"""
                **{rank}. {paper['title']}** {in_ref}  
                *Category: {paper['category']}*  
                <small>{paper['abstract'][:160]}...</small>
                """, unsafe_allow_html=True)

        with rec_col2:
            st.markdown("**Retrieved Under T1 (FP32 Golden Reference):**")
            for rank, idx in enumerate(ref_top5, start=1):
                paper = papers[idx] if idx < len(papers) else {"title": f"Paper #{idx}", "category": "AI", "abstract": "N/A"}
                st.markdown(f"""
                **{rank}. {paper['title']}**  
                *Category: {paper['category']}*  
                <small>{paper['abstract'][:160]}...</small>
                """, unsafe_allow_html=True)


# =============================================================================
# TAB 3: Full Scenario Simulation & Visualizer
# =============================================================================
with tab3:
    st.subheader("3. Full Multi-Epoch Scenario Simulation & Publication Charts")
    st.markdown("Run 60-epoch benchmark simulations reproducing the exact scenarios from Table 4 and Table 5 of the ICCA 2026 research paper.")

    sim_col1, sim_col2 = st.columns([2, 1])
    with sim_col1:
        sel_scenario = st.selectbox(
            "Evaluation Scenario",
            ["BATTERY_CRITICAL", "THERMAL_STRESS", "BACKGROUND_BATCH", "FOREGROUND_ACTIVE", "NETWORK_ASSISTED_OFFLOAD"],
            index=0
        )
    with sim_col2:
        num_epochs = st.slider("Total Epochs", min_value=10, max_value=100, value=60, step=10)

    start_sim = st.button("▶️ Run Complete Scenario Simulation", type="primary")

    if start_sim:
        sim_runner = SimulationRunner(CONFIG, executor=executor)
        progress_bar = st.progress(0.0)

        with st.spinner(f"Simulating {sel_scenario} across {num_epochs} epochs..."):
            sim_result = sim_runner.run_scenario(sel_scenario, total_epochs=num_epochs)
            progress_bar.progress(1.0)

        summary_ag = sim_result["summary"]["adaptgreen"]
        df_ag = sim_result["ag_df"]
        df_t1 = sim_result["t1_df"]
        df_t3 = sim_result["t3_df"]

        # Metric Tiles
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Energy", f"{summary_ag['total_energy_mj']:.1f} mJ", f"{summary_ag['saving_vs_t1_pct']:.1f}% vs T1")
        s2.metric("Savings vs Static T3", f"{summary_ag['saving_vs_t3_pct']:.1f}%")
        s3.metric("Semantic Fidelity", f"{summary_ag['fidelity_vs_t1']['mean_cosine_similarity']:.4f}")
        top5_acc_val = f"{summary_ag['recommendation_acc']['mean_top_k_overlap']*100.0:.1f}%" if summary_ag.get('recommendation_acc') else "N/A"
        s4.metric("Top-5 Recommender Overlap", top5_acc_val)

        # Multi-trace Interactive Plots
        st.markdown("### Simulation Time-Series Trajectories")

        fig_sim = go.Figure()
        # Temperature
        fig_sim.add_trace(go.Scatter(x=df_ag["epoch"], y=df_ag["temperature"], name="SoC Temp (°C)", line=dict(color="#d95f02", width=2.5)))
        # Battery
        fig_sim.add_trace(go.Scatter(x=df_ag["epoch"], y=df_ag["battery"], name="Battery (%)", line=dict(color="#7570b3", width=2.5, dash="dash")))
        # C3 Threshold line
        fig_sim.add_hline(y=CONFIG.constraints.thermal_max, line_dash="dot", line_color="#d73027", annotation_text="C3 Thermal Max (80°C)")

        fig_sim.update_layout(
            title=f"Environmental Telemetry: {sel_scenario}",
            xaxis_title="Simulation Epoch",
            yaxis_title="Value (°C / %)",
            height=360,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_sim, use_container_width=True)

        # Tier & Energy Chart
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            tier_num_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5}
            num_tiers = [tier_num_map.get(t, 1) for t in df_ag["selected_tier"]]
            fig_tier = go.Figure()
            fig_tier.add_trace(go.Scatter(
                x=df_ag["epoch"],
                y=num_tiers,
                mode="lines+markers",
                name="Selected Tier",
                line=dict(shape="hv", color="#1b7837", width=2.5),
            ))
            fig_tier.update_layout(
                title="PDU Selected Tier vs. Epoch",
                xaxis_title="Epoch",
                yaxis=dict(tickvals=[1, 2, 3, 4, 5], ticktext=["T1", "T2", "T3", "T4", "T5"]),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_tier, use_container_width=True)

        with col_c2:
            fig_energy = go.Figure()
            fig_energy.add_trace(go.Scatter(x=df_ag["epoch"], y=df_ag["energy_mj"], name="AdaptGreen", line=dict(color="#1b7837", width=2.5)))
            fig_energy.add_trace(go.Scatter(x=df_t1["epoch"], y=df_t1["energy_mj"], name="Static T1 (45 mJ)", line=dict(color="#2166ac", dash="dash")))
            fig_energy.add_trace(go.Scatter(x=df_t3["epoch"], y=df_t3["energy_mj"], name="Static T3 (15 mJ)", line=dict(color="#b2182b", dash="dot")))
            fig_energy.update_layout(
                title="Energy per Inference (mJ) vs. Baselines",
                xaxis_title="Epoch",
                yaxis_title="Energy (mJ)",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_energy, use_container_width=True)

        # CSV Download Button
        csv_data = df_ag.to_csv(index=False)
        st.download_button(
            label="📥 Download Telemetry CSV",
            data=csv_data,
            file_name=f"{sel_scenario.lower()}_telemetry.csv",
            mime="text/csv",
        )


# =============================================================================
# TAB 4: Multi-Device Profiles & Architecture Guide
# =============================================================================
with tab4:
    st.subheader("4. Multi-Device Profiles & Architecture Specification")
    st.markdown("Inspect hardware constraints and JSON profile tables for Flagship, Mid-Range, and Budget device classes.")

    dev_col1, dev_col2 = st.columns([1, 1])

    with dev_col1:
        st.markdown("### Hardware Presets Comparison")
        presets_data = [
            {
                "Device Model": "Samsung Galaxy S23 Ultra",
                "Target SoC": "Snapdragon 8 Gen 2 for Galaxy",
                "RAM": "12 GB",
                "RAM Safety Floor": "400 MB",
                "Thermal Ceiling": "83.0 °C",
                "Battery Floor": "20.0 %",
            },
            {
                "Device Model": "Google Pixel 8 Pro",
                "Target SoC": "Google Tensor G3 (Edge TPU)",
                "RAM": "12 GB",
                "RAM Safety Floor": "500 MB",
                "Thermal Ceiling": "81.0 °C",
                "Battery Floor": "20.0 %",
            },
            {
                "Device Model": "Redmi Note 12 / Budget",
                "Target SoC": "MediaTek Helio G99 / SD685",
                "RAM": "4 GB",
                "RAM Safety Floor": "800 MB",
                "Thermal Ceiling": "77.5 °C",
                "Battery Floor": "15.0 %",
            },
            {
                "Device Model": "Generic Flagship Preset",
                "Target SoC": "Snapdragon 8-series SoC",
                "RAM": "12 GB",
                "RAM Safety Floor": "400 MB",
                "Thermal Ceiling": "82.0 °C",
                "Battery Floor": "20.0 %",
            },
            {
                "Device Model": "Generic Mid-Range Preset",
                "Target SoC": "Snapdragon 6-series SoC",
                "RAM": "6 GB",
                "RAM Safety Floor": "600 MB",
                "Thermal Ceiling": "80.0 °C",
                "Battery Floor": "20.0 %",
            },
            {
                "Device Model": "Generic Budget Preset",
                "Target SoC": "MediaTek G-series SoC",
                "RAM": "4 GB",
                "RAM Safety Floor": "800 MB",
                "Thermal Ceiling": "78.0 °C",
                "Battery Floor": "20.0 %",
            },
        ]
        st.dataframe(pd.DataFrame(presets_data), use_container_width=True, hide_index=True)

        st.markdown("### 5-Tier Operational Characteristics")
        tier_specs = [
            {"Tier": "T1", "Type": "Full Precision FP32", "Energy": "45.0 mJ", "Fidelity": "1.000", "Footprint": "120 MB", "Execution": "Local PyTorch CPU"},
            {"Tier": "T2", "Type": "Structured Pruning (2%)", "Energy": "30.0 mJ", "Fidelity": "0.985", "Footprint": "85 MB", "Execution": "Local PyTorch CPU"},
            {"Tier": "T3", "Type": "Dynamic INT8 Quantized", "Energy": "15.0 mJ", "Fidelity": "0.995", "Footprint": "50 MB", "Execution": "Local PyTorch CPU"},
            {"Tier": "T4", "Type": "Cloud Offload REST", "Energy": "8.0 mJ", "Fidelity": "0.999", "Footprint": "15 MB", "Execution": "Remote REST Server"},
            {"Tier": "T5", "Type": "Vector Cache", "Energy": "1.0 mJ", "Fidelity": "1.000 (Hit)", "Footprint": "5 MB", "Execution": "In-Memory RAM"},
        ]
        st.dataframe(pd.DataFrame(tier_specs), use_container_width=True, hide_index=True)

    with dev_col2:
        st.markdown("### JSON Profile Table Inspector")
        sel_json_device = st.selectbox(
            "Inspect JSON Profile Asset:",
            [
                "samsung_galaxy_s23.json",
                "google_pixel_8.json",
                "budget_helio_g99.json",
                "flagship.json",
                "mid_range.json",
                "budget.json"
            ]
        )
        json_file_path = CONFIG.configs_dir / sel_json_device

        if json_file_path.exists():
            with open(json_file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            st.json(content)
        else:
            st.warning(f"File {json_file_path} not found.")

    # Real Hardware Profiler and Physical ADB Telemetry Section
    st.markdown("---")
    st.subheader("🔬 Real Hardware Benchmarking & Physical Android Telemetry Bridge")
    st.markdown("Execute real empirical benchmarks on this host PC or query physical Android hardware sensors (battery mA/mV, SoC thermal zones, DVFS frequencies) via ADB.")

    h_col1, h_col2 = st.columns([1, 1])

    with h_col1:
        st.markdown("#### 💻 Local Host CPU Hardware Profiler")
        st.caption("Runs multi-iteration CPU inference passes, measuring real wall-clock latency, peak RAM, and empirical energy.")
        
        prof_iters = st.slider("Benchmark Iterations per Tier", min_value=2, max_value=20, value=5, step=1)
        
        if st.button("▶️ Run Real Hardware Benchmark"):
            with st.spinner("Executing real PyTorch CPU forward passes across all 5 tiers..."):
                profiler = HardwareProfiler()
                results = profiler.profile_all_tiers(iterations=prof_iters)
                profiler.export_results(results)
                st.session_state["hw_profile_results"] = results
            st.success("Hardware benchmark completed!")

        if "hw_profile_results" in st.session_state:
            hw_res = st.session_state["hw_profile_results"]
            df_hw = pd.DataFrame(list(hw_res.values()))
            st.dataframe(
                df_hw[[
                    "tier", "latency_mean_ms", "latency_p95_ms", "memory_peak_mb", 
                    "energy_per_inference_mj", "fidelity_score", "top5_overlap_pct"
                ]],
                use_container_width=True,
                hide_index=True
            )

    with h_col2:
        st.markdown("#### 📱 Physical Android Device Bridge (via ADB)")
        st.caption("Reads real battery current/voltage (`/sys/class/power_supply/battery`) and SoC temperatures from an attached Android phone.")
        
        adb_runner = ADBDeviceRunner()
        if st.button("🔄 Poll Connected Android Device"):
            st.session_state["adb_telemetry"] = adb_runner.get_real_telemetry()

        if "adb_telemetry" in st.session_state:
            t = st.session_state["adb_telemetry"]
            if t.connected:
                st.success(f"Connected: {t.device_model} ({t.soc_hardware})")
                m1, m2, m3 = st.columns(3)
                m1.metric("SoC Temp", f"{t.soc_temp_c:.1f} °C")
                m2.metric("Battery Power", f"{t.power_mw:.1f} mW")
                m3.metric("RAM Available", f"{t.available_ram_mb:.0f} MB")
                st.caption(f"Voltage: {t.battery_voltage_mv:.0f} mV | Current: {t.battery_current_ma:.1f} mA | RSSI: {t.wifi_rssi_dbm:.0f} dBm")
            else:
                st.info(f"ℹ️ {t.status_note}")
                st.markdown("""
                **To connect a real Android phone:**
                1. Enable **Developer Options** and turn on **USB Debugging** on your phone.
                2. Connect your phone via USB cable to this PC.
                3. Ensure `adb.exe` is in your system PATH (or Android platform-tools).
                """)

st.markdown("---")
st.caption("AdaptGreen Python Simulation Prototype — Reproducing ICCA 2026 Research Paper Specifications.")
