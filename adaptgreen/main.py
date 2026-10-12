"""
AdaptGreen Main Execution Entrypoint.

Reproduces the complete Python simulation prototype of the research paper:
"AdaptGreen: A Context-Adaptive Runtime Framework for Energy-Efficient Mobile AI Inference"

Executes:
  1. Offline Q-learning policy training / loading (policy_table.json).
  2. Scenario 1: FOREGROUND_ACTIVE (60 epochs).
  3. Scenario 2: BACKGROUND_BATCH (60 epochs).
  4. Scenario 3: THERMAL_STRESS (60 epochs).
  5. Scenario 4: BATTERY_CRITICAL (60 epochs).
  6. Static T1 and Static T3 reference baselines.
  7. Telemetry generation (CSV logs for each scenario + combined CSV).
  8. Publication-grade visualization (9 standalone figures in results/figures/).
  9. Final reports (results/final_report.json and results/final_report.txt).
  10. Clean terminal summary comparison table.
"""

import sys
import os
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure proper module resolution whether executed from root or package directory
CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CURRENT_DIR.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from adaptgreen.config import CONFIG, SimulationConfig
    from adaptgreen.runtime.executor import RuntimeInferenceExecutor
    from adaptgreen.evaluation.simulator import SimulationRunner
    from adaptgreen.visualization.plots import Visualizer
except ImportError:
    from config import CONFIG, SimulationConfig
    from runtime.executor import RuntimeInferenceExecutor
    from evaluation.simulator import SimulationRunner
    from visualization.plots import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(CONFIG.results_dir / "simulation.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
# Keep root logger quiet on stdout so simulation prints remain clean
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logger = logging.getLogger("AdaptGreen.Main")
logger.setLevel(logging.INFO)


def print_simulation_header() -> None:
    """Prints formatted experiment header matching paper specifications."""
    header = f"""
==================================================
AdaptGreen Python Simulation
==================================================

Model:
{CONFIG.model.model_name}

Device Class:
{CONFIG.device_class}

Scenarios:
Foreground Active
Background Batch
Thermal Stress
Battery Critical

Epochs:
{CONFIG.total_epochs}

Energy Profiles (5 Tiers):
T1 (FP32)         = {CONFIG.energy.T1_MJ:.0f} mJ
T2 (Pruned)       = {CONFIG.energy.T2_MJ:.0f} mJ
T3 (Dynamic INT8) = {CONFIG.energy.T3_MJ:.0f} mJ
T4 (Cloud REST)   = {CONFIG.energy.T4_MJ:.0f} mJ
T5 (Cache)        = {CONFIG.energy.T5_MJ:.0f} mJ

Thermal threshold:
{CONFIG.constraints.thermal_max:.0f}°C

Battery threshold:
{CONFIG.constraints.battery_min:.0f}%

RAM threshold:
{CONFIG.constraints.ram_min:.0f} MB

Hysteresis:
{CONFIG.hysteresis.cooldown_epochs} epochs

==================================================
"""
    print(header)


def format_summary_table(summary: dict) -> str:
    """Formats ASCII terminal summary table of experimental results across all 5 tiers + AdaptGreen."""
    header = (
        f"{'Scenario':<24} | {'Method':<11} | {'Tot (mJ)':<9} | {'Avg (mJ)':<9} | "
        f"{'Avg Lat':<9} | {'Sav vs T1':<10} | {'Sav vs T3':<10} | {'Tiers (1/2/3/4/5)':<17} | {'Top-5 Acc':<9}"
    )
    separator = "-" * len(header)
    rows = [separator, header, separator]

    static_meta = [
        ("static_t1", "Static T1", "60/0/0/0/0", "0.0%", "N/A"),
        ("static_t2", "Static T2", "0/60/0/0/0", "33.3%", "-100.0%"),
        ("static_t3", "Static T3", "0/0/60/0/0", "66.7%", "0.0%"),
        ("static_t4", "Static T4", "0/0/0/60/0", "82.2%", "46.7%"),
        ("static_t5", "Static T5", "0/0/0/0/60", "97.8%", "93.3%"),
    ]

    for sc_name, sc_data in summary.items():
        for idx, (s_key, s_label, s_dist, s_sav_t1, s_sav_t3) in enumerate(static_meta):
            if s_key not in sc_data:
                continue
            s_data = sc_data[s_key]
            sc_col = sc_name if idx == 0 else ""
            top5_str = "100.0%" if s_key == "static_t1" else "N/A"
            if s_data.get("recommendation_acc"):
                top5_str = f"{s_data['recommendation_acc']['mean_top_k_overlap'] * 100.0:.1f}%"
            rows.append(
                f"{sc_col:<24} | {s_label:<11} | {s_data['total_energy_mj']:<9.1f} | "
                f"{s_data['avg_energy_mj']:<9.1f} | {s_data['avg_latency_ms']:<8.1f}ms | "
                f"{s_sav_t1:<10} | {s_sav_t3:<10} | {s_dist:<17} | {top5_str:<9}"
            )

        # AdaptGreen Row
        ag_data = sc_data["adaptgreen"]
        u = ag_data["tier_usage"]
        tier_dist = f"{u.get('T1', 0)}/{u.get('T2', 0)}/{u.get('T3', 0)}/{u.get('T4', 0)}/{u.get('T5', 0)}"
        top5_str = "N/A"
        if ag_data.get("recommendation_acc"):
            top5_str = f"{ag_data['recommendation_acc']['mean_top_k_overlap'] * 100.0:.1f}%"
        rows.append(
            f"{'':<24} | {'AdaptGreen':<11} | {ag_data['total_energy_mj']:<9.1f} | "
            f"{ag_data['avg_energy_mj']:<9.1f} | {ag_data['avg_latency_ms']:<8.1f}ms | "
            f"{ag_data['saving_vs_t1_pct']:<9.1f}% | {ag_data['saving_vs_t3_pct']:<9.1f}% | "
            f"{tier_dist:<17} | {top5_str:<9}"
        )
        rows.append(separator)

    return "\n".join(rows)


def generate_final_report_text(summary: dict) -> str:
    """Generates detailed textual report saved as results/final_report.txt."""
    lines = [
        "================================================================================",
        "                  ADAPTGREEN SIMULATION PROTOTYPE FINAL REPORT                  ",
        "================================================================================",
        "",
        "RESEARCH INTEGRITY DISCLAIMER:",
        "This experiment is a Python simulation prototype evaluating the AdaptGreen",
        "context-adaptive runtime architecture ahead of physical Android deployment.",
        "Simulated Elements: Battery, SoC Temperature, RAM, RSSI, Energy profiles (45/30/15/8/1 mJ).",
        "Actual Software Execution: PyTorch FP32 MiniLM (T1), L2 Structured Pruned MiniLM (T2),",
        "Dynamic INT8 Quantized PyTorch Model (T3), Cloud Offload REST Client (T4),",
        "In-Memory Embedding Cache (T5), Policy Decision Unit, Offline Q-learning, Hysteresis Logic.",
        "",
        "--------------------------------------------------------------------------------",
        "GLOBAL EXPERIMENTAL RESULTS SUMMARY",
        "--------------------------------------------------------------------------------",
        format_summary_table(summary),
        "",
        "--------------------------------------------------------------------------------",
        "DETAILED SCENARIO BREAKDOWN",
        "--------------------------------------------------------------------------------"
    ]

    for sc_name, sc_data in summary.items():
        ag = sc_data["adaptgreen"]
        fid = ag["fidelity_vs_t1"]
        u = ag["tier_usage"]

        rec_str = "N/A"
        ndcg_str = "N/A"
        mrr_str = "N/A"
        if ag.get("recommendation_acc"):
            rec_mean = ag["recommendation_acc"]["mean_top_k_overlap"] * 100.0
            rec_min = ag["recommendation_acc"]["min_top_k_overlap"] * 100.0
            rec_str = f"Mean={rec_mean:.1f}%, Min={rec_min:.1f}%"
            ndcg_str = f"{ag['recommendation_acc'].get('mean_ndcg_at_k', 0.0):.4f}"
            mrr_str = f"{ag['recommendation_acc'].get('mean_mrr', 0.0):.4f}"

        lines.extend([
            f"\n[ SCENARIO: {sc_name} ]",
            f"  - Total Epochs Simulated:       {sc_data['total_epochs']}",
            f"  - Average Execution Latency:    {ag['avg_latency_ms']:.2f} ms",
            f"  - Average Simulated Energy:     {ag['avg_energy_mj']:.2f} mJ",
            f"  - Total Simulated Energy:       {ag['total_energy_mj']:.2f} mJ",
            f"  - Tier Distribution:            T1={u.get('T1', 0)}, T2={u.get('T2', 0)}, T3={u.get('T3', 0)}, T4={u.get('T4', 0)}, T5={u.get('T5', 0)} epochs",
            f"  - Cache Hits / Misses:          {ag['cache_hit_count']} / {ag['cache_miss_count']}",
            f"  - Number of Tier Switches:      {ag['tier_switches']}",
            f"  - Cache Hit Rate:               {ag['cache_hit_rate_pct']:.1f}%",
            f"  - Energy Savings vs. Static T1: {ag['saving_vs_t1_pct']:.2f}%",
            f"  - Energy Savings vs. Static T2: {ag.get('saving_vs_t2_pct', 0.0):.2f}%",
            f"  - Energy Savings vs. Static T3: {ag['saving_vs_t3_pct']:.2f}%",
            f"  - Energy Savings vs. Static T4: {ag.get('saving_vs_t4_pct', 0.0):.2f}%",
            f"  - Energy Savings vs. Static T5: {ag.get('saving_vs_t5_pct', 0.0):.2f}%",
            f"  - Semantic Cosine Similarity:   Mean={fid['mean_cosine_similarity']:.4f}, Min={fid['min_cosine_similarity']:.4f}",
            f"  - Embeddings >= {fid['threshold_used']} Similarity:  {fid['pct_above_threshold']:.1f}%",
            f"  - Top-5 Recommender Overlap:    {rec_str}",
            f"  - Ranking Quality (NDCG@5/MRR): NDCG@5={ndcg_str}, MRR={mrr_str}",
            f"  - Metric Definition Note:       {fid['metric_definition_note']}"
        ])

    lines.append("\n================================================================================\n")
    return "\n".join(lines)


def run_experiment() -> None:
    """Executes the complete AdaptGreen experimental simulation pipeline."""
    print_simulation_header()

    # Shared runtime executor
    executor = RuntimeInferenceExecutor(CONFIG)
    runner = SimulationRunner(CONFIG, executor=executor)
    visualizer = Visualizer(CONFIG.figures_dir)

    all_ag_dfs: list[pd.DataFrame] = []
    all_t1_dfs: list[pd.DataFrame] = []
    all_t2_dfs: list[pd.DataFrame] = []
    all_t3_dfs: list[pd.DataFrame] = []
    all_t4_dfs: list[pd.DataFrame] = []
    all_t5_dfs: list[pd.DataFrame] = []
    global_summary: dict = {}

    scenarios = [
        "FOREGROUND_ACTIVE",
        "BACKGROUND_BATCH",
        "THERMAL_STRESS",
        "BATTERY_CRITICAL",
        "NETWORK_ASSISTED_OFFLOAD",
    ]

    for scenario_name in scenarios:
        print(f"\n{'='*50}")
        print(f"STARTING SCENARIO: {scenario_name}")
        print(f"{'='*50}\n")

        results = runner.run_scenario(scenario_name, total_epochs=CONFIG.total_epochs)

        global_summary[scenario_name] = results["summary"]
        all_ag_dfs.append(results["ag_df"])
        all_t1_dfs.append(results["t1_df"])
        all_t2_dfs.append(results["t2_df"])
        all_t3_dfs.append(results["t3_df"])
        all_t4_dfs.append(results["t4_df"])
        all_t5_dfs.append(results["t5_df"])

    # Combine all scenario telemetry into results/all_results.csv
    combined_ag_df = pd.concat(all_ag_dfs, ignore_index=True)
    combined_t1_df = pd.concat(all_t1_dfs, ignore_index=True)
    combined_t2_df = pd.concat(all_t2_dfs, ignore_index=True)
    combined_t3_df = pd.concat(all_t3_dfs, ignore_index=True)
    combined_t4_df = pd.concat(all_t4_dfs, ignore_index=True)
    combined_t5_df = pd.concat(all_t5_dfs, ignore_index=True)

    all_results_path = CONFIG.results_dir / "all_results.csv"
    combined_ag_df.to_csv(all_results_path, index=False)
    print(f"\n[TELEMETRY] Saved combined results to: {all_results_path}")

    # Generate Publication Figures
    print("\n[VISUALIZATION] Generating 9 publication-grade figures...")
    generated_plots = visualizer.generate_all_plots(
        ag_df=combined_ag_df,
        t1_df=combined_t1_df,
        t3_df=combined_t3_df,
        summary=global_summary,
        t2_df=combined_t2_df,
        t4_df=combined_t4_df,
        t5_df=combined_t5_df,
    )
    for plot_path in generated_plots:
        print(f"  -> Generated: {plot_path.name}")

    # Save final report JSON
    json_path = CONFIG.results_dir / "final_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)
    print(f"\n[REPORT] Saved JSON report to: {json_path}")

    # Save final report TXT
    txt_report = generate_final_report_text(global_summary)
    txt_path = CONFIG.results_dir / "final_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_report)
    print(f"[REPORT] Saved TXT report to: {txt_path}")

    # Print Clean Terminal Summary Table
    print("\n" + "=" * 80)
    print("                    ADAPTGREEN EXPERIMENT SUMMARY TABLE")
    print("=" * 80)
    print(format_summary_table(global_summary))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_experiment()
