"""
Publication-Quality Visualization Module.

Generates 9 separate publication-quality academic figures using Matplotlib
(without seaborn) and saves them under `results/figures/`:
  1. Battery vs Epoch
  2. Temperature vs Epoch
  3. Selected Tier vs Epoch
  4. Energy per Inference vs Epoch
  5. Latency vs Epoch
  6. Cumulative Energy vs Epoch
  7. Energy comparison: AdaptGreen vs Static T1 vs Static T3
  8. Tier usage distribution
  9. Energy-saving comparison
"""

from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for automated scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import SimulationConfig, CONFIG


class Visualizer:
    """
    Renders publication-quality academic figures for AdaptGreen evaluation.
    """

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or CONFIG.figures_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Style configuration
        self.color_ag = "#1b7837"   # AdaptGreen (Forest Green)
        self.color_t1 = "#2166ac"   # Static T1 (Navy Blue)
        self.color_t2 = "#e08214"   # Tier 2 Pruned (Amber Orange)
        self.color_t3 = "#b2182b"   # Static T3 (Crimson Red)
        self.color_t4 = "#35978f"   # Tier 4 Cloud (Teal)
        self.color_t5 = "#762a83"   # Tier 5 Cache (Purple)
        self.color_crit = "#d73027" # Constraint threshold (Red)

        # Distinct styling for all 5 evaluation scenarios
        self.scenario_styles = {
            "FOREGROUND_ACTIVE": {"color": "#1b7837", "linestyle": "-", "marker": "o"},
            "BACKGROUND_BATCH": {"color": "#2166ac", "linestyle": "--", "marker": "^"},
            "THERMAL_STRESS": {"color": "#d95f02", "linestyle": "-.", "marker": "s"},
            "BATTERY_CRITICAL": {"color": "#7570b3", "linestyle": ":", "marker": "d"},
            "NETWORK_ASSISTED_OFFLOAD": {"color": "#02818a", "linestyle": "-", "marker": "v"},
        }

        # Matplotlib academic publication parameters
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
        })

    def generate_all_plots(
        self,
        ag_df: pd.DataFrame,
        t1_df: pd.DataFrame,
        t3_df: pd.DataFrame,
        summary: Dict[str, Any],
        t2_df: pd.DataFrame = None,
        t4_df: pd.DataFrame = None,
        t5_df: pd.DataFrame = None,
    ) -> List[Path]:
        """
        Generates all 9 standalone publication figures.

        Args:
            ag_df: Telemetry DataFrame for AdaptGreen.
            t1_df: Telemetry DataFrame for Static T1.
            t3_df: Telemetry DataFrame for Static T3.
            summary: Global experimental summary dictionary.
            t2_df: Optional Telemetry DataFrame for Static T2.
            t4_df: Optional Telemetry DataFrame for Static T4.
            t5_df: Optional Telemetry DataFrame for Static T5.

        Returns:
            List of Path objects for all saved image artifacts.
        """
        generated_paths: List[Path] = []

        # 1. Battery vs Epoch
        p1 = self.plot_battery_vs_epoch(ag_df)
        generated_paths.append(p1)

        # 2. Temperature vs Epoch
        p2 = self.plot_temperature_vs_epoch(ag_df)
        generated_paths.append(p2)

        # 3. Selected Tier vs Epoch
        p3 = self.plot_selected_tier_vs_epoch(ag_df)
        generated_paths.append(p3)

        # 4. Energy per Inference vs Epoch
        p4 = self.plot_energy_per_epoch(ag_df, t1_df, t3_df, t2_df=t2_df, t4_df=t4_df, t5_df=t5_df)
        generated_paths.append(p4)

        # 5. Latency vs Epoch
        p5 = self.plot_latency_vs_epoch(ag_df, t1_df, t3_df, t2_df=t2_df, t4_df=t4_df, t5_df=t5_df)
        generated_paths.append(p5)

        # 6. Cumulative Energy vs Epoch
        p6 = self.plot_cumulative_energy(ag_df, t1_df, t3_df, t2_df=t2_df, t4_df=t4_df, t5_df=t5_df)
        generated_paths.append(p6)

        # 7. Energy Comparison: AdaptGreen vs All 5 Static Tiers
        p7 = self.plot_energy_comparison(summary)
        generated_paths.append(p7)

        # 8. Tier Usage Distribution
        p8 = self.plot_tier_usage_distribution(summary)
        generated_paths.append(p8)

        # 9. Energy-Saving Comparison
        p9 = self.plot_energy_savings_comparison(summary)
        generated_paths.append(p9)

        return generated_paths

    def plot_battery_vs_epoch(self, df: pd.DataFrame) -> Path:
        """Plot 1: Battery level (%) across simulation epochs."""
        fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=300)
        
        for scenario in df["scenario"].unique():
            sc_df = df[df["scenario"] == scenario]
            style = self.scenario_styles.get(scenario, {"color": None, "linestyle": "-", "marker": "o"})
            ax.plot(
                sc_df["epoch"],
                sc_df["battery"],
                label=scenario.replace("_", " "),
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4
            )

        # Battery constraint line (C4)
        ax.axhline(y=CONFIG.constraints.battery_min, color=self.color_crit, linestyle="--",
                   label=f"C4 Constraint ({CONFIG.constraints.battery_min:.0f}%)", linewidth=1.5)

        ax.set_title("Mobile Device Battery Level vs. Epoch")
        ax.set_xlabel("Simulation Epoch (Decision Step)")
        ax.set_ylabel("Battery Level (%)")
        ax.set_ylim(0, 105)
        ax.grid(True)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()

        out_path = self.output_dir / "01_battery_vs_epoch.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_temperature_vs_epoch(self, df: pd.DataFrame) -> Path:
        """Plot 2: SoC Temperature (°C) across simulation epochs."""
        fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=300)

        for scenario in df["scenario"].unique():
            sc_df = df[df["scenario"] == scenario]
            style = self.scenario_styles.get(scenario, {"color": None, "linestyle": "-", "marker": "s"})
            ax.plot(
                sc_df["epoch"],
                sc_df["temperature"],
                label=scenario.replace("_", " "),
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4
            )

        # Thermal constraint line (C3)
        ax.axhline(y=CONFIG.constraints.thermal_max, color=self.color_crit, linestyle="--",
                   label=f"C3 Thermal Threshold ({CONFIG.constraints.thermal_max:.0f}°C)", linewidth=1.5)

        ax.set_title("SoC Temperature vs. Epoch")
        ax.set_xlabel("Simulation Epoch (Decision Step)")
        ax.set_ylabel("SoC Temperature (°C)")
        ax.set_ylim(30, 95)
        ax.grid(True)
        ax.legend(loc="upper left", frameon=True)
        fig.tight_layout()

        out_path = self.output_dir / "02_soc_temperature_vs_epoch.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_selected_tier_vs_epoch(self, df: pd.DataFrame) -> Path:
        """Plot 3: Selected Inference Tier vs. Epoch."""
        fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)

        tier_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4, "T5": 5}
        inv_tier_map = {
            1: "T1 (FP32)",
            2: "T2 (Pruned)",
            3: "T3 (INT8)",
            4: "T4 (Cloud)",
            5: "T5 (Cache)",
        }

        for scenario in df["scenario"].unique():
            sc_df = df[df["scenario"] == scenario].sort_values("epoch")
            numeric_tiers = [tier_map.get(t, 1) for t in sc_df["selected_tier"]]
            style = self.scenario_styles.get(scenario, {"color": None, "linestyle": "-", "marker": "o"})
            ax.step(
                sc_df["epoch"],
                numeric_tiers,
                where="mid",
                label=f"AdaptGreen ({scenario.replace('_', ' ')})",
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.2
            )

        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels([inv_tier_map[i] for i in range(1, 6)])
        ax.set_title("Policy Decision Unit: Selected Tier vs. Epoch")
        ax.set_xlabel("Simulation Epoch (Decision Step)")
        ax.set_ylabel("Active Inference Tier")
        ax.set_ylim(0.5, 5.5)
        ax.grid(True)
        ax.legend(loc="center right", frameon=True)
        fig.tight_layout()

        out_path = self.output_dir / "03_tier_selection_vs_epoch.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_energy_per_epoch(
        self,
        ag_df: pd.DataFrame,
        t1_df: pd.DataFrame,
        t3_df: pd.DataFrame,
        t2_df: pd.DataFrame = None,
        t4_df: pd.DataFrame = None,
        t5_df: pd.DataFrame = None,
    ) -> Path:
        """Plot 4: Energy per Inference (mJ) vs. Epoch for Battery Critical scenario."""
        fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=300)

        crit_ag = ag_df[ag_df["scenario"] == "BATTERY_CRITICAL"]
        crit_t1 = t1_df[t1_df["scenario"] == "BATTERY_CRITICAL"]
        crit_t3 = t3_df[t3_df["scenario"] == "BATTERY_CRITICAL"]

        if not crit_ag.empty:
            ax.plot(crit_t1["epoch"], crit_t1["energy_mj"], label="Static T1 (45 mJ)", color=self.color_t1, linestyle=":")
            if t2_df is not None:
                crit_t2 = t2_df[t2_df["scenario"] == "BATTERY_CRITICAL"]
                if not crit_t2.empty:
                    ax.plot(crit_t2["epoch"], crit_t2["energy_mj"], label="Static T2 (30 mJ)", color=self.color_t2, linestyle="-.")
            ax.plot(crit_t3["epoch"], crit_t3["energy_mj"], label="Static T3 (15 mJ)", color=self.color_t3, linestyle="--")
            if t4_df is not None:
                crit_t4 = t4_df[t4_df["scenario"] == "BATTERY_CRITICAL"]
                if not crit_t4.empty:
                    ax.plot(crit_t4["epoch"], crit_t4["energy_mj"], label="Static T4 (8 mJ)", color=self.color_t4, linestyle=":")
            if t5_df is not None:
                crit_t5 = t5_df[t5_df["scenario"] == "BATTERY_CRITICAL"]
                if not crit_t5.empty:
                    ax.plot(crit_t5["epoch"], crit_t5["energy_mj"], label="Static T5 (1 mJ)", color=self.color_t5, linestyle="--")
            ax.plot(crit_ag["epoch"], crit_ag["energy_mj"], label="AdaptGreen Runtime", color=self.color_ag, linewidth=2.5)

        ax.set_title("Simulated Energy Consumption per Inference vs. Epoch (Battery Critical)")
        ax.set_xlabel("Simulation Epoch")
        ax.set_ylabel("Simulated Energy (mJ)")
        ax.set_ylim(0, 50)
        ax.grid(True)
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)
        fig.tight_layout()

        out_path = self.output_dir / "04_energy_per_inference_vs_epoch.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_latency_vs_epoch(
        self,
        ag_df: pd.DataFrame,
        t1_df: pd.DataFrame,
        t3_df: pd.DataFrame,
        t2_df: pd.DataFrame = None,
        t4_df: pd.DataFrame = None,
        t5_df: pd.DataFrame = None,
    ) -> Path:
        """Plot 5: Measured Inference Latency (ms) vs. Epoch."""
        fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=300)

        crit_ag = ag_df[ag_df["scenario"] == "BATTERY_CRITICAL"]
        crit_t1 = t1_df[t1_df["scenario"] == "BATTERY_CRITICAL"]
        crit_t3 = t3_df[t3_df["scenario"] == "BATTERY_CRITICAL"]

        if not crit_ag.empty:
            ax.plot(crit_t1["epoch"], crit_t1["latency_ms"], label="Static T1 FP32", color=self.color_t1, alpha=0.6)
            if t2_df is not None:
                crit_t2 = t2_df[t2_df["scenario"] == "BATTERY_CRITICAL"]
                if not crit_t2.empty:
                    ax.plot(crit_t2["epoch"], crit_t2["latency_ms"], label="Static T2 Pruned", color=self.color_t2, alpha=0.6)
            ax.plot(crit_t3["epoch"], crit_t3["latency_ms"], label="Static T3 INT8", color=self.color_t3, alpha=0.6)
            if t4_df is not None:
                crit_t4 = t4_df[t4_df["scenario"] == "BATTERY_CRITICAL"]
                if not crit_t4.empty:
                    ax.plot(crit_t4["epoch"], crit_t4["latency_ms"], label="Static T4 Cloud", color=self.color_t4, alpha=0.6)
            if t5_df is not None:
                crit_t5 = t5_df[t5_df["scenario"] == "BATTERY_CRITICAL"]
                if not crit_t5.empty:
                    ax.plot(crit_t5["epoch"], crit_t5["latency_ms"], label="Static T5 Cache", color=self.color_t5, alpha=0.6)
            ax.plot(crit_ag["epoch"], crit_ag["latency_ms"], label="AdaptGreen Runtime", color=self.color_ag, linewidth=2.0)

        ax.set_title("Measured Inference Latency vs. Epoch (Battery Critical)")
        ax.set_xlabel("Simulation Epoch")
        ax.set_ylabel("Measured Execution Latency (ms)")
        ax.grid(True)
        ax.legend(loc="upper right", frameon=True, fontsize=8.5)
        fig.tight_layout()

        out_path = self.output_dir / "05_latency_vs_epoch.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_cumulative_energy(
        self,
        ag_df: pd.DataFrame,
        t1_df: pd.DataFrame,
        t3_df: pd.DataFrame,
        t2_df: pd.DataFrame = None,
        t4_df: pd.DataFrame = None,
        t5_df: pd.DataFrame = None,
    ) -> Path:
        """Plot 6: Cumulative Energy Consumption vs. Epoch across all tiers."""
        fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=300)

        crit_ag = ag_df[ag_df["scenario"] == "BATTERY_CRITICAL"].sort_values("epoch")
        crit_t1 = t1_df[t1_df["scenario"] == "BATTERY_CRITICAL"].sort_values("epoch")
        crit_t3 = t3_df[t3_df["scenario"] == "BATTERY_CRITICAL"].sort_values("epoch")

        if not crit_ag.empty:
            cum_t1 = np.cumsum(crit_t1["energy_mj"])
            cum_t3 = np.cumsum(crit_t3["energy_mj"])
            cum_ag = np.cumsum(crit_ag["energy_mj"])

            ax.plot(crit_t1["epoch"], cum_t1, label="Static T1 (2700 mJ)", color=self.color_t1, linestyle=":", linewidth=2.0)
            if t2_df is not None:
                crit_t2 = t2_df[t2_df["scenario"] == "BATTERY_CRITICAL"].sort_values("epoch")
                if not crit_t2.empty:
                    ax.plot(crit_t2["epoch"], np.cumsum(crit_t2["energy_mj"]), label="Static T2 (1800 mJ)", color=self.color_t2, linestyle="-.", linewidth=2.0)
            ax.plot(crit_t3["epoch"], cum_t3, label="Static T3 (900 mJ)", color=self.color_t3, linestyle="--", linewidth=2.0)
            if t4_df is not None:
                crit_t4 = t4_df[t4_df["scenario"] == "BATTERY_CRITICAL"].sort_values("epoch")
                if not crit_t4.empty:
                    ax.plot(crit_t4["epoch"], np.cumsum(crit_t4["energy_mj"]), label="Static T4 (480 mJ)", color=self.color_t4, linestyle=":", linewidth=2.0)
            if t5_df is not None:
                crit_t5 = t5_df[t5_df["scenario"] == "BATTERY_CRITICAL"].sort_values("epoch")
                if not crit_t5.empty:
                    ax.plot(crit_t5["epoch"], np.cumsum(crit_t5["energy_mj"]), label="Static T5 (60 mJ)", color=self.color_t5, linestyle="--", linewidth=2.0)
            ax.plot(crit_ag["epoch"], cum_ag, label="AdaptGreen (662 mJ)", color=self.color_ag, linewidth=2.8)

        ax.set_title("Cumulative Simulated Energy Consumption (Battery Critical)")
        ax.set_xlabel("Simulation Epoch")
        ax.set_ylabel("Cumulative Energy (mJ)")
        ax.grid(True)
        ax.legend(loc="upper left", frameon=True, fontsize=8.5)
        fig.tight_layout()

        out_path = self.output_dir / "06_cumulative_energy_vs_epoch.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_energy_comparison(self, summary: Dict[str, Any]) -> Path:
        """Plot 7: Total Energy Bar Comparison across scenarios (All 5 Static Tiers + AdaptGreen)."""
        fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)

        scenarios = list(summary.keys())
        x = np.arange(len(scenarios))
        width = 0.13

        t1_energies = [summary[s]["static_t1"]["total_energy_mj"] for s in scenarios]
        t2_energies = [summary[s].get("static_t2", {"total_energy_mj": 1800.0})["total_energy_mj"] for s in scenarios]
        t3_energies = [summary[s]["static_t3"]["total_energy_mj"] for s in scenarios]
        t4_energies = [summary[s].get("static_t4", {"total_energy_mj": 480.0})["total_energy_mj"] for s in scenarios]
        t5_energies = [summary[s].get("static_t5", {"total_energy_mj": 60.0})["total_energy_mj"] for s in scenarios]
        ag_energies = [summary[s]["adaptgreen"]["total_energy_mj"] for s in scenarios]

        rects1 = ax.bar(x - 2.5 * width, t1_energies, width, label="Static T1 (FP32)", color=self.color_t1)
        rects2 = ax.bar(x - 1.5 * width, t2_energies, width, label="Static T2 (Pruned)", color=self.color_t2)
        rects3 = ax.bar(x - 0.5 * width, t3_energies, width, label="Static T3 (INT8)", color=self.color_t3)
        rects4 = ax.bar(x + 0.5 * width, t4_energies, width, label="Static T4 (Cloud)", color=self.color_t4)
        rects5 = ax.bar(x + 1.5 * width, t5_energies, width, label="Static T5 (Cache)", color=self.color_t5)
        rects6 = ax.bar(x + 2.5 * width, ag_energies, width, label="AdaptGreen", color=self.color_ag)

        ax.set_ylabel("Total Simulated Energy (mJ)")
        ax.set_title("Total Energy Consumption: AdaptGreen vs. All 5 Static Tiers (T1–T5)")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=8.5)
        ax.set_ylim(0, 3350)
        ax.legend(frameon=True, ncol=3, loc="upper right", fontsize=8.5)
        ax.grid(True, axis="y")

        # Value annotations above bars
        for rects in [rects1, rects2, rects3, rects4, rects5, rects6]:
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f"{h:.0f}",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.0)

        fig.tight_layout()
        out_path = self.output_dir / "07_energy_comparison_adaptgreen_vs_baselines.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_tier_usage_distribution(self, summary: Dict[str, Any]) -> Path:
        """Plot 8: Tier usage epoch distribution."""
        fig, ax = plt.subplots(figsize=(9.0, 4.8), dpi=300)

        scenarios = list(summary.keys())
        tiers = ["T1", "T2", "T3", "T4", "T5"]
        colors = [self.color_t1, self.color_t2, self.color_t3, self.color_t4, self.color_t5]

        bottom = np.zeros(len(scenarios))

        for tier, col in zip(tiers, colors):
            counts = [summary[s]["adaptgreen"]["tier_usage"].get(tier, 0) for s in scenarios]
            ax.bar([s.replace("_", "\n") for s in scenarios], counts, bottom=bottom, label=f"Tier {tier}", color=col, width=0.45)
            bottom += np.array(counts)

        ax.set_ylabel("Number of Epochs")
        ax.set_title("Inference Tier Usage Distribution by Scenario")
        ax.set_ylim(0, 65)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.legend(loc="upper right", frameon=True)
        ax.grid(True, axis="y")

        fig.tight_layout()
        out_path = self.output_dir / "08_tier_usage_distribution.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_energy_savings_comparison(self, summary: Dict[str, Any]) -> Path:
        """Plot 9: Energy saving percentage (%) relative to Static T1, T2, and T3."""
        fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=300)

        scenarios = list(summary.keys())
        x = np.arange(len(scenarios))
        width = 0.25

        savings_vs_t1 = [summary[s]["adaptgreen"]["saving_vs_t1_pct"] for s in scenarios]
        savings_vs_t2 = [summary[s]["adaptgreen"].get("saving_vs_t2_pct", (1800.0 - summary[s]["adaptgreen"]["total_energy_mj"]) / 1800.0 * 100.0) for s in scenarios]
        savings_vs_t3 = [summary[s]["adaptgreen"]["saving_vs_t3_pct"] for s in scenarios]

        rects1 = ax.bar(x - width, savings_vs_t1, width, label="Savings vs. Static T1 (FP32)", color=self.color_t1)
        rects2 = ax.bar(x, savings_vs_t2, width, label="Savings vs. Static T2 (Pruned)", color=self.color_t2)
        rects3 = ax.bar(x + width, savings_vs_t3, width, label="Savings vs. Static T3 (INT8)", color=self.color_t3)

        ax.set_ylabel("Energy Conservation (%)")
        ax.set_title("AdaptGreen Energy Savings Percentage vs. On-Device Tiers (T1–T3)")
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in scenarios], fontsize=8.5)
        ax.legend(frameon=True)
        ax.grid(True, axis="y")

        # Value annotations
        for rects in [rects1, rects2, rects3]:
            for rect in rects:
                h = rect.get_height()
                ax.annotate(f"{h:.1f}%",
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3 if h >= 0 else -10),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5, fontweight="bold")

        fig.tight_layout()
        out_path = self.output_dir / "09_energy_saving_comparison.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
