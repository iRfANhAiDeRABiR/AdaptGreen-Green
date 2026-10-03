"""
Simulation Runner and Telemetry Coordinator.

Coordinates full scenario executions for AdaptGreen and static baselines.
Collects epoch-by-epoch telemetry, evaluates semantic fidelity against T1 FP32 reference,
and writes structured CSV results to the results directory.
"""

from typing import List, Dict, Any, Tuple, Optional
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np

from ..config import SimulationConfig, CONFIG
from ..cme.simulator import ContextMonitoringEngine, DeviceState
from ..pdu.policy import PolicyDecisionUnit
from ..runtime.executor import RuntimeInferenceExecutor, InferenceResult
from .baselines import StaticBaselineExecutor
from .metrics import (
    aggregate_telemetry,
    evaluate_embedding_fidelity,
    compute_energy_saving_pct,
    compute_top_k_indices,
    compute_top_k_overlap,
    evaluate_recommendation_accuracy,
)

logger = logging.getLogger("AdaptGreen.Simulator")


# Realistic mobile assistant query workload with recurring patterns
DEFAULT_QUERY_CORPUS = [
    "Navigate to the nearest public electric vehicle charging station.",
    "Summarize unread text messages from emergency contacts.",
    "What is the traffic outlook for the evening commute?",
    "Set a reminder for the project team sync tomorrow morning.",
    "Find highly rated coffee shops with outdoor seating nearby.",
    "Check tomorrow's weather forecast for rain and wind speed.",
    "What is the conversion rate from US dollars to Euros?",
    "Show me the shortest walking route to the train station.",
    "Play the latest news briefing podcast on mobile speakers.",
    "Add milk, organic eggs, and sourdough bread to the grocery list.",
]


class SimulationRunner:
    """
    Orchestrates the AdaptGreen simulation across experimental scenarios.
    """

    def __init__(
        self,
        config: SimulationConfig = CONFIG,
        executor: RuntimeInferenceExecutor = None,
        query_corpus: List[str] = None
    ):
        self.cfg = config
        self.cme = ContextMonitoringEngine(seed=config.random_seed)
        self.executor = executor or RuntimeInferenceExecutor(config)
        self.pdu = PolicyDecisionUnit(config)
        self.baselines = StaticBaselineExecutor(self.executor, config)
        self.query_corpus = query_corpus or DEFAULT_QUERY_CORPUS

        # Warm up execution pipelines and seed realistic cache entries
        self.executor.warmup()
        self.executor.pre_populate_cache(self.query_corpus[:4])

        # Load academic paper corpus embeddings if present
        self.corpus_embeddings: Optional[np.ndarray] = None
        emb_path = self.cfg.base_dir / "data" / "corpus_embeddings.npy"
        if emb_path.exists():
            try:
                self.corpus_embeddings = np.load(emb_path)
                logger.info("Loaded pre-computed corpus embeddings: %s", self.corpus_embeddings.shape)
            except Exception as e:
                logger.warning("Failed to load corpus embeddings: %s", e)

    def generate_scenario_queries(self, num_epochs: int, seed: int) -> List[str]:
        """Generates recurring query sequences matching typical user interaction patterns."""
        rng = np.random.default_rng(seed)
        # 60% probability of selecting recurring query (enabling T5 cache utilization)
        queries: List[str] = []
        for i in range(num_epochs):
            idx = int(rng.integers(0, len(self.query_corpus)))
            queries.append(self.query_corpus[idx])
        return queries

    def run_scenario(self, scenario_name: str, total_epochs: int = 60) -> Dict[str, Any]:
        """
        Executes an experimental scenario comparing AdaptGreen against Static-T1 and Static-T3.

        Args:
            scenario_name: 'FOREGROUND_ACTIVE' or 'BATTERY_CRITICAL'.
            total_epochs: Number of epochs to simulate (default 60).

        Returns:
            Dictionary containing telemetry records, comparative summary metrics, and embeddings.
        """
        logger.info("Executing Scenario: %s (%d epochs)", scenario_name, total_epochs)
        self.pdu.reset()
        self.cme.set_seed(self.cfg.random_seed)

        # 1. Generate environmental device states for all epochs
        states = [
            self.cme.generate_state(scenario_name, ep, total_epochs)
            for ep in range(total_epochs)
        ]

        # 2. Generate identical queries for fair baseline comparison
        queries = self.generate_scenario_queries(total_epochs, self.cfg.random_seed)

        # 3. Run Static-T1 Reference Baseline
        logger.info("Running Static T1 Baseline for %s...", scenario_name)
        res_t1 = self.baselines.run_baseline("T1", queries, states, scenario_name)

        # 4. Run Static-T3 Quantized Baseline
        logger.info("Running Static T3 Baseline for %s...", scenario_name)
        res_t3 = self.baselines.run_baseline("T3", queries, states, scenario_name)

        # 5. Run AdaptGreen Adaptive Runtime Framework
        logger.info("Running AdaptGreen Adaptive Runtime for %s...", scenario_name)
        ag_records: List[Dict[str, Any]] = []
        ag_embeddings: List[np.ndarray] = []
        ag_latencies: List[float] = []
        ag_energies: List[float] = []
        tier_counts: Dict[str, int] = {"T1": 0, "T3": 0, "T5": 0}
        total_switches = 0

        for epoch_idx in range(total_epochs):
            epoch_num = epoch_idx + 1
            state = states[epoch_idx]
            query = queries[epoch_idx]

            # PDU Decision Step (Stage 1 Feasibility -> Stage 2 Q-table + Hysteresis)
            decision = self.pdu.decide(state, epoch_idx)

            if decision.tier_switched and epoch_idx > 0:
                total_switches += 1

            # Dispatch to Runtime Inference Executor
            result = self.executor.execute(text=query, tier=decision.selected_tier, rssi=state.rssi)

            tier_counts[result.tier] = tier_counts.get(result.tier, 0) + 1
            ag_embeddings.append(result.embedding)
            ag_latencies.append(result.latency_ms)
            ag_energies.append(result.energy_mj)

            # Compute epoch top-5 recommendation overlap if corpus available
            top5_overlap = 0.0
            if self.corpus_embeddings is not None and len(res_t1["embeddings"]) > epoch_idx:
                ref_top5 = compute_top_k_indices(res_t1["embeddings"][epoch_idx], self.corpus_embeddings, k=5)
                ev_top5 = compute_top_k_indices(result.embedding, self.corpus_embeddings, k=5)
                top5_overlap = compute_top_k_overlap(ev_top5, ref_top5, k=5)

            # Record Telemetry Record
            record = {
                "epoch": epoch_num,
                "scenario": scenario_name,
                "battery": state.battery,
                "temperature": state.temperature,
                "rssi": state.rssi,
                "user_activity": state.user_activity,
                "task_type": state.task_type,
                "available_ram": state.available_ram,
                "selected_tier": decision.selected_tier,
                "executed_tier": result.tier,
                "latency_ms": result.latency_ms,
                "energy_mj": result.energy_mj,
                "cache_hit": result.cache_hit,
                "cache_miss": result.cache_miss,
                "top5_overlap": round(top5_overlap, 4),
                "tier_switched": decision.tier_switched,
                "cooldown_remaining": decision.cooldown_remaining,
            }
            ag_records.append(record)

            # Formatted live epoch telemetry output
            print(f"Epoch {epoch_num:02d}")
            print(f"Battery: {state.battery:.1f}%")
            print(f"Temperature: {state.temperature:.1f}°C")
            print(f"RAM: {state.available_ram:.0f} MB")
            print(f"Tier: {decision.selected_tier}")
            print(f"Latency: {result.latency_ms:.1f} ms")
            print(f"Energy: {result.energy_mj:.1f} mJ")
            print(f"Top-5 Acc: {top5_overlap * 100.0:.1f}%\n")

        # Compute summary aggregations from the recorded telemetry rows.
        telemetry_summary = aggregate_telemetry(ag_records)
        ag_total_energy = telemetry_summary["total_energy_mj"]

        # Evaluate Semantic Fidelity against Static T1 FP32 ground truth
        fidelity = evaluate_embedding_fidelity(
            reference_embeddings=res_t1["embeddings"],
            evaluated_embeddings=ag_embeddings,
            threshold=self.cfg.constraints.min_similarity_threshold
        )

        t3_fidelity = evaluate_embedding_fidelity(
            reference_embeddings=res_t1["embeddings"],
            evaluated_embeddings=res_t3["embeddings"],
            threshold=self.cfg.constraints.min_similarity_threshold
        )

        # Evaluate Academic Paper Recommendation Accuracy
        rec_acc = None
        if self.corpus_embeddings is not None:
            rec_acc = evaluate_recommendation_accuracy(
                evaluated_queries=ag_embeddings,
                reference_queries=res_t1["embeddings"],
                corpus_embeddings=self.corpus_embeddings,
                k=5,
            )

        # Calculate energy savings
        saving_vs_t1 = compute_energy_saving_pct(res_t1["total_energy_mj"], ag_total_energy)
        saving_vs_t3 = compute_energy_saving_pct(res_t3["total_energy_mj"], ag_total_energy)

        summary = {
            "scenario": scenario_name,
            "total_epochs": total_epochs,
            "adaptgreen": {
                "total_energy_mj": telemetry_summary["total_energy_mj"],
                "avg_energy_mj": telemetry_summary["avg_energy_mj"],
                "avg_latency_ms": telemetry_summary["avg_latency_ms"],
                "tier_usage": telemetry_summary["tier_usage"],
                "selected_tier_usage": telemetry_summary["selected_tier_usage"],
                "tier_switches": total_switches,
                "cache_hit_count": telemetry_summary["cache_hit_count"],
                "cache_miss_count": telemetry_summary["cache_miss_count"],
                "cache_hit_rate_pct": telemetry_summary["cache_hit_rate_pct"],
                "fidelity_vs_t1": fidelity,
                "recommendation_acc": rec_acc,
                "saving_vs_t1_pct": saving_vs_t1,
                "saving_vs_t3_pct": saving_vs_t3,
            },
            "static_t1": {
                "total_energy_mj": res_t1["total_energy_mj"],
                "avg_energy_mj": res_t1["avg_energy_mj"],
                "avg_latency_ms": res_t1["avg_latency_ms"],
            },
            "static_t3": {
                "total_energy_mj": res_t3["total_energy_mj"],
                "avg_energy_mj": res_t3["avg_energy_mj"],
                "avg_latency_ms": res_t3["avg_latency_ms"],
                "fidelity_vs_t1": t3_fidelity,
            },
        }

        # Save Scenario CSV Telemetry
        clean_scenario_name = scenario_name.lower().replace(" ", "_")
        csv_filename = f"{clean_scenario_name}_results.csv"
        csv_path = self.cfg.results_dir / csv_filename
        df_ag = pd.DataFrame(ag_records)
        df_ag.to_csv(csv_path, index=False)
        logger.info("Saved telemetry to: %s", csv_path)

        return {
            "summary": summary,
            "ag_df": df_ag,
            "t1_df": pd.DataFrame(res_t1["records"]),
            "t3_df": pd.DataFrame(res_t3["records"]),
            "states": states,
        }
