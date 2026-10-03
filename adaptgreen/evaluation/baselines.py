"""
Static Baselines Evaluation Module.

Implements the non-adaptive reference baselines specified by AdaptGreen:
  1. Static T1 Baseline: Always executes full-precision FP32 PyTorch model (45.0 mJ).
  2. Static T3 Baseline: Always executes dynamic INT8 quantized PyTorch model (15.0 mJ).

Provides identical query workloads and device states to ensure fair,
reproducible head-to-head empirical comparison.
"""

from typing import List, Dict, Any
import numpy as np

from ..config import SimulationConfig, CONFIG
from ..cme.simulator import DeviceState
from ..runtime.executor import RuntimeInferenceExecutor, InferenceResult
from .metrics import aggregate_telemetry


class StaticBaselineExecutor:
    """
    Executes non-adaptive static tier baselines across simulation scenarios.
    """

    def __init__(self, executor: RuntimeInferenceExecutor, config: SimulationConfig = CONFIG):
        self.executor = executor
        self.cfg = config

    def run_baseline(
        self,
        tier: str,
        queries: List[str],
        states: List[DeviceState],
        scenario_name: str
    ) -> Dict[str, Any]:
        """
        Executes a static tier baseline over an entire scenario trace.

        Args:
            tier: Fixed tier identifier ('T1' or 'T3').
            queries: List of input query strings for each epoch.
            states: Pre-generated sequence of DeviceState objects.
            scenario_name: Name of evaluated scenario.

        Returns:
            Dictionary containing metrics, telemetry records, and embeddings.
        """
        records: List[Dict[str, Any]] = []
        embeddings: List[np.ndarray] = []
        latencies: List[float] = []
        energies: List[float] = []

        for epoch, (query, state) in enumerate(zip(queries, states)):
            result = self.executor.execute(text=query, tier=tier)

            records.append({
                "epoch": epoch + 1,
                "scenario": scenario_name,
                "strategy": f"Static_{tier}",
                "battery": state.battery,
                "temperature": state.temperature,
                "rssi": state.rssi,
                "user_activity": state.user_activity,
                "task_type": state.task_type,
                "available_ram": state.available_ram,
                "selected_tier": tier,
                "executed_tier": result.tier,
                "latency_ms": result.latency_ms,
                "energy_mj": result.energy_mj,
                "cache_hit": result.cache_hit,
                "cache_miss": result.cache_miss,
                "tier_switched": False,
                "cooldown_remaining": 0,
            })

            embeddings.append(result.embedding)
            latencies.append(result.latency_ms)
            energies.append(result.energy_mj)

        telemetry_summary = aggregate_telemetry(records)

        return {
            "tier": tier,
            "strategy": f"Static_{tier}",
            "scenario": scenario_name,
            "total_epochs": len(queries),
            "total_energy_mj": telemetry_summary["total_energy_mj"],
            "avg_energy_mj": telemetry_summary["avg_energy_mj"],
            "avg_latency_ms": telemetry_summary["avg_latency_ms"],
            "tier_usage": telemetry_summary["tier_usage"],
            "selected_tier_usage": telemetry_summary["selected_tier_usage"],
            "cache_hit_count": telemetry_summary["cache_hit_count"],
            "cache_miss_count": telemetry_summary["cache_miss_count"],
            "cache_hit_rate_pct": telemetry_summary["cache_hit_rate_pct"],
            "tier_switches": 0,
            "records": records,
            "embeddings": embeddings,
        }
