"""
Unit tests validating full 5-tier model library integration in AdaptGreen:
  - T1: Full Precision FP32 PyTorch MiniLM (45 mJ)
  - T2: L2 Structured Pruned PyTorch MiniLM (30 mJ)
  - T3: Dynamic INT8 Quantized PyTorch MiniLM (15 mJ)
  - T4: Cloud Offload REST Client (8 mJ)
  - T5: In-Memory Embedding Cache (1 mJ)
"""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptgreen.config import CONFIG, FeasibilityConstraints
from adaptgreen.cme.simulator import DeviceState
from adaptgreen.pdu.feasibility import FeasibilityFilter
from adaptgreen.pdu.q_learning import QLearningAgent
from adaptgreen.runtime.executor import RuntimeInferenceExecutor
from adaptgreen.models.t2_pruned import StructuredPrunedMiniLMModel
from adaptgreen.models.t4_cloud import CloudOffloadClient


class FiveTiersValidationTest(unittest.TestCase):
    def setUp(self):
        self.executor = RuntimeInferenceExecutor(CONFIG)
        self.feasibility_filter = FeasibilityFilter(CONFIG.constraints)
        self.q_agent = QLearningAgent(CONFIG.q_learning, CONFIG.energy)

    def test_all_five_actions_in_q_learning(self):
        self.assertEqual(self.q_agent.ACTIONS, ["T1", "T2", "T3", "T4", "T5"])
        q_vals = self.q_agent.get_q_values("B3_T1_R2_U1_K1_M2")
        for tier in ["T1", "T2", "T3", "T4", "T5"]:
            self.assertIn(tier, q_vals)

    def test_t2_structured_pruned_model_fidelity(self):
        t2_model = self.executor.t2_model
        text = "Evaluating on-device edge AI inference runtime efficiency."
        emb, latency = t2_model.encode(text)

        self.assertEqual(emb.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=4)
        self.assertGreater(latency, 0.0)

        # Check fidelity against T1
        t1_emb, _ = self.executor.t1_model.encode(text)
        cosine_sim = float(np.dot(emb, t1_emb) / (np.linalg.norm(emb) * np.linalg.norm(t1_emb)))
        self.assertGreaterEqual(cosine_sim, 0.95, "T2 pruned fidelity must meet C2 constraint threshold")

    def test_t4_cloud_client_execution(self):
        cloud_client = CloudOffloadClient()
        text = "Cloud-assisted mobile inference offload request."
        emb, latency = cloud_client.encode(text, rssi=-60.0)

        self.assertEqual(emb.shape, (384,))
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=4)
        # Latency reflects simulated network RTT
        self.assertGreaterEqual(latency, 20.0)

    def test_executor_executes_all_five_tiers(self):
        sample_query = "Test query across all five operational tiers."
        for tier in ["T1", "T2", "T3", "T4", "T5"]:
            res = self.executor.execute(text=sample_query, tier=tier, rssi=-60.0)
            self.assertEqual(res.embedding.shape, (384,))
            self.assertGreaterEqual(res.latency_ms, 0.0)
            self.assertIn(res.energy_mj, [45.0, 30.0, 15.0, 8.0, 1.0])

    def test_feasibility_c5_network_cutoff(self):
        # RSSI < -85 dBm drops T4
        bad_network_state = DeviceState(
            battery=80.0,
            temperature=40.0,
            rssi=-95.0,
            user_activity=1,
            task_type=1,
            available_ram=2000.0,
        )
        feasible, violations = self.feasibility_filter.evaluate(bad_network_state)
        self.assertNotIn("T4", feasible)
        self.assertTrue(any("C5 Network" in v for v in violations["T4"]))

    def test_feasibility_c3_thermal_restricts_t1_and_t2(self):
        hot_state = DeviceState(
            battery=80.0,
            temperature=82.0,
            rssi=-60.0,
            user_activity=1,
            task_type=1,
            available_ram=2000.0,
        )
        feasible, violations = self.feasibility_filter.evaluate(hot_state)
        self.assertNotIn("T1", feasible)
        self.assertNotIn("T2", feasible)
        self.assertTrue(any("C3 Thermal" in v for v in violations["T1"]))
        self.assertTrue(any("C3 Thermal" in v for v in violations["T2"]))


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
