"""
Unit tests for 2-Node Lumped RC Thermal Model, NDCG@5, and MRR metrics.
"""

import unittest
import numpy as np

from adaptgreen.cme.simulator import LumpedRCThermalModel
from adaptgreen.evaluation.metrics import (
    compute_mrr,
    compute_ndcg_at_k,
    evaluate_recommendation_accuracy,
)


class RCThermalModelTest(unittest.TestCase):
    """Verifies the physical 2-node lumped RC thermal circuit model."""

    def test_rc_thermal_heating_under_load(self):
        """Tests that silicon die temperature increases under active computing load."""
        model = LumpedRCThermalModel(t_ambient=25.0, initial_t_die=35.0, initial_t_skin=28.0)
        # Apply T1 heavy active power (4.8W) for 10 seconds
        for _ in range(10):
            t_die, t_skin = model.step_tier("T1", delta_time_s=1.0)

        self.assertGreater(t_die, 35.0, "Die temperature should rise under T1 load")
        self.assertGreater(t_skin, 28.0, "Skin temperature should absorb heat from die")
        self.assertGreater(t_die, t_skin, "Silicon die should be hotter than external skin")

    def test_rc_thermal_exponential_cooling_decay(self):
        """Tests that die temperature decays exponentially toward ambient when idling."""
        model = LumpedRCThermalModel(t_ambient=25.0, initial_t_die=82.0, initial_t_skin=42.0)
        
        # Step idle for 30 seconds
        temps = []
        for _ in range(30):
            t_die, _ = model.step_tier("IDLE", delta_time_s=1.0)
            temps.append(t_die)

        # Temperature should strictly decrease
        self.assertLess(temps[-1], 82.0, "Die must cool down when idle")
        self.assertLess(temps[-1], temps[0], "Die must cool progressively over time")

    def test_rc_thermal_trajectory_simulation(self):
        """Tests full trajectory simulation across a sequence of mixed tiers."""
        model = LumpedRCThermalModel()
        tier_seq = ["T1", "T1", "T3", "T5", "T5"]
        traj = model.simulate_trajectory(tier_seq, dt_per_step=1.0)
        self.assertEqual(len(traj), 5)
        self.assertIn("temperature_die_c", traj[0])
        self.assertIn("temperature_skin_c", traj[0])


class IRMetricsTest(unittest.TestCase):
    """Verifies NDCG@5 and MRR information retrieval metrics."""

    def test_mrr_perfect_match(self):
        """Rank 1 match should yield MRR = 1.0."""
        ref = [10, 20, 30, 40, 50]
        eval_q = [10, 25, 35, 45, 55]
        mrr = compute_mrr(eval_q, ref, k=5)
        self.assertEqual(mrr, 1.0)

    def test_mrr_second_rank_match(self):
        """Rank 2 match should yield MRR = 0.5."""
        ref = [10, 20, 30, 40, 50]
        eval_q = [99, 10, 35, 45, 55]
        mrr = compute_mrr(eval_q, ref, k=5)
        self.assertEqual(mrr, 0.5)

    def test_mrr_no_match(self):
        """No match in top-k should yield MRR = 0.0."""
        ref = [10, 20, 30, 40, 50]
        eval_q = [1, 2, 3, 4, 5]
        mrr = compute_mrr(eval_q, ref, k=5)
        self.assertEqual(mrr, 0.0)

    def test_ndcg_perfect_match(self):
        """Identical ranking should yield NDCG@5 = 1.0."""
        ref = [10, 20, 30, 40, 50]
        ndcg = compute_ndcg_at_k(ref, ref, k=5)
        self.assertAlmostEqual(ndcg, 1.0, places=4)

    def test_ndcg_partial_order_penalized(self):
        """Partially degraded ranking should have 0 < NDCG < 1.0."""
        ref = [10, 20, 30, 40, 50]
        permuted = [20, 10, 40, 30, 50]
        ndcg = compute_ndcg_at_k(permuted, ref, k=5)
        self.assertGreater(ndcg, 0.8)
        self.assertLess(ndcg, 1.0)

    def test_ndcg_completely_disjoint(self):
        """Disjoint ranking should yield NDCG = 0.0."""
        ref = [10, 20, 30, 40, 50]
        disjoint = [1, 2, 3, 4, 5]
        ndcg = compute_ndcg_at_k(disjoint, ref, k=5)
        self.assertEqual(ndcg, 0.0)

    def test_evaluate_recommendation_accuracy_includes_all_metrics(self):
        """Batch evaluation returns overlap, ndcg, and mrr."""
        q1 = np.array([1.0, 0.0, 0.0])
        q2 = np.array([0.9, 0.1, 0.0])
        corpus = np.array([
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        results = evaluate_recommendation_accuracy([q2], [q1], corpus, k=3)
        self.assertIn("mean_top_k_overlap", results)
        self.assertIn("mean_ndcg_at_k", results)
        self.assertIn("mean_mrr", results)
        self.assertGreaterEqual(results["mean_ndcg_at_k"], 0.8)
        self.assertGreaterEqual(results["mean_mrr"], 0.5)


if __name__ == "__main__":
    unittest.main()

