import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptgreen.cme.simulator import DeviceState
from adaptgreen.config import FeasibilityConstraints
from adaptgreen.pdu.feasibility import FeasibilityFilter


class C2AccuracyConstraintTest(unittest.TestCase):
    def setUp(self):
        self.state = DeviceState(
            battery=90.0,
            temperature=40.0,
            rssi=-50.0,
            user_activity=1,
            task_type=1,
            available_ram=2000.0,
        )

    def test_tier_meeting_c2_is_feasible(self):
        constraints = FeasibilityConstraints(min_similarity_threshold=0.99)
        feasible_tiers, violations = FeasibilityFilter(constraints).evaluate(self.state)

        self.assertIn("T3", feasible_tiers)
        self.assertNotIn("C2 Accuracy", " ".join(violations["T3"]))

    def test_tier_violating_c2_is_infeasible(self):
        constraints = FeasibilityConstraints(min_similarity_threshold=0.999)
        feasible_tiers, violations = FeasibilityFilter(constraints).evaluate(self.state)

        self.assertNotIn("T3", feasible_tiers)
        self.assertIn("C2 Accuracy", " ".join(violations["T3"]))

    def test_cached_t5_remains_feasible(self):
        constraints = FeasibilityConstraints(min_similarity_threshold=0.999)
        feasible_tiers, violations = FeasibilityFilter(constraints).evaluate(self.state)

        self.assertIn("T5", feasible_tiers)
        self.assertEqual(violations["T5"], [])


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    print("PASS: C2 accuracy constraint validation completed")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)