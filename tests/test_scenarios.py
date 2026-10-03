"""
Unit tests for all 4 AdaptGreen evaluation scenarios (ICCA 2026 Table 4):
  1. FOREGROUND_ACTIVE
  2. BACKGROUND_BATCH
  3. THERMAL_STRESS
  4. BATTERY_CRITICAL
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptgreen.cme.simulator import ContextMonitoringEngine, DeviceState
from adaptgreen.pdu.policy import PolicyDecisionUnit
from adaptgreen.config import CONFIG


class ScenarioValidationTest(unittest.TestCase):
    def setUp(self):
        self.cme = ContextMonitoringEngine(seed=42)
        self.pdu = PolicyDecisionUnit(CONFIG)

    def test_all_four_scenarios_generate_valid_states(self):
        scenarios = ["FOREGROUND_ACTIVE", "BACKGROUND_BATCH", "THERMAL_STRESS", "BATTERY_CRITICAL"]
        for sc in scenarios:
            for epoch in [0, 30, 59]:
                state = self.cme.generate_state(sc, epoch, 60)
                self.assertIsInstance(state, DeviceState)
                self.assertGreaterEqual(state.battery, 0.0)
                self.assertLessEqual(state.battery, 100.0)
                self.assertGreaterEqual(state.temperature, 20.0)
                self.assertLessEqual(state.temperature, 100.0)
                self.assertIn(state.user_activity, (0, 1))
                self.assertIn(state.task_type, (1, 2, 3))
                self.assertGreater(state.available_ram, 0.0)

    def test_background_batch_characteristics(self):
        """
        Background-Batch scenario:
        - user_activity = 0 (display off)
        - task_type = 2 (periodic refresh)
        - PDU prioritizes energy minimization (relaxes C1 latency ceiling)
        """
        self.pdu.reset()
        for epoch in range(60):
            state = self.cme.generate_state("BACKGROUND_BATCH", epoch, 60)
            self.assertEqual(state.user_activity, 0)
            self.assertEqual(state.task_type, 2)
            self.assertLess(state.temperature, CONFIG.constraints.thermal_max)

        # In background batch, PDU should select energy-minimal tiers (T3 or T5, never T1)
        self.pdu.reset()
        self.cme.set_seed(42)
        selected_tiers = []
        for epoch in range(60):
            state = self.cme.generate_state("BACKGROUND_BATCH", epoch, 60)
            decision = self.pdu.decide(state, epoch)
            selected_tiers.append(decision.selected_tier)

        self.assertNotIn("T1", selected_tiers, "T1 (high energy) should not be selected during background batch")
        self.assertTrue(all(t in ("T3", "T5") for t in selected_tiers))

    def test_thermal_stress_characteristics(self):
        """
        Thermal-Stress scenario:
        - Starts below 80°C C3 thermal threshold
        - Climbs under sustained compute to reach 83°C
        - C3 triggers and forces downgrade
        """
        initial_state = self.cme.generate_state("THERMAL_STRESS", epoch=0, total_epochs=60)
        final_state = self.cme.generate_state("THERMAL_STRESS", epoch=59, total_epochs=60)

        self.assertLess(initial_state.temperature, CONFIG.constraints.thermal_max)
        self.assertGreaterEqual(final_state.temperature, 82.5)  # Reaches ~83°C
        self.assertEqual(initial_state.user_activity, 1)
        self.assertEqual(final_state.user_activity, 1)

        # Run PDU over thermal stress and verify it downgrades when crossing 80°C
        self.pdu.reset()
        self.cme.set_seed(42)
        decisions = []
        for epoch in range(60):
            state = self.cme.generate_state("THERMAL_STRESS", epoch, 60)
            decision = self.pdu.decide(state, epoch)
            decisions.append(decision)

        # Starts in T1
        self.assertEqual(decisions[0].selected_tier, "T1")
        # Under thermal stress, at the end it must not be T1 (C3 constraint violated)
        self.assertNotEqual(decisions[-1].selected_tier, "T1")

    def test_network_assisted_offload_transitions_t3_t4_t5(self):
        """
        Tests the 3-stage dynamic transition T3 -> T4 -> T5:
          - Epochs 0..19: On-device INT8 (T3)
          - Epochs 20..39: Cloud Offload over Wi-Fi (T4)
          - Epochs 40..59: In-Memory Cache in background (T5)
        """
        self.pdu.reset()
        self.cme.set_seed(42)
        tiers = []
        for epoch in range(60):
            state = self.cme.generate_state("NETWORK_ASSISTED_OFFLOAD", epoch, 60)
            decision = self.pdu.decide(state, epoch)
            tiers.append(decision.selected_tier)

        # Check exact phase assignments
        self.assertTrue(all(t == "T3" for t in tiers[0:20]), "Phase 1 must run T3")
        self.assertTrue(all(t == "T4" for t in tiers[20:40]), "Phase 2 must run T4")
        self.assertTrue(all(t == "T5" for t in tiers[40:60]), "Phase 3 must run T5")

        # Verify transitions recorded
        switches = [s for s in self.pdu.switch_history if s["from_tier"] is not None]
        self.assertEqual(len(switches), 2)
        self.assertEqual(switches[0]["from_tier"], "T3")
        self.assertEqual(switches[0]["to_tier"], "T4")
        self.assertEqual(switches[1]["from_tier"], "T4")
        self.assertEqual(switches[1]["to_tier"], "T5")


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    raise SystemExit(0 if result.result.wasSuccessful() else 1)

