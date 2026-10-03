import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptgreen.cme.simulator import ContextMonitoringEngine, DeviceState
from adaptgreen.pdu.policy import PolicyDecisionUnit


class NoEpochTierForcingTest(unittest.TestCase):
    def test_same_context_has_same_activity_regardless_of_epoch(self):
        first_engine = ContextMonitoringEngine(seed=42)
        second_engine = ContextMonitoringEngine(seed=42)

        first_state = first_engine._generate_battery_critical(epoch=42, progress=0.8)
        second_state = second_engine._generate_battery_critical(epoch=43, progress=0.8)

        self.assertEqual(first_state.to_dict(), second_state.to_dict())

    def test_same_policy_state_cannot_switch_tier_from_epoch_alone(self):
        state = DeviceState(30.0, 79.0, -80.0, 0, 2, 700.0)
        first_pdu = self._configured_pdu()
        second_pdu = self._configured_pdu()

        first_decision = first_pdu.decide(state, epoch=42)
        second_decision = second_pdu.decide(state, epoch=43)

        self.assertEqual(first_decision.selected_tier, second_decision.selected_tier)
        self.assertEqual(first_decision.recommended_tier, second_decision.recommended_tier)

    @staticmethod
    def _configured_pdu():
        pdu = PolicyDecisionUnit.__new__(PolicyDecisionUnit)
        pdu.current_tier = "T3"
        pdu.cooldown_remaining = 0
        pdu.cooldown_period = 4
        pdu.allow_emergency = False
        pdu.switch_history = []
        pdu.feasibility_filter = type(
            "FixedFeasibility",
            (),
            {
                "evaluate": lambda self, state: (["T3", "T5"], {"T1": [], "T3": [], "T5": []}),
                "is_catastrophic_state": lambda self, state: False,
            },
        )()
        pdu.q_agent = type(
            "FixedPolicy",
            (),
            {"select_action": lambda self, state, feasible_tiers, epsilon=0.0: "T5"},
        )()
        return pdu


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    print("PASS: epoch-based tier forcing regression validation completed")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)