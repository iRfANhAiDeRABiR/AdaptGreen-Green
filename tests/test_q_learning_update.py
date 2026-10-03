import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptgreen.config import QLearningConfig, EnergyProfile
from adaptgreen.pdu.q_learning import QLearningAgent


class QLearningUpdateTest(unittest.TestCase):
    def setUp(self):
        self.agent = QLearningAgent(
            QLearningConfig(alpha=0.5, gamma=0.9),
            EnergyProfile(),
        )

    def test_future_state_value_affects_update(self):
        updated = self.agent.update_q_value(
            current_value=2.0,
            reward=3.0,
            best_next_value=10.0,
        )

        self.assertEqual(updated, 7.0)

    def test_terminal_update_ignores_future_state_value(self):
        updated = self.agent.update_q_value(
            current_value=2.0,
            reward=3.0,
            best_next_value=10.0,
            terminal=True,
        )

        self.assertEqual(updated, 2.5)


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    print("PASS: Q-learning update validation completed")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)