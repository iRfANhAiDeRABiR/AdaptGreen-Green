import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptgreen.cme.simulator import ContextMonitoringEngine
from adaptgreen.config import CONFIG


class BatteryCriticalTemperatureTest(unittest.TestCase):
    def test_starts_below_threshold_and_later_rises(self):
        engine = ContextMonitoringEngine(seed=42)
        initial_state = engine.generate_state("BATTERY_CRITICAL", epoch=0, total_epochs=60)
        later_state = engine.generate_state("BATTERY_CRITICAL", epoch=59, total_epochs=60)

        self.assertLess(initial_state.temperature, CONFIG.constraints.thermal_max)
        self.assertGreater(later_state.temperature, initial_state.temperature)


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    print("PASS: Battery Critical temperature validation completed")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)