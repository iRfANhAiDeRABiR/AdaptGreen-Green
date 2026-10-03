"""
Unit tests validating multi-device class configuration profiles:
  - Flagship: Snapdragon 8-series, 12GB RAM, 400 MB floor, 82°C thermal ceiling
  - Mid-Range: Snapdragon 6-series, 6GB RAM, 600 MB floor, 80°C thermal ceiling
  - Budget: MediaTek G-series, 4GB RAM, 800 MB floor, 78°C thermal ceiling
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptgreen.config import SimulationConfig, DEVICE_PRESETS
from adaptgreen.cme.simulator import DeviceState
from adaptgreen.pdu.feasibility import FeasibilityFilter


class DeviceClassesValidationTest(unittest.TestCase):
    def test_json_config_files_exist_and_valid(self):
        config = SimulationConfig()
        device_dir = config.configs_dir

        self.assertTrue(device_dir.exists(), f"Directory {device_dir} must exist")
        expected_files = ["flagship.json", "mid_range.json", "budget.json"]

        for filename in expected_files:
            file_path = device_dir / filename
            self.assertTrue(file_path.exists(), f"Configuration file {file_path} must exist")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("device_class", data)
            self.assertIn("constraints", data)
            self.assertIn("tier_profiles", data)

    def test_apply_device_presets(self):
        cfg = SimulationConfig()

        # Flagship profile
        cfg.apply_device_class("FLAGSHIP")
        self.assertEqual(cfg.device_class, "FLAGSHIP")
        self.assertEqual(cfg.constraints.ram_min, 400.0)
        self.assertEqual(cfg.constraints.thermal_max, 82.0)

        # Mid-range profile
        cfg.apply_device_class("MID_RANGE")
        self.assertEqual(cfg.device_class, "MID_RANGE")
        self.assertEqual(cfg.constraints.ram_min, 600.0)
        self.assertEqual(cfg.constraints.thermal_max, 80.0)

        # Budget profile
        cfg.apply_device_class("BUDGET")
        self.assertEqual(cfg.device_class, "BUDGET")
        self.assertEqual(cfg.constraints.ram_min, 800.0)
        self.assertEqual(cfg.constraints.thermal_max, 78.0)

    def test_budget_profile_stricter_ram_filtering(self):
        budget_cfg = SimulationConfig()
        budget_cfg.apply_device_class("BUDGET")
        filter_budget = FeasibilityFilter(budget_cfg.constraints)

        flagship_cfg = SimulationConfig()
        flagship_cfg.apply_device_class("FLAGSHIP")
        filter_flagship = FeasibilityFilter(flagship_cfg.constraints)

        # Moderate RAM condition: 700 MB
        state = DeviceState(
            battery=80.0,
            temperature=40.0,
            rssi=-50.0,
            user_activity=1,
            task_type=1,
            available_ram=700.0,
        )

        # Under Flagship (floor 400 MB), 700 MB is ample -> T1 is feasible
        feasible_flagship, _ = filter_flagship.evaluate(state)
        self.assertIn("T1", feasible_flagship)

        # Under Budget (floor 800 MB), 700 MB is constrained -> T1 is filtered
        feasible_budget, violations_budget = filter_budget.evaluate(state)
        self.assertNotIn("T1", feasible_budget)
        self.assertTrue(any("C6 Memory" in v for v in violations_budget["T1"]))


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
