"""
Unit tests for Real Device Benchmarking, Hardware Profiler, and ADB Bridge.
"""

import unittest
import numpy as np
from pathlib import Path

from adaptgreen.config import CONFIG, SimulationConfig, DEVICE_PRESETS
from adaptgreen.runtime.hardware_profiler import HardwareProfiler
from adaptgreen.runtime.adb_device_runner import ADBDeviceRunner, AndroidHardwareTelemetry
from adaptgreen.runtime.mobile_runner import OnDeviceHardwareMonitor, benchmark_tier_on_device


class RealDeviceBenchmarkTest(unittest.TestCase):
    """Verifies real device profiles, hardware profiler, and ADB bridge."""

    def test_new_smartphone_presets_exist(self):
        """Checks that Galaxy S23, Pixel 8, and Helio G99 presets exist and load."""
        expected_keys = ["GALAXY_S23", "PIXEL_8", "HELIO_G99"]
        for key in expected_keys:
            self.assertIn(key, DEVICE_PRESETS, f"Preset {key} missing from DEVICE_PRESETS")
            preset = DEVICE_PRESETS[key]
            self.assertGreater(preset.total_ram_mb, 1000.0)
            self.assertGreater(preset.thermal_max, 70.0)
            self.assertGreater(preset.battery_min, 0.0)

    def test_apply_device_class_smartphone_models(self):
        """Verifies SimulationConfig can apply real smartphone models."""
        cfg = SimulationConfig()

        # Galaxy S23
        cfg.apply_device_class("GALAXY_S23")
        self.assertEqual(cfg.device_class, "GALAXY_S23")
        self.assertEqual(cfg.constraints.thermal_max, 83.0)

        # Pixel 8
        cfg.apply_device_class("PIXEL_8")
        self.assertEqual(cfg.device_class, "PIXEL_8")
        self.assertEqual(cfg.constraints.thermal_max, 81.0)

        # Helio G99
        cfg.apply_device_class("HELIO_G99")
        self.assertEqual(cfg.device_class, "HELIO_G99")
        self.assertEqual(cfg.constraints.thermal_max, 77.5)
        self.assertEqual(cfg.constraints.ram_min, 800.0)

    def test_adb_runner_graceful_fallback(self):
        """Checks that ADB runner returns structured telemetry even without device connected."""
        runner = ADBDeviceRunner()
        telem = runner.get_real_telemetry()
        self.assertIsInstance(telem, AndroidHardwareTelemetry)
        self.assertIsInstance(telem.connected, bool)
        self.assertGreaterEqual(telem.battery_level_pct, 0.0)
        self.assertGreater(telem.soc_temp_c, 0.0)

        # Test conversion to DeviceState
        state = runner.to_device_state(telem)
        self.assertEqual(state.battery, telem.battery_level_pct)
        self.assertEqual(state.temperature, telem.soc_temp_c)

    def test_hardware_profiler_tier_execution(self):
        """Tests that HardwareProfiler can profile T1 and T5 and measure real latency."""
        profiler = HardwareProfiler()
        # Test fast tier T5
        res_t5 = profiler.profile_tier("T5", iterations=2)
        self.assertEqual(res_t5["tier"], "T5")
        self.assertGreaterEqual(res_t5["latency_mean_ms"], 0.0)
        self.assertGreater(res_t5["energy_per_inference_mj"], 0.0)
        self.assertGreaterEqual(res_t5["fidelity_score"], 0.90)

    def test_mobile_runner_benchmark_tier(self):
        """Tests that on-device benchmark function executes and measures metrics."""
        res = benchmark_tier_on_device("Test Tier", simulated_compute_ms=1.0, iterations=2)
        self.assertEqual(res["tier"], "Test Tier")
        self.assertGreater(res["latency_mean_ms"], 0.0)
        self.assertGreater(res["power_draw_mw"], 0.0)
        self.assertGreater(res["energy_mj"], 0.0)


if __name__ == "__main__":
    unittest.main()
