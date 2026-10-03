import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptgreen.evaluation.metrics import aggregate_telemetry


class TelemetryAggregationTest(unittest.TestCase):
    def test_uses_actual_execution_and_cache_telemetry(self):
        records = [
            {
                "selected_tier": "T3",
                "executed_tier": "T3",
                "latency_ms": 2.0,
                "energy_mj": 15.0,
                "cache_hit": False,
                "cache_miss": False,
            },
            {
                "selected_tier": "T5",
                "executed_tier": "T5",
                "latency_ms": 0.1,
                "energy_mj": 1.0,
                "cache_hit": True,
                "cache_miss": False,
            },
            {
                "selected_tier": "T5",
                "executed_tier": "T3",
                "latency_ms": 2.0,
                "energy_mj": 15.0,
                "cache_hit": False,
                "cache_miss": True,
            },
        ]

        summary = aggregate_telemetry(records)

        self.assertEqual(summary["total_energy_mj"], 31.0)
        self.assertEqual(summary["tier_usage"], {"T1": 0, "T3": 2, "T5": 1})
        self.assertEqual(summary["selected_tier_usage"], {"T3": 1, "T5": 2})
        self.assertEqual(summary["cache_hit_count"], 1)
        self.assertEqual(summary["cache_miss_count"], 1)
        self.assertEqual(summary["cache_hit_rate_pct"], 33.33)

    def test_tier_usage_includes_zero_count_supported_tiers(self):
        def records_for(tier, count):
            return [
                {
                    "selected_tier": tier,
                    "executed_tier": tier,
                    "latency_ms": 1.0,
                    "energy_mj": 15.0,
                    "cache_hit": False,
                    "cache_miss": False,
                }
                for _ in range(count)
            ]

        foreground_summary = aggregate_telemetry(records_for("T1", 60))
        battery_summary = aggregate_telemetry(
            records_for("T3", 43) + records_for("T5", 17)
        )

        self.assertEqual(
            foreground_summary["tier_usage"],
            {"T1": 60, "T3": 0, "T5": 0},
        )
        self.assertEqual(
            battery_summary["tier_usage"],
            {"T1": 0, "T3": 43, "T5": 17},
        )


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    print("PASS: telemetry aggregation validation completed")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)