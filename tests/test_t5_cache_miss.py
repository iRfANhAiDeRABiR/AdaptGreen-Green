import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptgreen.config import CONFIG
from adaptgreen.models.t5_cache import EmbeddingCache
from adaptgreen.runtime.executor import RuntimeInferenceExecutor


class DeterministicT3Model:
    def encode(self, text):
        return np.array([0.25, 0.5, 0.75], dtype=np.float32), 2.0


class DeterministicT1Model:
    def encode(self, text):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), 3.0


class T5CacheMissValidationTest(unittest.TestCase):
    def test_t5_miss_falls_back_then_hits(self):
        executor = RuntimeInferenceExecutor.__new__(RuntimeInferenceExecutor)
        executor.energy_profile = CONFIG.energy
        executor.t1_model = DeterministicT1Model()
        executor.t3_model = DeterministicT3Model()
        executor.t5_cache = EmbeddingCache()

        selected_tier = "T5"
        request = "deterministic uncached request"

        self.assertFalse(executor.t5_cache.contains(request))
        miss_result = executor.execute(request, selected_tier)

        self.assertEqual(selected_tier, "T5")
        self.assertFalse(miss_result.cache_hit)
        self.assertTrue(miss_result.cache_miss)
        self.assertNotEqual(miss_result.tier, "T5")
        self.assertEqual(miss_result.tier, "T3")
        self.assertEqual(miss_result.energy_mj, CONFIG.energy.T3_MJ)
        self.assertTrue(executor.t5_cache.contains(request))

        hit_result = executor.execute(request, selected_tier)

        self.assertTrue(hit_result.cache_hit)
        self.assertFalse(hit_result.cache_miss)
        self.assertEqual(hit_result.tier, "T5")
        self.assertEqual(hit_result.energy_mj, CONFIG.energy.T5_MJ)


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    print("PASS: T5 cache-miss validation completed")
    raise SystemExit(0 if result.result.wasSuccessful() else 1)