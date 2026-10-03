"""
Unit tests validating academic paper dataset and Top-5 recommendation metrics:
  - 200 academic paper abstracts corpus
  - compute_top_k_indices
  - compute_top_k_overlap
  - evaluate_recommendation_accuracy
"""

import json
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adaptgreen.config import CONFIG
from adaptgreen.evaluation.metrics import (
    compute_top_k_indices,
    compute_top_k_overlap,
    evaluate_recommendation_accuracy,
)


class TopKRecommendationTest(unittest.TestCase):
    def test_paper_corpus_json_validity(self):
        papers_file = CONFIG.papers_file
        self.assertTrue(papers_file.exists(), f"Papers dataset must exist at {papers_file}")

        with open(papers_file, "r", encoding="utf-8") as f:
            papers = json.load(f)

        self.assertEqual(len(papers), 200, "Dataset must contain exactly 200 paper abstracts")
        first = papers[0]
        self.assertIn("id", first)
        self.assertIn("title", first)
        self.assertIn("category", first)
        self.assertIn("abstract", first)

    def test_compute_top_k_indices(self):
        rng = np.random.default_rng(42)
        corpus = rng.normal(0, 1, size=(50, 16))
        # Query identical to item 7 in corpus
        query = corpus[7].copy()

        top_k = compute_top_k_indices(query, corpus, k=5)
        self.assertEqual(len(top_k), 5)
        # Item 7 must be the #1 nearest neighbor (cosine similarity = 1.0)
        self.assertEqual(top_k[0], 7)

    def test_compute_top_k_overlap_identity(self):
        ref_top_5 = [3, 7, 12, 18, 25]
        ev_top_5 = [3, 7, 12, 18, 25]

        overlap = compute_top_k_overlap(ev_top_5, ref_top_5, k=5)
        self.assertEqual(overlap, 1.0)

    def test_compute_top_k_overlap_partial_and_disjoint(self):
        ref_top_5 = [1, 2, 3, 4, 5]
        ev_partial = [1, 2, 3, 4, 99]  # 4 out of 5 match
        ev_disjoint = [6, 7, 8, 9, 10] # 0 out of 5 match

        self.assertEqual(compute_top_k_overlap(ev_partial, ref_top_5, k=5), 0.8)
        self.assertEqual(compute_top_k_overlap(ev_disjoint, ref_top_5, k=5), 0.0)

    def test_evaluate_recommendation_accuracy_batch(self):
        rng = np.random.default_rng(42)
        corpus = rng.normal(0, 1, size=(100, 32))

        # Query batch
        queries_ref = [corpus[i].copy() for i in range(10)]
        # Evaluated queries slightly perturbed (simulating quantization/pruning)
        queries_ev = [q + rng.normal(0, 0.05, size=q.shape) for q in queries_ref]

        result = evaluate_recommendation_accuracy(
            evaluated_queries=queries_ev,
            reference_queries=queries_ref,
            corpus_embeddings=corpus,
            k=5,
        )

        self.assertEqual(result["k"], 5)
        self.assertEqual(result["total_evaluated"], 10)
        self.assertGreater(result["mean_top_k_overlap"], 0.70)
        self.assertLessEqual(result["mean_top_k_overlap"], 1.0)


if __name__ == "__main__":
    result = unittest.main(verbosity=2, exit=False)
    raise SystemExit(0 if result.result.wasSuccessful() else 1)

