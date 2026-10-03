"""
Evaluation Metrics Module.

Implements rigorous semantic fidelity and energy-efficiency calculations:
  1. Cosine Similarity Fidelity: Measures embedding directional alignment relative
     to the FP32 golden reference (T1).
     NOTE ON RESEARCH INTEGRITY: In accordance with natural language processing rigor,
     cosine similarity measures embedding representation fidelity, NOT categorical
     classification accuracy.
  2. Energy Conservation Calculations: Computes percentage energy savings relative
     to Static T1 and Static T3 baselines.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def aggregate_telemetry(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregates energy, latency, tier, and cache metrics from telemetry rows."""
    if not records:
        return {
            "total_energy_mj": 0.0,
            "avg_energy_mj": 0.0,
            "avg_latency_ms": 0.0,
            "tier_usage": {},
            "selected_tier_usage": {},
            "cache_hit_count": 0,
            "cache_miss_count": 0,
            "cache_hit_rate_pct": 0.0,
        }

    executed_tier_usage: Dict[str, int] = {"T1": 0, "T3": 0, "T5": 0}
    selected_tier_usage: Dict[str, int] = {}
    cache_hit_count = 0
    cache_miss_count = 0

    for record in records:
        executed_tier = record["executed_tier"]
        selected_tier = record["selected_tier"]
        executed_tier_usage[executed_tier] = executed_tier_usage.get(executed_tier, 0) + 1
        selected_tier_usage[selected_tier] = selected_tier_usage.get(selected_tier, 0) + 1
        cache_hit_count += int(record["cache_hit"])
        cache_miss_count += int(record["cache_miss"])

    total_rows = len(records)
    return {
        "total_energy_mj": round(float(sum(record["energy_mj"] for record in records)), 2),
        "avg_energy_mj": round(float(np.mean([record["energy_mj"] for record in records])), 2),
        "avg_latency_ms": round(float(np.mean([record["latency_ms"] for record in records])), 2),
        "tier_usage": executed_tier_usage,
        "selected_tier_usage": selected_tier_usage,
        "cache_hit_count": cache_hit_count,
        "cache_miss_count": cache_miss_count,
        "cache_hit_rate_pct": round((cache_hit_count / total_rows) * 100.0, 2),
    }


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D embedding vectors:
      CosineSim(u, v) = (u . v) / (||u||_2 * ||v||_2)
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot_product = float(np.dot(vec_a, vec_b))
    similarity = dot_product / (norm_a * norm_b)
    # Clip to valid numeric domain [-1.0, 1.0]
    return float(np.clip(similarity, -1.0, 1.0))


def evaluate_embedding_fidelity(
    reference_embeddings: List[np.ndarray],
    evaluated_embeddings: List[np.ndarray],
    threshold: float = 0.95
) -> Dict[str, Any]:
    """
    Evaluates semantic fidelity of evaluated embeddings against reference FP32 embeddings.

    Args:
        reference_embeddings: List of T1 FP32 ground truth embeddings.
        evaluated_embeddings: List of embeddings produced by evaluated tier (e.g. T3 or AdaptGreen).
        threshold: Minimum acceptable cosine similarity threshold (default 0.95).

    Returns:
        Dictionary containing:
          - mean_cosine_similarity
          - min_cosine_similarity
          - max_cosine_similarity
          - pct_above_threshold (% of embeddings exceeding similarity threshold)
          - total_evaluated
          - metric_definition_note
    """
    if len(reference_embeddings) != len(evaluated_embeddings) or len(reference_embeddings) == 0:
        return {
            "mean_cosine_similarity": 0.0,
            "min_cosine_similarity": 0.0,
            "max_cosine_similarity": 0.0,
            "pct_above_threshold": 0.0,
            "total_evaluated": 0,
            "metric_definition_note": "Invalid or empty embeddings list."
        }

    similarities = [
        compute_cosine_similarity(ref, ev)
        for ref, ev in zip(reference_embeddings, evaluated_embeddings)
    ]

    sim_array = np.array(similarities, dtype=np.float64)
    above_threshold = np.sum(sim_array >= threshold)

    return {
        "mean_cosine_similarity": round(float(np.mean(sim_array)), 4),
        "min_cosine_similarity": round(float(np.min(sim_array)), 4),
        "max_cosine_similarity": round(float(np.max(sim_array)), 4),
        "pct_above_threshold": round(float((above_threshold / len(sim_array)) * 100.0), 2),
        "threshold_used": threshold,
        "total_evaluated": len(sim_array),
        "metric_definition_note": (
            "Cosine similarity measures geometric alignment in 384-dimensional latent embedding space "
            "relative to full FP32 PyTorch inference. It does NOT represent classification accuracy."
        )
    }


def compute_energy_saving_pct(baseline_energy: float, adaptgreen_energy: float) -> float:
    """
    Calculates percentage energy conservation:
      saving_pct = ((baseline_energy - adaptgreen_energy) / baseline_energy) * 100.0
    """
    if baseline_energy <= 0.0:
        return 0.0
    savings = ((baseline_energy - adaptgreen_energy) / baseline_energy) * 100.0
    return float(round(savings, 2))


def compute_top_k_indices(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int = 5,
) -> List[int]:
    """
    Computes top-k nearest neighbor document indices in corpus by cosine similarity.

    Args:
        query_embedding: 1D embedding vector of shape (D,).
        corpus_embeddings: 2D embedding matrix of shape (N, D).
        k: Number of nearest items to retrieve.

    Returns:
        List of integer indices corresponding to the top-k highest similarity documents.
    """
    q_norm = np.linalg.norm(query_embedding)
    if q_norm == 0.0 or corpus_embeddings.shape[0] == 0:
        return list(range(min(k, corpus_embeddings.shape[0])))

    corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)
    corpus_norms = np.where(corpus_norms == 0.0, 1e-12, corpus_norms)

    dot_products = np.dot(corpus_embeddings, query_embedding)
    similarities = dot_products / (corpus_norms * q_norm)

    k_actual = min(k, len(similarities))
    top_indices = np.argsort(similarities)[-k_actual:][::-1].tolist()
    return top_indices


def compute_top_k_overlap(
    evaluated_top_k: List[int],
    reference_top_k: List[int],
    k: int = 5,
) -> float:
    """
    Computes top-k ranking overlap ratio (intersection relative to reference):
      overlap = |set(evaluated_top_k) & set(reference_top_k)| / k

    Args:
        evaluated_top_k: List of retrieved indices from evaluated tier.
        reference_top_k: List of retrieved indices from T1 golden reference.
        k: Top-k cutoff (default 5).

    Returns:
        Overlap ratio in [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    intersection = set(evaluated_top_k[:k]).intersection(set(reference_top_k[:k]))
    return float(len(intersection) / float(k))


def compute_mrr(
    evaluated_top_k: List[int],
    reference_top_k: List[int],
    k: int = 5,
) -> float:
    """
    Computes Mean Reciprocal Rank (MRR) for the golden rank-1 item:
      RR = 1 / rank(reference[0]) if found in evaluated_top_k[:k] else 0.0

    Args:
        evaluated_top_k: List of retrieved indices from evaluated tier.
        reference_top_k: List of retrieved indices from T1 golden reference.
        k: Maximum rank cutoff.

    Returns:
        Reciprocal rank float in [0.0, 1.0].
    """
    if not reference_top_k or not evaluated_top_k or k <= 0:
        return 0.0

    target = reference_top_k[0]
    eval_slice = evaluated_top_k[:k]
    if target in eval_slice:
        rank = eval_slice.index(target) + 1  # 1-indexed rank
        return float(1.0 / rank)
    return 0.0


def compute_ndcg_at_k(
    evaluated_top_k: List[int],
    reference_top_k: List[int],
    k: int = 5,
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank K (NDCG@K)
    using graded relevance based on golden reference rankings.

    Relevance of item at golden rank j (0-indexed) is (k - j). Unranked items have relevance 0.

    Args:
        evaluated_top_k: List of retrieved indices from evaluated tier.
        reference_top_k: List of retrieved indices from T1 golden reference.
        k: Evaluation rank cutoff (default 5).

    Returns:
        NDCG score float in [0.0, 1.0].
    """
    if not reference_top_k or not evaluated_top_k or k <= 0:
        return 0.0

    # Build relevance map from reference ranking
    ref_slice = reference_top_k[:k]
    rel_map = {idx: float(k - j) for j, idx in enumerate(ref_slice)}

    # Compute DCG@K
    eval_slice = evaluated_top_k[:k]
    dcg = 0.0
    for i, idx in enumerate(eval_slice):
        rel = rel_map.get(idx, 0.0)
        # Standard formulation: (2^rel - 1) / log2(i + 2)
        dcg += (2.0 ** rel - 1.0) / np.log2(i + 2.0)

    # Compute Ideal DCG@K (IDCG)
    idcg = 0.0
    for i in range(len(ref_slice)):
        ideal_rel = float(k - i)
        idcg += (2.0 ** ideal_rel - 1.0) / np.log2(i + 2.0)

    if idcg <= 0.0:
        return 1.0 if dcg <= 0.0 else 0.0

    return float(np.clip(dcg / idcg, 0.0, 1.0))


def evaluate_recommendation_accuracy(
    evaluated_queries: List[np.ndarray],
    reference_queries: List[np.ndarray],
    corpus_embeddings: np.ndarray,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Evaluates downstream Top-K recommendation accuracy across all queries,
    reporting Top-K Jaccard overlap, NDCG@K, and MRR.
    """
    if not evaluated_queries or not reference_queries or len(evaluated_queries) != len(reference_queries):
        return {
            "mean_top_k_overlap": 0.0,
            "min_top_k_overlap": 0.0,
            "max_top_k_overlap": 0.0,
            "mean_ndcg_at_k": 0.0,
            "mean_mrr": 0.0,
            "k": k,
            "total_evaluated": 0,
        }

    overlaps: List[float] = []
    ndcgs: List[float] = []
    mrrs: List[float] = []

    for ev_q, ref_q in zip(evaluated_queries, reference_queries):
        ref_top_k = compute_top_k_indices(ref_q, corpus_embeddings, k=k)
        ev_top_k = compute_top_k_indices(ev_q, corpus_embeddings, k=k)
        
        overlaps.append(compute_top_k_overlap(ev_top_k, ref_top_k, k=k))
        ndcgs.append(compute_ndcg_at_k(ev_top_k, ref_top_k, k=k))
        mrrs.append(compute_mrr(ev_top_k, ref_top_k, k=k))

    overlap_arr = np.array(overlaps, dtype=np.float64)
    ndcg_arr = np.array(ndcgs, dtype=np.float64)
    mrr_arr = np.array(mrrs, dtype=np.float64)

    return {
        "mean_top_k_overlap": round(float(np.mean(overlap_arr)), 4),
        "min_top_k_overlap": round(float(np.min(overlap_arr)), 4),
        "max_top_k_overlap": round(float(np.max(overlap_arr)), 4),
        "mean_ndcg_at_k": round(float(np.mean(ndcg_arr)), 4),
        "mean_mrr": round(float(np.mean(mrr_arr)), 4),
        "k": k,
        "total_evaluated": len(overlaps),
    }
