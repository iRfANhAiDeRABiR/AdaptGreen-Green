"""
Tier 5 (T5) — In-Memory Embedding Cache.

Implements an ultra-low energy caching tier (1.0 mJ simulated energy)
that stores and retrieves previously computed sentence embeddings.
If a requested query text has been encountered before, the cached 384-dimensional
vector is returned immediately with sub-millisecond memory lookup latency.
"""

import time
import logging
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger("AdaptGreen.T5_Cache")


class EmbeddingCache:
    """
    Tier 5 (T5) In-Memory Cache.
    Key: Normalized text string.
    Value: 1D numpy ndarray embedding vector.
    """

    def __init__(self, max_capacity: int = 10000):
        self.max_capacity = max_capacity
        self.cache: Dict[str, np.ndarray] = {}
        self.hits: int = 0
        self.misses: int = 0

    @staticmethod
    def _normalize_key(text: str) -> str:
        """Normalizes text for consistent cache lookup."""
        return text.strip().lower()

    def get(self, text: str) -> Tuple[Optional[np.ndarray], float, bool]:
        """
        Retrieves embedding from cache.

        Args:
            text: Query string.

        Returns:
            Tuple of (embedding or None, lookup latency in ms, is_hit boolean).
        """
        start_time = time.perf_counter()
        key = self._normalize_key(text)

        if key in self.cache:
            self.hits += 1
            embedding = self.cache[key].copy()
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return embedding, latency_ms, True

        self.misses += 1
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return None, latency_ms, False

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Stores computed embedding vector in cache."""
        key = self._normalize_key(text)
        if len(self.cache) >= self.max_capacity:
            # Simple FIFO eviction of the oldest inserted element
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        self.cache[key] = embedding.copy()

    def contains(self, text: str) -> bool:
        """Checks if text is currently cached."""
        return self._normalize_key(text) in self.cache

    def clear(self) -> None:
        """Clears cache entries and resets statistics."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Calculates cache hit rate as a fraction between 0.0 and 1.0."""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0

    def get_stats(self) -> Dict[str, object]:
        """Returns cache telemetry stats."""
        return {
            "size": len(self.cache),
            "capacity": self.max_capacity,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_pct": round(self.hit_rate * 100.0, 2)
        }
