"""
AdaptGreen Model Tiers Package (5-Tier Library).

Contains:
- Tier 1 (T1): Full FP32 PyTorch Sentence Transformer.
- Tier 2 (T2): Structured Magnitude Pruned Model.
- Tier 3 (T3): Dynamic INT8 Quantized PyTorch Model.
- Tier 4 (T4): Cloud Offload REST Client.
- Tier 5 (T5): In-Memory Embedding Cache.
"""

from .t1_fp32 import FP32MiniLMModel
from .t2_pruned import StructuredPrunedMiniLMModel
from .t3_int8 import INT8MiniLMModel
from .t4_cloud import CloudOffloadClient
from .t5_cache import EmbeddingCache

__all__ = [
    "FP32MiniLMModel",
    "StructuredPrunedMiniLMModel",
    "INT8MiniLMModel",
    "CloudOffloadClient",
    "EmbeddingCache",
]
