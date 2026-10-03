"""
Runtime Inference Executor Module.

Dispatches inference workloads to the active tier selected by the PDU:
  - T1: Full FP32 PyTorch Sentence Transformer.
  - T3: Dynamic INT8 Quantized PyTorch Model.
  - T5: In-Memory Embedding Cache.

Measures actual wall-clock execution latency via high-resolution timing
and assigns configured simulated energy consumption (T1: 45 mJ, T3: 15 mJ, T5: 1 mJ).
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import logging
import numpy as np

from ..config import SimulationConfig, CONFIG
from ..models.t1_fp32 import FP32MiniLMModel
from ..models.t2_pruned import StructuredPrunedMiniLMModel
from ..models.t3_int8 import INT8MiniLMModel
from ..models.t4_cloud import CloudOffloadClient, NetworkUnavailableError
from ..models.t5_cache import EmbeddingCache

logger = logging.getLogger("AdaptGreen.RuntimeExecutor")


@dataclass
class InferenceResult:
    """Encapsulates the execution output from an inference tier."""
    tier: str
    embedding: np.ndarray
    latency_ms: float           # Measured wall-clock latency in milliseconds
    energy_mj: float            # SIMULATED energy according to AdaptGreen paper profile
    cache_hit: bool
    cache_miss: bool
    is_simulated_energy: bool = True  # Explicit disclaimer label


class RuntimeInferenceExecutor:
    """
    Runtime Inference Executor.
    Maintains active model tier instances and manages workload dispatch.
    """

    def __init__(self, config: SimulationConfig = CONFIG):
        self.cfg = config
        self.energy_profile = config.energy

        logger.info("Initializing Runtime Inference Executor tiers (5-tier library)...")
        # 1. Initialize FP32 Tier (T1)
        self.t1_model = FP32MiniLMModel(
            model_name=config.model.model_name,
            max_seq_length=config.model.max_seq_length
        )

        # 2. Initialize Structured Pruning Tier (T2)
        self.t2_model = StructuredPrunedMiniLMModel(
            fp32_model=self.t1_model.get_underlying_model(),
            tokenizer=self.t1_model.tokenizer,
            prune_ratio=0.02,
            max_seq_length=config.model.max_seq_length
        )

        # 3. Initialize Dynamic INT8 Tier (T3)
        self.t3_model = INT8MiniLMModel(
            fp32_model=self.t1_model.get_underlying_model(),
            tokenizer=self.t1_model.tokenizer,
            max_seq_length=config.model.max_seq_length
        )

        # 4. Initialize Cloud Offload REST Client (T4)
        self.t4_model = CloudOffloadClient(
            server_model=self.t1_model,
            min_rssi_dbm=config.constraints.network_rssi_min
        )

        # 5. Initialize In-Memory Vector Cache (T5)
        self.t5_cache = EmbeddingCache()

    def warmup(self) -> None:
        """Warms up PyTorch execution graphs to prevent cold-start latency artifacts."""
        warmup_text = "AdaptGreen runtime initialization and warmup sequence."
        logger.info("Performing warmup pass across T1, T2, and T3 models...")
        vec_t1, _ = self.t1_model.encode(warmup_text)
        vec_t2, _ = self.t2_model.encode(warmup_text)
        vec_t3, _ = self.t3_model.encode(warmup_text)
        self.t5_cache.put(warmup_text, vec_t1)
        logger.info("Warmup complete.")

    def pre_populate_cache(self, corpus: List[str]) -> None:
        """
        Pre-populates the T5 cache with embeddings for recurring query patterns
        to simulate realistic mobile assistant cache hit dynamics.
        """
        logger.info("Pre-populating T5 cache with %d reference queries...", len(corpus))
        for text in corpus:
            vec, _ = self.t1_model.encode(text)
            self.t5_cache.put(text, vec)

    def execute(self, text: str, tier: str, rssi: float = -50.0) -> InferenceResult:
        """
        Executes inference for the given text using the specified tier.

        Args:
            text: Query text string.
            tier: Selected tier identifier ('T1', 'T2', 'T3', 'T4', or 'T5').
            rssi: Network signal strength for cloud offloading (T4).

        Returns:
            InferenceResult containing computed embedding, measured latency,
            and simulated energy.
        """
        normalized_tier = tier.strip().upper()

        if normalized_tier == "T1":
            # T1: Full FP32 PyTorch inference
            embedding, latency_ms = self.t1_model.encode(text)
            energy_mj = self.energy_profile.T1_MJ
            cache_hit = False
            cache_miss = False
            self.t5_cache.put(text, embedding)

        elif normalized_tier == "T2":
            # T2: Structured Pruning inference
            embedding, latency_ms = self.t2_model.encode(text)
            energy_mj = self.energy_profile.T2_MJ
            cache_hit = False
            cache_miss = False
            self.t5_cache.put(text, embedding)

        elif normalized_tier == "T3":
            # T3: Dynamic INT8 quantized inference
            embedding, latency_ms = self.t3_model.encode(text)
            energy_mj = self.energy_profile.T3_MJ
            cache_hit = False
            cache_miss = False
            self.t5_cache.put(text, embedding)

        elif normalized_tier == "T4":
            # T4: Cloud Offload REST API
            try:
                embedding, latency_ms = self.t4_model.encode(text, rssi=rssi)
                energy_mj = self.energy_profile.T4_MJ
                cache_hit = False
                cache_miss = False
                self.t5_cache.put(text, embedding)
            except NetworkUnavailableError as e:
                # Network failed; fallback to on-device T3
                logger.warning("Cloud offload failed (%s). Falling back to on-device T3.", e)
                embedding, latency_ms = self.t3_model.encode(text)
                normalized_tier = "T3"
                energy_mj = self.energy_profile.T3_MJ
                cache_hit = False
                cache_miss = False

        elif normalized_tier == "T5":
            # T5: In-memory cache lookup
            cached_embedding, latency_ms, is_hit = self.t5_cache.get(text)

            if is_hit and cached_embedding is not None:
                embedding = cached_embedding
                energy_mj = self.energy_profile.T5_MJ
                cache_hit = True
                cache_miss = False
            else:
                # T5 miss fallback to T3 execution
                embedding, comp_latency = self.t3_model.encode(text)
                latency_ms += comp_latency
                normalized_tier = "T3"
                energy_mj = self.energy_profile.T3_MJ
                cache_hit = False
                cache_miss = True
                self.t5_cache.put(text, embedding)

        else:
            raise ValueError(f"Unknown tier: '{tier}'. Supported: 'T1', 'T2', 'T3', 'T4', 'T5'")

        return InferenceResult(
            tier=normalized_tier,
            embedding=embedding,
            latency_ms=round(latency_ms, 2),
            energy_mj=round(energy_mj, 2),
            cache_hit=cache_hit,
            cache_miss=cache_miss,
            is_simulated_energy=True
        )
