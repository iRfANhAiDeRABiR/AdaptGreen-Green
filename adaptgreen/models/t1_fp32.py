"""
Tier 1 (T1) — Full FP32 Inference Model.

Performs full-precision (FP32) neural inference using the 384-dimensional
sentence transformer model `sentence-transformers/all-MiniLM-L6-v2`.
Executes real forward passes and returns normalized embedding vectors.
"""

import time
import logging
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger("AdaptGreen.T1_FP32")


class FP32MiniLMModel:
    """
    Tier 1 (T1) Runtime Inference Model.
    Runs full 32-bit floating point inference on CPU.
    Serves as the golden reference for embedding accuracy and baseline energy.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_seq_length: int = 128):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.device = torch.device("cpu")

        logger.info("Initializing Tier 1 (FP32) with model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

        # Confirm model parameters are float32
        for p in self.model.parameters():
            p.requires_grad = False

    @staticmethod
    def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies mean pooling over token embeddings weighted by attention mask,
        matching the official sentence-transformers architecture.
        """
        token_embeddings = model_output[0]  # First element contains hidden states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, text: str) -> Tuple[np.ndarray, float]:
        """
        Executes real FP32 forward pass on input text.
        
        Args:
            text: Input sentence or query string.

        Returns:
            Tuple of (normalized embedding as 1D numpy array, execution latency in milliseconds).
        """
        start_time = time.perf_counter()

        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
            # L2 normalize embeddings
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        embedding_vec = sentence_embeddings.cpu().squeeze(0).numpy()

        return embedding_vec, latency_ms

    def get_underlying_model(self) -> nn.Module:
        """Returns the underlying PyTorch nn.Module for quantization."""
        return self.model
