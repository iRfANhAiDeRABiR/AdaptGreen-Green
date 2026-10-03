"""
Tier 3 (T3) — Dynamic INT8 Quantized Inference Model.

Performs dynamic 8-bit integer (INT8) quantized inference on CPU.
Quantizes all linear projection and feed-forward layers within the
transformer encoder sub-modules of `all-MiniLM-L6-v2`.
Preserves high semantic fidelity while reducing CPU compute and energy demand.
"""

import copy
import time
import logging
from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

logger = logging.getLogger("AdaptGreen.T3_INT8")


class INT8MiniLMModel:
    """
    Tier 3 (T3) Runtime Inference Model.
    Dynamically quantized model targeting torch.nn.Linear layers with torch.qint8 weights.
    
    Quantized Layers Documented:
      - 6 Transformer Encoder Layers:
          - BertSelfAttention: query, key, value (384 -> 384)
          - BertSelfOutput: dense (384 -> 384)
          - BertIntermediate: dense (384 -> 1536)
          - BertOutput: dense (1536 -> 384)
      - BertPooler: dense (384 -> 384)
    Total: 37 linear layers dynamically quantized to 8-bit integer precision.
    """

    def __init__(self, fp32_model: nn.Module, tokenizer: AutoTokenizer, max_seq_length: int = 128):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = torch.device("cpu")

        logger.info("Applying Dynamic INT8 Quantization to PyTorch MiniLM model...")
        
        # Determine available dynamic quantization API
        try:
            import torch.ao.quantization as quantization_api
            logger.info("Using modern torch.ao.quantization API")
        except ImportError:
            import torch.quantization as quantization_api
            logger.info("Using legacy torch.quantization API")

        # Deep-copy FP32 model to avoid mutating the T1 reference
        model_copy = copy.deepcopy(fp32_model).to("cpu")
        model_copy.eval()

        # Quantize all nn.Linear layers to torch.qint8 dynamically
        self.quantized_model = quantization_api.quantize_dynamic(
            model_copy,
            {nn.Linear},
            dtype=torch.qint8
        )
        self.quantized_model.eval()

        # Document and audit quantized modules
        self.quantized_layer_names: List[str] = []
        for name, module in self.quantized_model.named_modules():
            if "quantized" in module.__class__.__name__.lower() or "dynamic" in module.__class__.__name__.lower():
                self.quantized_layer_names.append(f"{name} ({module.__class__.__name__})")

        logger.info(
            "Dynamic INT8 quantization completed successfully: %d linear layers converted.",
            len(self.quantized_layer_names)
        )

    @staticmethod
    def mean_pooling(model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean pooling weighted by attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, text: str) -> Tuple[np.ndarray, float]:
        """
        Executes real INT8 forward pass on input text.

        Args:
            text: Input query string.

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
            model_output = self.quantized_model(**encoded_input)
            sentence_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        embedding_vec = sentence_embeddings.cpu().squeeze(0).numpy()

        return embedding_vec, latency_ms

    def get_quantization_summary(self) -> Dict[str, object]:
        """Returns structured metadata about quantized architecture."""
        return {
            "num_quantized_layers": len(self.quantized_layer_names),
            "target_layer_type": "torch.nn.Linear",
            "quantized_dtype": "torch.qint8",
            "sample_quantized_modules": self.quantized_layer_names[:5]
        }
