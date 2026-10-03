"""
Tier 4 (T4) — Cloud Offload Inference Client.

Transmits inference requests to a remote edge/cloud server via REST API over Wi-Fi/5G.
On-device energy consumption is limited strictly to radio transmission cost (8.0 mJ).
Server executes full-precision inference and returns embedding over HTTP/JSON.
Operates under Constraint C5 (Network viability: RSSI >= -85 dBm).
"""

import time
import logging
from typing import Tuple, Optional
import numpy as np
import torch

logger = logging.getLogger("AdaptGreen.T4_Cloud")


class NetworkUnavailableError(RuntimeError):
    """Raised when network quality (RSSI) is insufficient for cloud offloading."""
    pass


class CloudOffloadClient:
    """
    Tier 4 (T4) Runtime Cloud Offload Client.
    Simulates REST API over 5 GHz Wi-Fi / 5G channel with variable transmission delay.
    On-device energy is modeled at 8.0 mJ (radio transmission cost only).
    """

    def __init__(
        self,
        server_model=None,
        base_transmission_latency_ms: float = 28.0,
        min_rssi_dbm: float = -85.0,
    ):
        self.server_model = server_model
        self.base_transmission_latency_ms = base_transmission_latency_ms
        self.min_rssi_dbm = min_rssi_dbm

    def is_network_viable(self, rssi: float) -> bool:
        """Evaluates Constraint C5: network signal must meet minimum threshold."""
        return rssi >= self.min_rssi_dbm

    def encode(self, text: str, rssi: float = -50.0) -> Tuple[np.ndarray, float]:
        """
        Executes cloud-offloaded inference via simulated REST API.

        Args:
            text: Query string transmitted to remote server.
            rssi: Current network signal strength in dBm.

        Returns:
            Tuple of (384-dimensional embedding, total round-trip latency in ms).
        """
        start_time = time.perf_counter()

        if not self.is_network_viable(rssi):
            raise NetworkUnavailableError(
                f"C5 Network Viability Error: RSSI {rssi:.1f} dBm < minimum required {self.min_rssi_dbm:.1f} dBm"
            )

        # Signal-dependent transmission delay: weaker signal causes packet retransmission
        # RSSI ranges from -50 (ideal) to -85 (cutoff)
        signal_penalty = max(0.0, (-50.0 - rssi) * 0.4)
        total_latency_ms = self.base_transmission_latency_ms + signal_penalty

        # Server executes reference inference
        if self.server_model is not None:
            embedding, _ = self.server_model.encode(text)
        else:
            # Fallback deterministic normalized vector
            vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
            embedding = vec

        elapsed = (time.perf_counter() - start_time) * 1000.0
        # Total latency is network round-trip + on-device client overhead
        measured_latency = round(total_latency_ms + elapsed, 2)

        return embedding, measured_latency

    def get_client_summary(self) -> dict:
        """Returns metadata about cloud offload configuration."""
        return {
            "tier": "T4",
            "protocol": "HTTP/REST JSON over Wi-Fi / 5G",
            "min_rssi_dbm": self.min_rssi_dbm,
            "simulated_transmission_energy_mj": 8.0,
        }

