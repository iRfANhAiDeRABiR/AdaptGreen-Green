"""
AdaptGreen Evaluation Package.
"""

from .metrics import compute_cosine_similarity, evaluate_embedding_fidelity, compute_energy_saving_pct
from .baselines import StaticBaselineExecutor
from .simulator import SimulationRunner

__all__ = [
    "compute_cosine_similarity",
    "evaluate_embedding_fidelity",
    "compute_energy_saving_pct",
    "StaticBaselineExecutor",
    "SimulationRunner",
]
