"""
Round-stage package for OPAL execution.
"""

from __future__ import annotations

from .scoring import stage_scoring
from .training import stage_training
from .x_matrices import stage_x_matrices

__all__ = [
    "stage_scoring",
    "stage_training",
    "stage_x_matrices",
]
