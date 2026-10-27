"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/stage_a/stage_a_types.py

Stage-A data types for PWM sampling.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class FimoCandidate:
    seq: str
    score: float
    start: int
    stop: int
    strand: str
    matched_sequence: Optional[str] = None


@dataclass(frozen=True)
class SelectionMeta:
    selection_rank: int
    selection_utility: float | None
    selection_score_norm: float | None = None
    nearest_selected_similarity: float | None = None
    nearest_selected_distance: float | None = None
    nearest_selected_distance_norm: float | None = None

    def __post_init__(self) -> None:
        if int(self.selection_rank) <= 0:
            raise ValueError("Selection rank must be >= 1.")
        if self.selection_score_norm is not None:
            value = float(self.selection_score_norm)
            if value < -1e-6 or value > 1.0 + 1e-6:
                raise ValueError("Selection score norm must be in [0, 1].")
        if self.nearest_selected_distance_norm is not None:
            value = float(self.nearest_selected_distance_norm)
            if value < -1e-6 or value > 1.0 + 1e-6:
                raise ValueError("Selection nearest distance norm must be in [0, 1].")

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_rank": int(self.selection_rank),
            "selection_utility": self.selection_utility,
            "selection_score_norm": self.selection_score_norm,
            "nearest_selected_similarity": self.nearest_selected_similarity,
            "nearest_selected_distance": self.nearest_selected_distance,
            "nearest_selected_distance_norm": self.nearest_selected_distance_norm,
        }


@dataclass(frozen=True)
class PWMMotif:
    motif_id: str
    matrix: List[dict[str, float]]
    background: dict[str, float]
    log_odds: Optional[List[dict[str, float]]] = None
