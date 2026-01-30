"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/adapters/sources/stage_a_types.py

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
    nearest_selected_similarity: float | None

    def __post_init__(self) -> None:
        if int(self.selection_rank) <= 0:
            raise ValueError("Selection rank must be >= 1.")

    def to_dict(self) -> dict[str, object]:
        return {
            "selection_rank": int(self.selection_rank),
            "selection_utility": self.selection_utility,
            "nearest_selected_similarity": self.nearest_selected_similarity,
        }


@dataclass(frozen=True)
class PWMMotif:
    motif_id: str
    matrix: List[dict[str, float]]
    background: dict[str, float]
    log_odds: Optional[List[dict[str, float]]] = None
