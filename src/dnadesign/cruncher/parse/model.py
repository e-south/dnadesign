"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/motif/model.py

Lightweight PWM container used throughout cruncher.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(slots=True, frozen=True)
class PWM:
    name: str
    matrix: np.ndarray  # position‐specific probability matrix (L×4)
    alphabet: Sequence[str] = ("A", "C", "G", "T")
    nsites: Optional[int] = None
    evalue: Optional[float] = None
    log_odds_matrix: Optional[np.ndarray] = None  # override if MEME log-odds provided

    @property
    def length(self) -> int:
        return self.matrix.shape[0]

    def information_bits(self) -> float:
        """Shannon information in bits across the motif."""
        p = self.matrix + 1e-9
        return float((2 + (p * np.log2(p)).sum(axis=1)).sum())

    def log_odds(
        self,
        background: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    ) -> np.ndarray:
        """
        Position-specific log-odds matrix.
        If the MEME log-odds block was parsed, use that; otherwise compute from self.matrix.
        """
        if self.log_odds_matrix is not None:
            return self.log_odds_matrix
        p = self.matrix + 1e-9
        bg = np.array(background, dtype=float)
        return np.log2(p / bg)
