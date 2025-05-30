"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/parse/model.py

PWM container with probability and log-odds support.

Stores a position-specific frequency matrix and optionally pre-parsed log-odds.
Provides methods to compute Shannon information and log-odds under any zero-order background.

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
    """
    Position-weight matrix (probabilities) and log-odds calculation.

    Attributes:
      name: identifier for the motif
      matrix: (Lx4) numpy array of base probabilities at each position
      alphabet: order of bases corresponding to matrix columns
      nsites: number of sequences used to build the PWM (optional)
      evalue: associated E-value (optional)
      log_odds_matrix: precomputed log-odds (optional)
    """

    name: str
    matrix: np.ndarray
    alphabet: Sequence[str] = ("A", "C", "G", "T")
    nsites: Optional[int] = None
    evalue: Optional[float] = None
    log_odds_matrix: Optional[np.ndarray] = None

    @property
    def length(self) -> int:
        return self.matrix.shape[0]

    def information_bits(self) -> float:
        """
        Shannon information content (sum of 2 + sum(p_ij log2 p_ij)).
        """
        p = self.matrix + 1e-9
        return float((2 + (p * np.log2(p)).sum(axis=1)).sum())

    def log_odds(
        self,
        background: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
    ) -> np.ndarray:
        """
        Return or compute the log-odds matrix:
          L[i,j] = log2(p[i,j] / background[j])

        Args:
          background: length-4 array of bg frequencies
        """
        if self.log_odds_matrix is not None:
            return self.log_odds_matrix
        p = self.matrix + 1e-9
        bg = np.array(background, float)
        return np.log2(p / bg)
