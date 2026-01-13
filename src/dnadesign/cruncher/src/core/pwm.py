"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/core/pwm.py

Author(s): Eric J. South
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
    source_length: Optional[int] = None
    window_start: Optional[int] = None
    window_strategy: Optional[str] = None
    window_score: Optional[float] = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != 4:
            raise ValueError("PWM.matrix must be a 2D array with shape (L, 4)")
        if matrix.shape[0] < 1:
            raise ValueError("PWM.matrix must have at least one position")
        if np.any(matrix < 0):
            raise ValueError("PWM.matrix values must be non-negative probabilities")
        row_sums = matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("PWM.matrix rows must sum to 1.0")
        object.__setattr__(self, "matrix", matrix)

        if self.log_odds_matrix is not None:
            lom = np.asarray(self.log_odds_matrix, dtype=float)
            if lom.shape != matrix.shape:
                raise ValueError("log_odds_matrix must match PWM.matrix shape")
            object.__setattr__(self, "log_odds_matrix", lom)

        if self.source_length is not None and self.source_length < self.length:
            raise ValueError("source_length must be >= PWM length")
        if self.window_start is not None:
            if self.source_length is None:
                raise ValueError("window_start requires source_length")
            if self.window_start < 0 or (self.window_start + self.length) > self.source_length:
                raise ValueError("window_start is out of bounds for source_length")

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
