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
from typing import Sequence

import numpy as np


@dataclass(slots=True, frozen=True)
class PWM:
    name: str
    matrix: np.ndarray  # shape (L,4)
    alphabet: Sequence[str] = ("A", "C", "G", "T")

    @property
    def length(self) -> int:
        return self.matrix.shape[0]

    def information_bits(self) -> float:
        """Shannon information in bits across the motif."""
        p = self.matrix + 1e-9  # avoid log(0)
        return float((2 + (p * np.log2(p)).sum(axis=1)).sum())
