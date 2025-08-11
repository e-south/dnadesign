"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/protocols/base.py

Protocol ABC defining the contract for all permutation protocols.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable

import numpy as np


class Protocol(ABC):
    """Abstract base class for all permutation protocols."""

    id: str  # e.g., "scan_dna"
    version: str = "1.0"

    @abstractmethod
    def validate_cfg(self, *, params: Dict) -> None:
        """Raise ValueError with a precise message if params are invalid."""

    @abstractmethod
    def generate(
        self, *, ref_entry: Dict, params: Dict, rng: np.random.Generator
    ) -> Iterable[Dict]:
        """
        Yield variant dicts (streaming). Each variant MUST include:
          - 'sequence': str (full assembled, uppercase A/C/G/T)
          - 'modifications': list[str] (first entry = compact key=value summary)
        Additional protocol-specific flat keys are allowed (namespaced).
        """
        ...
