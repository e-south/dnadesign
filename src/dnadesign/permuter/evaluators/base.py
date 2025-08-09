"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/evaluators/base.py

Abstract base class for sequence-scoring modules.
Defines the interface that all concrete evaluators must implement.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import abc
from typing import List, Optional

import numpy as np


class Evaluator(abc.ABC):
    """
    Abstract API for sequence scorers.  Subclasses may cache heavy resources
    (models, tokenizers, etc.) in __init__.
    """

    def __init__(self, **params) -> None:
        """
        params: free-form dict of evaluator-specific parameters.
        """
        self.params = params

    @abc.abstractmethod
    def score(
        self,
        sequences: List[str],
        *,
        metric: str,
        ref_sequence: Optional[str] = None,
        ref_embedding: Optional[np.ndarray] = None,
    ) -> List[float]:
        """
        Score a batch of sequences.

        Args:
          sequences: list of strings to score
          metric: which scoring mode
          ref_sequence: for ratio/distance metrics
          ref_embedding: optional precomputed embedding for distance metrics

        Returns:
          list of floats (same order as `sequences`), where higher is better.
        """
        ...
