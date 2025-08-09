"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/evaluators/placeholder.py

Deterministic, side-effect-free stub for testing & dry-runs.
Gives reproducible pseudo-random floats via hashing or RNG.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import random
from typing import List, Optional

import numpy as np

from .base import Evaluator


def _stable_float(seed: str) -> float:
    """
    Convert a string → deterministic float in [-1,1].
    Useful for reproducible stub scores.
    """
    h = hashlib.sha256(seed.encode()).hexdigest()[:16]
    # map 0..2^64 → -1..1
    return (int(h, 16) / 2**63) * 2 - 1  # type: ignore[return-value]


class PlaceholderEvaluator(Evaluator):
    """
    A toy evaluator that:
      - log_likelihood: returns stable hashed floats
      - log_likelihood_ratio: subtracts reference's stub-score
      - embedding_distance: returns negative L2 distance in a fixed pseudo-embedding space
    """

    def __init__(self, **params):
        super().__init__(**params)
        # fixed RNG for embedding reproducibility
        self._rng = random.Random(42)

    def _embed(self, seq: str, dim: int = 8) -> np.ndarray:
        """Produce a deterministic pseudo-embedding vector for each sequence."""
        self._rng.seed(seq)
        return np.array([self._rng.random() for _ in range(dim)], dtype=np.float32)

    def score(
        self,
        sequences: List[str],
        *,
        metric: str,
        ref_sequence: Optional[str] = None,
        ref_embedding: Optional[np.ndarray] = None,
    ) -> List[float]:
        """
        Compute scores according to `metric`:
          - "log_likelihood": hashed stub
          - "log_likelihood_ratio": stub(seq) - stub(ref_sequence)
          - "embedding_distance": -||embed(seq)-ref_embedding|| (higher=better)
        """
        if metric == "log_likelihood":
            return [_stable_float(s) for s in sequences]

        if metric == "log_likelihood_ratio":
            if ref_sequence is None:
                raise ValueError("ref_sequence required for log_likelihood_ratio")
            base = _stable_float(ref_sequence)
            return [_stable_float(s) - base for s in sequences]

        if metric == "embedding_distance":
            # If a reference embedding wasn't passed, compute from ref_sequence.
            if ref_embedding is None:
                if ref_sequence is None:
                    raise ValueError("ref_sequence required for embedding_distance")
                ref_embedding = self._embed(ref_sequence)
            ref_embedding = np.asarray(ref_embedding, dtype=np.float32)
            return [
                -float(np.linalg.norm(self._embed(s) - ref_embedding))
                for s in sequences
            ]

        raise ValueError(f"Unsupported metric: {metric}")
