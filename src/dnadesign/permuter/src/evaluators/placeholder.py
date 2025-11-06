"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/evaluators/placeholder.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import random
from typing import List, Optional

import numpy as np

from dnadesign.permuter.src.evaluators.base import Evaluator


def _stable_float(seed: str) -> float:
    # map 64-bit integer to [-1, 1)
    h = hashlib.sha256(seed.encode()).hexdigest()[:16]
    x = int(h, 16)
    return (x / (2**64 - 1)) * 2.0 - 1.0


class PlaceholderEvaluator(Evaluator):
    def __init__(self, **params):
        super().__init__(**params)
        self._rng = random.Random(42)

    def _embed(self, seq: str, dim: int = 8) -> np.ndarray:
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
        if metric == "log_likelihood":
            return [_stable_float(s) for s in sequences]
        if metric == "log_likelihood_ratio":
            if ref_sequence is None:
                raise ValueError("ref_sequence required for log_likelihood_ratio")
            base = _stable_float(ref_sequence)
            return [_stable_float(s) - base for s in sequences]
        if metric == "embedding_distance":
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
