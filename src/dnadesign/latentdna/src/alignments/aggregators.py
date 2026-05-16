"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/alignments/aggregators.py

Alignment-aware row aggregation helpers for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np

from ..contracts.errors import AlignmentError, ContractViolationError


def aggregate_rows(matrix: np.ndarray, indices: list[int], *, mode: str) -> np.ndarray:
    if mode == "error":
        if len(indices) != 1:
            raise AlignmentError(f"alignment expected one row but found {len(indices)} for aggregation=error")
        return np.asarray(matrix[indices[0]], dtype=np.float32)
    if mode == "first":
        return np.asarray(matrix[indices[0]], dtype=np.float32)

    block = np.asarray(matrix[indices], dtype=np.float32)
    if mode == "mean":
        return block.mean(axis=0)
    if mode == "medoid":
        centroid = block.mean(axis=0, keepdims=True)
        distances = np.square(block - centroid).sum(axis=1)
        return block[int(np.argmin(distances))]
    raise ContractViolationError(f"unsupported aggregation mode: {mode}")
