"""Shared preprocessing helpers for cosine-based LatentDNA geometry."""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..contracts.errors import ContractViolationError

_DEFAULT_EPS = 1e-8


def standardize_and_l2_normalize(
    matrix: np.ndarray,
    *,
    center: bool = True,
    scale: bool = True,
    eps: float = _DEFAULT_EPS,
    nonfinite_policy: Literal["error", "coerce"] = "error",
    zero_variance_policy: Literal["error", "drop_or_zero"] = "error",
    zero_row_policy: Literal["error", "zero"] = "error",
) -> np.ndarray:
    """Apply the view-level standardization and row-normalization contract.

    The helper keeps the original dimensionality. Tolerant behavior for
    non-finite values, zero-variance columns, or zero-norm rows is opt-in so
    callers must choose it explicitly.
    """

    if nonfinite_policy not in {"coerce", "error"}:
        raise ContractViolationError(f"unsupported nonfinite_policy: {nonfinite_policy!r}")
    if zero_variance_policy not in {"drop_or_zero", "error"}:
        raise ContractViolationError(f"unsupported zero_variance_policy: {zero_variance_policy!r}")
    if zero_row_policy not in {"error", "zero"}:
        raise ContractViolationError(f"unsupported zero_row_policy: {zero_row_policy!r}")
    if eps <= 0:
        raise ContractViolationError("eps must be positive for standardize_and_l2_normalize")

    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise ContractViolationError("standardize_and_l2_normalize expects a 2D matrix")

    if not np.isfinite(array).all():
        if nonfinite_policy == "error":
            raise ContractViolationError("standardize_and_l2_normalize encountered non-finite values")
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    working = np.asarray(array, dtype=np.float32)
    if center:
        working = np.asarray(working - working.mean(axis=0, keepdims=True), dtype=np.float32)
    if scale:
        scales = np.asarray(working.std(axis=0, keepdims=True), dtype=np.float32)
        zero_mask = scales <= eps
        if np.any(zero_mask):
            if zero_variance_policy == "error":
                raise ContractViolationError("standardize_and_l2_normalize encountered zero-variance columns")
            scales = np.where(zero_mask, 1.0, scales)
        working = np.asarray(working / np.maximum(scales, eps), dtype=np.float32)
        if np.any(zero_mask):
            working[:, np.asarray(zero_mask[0], dtype=bool)] = 0.0

    norms = np.linalg.norm(working, axis=1, keepdims=True)
    normalized = np.asarray(working / np.maximum(norms, eps), dtype=np.float32)
    zero_rows = np.asarray((norms[:, 0] <= eps), dtype=bool)
    if np.any(zero_rows):
        if zero_row_policy == "error":
            raise ContractViolationError("standardize_and_l2_normalize encountered zero-norm rows")
        normalized[zero_rows] = 0.0
    return np.ascontiguousarray(normalized)


def l2_normalize_vector(
    vector: np.ndarray,
    *,
    eps: float = _DEFAULT_EPS,
) -> np.ndarray:
    """Return a unit-norm copy of ``vector`` or raise on a degenerate input."""

    if eps <= 0:
        raise ContractViolationError("eps must be positive for l2_normalize_vector")
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm < eps:
        raise ContractViolationError("vector norm is zero after normalization")
    return np.asarray(array / norm, dtype=np.float32)


def try_l2_normalize_vector(
    vector: np.ndarray,
    *,
    eps: float = _DEFAULT_EPS,
) -> np.ndarray | None:
    """Return a unit-norm copy of ``vector`` or ``None`` if it is degenerate."""

    if eps <= 0:
        raise ContractViolationError("eps must be positive for try_l2_normalize_vector")
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm < eps:
        return None
    return np.asarray(array / norm, dtype=np.float32)
