from __future__ import annotations

import numpy as np
import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.geometry.preprocessing import (
    l2_normalize_vector,
    standardize_and_l2_normalize,
    try_l2_normalize_vector,
)


def test_standardize_and_l2_normalize_zero_variance_columns_and_zero_rows_require_explicit_policies() -> None:
    matrix = np.asarray(
        [
            [1.0, 5.0, 0.0],
            [1.0, 5.0, 0.0],
            [1.0, 5.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ContractViolationError, match="zero-variance columns"):
        standardize_and_l2_normalize(matrix)

    normalized = standardize_and_l2_normalize(
        matrix,
        zero_variance_policy="drop_or_zero",
        zero_row_policy="zero",
    )

    assert normalized.shape == (3, 3)
    assert np.allclose(normalized, 0.0)


def test_standardize_and_l2_normalize_rejects_nonfinite_values_by_default() -> None:
    matrix = np.asarray(
        [
            [np.nan, 1.0, 0.0],
            [np.inf, -np.inf, 3.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ContractViolationError, match="non-finite values"):
        standardize_and_l2_normalize(matrix)


def test_standardize_and_l2_normalize_can_coerce_nonfinite_when_explicit() -> None:
    matrix = np.asarray(
        [
            [np.nan, 1.0, 0.0],
            [np.inf, -np.inf, 3.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    normalized = standardize_and_l2_normalize(
        matrix,
        nonfinite_policy="coerce",
        zero_variance_policy="drop_or_zero",
        zero_row_policy="zero",
    )

    assert np.isfinite(normalized).all()
    assert np.all(np.linalg.norm(normalized, axis=1) <= 1.0 + 1e-6)


def test_standardize_and_l2_normalize_can_fail_on_zero_variance_columns() -> None:
    matrix = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)

    with pytest.raises(ContractViolationError, match="zero-variance columns"):
        standardize_and_l2_normalize(matrix)


def test_standardize_and_l2_normalize_rejects_zero_norm_rows_by_default() -> None:
    matrix = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    with pytest.raises(ContractViolationError, match="zero-norm rows"):
        standardize_and_l2_normalize(
            matrix,
            center=False,
            scale=False,
        )


def test_try_l2_normalize_vector_returns_none_for_zero_vector() -> None:
    assert try_l2_normalize_vector(np.zeros(4, dtype=np.float32)) is None


def test_l2_normalize_vector_rejects_zero_vector() -> None:
    with pytest.raises(ContractViolationError, match="vector norm is zero"):
        l2_normalize_vector(np.zeros(3, dtype=np.float32))
