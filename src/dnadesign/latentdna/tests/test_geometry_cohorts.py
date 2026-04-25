from __future__ import annotations

import numpy as np
import pytest

from dnadesign.latentdna.src.contracts.errors import ContractViolationError
from dnadesign.latentdna.src.geometry.cohorts import (
    aligned_cohort_distance_vectors,
    bootstrap_ci,
    cohort_distance_vector,
    group_indices,
    separation_ratio_from_groups,
)


def test_group_indices_respects_allowed_and_excluded_values() -> None:
    rows = [
        {"design_family": "background_only"},
        {"design_family": "ethanol"},
        {"design_family": "ethanol"},
        {"design_family": "ciprofloxacin"},
    ]

    grouped = group_indices(
        rows,
        column="design_family",
        exclude_values={"background_only"},
        allowed_values={"ethanol", "ciprofloxacin"},
    )

    assert grouped == {"ethanol": [1, 2], "ciprofloxacin": [3]}


def test_separation_ratio_from_groups_is_finite_for_separated_groups() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [-1.0, 0.0],
            [-0.8, -0.2],
        ],
        dtype=np.float32,
    )
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    groups = {"left": [0, 1], "right": [2, 3]}

    ratio = separation_ratio_from_groups(matrix, groups)

    assert np.isfinite(ratio)
    assert ratio > 1.0


def test_cohort_distance_vector_returns_upper_triangle_distances() -> None:
    matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ],
        dtype=np.float32,
    )
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    rows = [
        {"sig35_variant": "a"},
        {"sig35_variant": "a"},
        {"sig35_variant": "b"},
        {"sig35_variant": "b"},
        {"sig35_variant": "c"},
        {"sig35_variant": "c"},
    ]

    distances = cohort_distance_vector(matrix, rows, column="sig35_variant")

    assert distances.shape == (3,)
    assert np.isfinite(distances).all()
    assert (distances > 0.0).all()


def test_bootstrap_ci_tracks_finite_metric_outputs() -> None:
    values = iter([1.0, float("nan"), 2.0, 3.0, 4.0])

    ci_lower, ci_upper = bootstrap_ci(lambda: next(values), iterations=5)

    assert ci_lower is not None
    assert ci_upper is not None
    assert ci_lower <= ci_upper


def test_aligned_cohort_distance_vectors_reject_mismatched_non_degenerate_pairs() -> None:
    rows = [
        {"sig35_variant": "a"},
        {"sig35_variant": "a"},
        {"sig35_variant": "b"},
        {"sig35_variant": "b"},
        {"sig35_variant": "c"},
        {"sig35_variant": "c"},
    ]
    left = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    right = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ],
        dtype=np.float32,
    )
    left = left / np.linalg.norm(left, axis=1, keepdims=True)
    right = right / np.linalg.norm(right, axis=1, keepdims=True)

    with pytest.raises(ContractViolationError, match="matching non-degenerate cohort pairs"):
        aligned_cohort_distance_vectors(left, right, rows, column="sig35_variant")
