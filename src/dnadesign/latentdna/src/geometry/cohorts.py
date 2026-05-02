"""Cohort-level geometry helpers for pre-assay LatentDNA summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np

from ..contracts.errors import ContractViolationError
from .preprocessing import try_l2_normalize_vector


def group_indices(
    rows: list[dict[str, object]],
    *,
    column: str,
    exclude_values: set[str] | None = None,
    allowed_values: set[str] | None = None,
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        value = row.get(column)
        if value is None:
            continue
        text = str(value)
        if exclude_values is not None and text in exclude_values:
            continue
        if allowed_values is not None and text not in allowed_values:
            continue
        grouped[text].append(index)
    return {key: value for key, value in grouped.items() if value}


def centroid_map(matrix: np.ndarray, groups: dict[str, list[int]]) -> dict[str, np.ndarray]:
    centroids: dict[str, np.ndarray] = {}
    for key, indices in groups.items():
        centroid = try_l2_normalize_vector(np.asarray(matrix[indices].mean(axis=0), dtype=np.float32))
        if centroid is not None:
            centroids[key] = centroid
    return centroids


def separation_ratio_from_groups(
    matrix: np.ndarray,
    groups: dict[str, list[int]],
    *,
    eps: float = 1e-8,
) -> float:
    if len(groups) < 2:
        return float("nan")
    centroids = centroid_map(matrix, groups)
    if len(centroids) < 2:
        return float("nan")
    ordered = sorted(centroids)
    between = [
        1.0 - float(np.dot(centroids[left], centroids[right]))
        for left_index, left in enumerate(ordered)
        for right in ordered[left_index + 1 :]
    ]
    within: list[float] = []
    for key, indices in groups.items():
        if key not in centroids:
            continue
        centroid = centroids[key]
        within.extend((1.0 - np.asarray(matrix[indices] @ centroid, dtype=np.float32)).astype(np.float64).tolist())
    if not between or not within:
        return float("nan")
    return float(np.mean(np.asarray(between, dtype=np.float64)) / (np.mean(np.asarray(within, dtype=np.float64)) + eps))


def resample_groups(groups: dict[str, list[int]], *, rng: np.random.Generator) -> dict[str, list[int]]:
    return {
        key: rng.choice(np.asarray(indices, dtype=np.int64), size=len(indices), replace=True).astype(int).tolist()
        for key, indices in groups.items()
    }


def bootstrap_ci(
    metric_fn: Callable[[], float],
    *,
    iterations: int,
) -> tuple[float | None, float | None]:
    values: list[float] = []
    for _ in range(iterations):
        value = float(metric_fn())
        if np.isfinite(value):
            values.append(value)
    if not values:
        return None, None
    array = np.asarray(values, dtype=np.float64)
    return float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))


def balanced_group_indices(
    rows: list[dict[str, object]],
    *,
    group_column: str,
    balance_columns: list[str],
    required_group_values: set[str] | None,
    exclude_group_values: set[str] | None,
    rng: np.random.Generator,
) -> dict[str, list[int]]:
    strata: dict[tuple[object, ...], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        group_value = row.get(group_column)
        if group_value is None:
            continue
        group_text = str(group_value)
        if exclude_group_values is not None and group_text in exclude_group_values:
            continue
        if required_group_values is not None and group_text not in required_group_values:
            continue
        stratum_values: list[object] = []
        skip = False
        for column in balance_columns:
            value = row.get(column)
            if value is None:
                skip = True
                break
            stratum_values.append(value)
        if skip:
            continue
        strata[tuple(stratum_values)][group_text].append(index)

    balanced: dict[str, list[int]] = defaultdict(list)
    for group_map in strata.values():
        present_groups = set(group_map)
        if required_group_values is not None:
            if not required_group_values.issubset(present_groups):
                continue
            active_groups = sorted(required_group_values)
        else:
            if len(present_groups) < 2:
                continue
            active_groups = sorted(present_groups)
        count = min(len(group_map[group]) for group in active_groups)
        if count <= 0:
            continue
        for group in active_groups:
            selected = rng.choice(np.asarray(group_map[group], dtype=np.int64), size=count, replace=False)
            balanced[group].extend(int(value) for value in selected.tolist())
    return {key: sorted(value) for key, value in balanced.items() if value}


def cohort_distance_vector(
    matrix: np.ndarray,
    rows: list[dict[str, object]],
    *,
    column: str,
    exclude_values: set[str] | None = None,
) -> np.ndarray:
    distances = cohort_distance_map(matrix, rows, column=column, exclude_values=exclude_values)
    return np.asarray([distances[key] for key in sorted(distances)], dtype=np.float64)


def cohort_distance_map(
    matrix: np.ndarray,
    rows: list[dict[str, object]],
    *,
    column: str,
    exclude_values: set[str] | None = None,
) -> dict[tuple[str, str], float]:
    groups = group_indices(rows, column=column, exclude_values=exclude_values)
    if len(groups) < 2:
        return {}
    centroids = centroid_map(matrix, groups)
    if len(centroids) < 2:
        return {}
    ordered = sorted(centroids)
    distances: dict[tuple[str, str], float] = {}
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            distances[(left, right)] = 1.0 - float(np.dot(centroids[left], centroids[right]))
    return distances


def aligned_cohort_distance_vectors(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    rows: list[dict[str, object]],
    *,
    column: str,
    exclude_values: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    left_distances = cohort_distance_map(left_matrix, rows, column=column, exclude_values=exclude_values)
    right_distances = cohort_distance_map(right_matrix, rows, column=column, exclude_values=exclude_values)
    if not left_distances and not right_distances:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty
    if set(left_distances) != set(right_distances):
        raise ContractViolationError(
            f"aligned cohort-distance vectors require matching non-degenerate cohort pairs for {column!r}"
        )
    ordered_pairs = sorted(left_distances)
    return (
        np.asarray([left_distances[pair] for pair in ordered_pairs], dtype=np.float64),
        np.asarray([right_distances[pair] for pair in ordered_pairs], dtype=np.float64),
    )


def ordinal_gap_and_distance_vectors(
    *,
    centroids: dict[str, np.ndarray],
    ranks: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(variant for variant in centroids if variant in ranks)
    gaps: list[float] = []
    distances: list[float] = []
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            gaps.append(abs(float(ranks[left]) - float(ranks[right])))
            distances.append(1.0 - float(np.dot(centroids[left], centroids[right])))
    return np.asarray(gaps, dtype=np.float64), np.asarray(distances, dtype=np.float64)
