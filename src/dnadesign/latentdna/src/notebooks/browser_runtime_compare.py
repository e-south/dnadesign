"""
Comparison math and visualization helpers for generated latentdna marimo notebooks.
"""

from __future__ import annotations

from pathlib import Path

import marimo as mo
import numpy as np
import pandas as pd

from .browser_runtime_support import (
    CONTROL_PLANE_PALETTE,
    fig_to_image,
    load_table,
    load_view_matrix,
    load_view_rows,
    shared_join_key,
)


def deterministic_take(length: int, max_rows: int) -> np.ndarray:
    if length <= max_rows:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=max_rows, dtype=np.int64)


def explicit_compare_basis(
    left_view_id: str,
    right_view_id: str,
    comparison_bases: list[dict[str, object]],
) -> tuple[dict[str, object] | None, bool]:
    for basis in comparison_bases:
        if str(basis.get("left_view")) == left_view_id and str(basis.get("right_view")) == right_view_id:
            return basis, False
        if str(basis.get("left_view")) == right_view_id and str(basis.get("right_view")) == left_view_id:
            return basis, True
    return None, False


def resolve_alignment_pair_indices(
    alignment_id: str,
    *,
    output_root: Path,
    swapped: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    mapping = load_table(output_root / "alignments" / alignment_id / "mapping.parquet")
    if mapping.empty:
        return None
    required = {"left_indices", "right_indices", "left_count", "right_count"}
    if not required.issubset(set(mapping.columns)):
        return None
    mapping = mapping[(mapping["left_count"] == 1) & (mapping["right_count"] == 1)].copy()
    if mapping.empty:
        return None
    left_indices = np.asarray([int(values[0]) for values in mapping["left_indices"].tolist()], dtype=np.int64)
    right_indices = np.asarray([int(values[0]) for values in mapping["right_indices"].tolist()], dtype=np.int64)
    if swapped:
        return right_indices, left_indices
    return left_indices, right_indices


def resolve_shared_key_pair_indices(
    left_rows: pd.DataFrame,
    right_rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    join_key = shared_join_key(left_rows, right_rows)
    if join_key is None:
        return None
    if left_rows[join_key].duplicated().any() or right_rows[join_key].duplicated().any():
        raise ValueError(f"shared-key comparison requires unique `{join_key}` rows on both sides")
    merged = left_rows.reset_index().merge(
        right_rows.reset_index(),
        on=join_key,
        how="inner",
        suffixes=("_left", "_right"),
    )
    if merged.empty:
        return None
    return (
        merged["index_left"].to_numpy(dtype=np.int64),
        merged["index_right"].to_numpy(dtype=np.int64),
        join_key,
    )


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return matrix / norms


def cosine_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(matrix)
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    return np.clip(1.0 - similarity, 0.0, 2.0)


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    tri = np.triu_indices(matrix.shape[0], k=1)
    return matrix[tri].astype(np.float64, copy=False)


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if float(np.std(x_rank)) == 0.0 or float(np.std(y_rank)) == 0.0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def linear_cka(left_matrix: np.ndarray, right_matrix: np.ndarray) -> float | None:
    if left_matrix.shape[0] != right_matrix.shape[0] or left_matrix.shape[0] < 2:
        return None
    left_centered = left_matrix - left_matrix.mean(axis=0, keepdims=True)
    right_centered = right_matrix - right_matrix.mean(axis=0, keepdims=True)
    numerator = float(np.linalg.norm(right_centered.T @ left_centered, ord="fro") ** 2)
    denom_left = float(np.linalg.norm(left_centered.T @ left_centered, ord="fro"))
    denom_right = float(np.linalg.norm(right_centered.T @ right_centered, ord="fro"))
    if denom_left == 0.0 or denom_right == 0.0:
        return None
    return numerator / (denom_left * denom_right)


def mean_knn_overlap(left_matrix: np.ndarray, right_matrix: np.ndarray, *, k: int) -> float | None:
    if left_matrix.shape[0] != right_matrix.shape[0] or left_matrix.shape[0] <= 2:
        return None
    max_k = min(int(k), left_matrix.shape[0] - 1)
    if max_k < 1:
        return None
    left_distance = cosine_distance_matrix(left_matrix)
    right_distance = cosine_distance_matrix(right_matrix)
    np.fill_diagonal(left_distance, np.inf)
    np.fill_diagonal(right_distance, np.inf)
    left_neighbors = np.argpartition(left_distance, kth=max_k - 1, axis=1)[:, :max_k]
    right_neighbors = np.argpartition(right_distance, kth=max_k - 1, axis=1)[:, :max_k]
    overlaps = []
    for row_index in range(left_neighbors.shape[0]):
        left_set = set(int(item) for item in left_neighbors[row_index].tolist())
        right_set = set(int(item) for item in right_neighbors[row_index].tolist())
        union_size = len(left_set.union(right_set))
        overlaps.append(0.0 if union_size == 0 else len(left_set.intersection(right_set)) / union_size)
    return float(np.mean(overlaps))


def compare_pair_payload(
    left_view_id: str,
    right_view_id: str,
    *,
    geometry_rows_by_id: dict[str, dict[str, object]],
    comparison_bases: list[dict[str, object]],
    compare_metrics: dict[str, object],
    output_root: Path,
) -> dict[str, object]:
    left_rows = load_view_rows(left_view_id, output_root=output_root)
    right_rows = load_view_rows(right_view_id, output_root=output_root)
    if left_rows.empty or right_rows.empty:
        return {"status": "missing", "error": "one or both selected views are not materialized"}
    basis, swapped = explicit_compare_basis(left_view_id, right_view_id, comparison_bases)
    try:
        if basis is not None:
            indices = resolve_alignment_pair_indices(
                str(basis["alignment_id"]),
                output_root=output_root,
                swapped=swapped,
            )
            basis_label = f"alignment:{basis['alignment_id']}"
        else:
            shared = resolve_shared_key_pair_indices(left_rows, right_rows)
            if shared is None:
                return {"status": "missing", "error": "no explicit alignment or shared unique row key exists"}
            left_indices, right_indices, join_key = shared
            indices = (left_indices, right_indices)
            basis_label = f"shared_key:{join_key}"
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}
    if indices is None:
        return {"status": "missing", "error": "comparison basis exists but does not expose one-to-one support"}
    left_indices, right_indices = indices
    if left_indices.size == 0:
        return {"status": "missing", "error": "comparison support is empty"}
    take = deterministic_take(int(left_indices.size), int(compare_metrics.get("sample_rows", 192)))
    left_matrix = load_view_matrix(left_view_id, output_root=output_root)
    right_matrix = load_view_matrix(right_view_id, output_root=output_root)
    if left_matrix is None or right_matrix is None:
        return {"status": "missing", "error": "one or both selected matrices are not materialized"}
    left_sample = np.asarray(left_matrix[left_indices[take]], dtype=np.float32)
    right_sample = np.asarray(right_matrix[right_indices[take]], dtype=np.float32)
    left_distance = cosine_distance_matrix(left_sample)
    right_distance = cosine_distance_matrix(right_sample)
    left_distance_values = upper_triangle_values(left_distance)
    right_distance_values = upper_triangle_values(right_distance)
    pair_limit = int(compare_metrics.get("distance_pair_limit", 4096))
    if left_distance_values.size > pair_limit:
        pair_take = deterministic_take(int(left_distance_values.size), pair_limit)
        left_distance_values = left_distance_values[pair_take]
        right_distance_values = right_distance_values[pair_take]
    same_dims = int(left_sample.shape[1]) == int(right_sample.shape[1])
    same_coordinate_space = str((geometry_rows_by_id.get(left_view_id) or {}).get("coordinate_space_id") or "") == str(
        (geometry_rows_by_id.get(right_view_id) or {}).get("coordinate_space_id") or ""
    )
    rowwise_cosine = None
    rowwise_diff_norm = None
    coordinate_r2 = None
    if same_dims and same_coordinate_space:
        left_norm = l2_normalize(left_sample)
        right_norm = l2_normalize(right_sample)
        rowwise_cosine = np.sum(left_norm * right_norm, axis=1).astype(np.float64)
        rowwise_diff_norm = np.linalg.norm(left_sample - right_sample, axis=1).astype(np.float64)
        if left_sample.shape == right_sample.shape:
            left_flat = left_sample.reshape(-1).astype(np.float64)
            right_flat = right_sample.reshape(-1).astype(np.float64)
            if float(np.std(left_flat)) > 0.0 and float(np.std(right_flat)) > 0.0:
                coordinate_r = float(np.corrcoef(left_flat, right_flat)[0, 1])
                coordinate_r2 = coordinate_r * coordinate_r
    return {
        "status": "ok",
        "basis": basis_label,
        "rows": int(left_sample.shape[0]),
        "left_dims": int(left_sample.shape[1]),
        "right_dims": int(right_sample.shape[1]),
        "distance_pairs": int(left_distance_values.size),
        "distance_x": left_distance_values,
        "distance_y": right_distance_values,
        "rowwise_cosine": rowwise_cosine,
        "rowwise_diff_norm": rowwise_diff_norm,
        "same_dims": same_dims,
        "same_coordinate_space": same_coordinate_space,
        "metrics": {
            "distance_spearman": spearman_correlation(left_distance_values, right_distance_values),
            "linear_cka": linear_cka(left_sample, right_sample),
            "mean_knn_overlap": mean_knn_overlap(
                left_sample,
                right_sample,
                k=int(compare_metrics.get("knn_k", 10)),
            ),
            "coordinate_r2_diagnostic": coordinate_r2,
            "median_rowwise_cosine": None if rowwise_cosine is None else float(np.median(rowwise_cosine)),
            "median_rowwise_diff_norm": (None if rowwise_diff_norm is None else float(np.median(rowwise_diff_norm))),
        },
    }


def render_distance_correlation(payload: dict[str, object], *, title: str):
    if str(payload.get("status")) != "ok":
        return mo.callout(str(payload.get("error") or "Comparison payload is unavailable."), kind="warn")
    distance_x = np.asarray(payload.get("distance_x", []), dtype=np.float64)
    distance_y = np.asarray(payload.get("distance_y", []), dtype=np.float64)
    if distance_x.size < 2 or distance_y.size < 2:
        return mo.callout("Not enough paired distances are available to render a correlation plot.", kind="warn")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    mesh = ax.hexbin(distance_x, distance_y, gridsize=40, cmap="viridis", mincnt=1)
    lower = float(min(distance_x.min(), distance_y.min()))
    upper = float(max(distance_x.max(), distance_y.max()))
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.0, color="#5C6874", alpha=0.8)
    fig.colorbar(mesh, ax=ax, shrink=0.84, label="Pair count")
    ax.set_title(title, fontsize=11, fontweight="semibold")
    ax.set_xlabel("Left-view cosine distance")
    ax.set_ylabel("Right-view cosine distance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#D5DCE4", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    return fig_to_image(fig)


def render_rowwise_distribution(payload: dict[str, object], *, value_key: str, title: str, xlabel: str):
    if str(payload.get("status")) != "ok":
        return mo.callout(str(payload.get("error") or "Comparison payload is unavailable."), kind="warn")
    values = payload.get(value_key)
    if values is None:
        return mo.callout(
            "This row-wise metric is only defined when both selected views share the same coordinate "
            "space and dimensionality.",
            kind="info",
        )
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return mo.callout("No row-wise metric values are available for the selected pair.", kind="warn")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.hist(values, bins=min(40, max(12, int(values.size / 3))), color=CONTROL_PLANE_PALETTE[0], alpha=0.8)
    ax.set_title(title, fontsize=11, fontweight="semibold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#D5DCE4", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    return fig_to_image(fig)
