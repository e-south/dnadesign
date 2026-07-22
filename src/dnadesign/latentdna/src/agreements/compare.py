"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/agreements/compare.py

Agreement artifact builders for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..io.json_io import write_json
from ..io.matrix_io import read_matrix
from ..io.parquet_io import read_table, write_table
from ..sources.resolver import read_records_table, resolve_source
from ..workspaces.loader import WorkspaceContext


def _select_indices(rows: list[dict[str, Any]], where: dict[str, Any]) -> list[int]:
    column = where.get("column")
    if not isinstance(column, str):
        raise ContractViolationError("landmark where clause requires a 'column' field")
    if "equals" in where:
        target = where["equals"]
        return [index for index, row in enumerate(rows) if row.get(column) == target]
    if "in" in where:
        targets = set(where["in"])
        return [index for index, row in enumerate(rows) if row.get(column) in targets]
    raise ContractViolationError("landmark where clause requires either 'equals' or 'in'")


def _alignment_input_uses_source(context: WorkspaceContext, ref_id: str, source_id: str) -> bool:
    if ref_id == source_id:
        return True
    view = context.config.views.get(ref_id)
    return bool(view is not None and getattr(view, "source", None) == source_id)


def _neighbor_artifact_paths(context: WorkspaceContext, neighbor_id: str) -> tuple[Path, Path, Path]:
    artifact_dir = context.output_root / "neighbors" / neighbor_id
    indices_path = artifact_dir / "indices.npy"
    rows_path = artifact_dir / "rows.parquet"
    manifest_path = artifact_dir / "manifest.json"
    for required in [indices_path, rows_path, manifest_path]:
        if not required.exists():
            raise MissingArtifactError(f"neighbor artifact is missing for agreement comparison: {required}")
    return indices_path, rows_path, manifest_path


def _cluster_artifact_paths(context: WorkspaceContext, cluster_id: str) -> tuple[Path, Path, Path]:
    artifact_dir = context.output_root / "clusters" / cluster_id
    assignments_path = artifact_dir / "assignments.parquet"
    summary_path = artifact_dir / "summary.json"
    manifest_path = artifact_dir / "manifest.json"
    for required in [assignments_path, summary_path, manifest_path]:
        if not required.exists():
            raise MissingArtifactError(f"cluster artifact is missing for agreement comparison: {required}")
    return assignments_path, summary_path, manifest_path


def _require_same_rows(left_rows: pa.Table, right_rows: pa.Table, *, label: str) -> None:
    if left_rows.column_names != right_rows.column_names or left_rows.to_pylist() != right_rows.to_pylist():
        raise ContractViolationError(f"{label} requires the same ordered row basis on both inputs")


def _row_basis(rows: list[dict[str, Any]], *, exclude_columns: set[str]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key not in exclude_columns} for row in rows]


def _table_from_rows(rows: list[dict[str, Any]]) -> pa.Table:
    column_order: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            column_order.append(key)
    normalized = [{column: row.get(column) for column in column_order} for row in rows]
    return pa.Table.from_pylist(normalized)


def _compare_neighbor_rows(
    left_rows: pa.Table,
    left_indices: np.ndarray,
    *,
    left_neighbors_id: str,
    right_rows: pa.Table,
    right_indices: np.ndarray,
    right_neighbors_id: str,
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, float | int | str]]:
    overlap_rows: list[dict[str, Any]] = []
    shared_counts: list[int] = []
    overlap_fractions: list[float] = []
    for row, left_row_indices, right_row_indices in zip(
        left_rows.to_pylist(),
        left_indices,
        right_indices,
        strict=True,
    ):
        shared = len(set(left_row_indices.tolist()).intersection(int(value) for value in right_row_indices.tolist()))
        fraction = shared / k
        shared_counts.append(shared)
        overlap_fractions.append(fraction)
        overlap_rows.append(
            {
                **row,
                "method": "knn_overlap",
                "shared_neighbor_count": shared,
                "neighbor_overlap_fraction": float(fraction),
                "overlap_fraction": float(fraction),
            }
        )

    summary = {
        "method": "knn_overlap",
        "left_neighbors": left_neighbors_id,
        "right_neighbors": right_neighbors_id,
        "rows": len(overlap_rows),
        "k": k,
        "mean_shared_neighbor_count": float(np.mean(shared_counts, dtype=np.float64)),
        "mean_neighbor_overlap_fraction": float(np.mean(overlap_fractions, dtype=np.float64)),
        "mean_overlap_fraction": float(np.mean(overlap_fractions, dtype=np.float64)),
        "min_overlap_fraction": float(np.min(overlap_fractions)),
        "max_overlap_fraction": float(np.max(overlap_fractions)),
    }
    return overlap_rows, summary


def _landmark_seed_indices(
    context: WorkspaceContext,
    landmark_id: str,
    *,
    rows_table: pa.Table,
    scope_kind: str,
    scope_id: str | None,
) -> list[int]:
    landmark = context.require_landmark(landmark_id)
    selector_column = landmark.where.get("column")
    if not isinstance(selector_column, str):
        raise ContractViolationError(f"landmark {landmark_id} is missing a selector column")

    rows = rows_table.to_pylist()
    if selector_column in rows_table.column_names:
        indices = _select_indices(rows, landmark.where)
        if indices:
            return indices

    resolved_scope_kind = scope_kind
    resolved_scope_id = scope_id
    if scope_kind == "reduced_view" and scope_id is not None:
        reduced_manifest = context.read_manifest(context.output_root / "reduced_views" / scope_id / "manifest.json")
        upstream_scope_kind = reduced_manifest["params"].get("fit_scope_kind")
        upstream_scope_id = reduced_manifest["params"].get("fit_scope_id")
        if isinstance(upstream_scope_kind, str):
            resolved_scope_kind = upstream_scope_kind
            resolved_scope_id = str(upstream_scope_id) if upstream_scope_id is not None else None

    if resolved_scope_kind != "alignment_set" or resolved_scope_id is None:
        raise ContractViolationError(
            "landmark "
            f"{landmark_id} cannot be selected from neighbor rows without "
            f"{selector_column!r} in scope {scope_kind!r}"
        )

    alignment_manifest = context.read_manifest(context.output_root / "alignments" / resolved_scope_id / "manifest.json")
    left_key_columns = [str(name) for name in alignment_manifest["params"]["key_columns"]]
    right_key_columns = [str(name) for name in alignment_manifest["params"].get("right_key_columns", left_key_columns)]
    right_ref = str(alignment_manifest["params"]["right"])
    source_key_columns = (
        right_key_columns if _alignment_input_uses_source(context, right_ref, landmark.source) else left_key_columns
    )
    source = context.require_source(landmark.source)
    resolved = resolve_source(landmark.source, source, workspace_dir=context.workspace_dir)
    source_rows = read_records_table(
        resolved,
        columns=list(dict.fromkeys([*source_key_columns, selector_column])),
    ).to_pylist()
    matched_source_indices = _select_indices(source_rows, landmark.where)
    if not matched_source_indices:
        raise ContractViolationError(f"landmark {landmark_id} matched no rows in source {landmark.source!r}")
    key_set = {tuple(source_rows[index][column] for column in source_key_columns) for index in matched_source_indices}
    aligned_indices = [
        index for index, row in enumerate(rows) if tuple(row.get(column) for column in left_key_columns) in key_set
    ]
    if not aligned_indices:
        raise ContractViolationError(f"landmark {landmark_id} matched no rows in agreement scope {resolved_scope_id!r}")
    return aligned_indices


def _compare_landmark_neighbor_rows(
    context: WorkspaceContext,
    *,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
    rows_table: pa.Table,
    scope_kind: str,
    scope_id: str | None,
    landmark_ids: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not landmark_ids:
        return [], {}

    landmark_rows: list[dict[str, Any]] = []
    jaccards: list[float] = []
    for landmark_id in landmark_ids:
        seed_indices = _landmark_seed_indices(
            context,
            landmark_id,
            rows_table=rows_table,
            scope_kind=scope_kind,
            scope_id=scope_id,
        )
        left_neighbors = {int(index) for index in left_indices[seed_indices].ravel().tolist()}
        right_neighbors = {int(index) for index in right_indices[seed_indices].ravel().tolist()}
        shared = len(left_neighbors.intersection(right_neighbors))
        union = len(left_neighbors.union(right_neighbors))
        jaccard = 1.0 if union == 0 else shared / union
        left_fraction = 1.0 if not left_neighbors else shared / len(left_neighbors)
        right_fraction = 1.0 if not right_neighbors else shared / len(right_neighbors)
        jaccards.append(jaccard)
        landmark_rows.append(
            {
                "method": "landmark_neighbor_overlap",
                "landmark_id": landmark_id,
                "seed_count": len(seed_indices),
                "left_neighbor_count": len(left_neighbors),
                "right_neighbor_count": len(right_neighbors),
                "shared_neighbor_count": shared,
                "jaccard_overlap": float(jaccard),
                "left_overlap_fraction": float(left_fraction),
                "right_overlap_fraction": float(right_fraction),
            }
        )

    summary = {
        "method": "landmark_neighbor_overlap",
        "rows": len(landmark_rows),
        "landmarks": landmark_ids,
        "mean_jaccard_overlap": float(np.mean(jaccards, dtype=np.float64)),
        "min_jaccard_overlap": float(np.min(jaccards)),
        "max_jaccard_overlap": float(np.max(jaccards)),
    }
    return landmark_rows, summary


def _contingency_matrix(left_labels: np.ndarray, right_labels: np.ndarray) -> np.ndarray:
    left_unique, left_inverse = np.unique(left_labels, return_inverse=True)
    right_unique, right_inverse = np.unique(right_labels, return_inverse=True)
    contingency = np.zeros((len(left_unique), len(right_unique)), dtype=np.int64)
    np.add.at(contingency, (left_inverse, right_inverse), 1)
    return contingency


def _adjusted_rand_index(contingency: np.ndarray) -> float:
    n_samples = int(contingency.sum())
    if n_samples < 2:
        return 1.0

    def _comb2(values: np.ndarray) -> float:
        values = values.astype(np.float64, copy=False)
        return float(np.sum(values * (values - 1.0) / 2.0, dtype=np.float64))

    sum_comb = _comb2(contingency)
    sum_comb_left = _comb2(contingency.sum(axis=1))
    sum_comb_right = _comb2(contingency.sum(axis=0))
    total_comb = (n_samples * (n_samples - 1.0)) / 2.0
    if total_comb == 0.0:
        return 1.0
    expected = (sum_comb_left * sum_comb_right) / total_comb
    max_index = 0.5 * (sum_comb_left + sum_comb_right)
    denominator = max_index - expected
    if denominator == 0.0:
        return 1.0
    return float((sum_comb - expected) / denominator)


def _normalized_mutual_information(contingency: np.ndarray) -> float:
    total = float(contingency.sum())
    if total == 0.0:
        return 1.0
    left = contingency.sum(axis=1).astype(np.float64, copy=False)
    right = contingency.sum(axis=0).astype(np.float64, copy=False)
    contingency = contingency.astype(np.float64, copy=False)
    mutual_information = 0.0
    for left_index in range(contingency.shape[0]):
        for right_index in range(contingency.shape[1]):
            count = contingency[left_index, right_index]
            if count == 0.0:
                continue
            mutual_information += (count / total) * math.log((count * total) / (left[left_index] * right[right_index]))

    def _entropy(counts: np.ndarray) -> float:
        probabilities = counts[counts > 0.0] / total
        if probabilities.size == 0:
            return 0.0
        return float(-np.sum(probabilities * np.log(probabilities), dtype=np.float64))

    left_entropy = _entropy(left)
    right_entropy = _entropy(right)
    denominator = math.sqrt(left_entropy * right_entropy)
    if denominator == 0.0:
        return 1.0
    return float(mutual_information / denominator)


def _compare_cluster_rows(
    left_assignments: pa.Table,
    *,
    left_cluster_id: str,
    right_assignments: pa.Table,
    right_cluster_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if "cluster_label" not in left_assignments.column_names or "cluster_label" not in right_assignments.column_names:
        raise ContractViolationError("cluster comparison requires cluster_label columns in both assignment tables")

    left_rows = left_assignments.to_pylist()
    right_rows = right_assignments.to_pylist()
    left_basis = _row_basis(left_rows, exclude_columns={"cluster_label"})
    right_basis = _row_basis(right_rows, exclude_columns={"cluster_label"})
    if left_basis != right_basis:
        raise ContractViolationError("cluster comparison requires the same ordered row basis on both cluster sets")

    left_labels = np.asarray([row["cluster_label"] for row in left_rows], dtype=np.int64)
    right_labels = np.asarray([row["cluster_label"] for row in right_rows], dtype=np.int64)
    contingency = _contingency_matrix(left_labels, right_labels)
    ari = _adjusted_rand_index(contingency)
    nmi = _normalized_mutual_information(contingency)
    rows = [
        {
            "method": "cluster_agreement",
            "metric": "adjusted_rand_index",
            "value": float(ari),
        },
        {
            "method": "cluster_agreement",
            "metric": "normalized_mutual_information",
            "value": float(nmi),
        },
    ]
    summary = {
        "method": "cluster_agreement",
        "left_clusters": left_cluster_id,
        "right_clusters": right_cluster_id,
        "rows": len(left_labels),
        "adjusted_rand_index": float(ari),
        "normalized_mutual_information": float(nmi),
    }
    return rows, summary


def compare_agreement_artifact(
    context: WorkspaceContext,
    *,
    agreement_id: str,
    left_neighbors_id: str | None = None,
    right_neighbors_id: str | None = None,
    left_cluster_id: str | None = None,
    right_cluster_id: str | None = None,
    landmark_ids: list[str] | None = None,
) -> tuple[Path, int, dict[str, Any]]:
    if not any([left_neighbors_id, right_neighbors_id, left_cluster_id, right_cluster_id]):
        raise ContractViolationError("agreement comparison requires neighbor and/or cluster inputs")
    if (left_neighbors_id is None) != (right_neighbors_id is None):
        raise ContractViolationError("agreement comparison requires both left and right neighbor ids together")
    if (left_cluster_id is None) != (right_cluster_id is None):
        raise ContractViolationError("agreement comparison requires both left and right cluster ids together")
    if landmark_ids and left_neighbors_id is None:
        raise ContractViolationError("landmark neighbor overlap requires left/right neighbor ids")

    output_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    methods: list[str] = []

    if left_neighbors_id is not None and right_neighbors_id is not None:
        left_indices_path, left_rows_path, left_manifest_path = _neighbor_artifact_paths(context, left_neighbors_id)
        right_indices_path, right_rows_path, right_manifest_path = _neighbor_artifact_paths(context, right_neighbors_id)
        left_manifest = context.read_manifest(left_manifest_path)
        right_manifest = context.read_manifest(right_manifest_path)

        left_k = int(left_manifest["params"]["k"])
        right_k = int(right_manifest["params"]["k"])
        if left_k != right_k:
            raise ContractViolationError(f"agreement comparison requires equal k values: {left_k} vs {right_k}")

        left_rows = read_table(left_rows_path)
        right_rows = read_table(right_rows_path)
        _require_same_rows(left_rows, right_rows, label="agreement comparison")

        left_indices = np.asarray(read_matrix(left_indices_path, mmap_mode=None), dtype=np.int64)
        right_indices = np.asarray(read_matrix(right_indices_path, mmap_mode=None), dtype=np.int64)
        if left_indices.shape != right_indices.shape:
            raise ContractViolationError(
                "agreement comparison requires matching neighbor matrix shapes: "
                f"{left_indices.shape} vs {right_indices.shape}"
            )

        knn_rows, knn_summary = _compare_neighbor_rows(
            left_rows,
            left_indices,
            left_neighbors_id=left_neighbors_id,
            right_rows=right_rows,
            right_indices=right_indices,
            right_neighbors_id=right_neighbors_id,
            k=left_k,
        )
        output_rows.extend(knn_rows)
        summary["knn_overlap"] = knn_summary
        methods.append("knn_overlap")
        for key in [
            "rows",
            "k",
            "mean_shared_neighbor_count",
            "mean_overlap_fraction",
            "min_overlap_fraction",
            "max_overlap_fraction",
        ]:
            summary[key] = knn_summary[key]

        landmark_rows, landmark_summary = _compare_landmark_neighbor_rows(
            context,
            left_indices=left_indices,
            right_indices=right_indices,
            rows_table=left_rows,
            scope_kind=str(left_manifest["params"]["scope_kind"]),
            scope_id=left_manifest["params"].get("scope_id"),
            landmark_ids=landmark_ids or [],
        )
        if landmark_rows:
            output_rows.extend(landmark_rows)
            summary["landmark_neighbor_overlap"] = landmark_summary
            methods.append("landmark_neighbor_overlap")

    if left_cluster_id is not None and right_cluster_id is not None:
        left_assignments_path, _, left_manifest_path = _cluster_artifact_paths(context, left_cluster_id)
        right_assignments_path, _, right_manifest_path = _cluster_artifact_paths(context, right_cluster_id)
        left_manifest = context.read_manifest(left_manifest_path)
        right_manifest = context.read_manifest(right_manifest_path)

        left_assignments = read_table(left_assignments_path)
        right_assignments = read_table(right_assignments_path)
        cluster_rows, cluster_summary = _compare_cluster_rows(
            left_assignments,
            left_cluster_id=left_cluster_id,
            right_assignments=right_assignments,
            right_cluster_id=right_cluster_id,
        )
        output_rows.extend(cluster_rows)
        summary["cluster_agreement"] = cluster_summary
        methods.append("cluster_agreement")

        left_scope = (
            str(left_manifest["params"].get("scope_kind")),
            left_manifest["params"].get("scope_id"),
        )
        right_scope = (
            str(right_manifest["params"].get("scope_kind")),
            right_manifest["params"].get("scope_id"),
        )
        if left_scope != right_scope:
            raise ContractViolationError(
                "cluster comparison requires both cluster sets to use the same declared scope kind and scope id"
            )

    if not output_rows:
        raise ContractViolationError("agreement comparison produced no rows")

    summary["methods"] = sorted(methods)
    artifact_dir = context.output_root / "agreements" / agreement_id
    write_table(_table_from_rows(output_rows), artifact_dir / "table.parquet")
    write_json(artifact_dir / "summary.json", summary)
    return artifact_dir, len(output_rows), summary
