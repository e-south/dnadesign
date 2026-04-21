"""
Derived view builders for latentdna.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ..alignments.aggregators import aggregate_rows
from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..contracts.workspace import DerivedViewConfig
from ..io.matrix_io import read_matrix, write_matrix
from ..io.parquet_io import read_table, write_table
from ..workspaces.loader import WorkspaceContext
from ..workspaces.validation import _view_declares_reduced_space

_JOIN_KEY_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("construct__anchor_id", "construct__anchor_id"),
    ("construct__anchor_id", "id"),
    ("id", "construct__anchor_id"),
    ("id", "id"),
    ("subject_id", "subject_id"),
    ("context_id", "context_id"),
)


def _view_artifact_paths(context: WorkspaceContext, view_id: str) -> tuple[Path, Path, Path]:
    view_dir = context.output_root / "views" / view_id
    matrix_path = view_dir / "matrix.npy"
    rows_path = view_dir / "rows.parquet"
    manifest_path = view_dir / "manifest.json"
    for required in [matrix_path, rows_path, manifest_path]:
        if not required.exists():
            raise MissingArtifactError(f"missing prerequisite artifact for derived view {view_id}: {required}")
    return matrix_path, rows_path, manifest_path


def _load_view_artifact(context: WorkspaceContext, view_id: str) -> tuple[np.ndarray, pa.Table, dict[str, Any]]:
    matrix_path, rows_path, manifest_path = _view_artifact_paths(context, view_id)
    return (
        read_matrix(matrix_path),
        read_table(rows_path),
        context.read_manifest(manifest_path),
    )


def _resolve_key_column(rows_table: pa.Table, manifest: dict[str, Any], key: str) -> str:
    params = manifest.get("params", {})
    if key in {"record_key", "subject_key", "context_key"}:
        candidate = params.get(key)
        if isinstance(candidate, str) and candidate:
            key = candidate
    if key not in rows_table.column_names:
        raise ContractViolationError(f"derived view grouping key is missing from row ledger: {key!r}")
    return key


def _constant_group_columns(rows: list[dict[str, Any]], column_names: list[str]) -> list[str]:
    kept: list[str] = []
    for column in column_names:
        if all(len({row[column] for row in group}) == 1 for group in rows):
            kept.append(column)
    return kept


def _resolve_join_keys(left_rows: pa.Table, right_rows: pa.Table) -> tuple[str, str] | None:
    left_columns = set(left_rows.column_names)
    right_columns = set(right_rows.column_names)
    for left_key, right_key in _JOIN_KEY_CANDIDATES:
        if left_key in left_columns and right_key in right_columns:
            return left_key, right_key
    return None


def _candidate_join_keys(left_rows: pa.Table, right_rows: pa.Table) -> list[tuple[str, str]]:
    left_columns = set(left_rows.column_names)
    right_columns = set(right_rows.column_names)
    return [
        (left_key, right_key)
        for left_key, right_key in _JOIN_KEY_CANDIDATES
        if left_key in left_columns and right_key in right_columns
    ]


def _project_matrix_to_reference_rows_for_keys(
    reference_rows: pa.Table,
    candidate_rows: pa.Table,
    candidate_matrix: np.ndarray,
    *,
    input_view: str,
    left_key: str,
    right_key: str,
) -> np.ndarray:
    seen_reference_keys: set[object] = set()
    duplicate_reference_keys: list[object] = []
    for row in reference_rows.select([left_key]).to_pylist():
        key = row[left_key]
        if key in seen_reference_keys:
            duplicate_reference_keys.append(key)
            continue
        seen_reference_keys.add(key)
    if duplicate_reference_keys:
        preview = ", ".join(str(value) for value in duplicate_reference_keys[:5])
        raise ContractViolationError(f"concatenate reference rows are non-unique on {left_key!r}: {preview}")
    candidate_index_by_key: dict[object, int] = {}
    for index, row in enumerate(candidate_rows.select([right_key]).to_pylist()):
        key = row[right_key]
        if key in candidate_index_by_key:
            raise ContractViolationError(f"concatenate input {input_view!r} is non-unique on {right_key!r}")
        candidate_index_by_key[key] = index

    ordered_indices: list[int] = []
    missing_keys: list[object] = []
    for row in reference_rows.select([left_key]).to_pylist():
        key = row[left_key]
        index = candidate_index_by_key.get(key)
        if index is None:
            missing_keys.append(key)
            continue
        ordered_indices.append(index)

    if missing_keys:
        preview = ", ".join(str(value) for value in missing_keys[:5])
        raise ContractViolationError(
            f"concatenate input {input_view!r} is missing aligned rows on {right_key!r}: {preview}"
        )
    if len(ordered_indices) != len(candidate_index_by_key):
        raise ContractViolationError(
            f"concatenate input {input_view!r} has extra rows outside the reference support on {right_key!r}"
        )
    return np.ascontiguousarray(candidate_matrix[np.asarray(ordered_indices, dtype=np.int64)], dtype=np.float32)


def _project_matrix_to_reference_rows(
    reference_rows: pa.Table,
    candidate_rows: pa.Table,
    candidate_matrix: np.ndarray,
    *,
    input_view: str,
) -> np.ndarray:
    candidates = _candidate_join_keys(reference_rows, candidate_rows)
    if not candidates:
        raise ContractViolationError(
            f"concatenate requires matching rows or joinable key support; {input_view!r} shares no supported join key"
        )
    failures: list[str] = []
    for left_key, right_key in candidates:
        try:
            return _project_matrix_to_reference_rows_for_keys(
                reference_rows,
                candidate_rows,
                candidate_matrix,
                input_view=input_view,
                left_key=left_key,
                right_key=right_key,
            )
        except ContractViolationError as exc:
            failures.append(f"{left_key}->{right_key}: {exc}")
    if len(failures) == 1:
        raise ContractViolationError(failures[0].split(": ", 1)[1])
    raise ContractViolationError(
        f"concatenate could not align {input_view!r} on any supported join key: {'; '.join(failures)}"
    )


def _derive_vector_difference_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    view: DerivedViewConfig,
    artifact_dir: Path,
) -> tuple[Path, int, int, str, list[str]]:
    left_matrix, left_rows_table, left_manifest = _load_view_artifact(context, view.derive.left)
    right_matrix, right_rows_table, right_manifest = _load_view_artifact(context, view.derive.right)
    alignment_dir = context.output_root / "alignments" / view.derive.alignment
    required_paths = [
        alignment_dir / "mapping.parquet",
        alignment_dir / "rows.parquet",
        alignment_dir / "manifest.json",
    ]
    for required in required_paths:
        if not required.exists():
            raise MissingArtifactError(f"missing prerequisite artifact for derived view {view_id}: {required}")

    if left_manifest["params"]["coordinate_space_id"] != right_manifest["params"]["coordinate_space_id"]:
        raise ContractViolationError(
            "vector_difference requires matching coordinate spaces: "
            f"{left_manifest['params']['coordinate_space_id']!r} vs "
            f"{right_manifest['params']['coordinate_space_id']!r}"
        )
    if left_matrix.shape[1] != right_matrix.shape[1]:
        raise ContractViolationError(
            f"vector_difference requires matching dimensions: {left_matrix.shape[1]} vs {right_matrix.shape[1]}"
        )

    alignment_manifest = context.read_manifest(alignment_dir / "manifest.json")
    key_columns = list(alignment_manifest["params"]["key_columns"])
    if len(key_columns) != 1:
        raise ContractViolationError("vector_difference currently requires an alignment with exactly one key column")

    mapping_rows = read_table(alignment_dir / "mapping.parquet").to_pylist()
    rows_table = read_table(alignment_dir / "rows.parquet")
    left_rows = left_rows_table.to_pylist()
    right_rows = right_rows_table.to_pylist()
    left_mode = str(alignment_manifest["params"]["left_aggregation"])
    right_mode = str(alignment_manifest["params"]["right_aggregation"])
    output_matrix = _vector_difference_matrix(
        left_matrix,
        right_matrix,
        mapping_rows=mapping_rows,
        left_mode=left_mode,
        right_mode=right_mode,
        output_path=artifact_dir / "matrix.npy",
    )

    output_rows: list[dict[str, Any]] = []
    for key_row, mapping_row in zip(rows_table.to_pylist(), mapping_rows, strict=True):
        output_row = dict(key_row)
        for source_rows, index_field in ((right_rows, "right_indices"), (left_rows, "left_indices")):
            indices = [int(index) for index in mapping_row[index_field]]
            if not indices:
                continue
            candidate_rows = [source_rows[index] for index in indices]
            for column in candidate_rows[0]:
                if column in output_row:
                    continue
                first_value = candidate_rows[0][column]
                if all(row[column] == first_value for row in candidate_rows):
                    output_row[column] = candidate_rows[0][column]
        output_rows.append(output_row)
    rows_table = pa.Table.from_pylist(output_rows)

    write_table(rows_table, artifact_dir / "rows.parquet")
    return artifact_dir, output_matrix.shape[0], output_matrix.shape[1], key_columns[0], list(rows_table.column_names)


def _difference_batch_rows(*, dims: int) -> int:
    target_bytes = 128 * 1024**2
    row_bytes = max(dims, 1) * np.dtype(np.float32).itemsize * 3
    return max(int(target_bytes // max(row_bytes, 1)), 128)


def _single_index_pairs(mapping_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray] | None:
    left_indices: list[int] = []
    right_indices: list[int] = []
    for row in mapping_rows:
        left = list(row["left_indices"])
        right = list(row["right_indices"])
        if len(left) != 1 or len(right) != 1:
            return None
        left_indices.append(int(left[0]))
        right_indices.append(int(right[0]))
    return np.asarray(left_indices, dtype=np.int64), np.asarray(right_indices, dtype=np.int64)


def _vector_difference_matrix(
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    *,
    mapping_rows: list[dict[str, Any]],
    left_mode: str,
    right_mode: str,
    output_path: Path,
) -> np.ndarray:
    direct_pairs = None
    if left_mode == "error" and right_mode == "error":
        direct_pairs = _single_index_pairs(mapping_rows)
    if direct_pairs is None:
        output_matrix = np.vstack(
            [
                aggregate_rows(left_matrix, list(row["left_indices"]), mode=left_mode)
                - aggregate_rows(right_matrix, list(row["right_indices"]), mode=right_mode)
                for row in mapping_rows
            ]
        ).astype(np.float32, copy=False)
        output_matrix = np.ascontiguousarray(output_matrix)
        write_matrix(output_path, output_matrix)
        return output_matrix

    output_path.parent.mkdir(parents=True, exist_ok=True)
    left_indices, right_indices = direct_pairs
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(left_indices.shape[0]), int(left_matrix.shape[1])),
    )
    batch_rows = _difference_batch_rows(dims=int(left_matrix.shape[1]))
    for start in range(0, int(left_indices.shape[0]), batch_rows):
        stop = min(start + batch_rows, int(left_indices.shape[0]))
        output[start:stop] = np.asarray(left_matrix[left_indices[start:stop]], dtype=np.float32) - np.asarray(
            right_matrix[right_indices[start:stop]],
            dtype=np.float32,
        )
    output.flush()
    del output
    return np.load(output_path, mmap_mode="r")


def _derive_normalize_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    view: DerivedViewConfig,
    artifact_dir: Path,
) -> tuple[Path, int, int, str, list[str]]:
    source_matrix, rows_table, manifest = _load_view_artifact(context, view.derive.view)
    if view.derive.method == "l2":
        norms = np.linalg.norm(source_matrix, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        output_matrix = source_matrix / norms
    elif view.derive.method == "zscore":
        mean = source_matrix.mean(axis=0, keepdims=True)
        std = np.clip(source_matrix.std(axis=0, keepdims=True), a_min=1e-12, a_max=None)
        output_matrix = (source_matrix - mean) / std
    else:  # pragma: no cover - constrained by config
        raise ContractViolationError(f"unsupported normalize method: {view.derive.method}")
    output_matrix = np.ascontiguousarray(np.asarray(output_matrix, dtype=np.float32))
    write_matrix(artifact_dir / "matrix.npy", output_matrix)
    write_table(rows_table, artifact_dir / "rows.parquet")
    record_key = str(manifest["params"]["record_key"])
    return artifact_dir, output_matrix.shape[0], output_matrix.shape[1], record_key, list(rows_table.column_names)


def _derive_aggregate_by_key_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    view: DerivedViewConfig,
    artifact_dir: Path,
) -> tuple[Path, int, int, str, list[str]]:
    source_matrix, rows_table, manifest = _load_view_artifact(context, view.derive.view)
    key_column = _resolve_key_column(rows_table, manifest, view.derive.key)
    grouped_indices: dict[Any, list[int]] = {}
    for index, row in enumerate(rows_table.select([key_column]).to_pylist()):
        grouped_indices.setdefault(row[key_column], []).append(index)

    ordered_keys = sorted(grouped_indices, key=lambda value: str(value))
    output_matrix = np.vstack(
        [aggregate_rows(source_matrix, grouped_indices[key], mode=view.derive.aggregation) for key in ordered_keys]
    ).astype(np.float32, copy=False)
    grouped_rows = [
        rows_table.take(pa.array(indices, type=pa.int64())).to_pylist() for indices in grouped_indices.values()
    ]
    kept_columns = _constant_group_columns(grouped_rows, list(rows_table.column_names))
    output_rows = []
    for key in ordered_keys:
        row_group = rows_table.take(pa.array(grouped_indices[key], type=pa.int64())).to_pylist()
        output_rows.append({column: row_group[0][column] for column in kept_columns})

    output_matrix = np.ascontiguousarray(output_matrix)
    output_table = pa.Table.from_pylist(output_rows)
    write_matrix(artifact_dir / "matrix.npy", output_matrix)
    write_table(output_table, artifact_dir / "rows.parquet")
    return artifact_dir, output_matrix.shape[0], output_matrix.shape[1], key_column, list(output_table.column_names)


def _derive_apply_reducer_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    view: DerivedViewConfig,
    artifact_dir: Path,
) -> tuple[Path, int, int, str, list[str]]:
    source_matrix, rows_table, manifest = _load_view_artifact(context, view.derive.view)
    reducer_path = context.output_root / "reducers" / view.derive.reducer / "state.npz"
    if not reducer_path.exists():
        raise MissingArtifactError(f"missing reducer artifact for derived view {view_id}: {reducer_path}")
    state = np.load(reducer_path)
    mean = np.asarray(state["mean"], dtype=np.float32)
    components = np.asarray(state["components"], dtype=np.float32)
    if source_matrix.shape[1] != mean.shape[0]:
        raise ContractViolationError(
            f"apply_reducer requires matching input dimensions: {source_matrix.shape[1]} vs {mean.shape[0]}"
        )
    output_matrix = np.ascontiguousarray((source_matrix - mean) @ components.T, dtype=np.float32)
    write_matrix(artifact_dir / "matrix.npy", output_matrix)
    write_table(rows_table, artifact_dir / "rows.parquet")
    record_key = str(manifest["params"]["record_key"])
    return artifact_dir, output_matrix.shape[0], output_matrix.shape[1], record_key, list(rows_table.column_names)


def _derive_concatenate_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    view: DerivedViewConfig,
    artifact_dir: Path,
) -> tuple[Path, int, int, str, list[str]]:
    matrices: list[np.ndarray] = []
    rows_table: pa.Table | None = None
    record_key: str | None = None
    row_columns: list[str] | None = None
    input_spaces = {
        input_view: context.require_view(input_view).coordinate_space_id for input_view in view.derive.inputs
    }
    if len(set(input_spaces.values())) > 1 and not all(
        _view_declares_reduced_space(context.config, input_view) for input_view in view.derive.inputs
    ):
        rendered = ", ".join(f"{input_view}={space}" for input_view, space in input_spaces.items())
        raise ContractViolationError(
            f"concatenate inputs must share one coordinate space or all be reduced; got {rendered}"
        )
    for input_view in view.derive.inputs:
        matrix, candidate_rows, manifest = _load_view_artifact(context, input_view)
        if rows_table is None:
            rows_table = candidate_rows
            record_key = str(manifest["params"]["record_key"])
            row_columns = list(candidate_rows.column_names)
        elif not rows_table.equals(candidate_rows, check_metadata=False):
            matrix = _project_matrix_to_reference_rows(
                rows_table,
                candidate_rows,
                matrix,
                input_view=input_view,
            )
        matrices.append(matrix)
    assert rows_table is not None and record_key is not None and row_columns is not None
    output_matrix = np.ascontiguousarray(np.column_stack(matrices), dtype=np.float32)
    write_matrix(artifact_dir / "matrix.npy", output_matrix)
    write_table(rows_table, artifact_dir / "rows.parquet")
    return artifact_dir, output_matrix.shape[0], output_matrix.shape[1], record_key, row_columns


def derive_view_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    artifact_dir: Path | None = None,
) -> tuple[Path, int, int, str, list[str]]:
    view = context.require_view(view_id)
    if not isinstance(view, DerivedViewConfig):
        raise ContractViolationError(f"view {view_id} is not a derived view declaration")
    target_dir = artifact_dir or (context.output_root / "views" / view_id)
    if view.derive.kind == "vector_difference":
        return _derive_vector_difference_artifact(context, view_id=view_id, view=view, artifact_dir=target_dir)
    if view.derive.kind == "normalize":
        return _derive_normalize_artifact(context, view_id=view_id, view=view, artifact_dir=target_dir)
    if view.derive.kind == "aggregate_by_key":
        return _derive_aggregate_by_key_artifact(context, view_id=view_id, view=view, artifact_dir=target_dir)
    if view.derive.kind == "apply_reducer":
        return _derive_apply_reducer_artifact(context, view_id=view_id, view=view, artifact_dir=target_dir)
    if view.derive.kind == "concatenate":
        return _derive_concatenate_artifact(context, view_id=view_id, view=view, artifact_dir=target_dir)
    raise ContractViolationError(f"unsupported derived view kind: {view.derive.kind}")
