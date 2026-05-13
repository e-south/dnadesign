"""
Shared scoped-matrix helpers for latentdna view-backed artifacts.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from ..alignments.aggregators import aggregate_rows
from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..io.matrix_io import read_matrix
from ..io.parquet_io import read_table
from ..workspaces.loader import WorkspaceContext

_SAMPLE_SCOPE_CACHE_MAX_ENTRY_BYTES = 128 * 1024**2
_SAMPLE_SCOPE_CACHE_MAX_TOTAL_BYTES = 512 * 1024**2
_SAMPLE_SCOPE_CACHE_MAX_ENTRIES = 16


@dataclass(frozen=True, slots=True)
class _PathStamp:
    path: str
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class _SampleScopeCacheKey:
    workspace_id: str
    view_id: str
    sample_id: str
    record_key: str
    matrix: _PathStamp
    view_rows: _PathStamp
    sample_rows: _PathStamp
    manifest: _PathStamp


_SAMPLE_SCOPE_CACHE: OrderedDict[_SampleScopeCacheKey, tuple[np.ndarray, pa.Table]] = OrderedDict()
_SAMPLE_SCOPE_CACHE_BYTES = 0


def _path_stamp(path: Path) -> _PathStamp:
    stat = path.stat()
    return _PathStamp(path=path.resolve().as_posix(), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def _sample_scope_cache_key(
    context: WorkspaceContext,
    *,
    view_id: str,
    sample_id: str,
    record_key: str,
    matrix_path: Path,
    rows_path: Path,
    sample_rows_path: Path,
    manifest_path: Path,
) -> _SampleScopeCacheKey:
    return _SampleScopeCacheKey(
        workspace_id=context.workspace_id,
        view_id=view_id,
        sample_id=sample_id,
        record_key=record_key,
        matrix=_path_stamp(matrix_path),
        view_rows=_path_stamp(rows_path),
        sample_rows=_path_stamp(sample_rows_path),
        manifest=_path_stamp(manifest_path),
    )


def _cache_sample_scope(key: _SampleScopeCacheKey, matrix: np.ndarray, rows: pa.Table) -> None:
    global _SAMPLE_SCOPE_CACHE_BYTES
    entry_bytes = int(matrix.nbytes)
    if entry_bytes > _SAMPLE_SCOPE_CACHE_MAX_ENTRY_BYTES:
        return
    while _SAMPLE_SCOPE_CACHE and (
        len(_SAMPLE_SCOPE_CACHE) >= _SAMPLE_SCOPE_CACHE_MAX_ENTRIES
        or _SAMPLE_SCOPE_CACHE_BYTES + entry_bytes > _SAMPLE_SCOPE_CACHE_MAX_TOTAL_BYTES
    ):
        _, (evicted_matrix, _) = _SAMPLE_SCOPE_CACHE.popitem(last=False)
        _SAMPLE_SCOPE_CACHE_BYTES -= int(evicted_matrix.nbytes)
    if entry_bytes > _SAMPLE_SCOPE_CACHE_MAX_TOTAL_BYTES:
        return
    cached_matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    cached_matrix.setflags(write=False)
    _SAMPLE_SCOPE_CACHE[key] = (cached_matrix, rows)
    _SAMPLE_SCOPE_CACHE_BYTES += int(cached_matrix.nbytes)


def clear_scope_caches() -> None:
    """Clear process-local scoped matrix caches."""

    global _SAMPLE_SCOPE_CACHE_BYTES
    _SAMPLE_SCOPE_CACHE.clear()
    _SAMPLE_SCOPE_CACHE_BYTES = 0


def _ordered_indices(view_rows: list[dict], sample_rows: list[dict], *, record_key: str) -> list[int]:
    index_by_key = {row[record_key]: index for index, row in enumerate(view_rows)}
    indices: list[int] = []
    missing: list[str] = []
    for row in sample_rows:
        key = row[record_key]
        if key not in index_by_key:
            missing.append(str(key))
            continue
        indices.append(index_by_key[key])
    if missing:
        raise ContractViolationError(f"sample rows are not aligned to view on {record_key}: missing {missing[:5]}")
    return indices


def _matrix_source_paths(
    context: WorkspaceContext,
    *,
    view_id: str | None,
    reduced_view_id: str | None,
) -> tuple[Path, Path, dict[str, object], str, str]:
    if (view_id is None) == (reduced_view_id is None):
        raise ContractViolationError("matrix scope requires exactly one of view_id or reduced_view_id")
    artifact_kind = "view" if view_id is not None else "reduced_view"
    artifact_id = str(view_id if view_id is not None else reduced_view_id)
    artifact_dir = context.output_root / ("views" if artifact_kind == "view" else "reduced_views") / artifact_id
    matrix_path = artifact_dir / "matrix.npy"
    rows_path = artifact_dir / "rows.parquet"
    manifest_path = artifact_dir / "manifest.json"
    for required in [matrix_path, rows_path, manifest_path]:
        if not required.exists():
            raise MissingArtifactError(f"view artifact is missing for scoped view access: {required}")
    return matrix_path, rows_path, context.read_manifest(manifest_path), artifact_kind, artifact_id


def view_artifact_paths(context: WorkspaceContext, view_id: str) -> tuple[Path, Path, dict[str, object]]:
    matrix_path, rows_path, manifest, _, _ = _matrix_source_paths(context, view_id=view_id, reduced_view_id=None)
    return matrix_path, rows_path, manifest


def _view_artifact_paths_with_manifest_path(
    context: WorkspaceContext,
    view_id: str,
) -> tuple[Path, Path, Path, dict[str, object]]:
    matrix_path, rows_path, manifest, _, _ = _matrix_source_paths(context, view_id=view_id, reduced_view_id=None)
    return matrix_path, rows_path, rows_path.parent / "manifest.json", manifest


def matrix_input_digest_path(
    context: WorkspaceContext,
    *,
    view_id: str | None,
    reduced_view_id: str | None,
) -> tuple[str, str, Path]:
    matrix_path, _, _, artifact_kind, artifact_id = _matrix_source_paths(
        context,
        view_id=view_id,
        reduced_view_id=reduced_view_id,
    )
    input_kind = "view_matrix" if artifact_kind == "view" else "reduced_view"
    return input_kind, artifact_id, matrix_path


def _full_scope(context: WorkspaceContext, view_id: str) -> tuple[np.ndarray, pa.Table, str, str | None]:
    matrix_path, rows_path, _ = view_artifact_paths(context, view_id)
    return np.asarray(read_matrix(matrix_path), dtype=np.float32), read_table(rows_path), "full_view", view_id


def _sample_scope(
    context: WorkspaceContext,
    view_id: str,
    *,
    sample_id: str,
) -> tuple[np.ndarray, pa.Table, str, str | None]:
    matrix_path, rows_path, manifest_path, manifest = _view_artifact_paths_with_manifest_path(context, view_id)
    sample_rows_path = context.output_root / "samples" / sample_id / "rows.parquet"
    if not sample_rows_path.exists():
        raise MissingArtifactError(f"sample artifact is missing for scoped view access: {sample_id}")
    record_key = str(manifest["params"]["record_key"])
    cache_key = _sample_scope_cache_key(
        context,
        view_id=view_id,
        sample_id=sample_id,
        record_key=record_key,
        matrix_path=matrix_path,
        rows_path=rows_path,
        sample_rows_path=sample_rows_path,
        manifest_path=manifest_path,
    )
    cached = _SAMPLE_SCOPE_CACHE.get(cache_key)
    if cached is not None:
        _SAMPLE_SCOPE_CACHE.move_to_end(cache_key)
        cached_matrix, cached_rows = cached
        return cached_matrix, cached_rows, "sample_set", sample_id

    matrix = np.asarray(read_matrix(matrix_path), dtype=np.float32)
    view_rows = read_table(rows_path).to_pylist()
    sample_rows = read_table(sample_rows_path).to_pylist()
    indices = _ordered_indices(view_rows, sample_rows, record_key=record_key)
    scoped_matrix = np.ascontiguousarray(np.asarray(matrix[indices], dtype=np.float32))
    rows = pa.Table.from_pylist(sample_rows)
    _cache_sample_scope(cache_key, scoped_matrix, rows)
    return scoped_matrix, rows, "sample_set", sample_id


def _group_candidate_rows(candidate_rows: pa.Table, *, key_columns: list[str]) -> dict[tuple[object, ...], list[int]]:
    missing_columns = [column for column in key_columns if column not in candidate_rows.column_names]
    if missing_columns:
        raise ContractViolationError(f"aligned metadata projection is missing required key columns: {missing_columns}")
    grouped: dict[tuple[object, ...], list[int]] = {}
    for index, row in enumerate(candidate_rows.select(key_columns).to_pylist()):
        key = tuple(row[column] for column in key_columns)
        grouped.setdefault(key, []).append(index)
    return grouped


def _project_rows_to_alignment(
    alignment_rows: pa.Table,
    candidate_rows: pa.Table,
    *,
    alignment_key_columns: list[str],
    candidate_key_columns: list[str],
    label: str,
) -> list[list[int]]:
    grouped = _group_candidate_rows(candidate_rows, key_columns=candidate_key_columns)
    index_groups: list[list[int]] = []
    missing: list[tuple[object, ...]] = []
    for row in alignment_rows.select(alignment_key_columns).to_pylist():
        key = tuple(row[column] for column in alignment_key_columns)
        indices = grouped.get(key)
        if indices is None:
            missing.append(key)
            continue
        index_groups.append(indices)
    if missing:
        raise ContractViolationError(f"{label} is missing rows for aligned keys: {missing[:5]}")
    return index_groups


def _project_metadata_to_alignment_rows(
    alignment_rows: pa.Table,
    candidate_rows: pa.Table,
    *,
    index_groups: list[list[int]],
    label: str,
) -> pa.Table:
    if len(index_groups) != alignment_rows.num_rows:
        raise ContractViolationError(
            f"{label} produced {len(index_groups)} index groups for {alignment_rows.num_rows} aligned rows"
        )
    projected = alignment_rows
    for field in candidate_rows.schema:
        column_name = field.name
        if column_name in projected.column_names:
            continue
        source_values = candidate_rows[column_name].to_pylist()
        projected_values: list[object] = []
        for indices in index_groups:
            first_value = source_values[indices[0]]
            for index in indices[1:]:
                if source_values[index] != first_value:
                    raise ContractViolationError(
                        f"{label} metadata column {column_name!r} disagrees within one aligned row"
                    )
            projected_values.append(first_value)
        projected = projected.append_column(column_name, pa.array(projected_values, type=field.type))
    return projected


def _alignment_scope(
    context: WorkspaceContext,
    view_id: str,
    *,
    alignment_id: str,
) -> tuple[np.ndarray, pa.Table, str, str | None]:
    matrix_path, view_rows_path, _ = view_artifact_paths(context, view_id)
    alignment_dir = context.output_root / "alignments" / alignment_id
    alignment_manifest_path = alignment_dir / "manifest.json"
    mapping_path = alignment_dir / "mapping.parquet"
    rows_path = alignment_dir / "rows.parquet"
    for required in [alignment_manifest_path, mapping_path, rows_path]:
        if not required.exists():
            raise MissingArtifactError(f"alignment artifact is missing for scoped view access: {required}")

    alignment_manifest = context.read_manifest(alignment_manifest_path)
    matrix = np.asarray(read_matrix(matrix_path), dtype=np.float32)
    mapping_rows = read_table(mapping_path).to_pylist()
    rows = read_table(rows_path)
    candidate_rows = read_table(view_rows_path)
    left_key_columns = [str(name) for name in alignment_manifest["params"]["key_columns"]]
    right_key_columns = [str(name) for name in alignment_manifest["params"].get("right_key_columns", left_key_columns)]
    if alignment_manifest["params"]["left"] == view_id:
        index_field = "left_indices"
        mode = str(alignment_manifest["params"]["left_aggregation"])
    elif alignment_manifest["params"]["right"] == view_id:
        index_field = "right_indices"
        mode = str(alignment_manifest["params"]["right_aggregation"])
    else:
        raise ContractViolationError(f"alignment {alignment_id} does not include view {view_id}")

    candidate_key_columns = (
        left_key_columns
        if set(left_key_columns).issubset(set(candidate_rows.column_names))
        else right_key_columns
        if set(right_key_columns).issubset(set(candidate_rows.column_names))
        else []
    )
    if not candidate_key_columns:
        raise ContractViolationError(
            f"alignment {alignment_id} shares neither {left_key_columns} nor {right_key_columns} with {view_id!r}"
        )
    index_groups = _project_rows_to_alignment(
        rows,
        candidate_rows,
        alignment_key_columns=left_key_columns,
        candidate_key_columns=candidate_key_columns,
        label=f"alignment scope {alignment_id}:{view_id}",
    )
    rows = _project_metadata_to_alignment_rows(
        rows,
        candidate_rows,
        index_groups=index_groups,
        label=f"alignment scope {alignment_id}:{view_id}",
    )

    aligned_matrix = np.vstack(
        [aggregate_rows(matrix, list(row[index_field]), mode=mode) for row in mapping_rows]
    ).astype(np.float32, copy=False)
    return np.ascontiguousarray(aligned_matrix), rows, "alignment_set", alignment_id


def resolve_view_scope(
    context: WorkspaceContext,
    *,
    view_id: str,
    sample_id: str | None,
    alignment_id: str | None,
) -> tuple[np.ndarray, pa.Table, str, str | None]:
    if sample_id and alignment_id:
        raise ContractViolationError("scope-aware view access accepts at most one scope of sample or alignment")
    if alignment_id:
        return _alignment_scope(context, view_id, alignment_id=alignment_id)
    if sample_id:
        return _sample_scope(context, view_id, sample_id=sample_id)
    return _full_scope(context, view_id)


def resolve_feature_scope(
    context: WorkspaceContext,
    *,
    view_id: str | None,
    reduced_view_id: str | None,
    sample_id: str | None,
    alignment_id: str | None,
) -> tuple[np.ndarray, pa.Table, str, str | None]:
    if reduced_view_id is not None:
        if sample_id is not None or alignment_id is not None:
            raise ContractViolationError(
                "reduced-view scoped access does not accept sample or alignment; "
                "reduce the scoped view first and pass --reduced-view"
            )
        matrix_path, rows_path, _, _, artifact_id = _matrix_source_paths(
            context,
            view_id=None,
            reduced_view_id=reduced_view_id,
        )
        return (
            np.asarray(read_matrix(matrix_path), dtype=np.float32),
            read_table(rows_path),
            "reduced_view",
            artifact_id,
        )
    if view_id is None:
        raise ContractViolationError("scope-aware matrix access requires exactly one of view_id or reduced_view_id")
    return resolve_view_scope(
        context,
        view_id=view_id,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )


def scope_input_digest_path(
    context: WorkspaceContext,
    *,
    view_id: str | None,
    reduced_view_id: str | None,
    sample_id: str | None,
    alignment_id: str | None,
) -> tuple[str, str, Path]:
    if reduced_view_id is not None:
        if sample_id is not None or alignment_id is not None:
            raise ContractViolationError(
                "reduced-view digest scope does not accept sample or alignment; "
                "reduce the scoped view first and pass --reduced-view"
            )
        _, rows_path, _, artifact_kind, artifact_id = _matrix_source_paths(
            context,
            view_id=None,
            reduced_view_id=reduced_view_id,
        )
        return artifact_kind, artifact_id, rows_path
    if alignment_id is not None:
        return "alignment_set", alignment_id, context.output_root / "alignments" / alignment_id / "rows.parquet"
    if sample_id is not None:
        return "sample_set", sample_id, context.output_root / "samples" / sample_id / "rows.parquet"
    return "view_rows", str(view_id), context.output_root / "views" / str(view_id) / "rows.parquet"
