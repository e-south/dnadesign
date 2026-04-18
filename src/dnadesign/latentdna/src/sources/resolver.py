"""
Source resolution and inspection helpers for latentdna.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts.errors import SourceResolutionError, WorkspaceValidationError
from ..contracts.workspace import MatrixBundleSourceConfig, ParquetSourceConfig, SourceConfig, USRSourceConfig
from ..io.hashing import sha256_payload
from ..io.matrix_io import read_matrix
from ..io.parquet_io import read_row_count, read_schema, read_table
from . import parquet_source, usr_source
from .provenance import source_provenance_digest


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    source_id: str
    source: SourceConfig
    workspace_dir: Path
    records_path: Path | None
    matrix_path: Path | None = None
    rows_path: Path | None = None
    manifest_path: Path | None = None


def _load_usr_dataset(resolved: ResolvedSource):
    source = resolved.source
    if not isinstance(source, USRSourceConfig):
        raise SourceResolutionError(f"source {resolved.source_id} is not a USR dataset")
    return usr_source.load_dataset(source.root, source.dataset, workspace_dir=resolved.workspace_dir)


def resolve_source(source_id: str, source: SourceConfig, *, workspace_dir: Path) -> ResolvedSource:
    if isinstance(source, USRSourceConfig):
        records = usr_source.records_path(source.root, source.dataset, workspace_dir=workspace_dir)
        return ResolvedSource(source_id=source_id, source=source, workspace_dir=workspace_dir, records_path=records)
    if isinstance(source, ParquetSourceConfig):
        records = parquet_source.records_path(source.path, workspace_dir=workspace_dir)
        return ResolvedSource(source_id=source_id, source=source, workspace_dir=workspace_dir, records_path=records)
    if isinstance(source, MatrixBundleSourceConfig):
        bundle = parquet_source.records_path(source.path, workspace_dir=workspace_dir)
        rows_path = bundle / "rows.parquet"
        matrix_path = bundle / "matrix.npy"
        if not matrix_path.exists():
            matrix_path = bundle / "matrix.npz"
        return ResolvedSource(
            source_id=source_id,
            source=source,
            workspace_dir=workspace_dir,
            records_path=None,
            matrix_path=matrix_path,
            rows_path=rows_path,
            manifest_path=bundle / "manifest.json",
        )
    raise SourceResolutionError(f"unsupported source kind for {source_id}: {source.kind}")


def require_records_path(resolved: ResolvedSource) -> Path:
    if resolved.records_path is None:
        raise SourceResolutionError(f"source {resolved.source_id} does not expose a records.parquet table")
    if not resolved.records_path.exists():
        raise SourceResolutionError(
            f"records.parquet not found for source {resolved.source_id}: {resolved.records_path}"
        )
    return resolved.records_path


def require_matrix_bundle_paths(resolved: ResolvedSource) -> tuple[Path, Path, Path]:
    rows_path = resolved.rows_path
    matrix_path = resolved.matrix_path
    manifest_path = resolved.manifest_path
    if rows_path is None or matrix_path is None or manifest_path is None:
        raise SourceResolutionError(f"matrix bundle source {resolved.source_id} is missing rows/matrix/manifest paths")
    if not rows_path.exists():
        raise SourceResolutionError(f"rows.parquet not found for source {resolved.source_id}: {rows_path}")
    if not matrix_path.exists():
        raise SourceResolutionError(f"matrix file not found for source {resolved.source_id}: {matrix_path}")
    if not manifest_path.exists():
        raise SourceResolutionError(f"manifest.json not found for source {resolved.source_id}: {manifest_path}")
    return rows_path, matrix_path, manifest_path


def inspect_source_schema(resolved: ResolvedSource) -> dict[str, Any]:
    if resolved.records_path is not None:
        records_path = require_records_path(resolved)
        if isinstance(resolved.source, USRSourceConfig):
            schema = _load_usr_dataset(resolved).schema()
        else:
            schema = read_schema(records_path)
        vector_columns = [
            field.name
            for field in schema
            if pa.types.is_list(field.type)
            or pa.types.is_large_list(field.type)
            or pa.types.is_fixed_size_list(field.type)
        ]
        return {
            "path": records_path.as_posix(),
            "row_count": read_row_count(records_path),
            "columns": [field.name for field in schema],
            "vector_columns": vector_columns,
        }

    rows_path, matrix_path, _ = require_matrix_bundle_paths(resolved)
    schema = read_schema(rows_path)
    matrix = np.asarray(read_matrix(matrix_path))
    if matrix.ndim != 2:
        raise SourceResolutionError(f"matrix bundle source {resolved.source_id} must expose a 2D matrix")
    if matrix.shape[0] != read_row_count(rows_path):
        raise SourceResolutionError(
            f"matrix bundle source {resolved.source_id} row count mismatch: "
            f"matrix has {matrix.shape[0]} rows but rows.parquet has {read_row_count(rows_path)}"
        )
    return {
        "path": rows_path.parent.as_posix(),
        "row_count": read_row_count(rows_path),
        "columns": [field.name for field in schema],
        "vector_columns": ["bundle_matrix"],
    }


def read_records_table(
    resolved: ResolvedSource,
    *,
    columns: list[str] | None = None,
) -> pa.Table:
    if isinstance(resolved.source, USRSourceConfig):
        dataset = _load_usr_dataset(resolved)
        batches = list(
            dataset.scan(
                columns=columns,
                include_overlays=True,
                include_deleted=False,
                batch_size=65536,
            )
        )
        if batches:
            return pa.Table.from_batches(batches)
        schema = dataset.schema()
        if columns is not None:
            fields = [schema.field(name) for name in columns]
            schema = pa.schema(fields)
        return pa.Table.from_batches([], schema=schema)
    records_path = require_records_path(resolved)
    return read_table(records_path, columns=columns)


def iter_records_batches(
    resolved: ResolvedSource,
    *,
    columns: list[str] | None = None,
    batch_size: int = 4096,
):
    if isinstance(resolved.source, USRSourceConfig):
        dataset = _load_usr_dataset(resolved)
        yield from dataset.scan(
            columns=columns,
            include_overlays=True,
            include_deleted=False,
            batch_size=batch_size,
        )
        return
    records_path = require_records_path(resolved)
    yield from pq.ParquetFile(records_path).iter_batches(columns=columns, batch_size=batch_size)


def source_provenance(
    resolved: ResolvedSource,
    *,
    columns: list[str] | None = None,
) -> list[dict[str, object]]:
    if isinstance(resolved.source, USRSourceConfig):
        return usr_source.source_provenance(
            resolved.source.root,
            resolved.source.dataset,
            workspace_dir=resolved.workspace_dir,
            columns=columns,
        )
    if isinstance(resolved.source, ParquetSourceConfig):
        return parquet_source.source_provenance(resolved.source.path, workspace_dir=resolved.workspace_dir)
    if isinstance(resolved.source, MatrixBundleSourceConfig):
        rows_path, matrix_path, manifest_path = require_matrix_bundle_paths(resolved)
        return [
            {
                "kind": "file",
                "id": rows_path.name,
                "path": rows_path.as_posix(),
                "role": "rows",
            },
            {
                "kind": "file",
                "id": matrix_path.name,
                "path": matrix_path.as_posix(),
                "role": "matrix",
            },
            {
                "kind": "file",
                "id": manifest_path.name,
                "path": manifest_path.as_posix(),
                "role": "manifest",
            },
        ]
    raise SourceResolutionError(f"unsupported source kind for provenance: {resolved.source.kind}")


def source_digest(
    resolved: ResolvedSource,
    *,
    columns: list[str] | None = None,
) -> tuple[str, list[dict[str, object]], dict[str, str]]:
    entries = source_provenance(resolved, columns=columns)
    serialized: list[dict[str, object]] = []
    input_digests: dict[str, str] = {}
    for entry in entries:
        digest = source_provenance_digest(entry)
        item = dict(entry)
        item["digest"] = digest
        serialized.append(item)
        input_digests[f"{item['role']}:{item['id']}"] = digest
    composite_digest = sha256_payload(
        [
            {
                "kind": item["kind"],
                "id": item["id"],
                "role": item["role"],
                "path": item["path"],
                "namespace": item.get("namespace"),
                "columns": item.get("columns"),
                "digest_mode": item.get("digest_mode"),
                "digest": item["digest"],
            }
            for item in serialized
        ]
    )
    return composite_digest, serialized, input_digests


def ensure_unique_keys(table: pa.Table, *, key_names: list[str], label: str) -> None:
    seen: set[tuple[Any, ...]] = set()
    for row in table.select(key_names).to_pylist():
        key = tuple(row[name] for name in key_names)
        if key in seen:
            raise WorkspaceValidationError(f"{label} must be unique for keys {key_names}: duplicate {key!r}")
        seen.add(key)
