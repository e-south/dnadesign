"""
Matrix export builders for latentdna.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ..alignments.aggregators import aggregate_rows
from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..contracts.workspace import ReducedViewExportBlockConfig, TableColumnsExportBlockConfig
from ..io.matrix_io import read_matrix, write_matrix
from ..io.parquet_io import read_table, write_table
from ..workspaces.loader import WorkspaceContext


@dataclass(frozen=True, slots=True)
class ExportBlockPayload:
    matrix: np.ndarray
    feature_rows: list[dict[str, object]]
    block_row: list[dict[str, object]]


def _resolve_rows_artifact(context: WorkspaceContext, artifact_id: str) -> tuple[Path, pa.Table]:
    candidates = [
        context.output_root / "alignments" / artifact_id / "rows.parquet",
        context.output_root / "samples" / artifact_id / "rows.parquet",
        context.output_root / "reduced_views" / artifact_id / "rows.parquet",
        context.output_root / "views" / artifact_id / "rows.parquet",
        context.output_root / "scalars" / artifact_id / "table.parquet",
        context.output_root / "distances" / artifact_id / "table.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path, read_table(path)
    raise MissingArtifactError(f"row-basis artifact not found: {artifact_id}")


def _resolve_reduced_view(context: WorkspaceContext, artifact_id: str) -> tuple[Path, Path, np.ndarray, pa.Table]:
    matrix_path = context.output_root / "reduced_views" / artifact_id / "matrix.npy"
    rows_path = context.output_root / "reduced_views" / artifact_id / "rows.parquet"
    if not matrix_path.exists() or not rows_path.exists():
        raise MissingArtifactError(f"reduced_view artifact not found: {artifact_id}")
    return matrix_path, rows_path, np.asarray(read_matrix(matrix_path), dtype=np.float32), read_table(rows_path)


def _resolve_table_artifact(context: WorkspaceContext, artifact_id: str) -> tuple[Path, pa.Table]:
    candidates = [
        context.output_root / "scalars" / artifact_id / "table.parquet",
        context.output_root / "distances" / artifact_id / "table.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path, read_table(path)
    raise MissingArtifactError(f"table artifact not found: {artifact_id}")


def _shared_basis_columns(left: pa.Table, right: pa.Table) -> list[str]:
    return [name for name in left.column_names if name in right.column_names]


def _assert_row_alignment(basis: pa.Table, candidate: pa.Table, *, label: str) -> None:
    shared_columns = _shared_basis_columns(basis, candidate)
    if not shared_columns:
        raise ContractViolationError(f"{label} shares no row-basis columns with the export row basis")
    if basis.num_rows != candidate.num_rows:
        raise ContractViolationError(
            f"{label} row count {candidate.num_rows} does not match export row basis {basis.num_rows}"
        )
    if not basis.select(shared_columns).equals(candidate.select(shared_columns), check_metadata=False):
        raise ContractViolationError(f"{label} row ordering does not match the export row basis")


def _require_columns(table: pa.Table, columns: list[str], *, label: str) -> None:
    missing = [column for column in columns if column not in table.column_names]
    if missing:
        raise ContractViolationError(f"{label} is missing required columns: {missing}")


def _load_alignment_projection(
    context: WorkspaceContext,
    *,
    alignment_id: str,
) -> tuple[Path, pa.Table, list[str], list[str]]:
    alignment_dir = context.output_root / "alignments" / alignment_id
    manifest_path = alignment_dir / "manifest.json"
    rows_path = alignment_dir / "rows.parquet"
    if not manifest_path.exists() or not rows_path.exists():
        raise MissingArtifactError(f"alignment artifact not found for export projection: {alignment_id}")
    manifest = context.read_manifest(manifest_path)
    left_key_columns = [str(name) for name in manifest["params"]["key_columns"]]
    right_key_columns = [str(name) for name in manifest["params"].get("right_key_columns", left_key_columns)]
    return rows_path, read_table(rows_path), left_key_columns, right_key_columns


def _group_candidate_rows(candidate_rows: pa.Table, *, key_columns: list[str]) -> dict[tuple[Any, ...], list[int]]:
    _require_columns(candidate_rows, key_columns, label="export alignment candidate")
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(candidate_rows.select(key_columns).to_pylist()):
        key = tuple(row[name] for name in key_columns)
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
    missing: list[tuple[Any, ...]] = []
    for row in alignment_rows.select(alignment_key_columns).to_pylist():
        key = tuple(row[name] for name in alignment_key_columns)
        indices = grouped.get(key)
        if indices is None:
            missing.append(key)
            continue
        index_groups.append(indices)
    if missing:
        raise ContractViolationError(f"{label} is missing rows for aligned keys: {missing[:5]}")
    return index_groups


def resolve_export_basis(context: WorkspaceContext, *, export_id: str) -> tuple[Path, pa.Table]:
    export = context.require_export(export_id)
    return _resolve_rows_artifact(context, export.row_basis)


def _reduced_view_block_payload(
    context: WorkspaceContext,
    *,
    block: ReducedViewExportBlockConfig,
    block_order: int,
    basis_table: pa.Table,
) -> ExportBlockPayload:
    source_path, rows_path, matrix, rows_table = _resolve_reduced_view(context, block.source)
    alignment_path: Path | None = None
    if block.alignment is not None:
        alignment_path, aligned_rows, left_key_columns, right_key_columns = _load_alignment_projection(
            context,
            alignment_id=block.alignment,
        )
        candidate_key_columns = (
            left_key_columns
            if set(left_key_columns).issubset(set(rows_table.column_names))
            else right_key_columns
            if set(right_key_columns).issubset(set(rows_table.column_names))
            else []
        )
        if not candidate_key_columns:
            raise ContractViolationError(
                f"export block {block.block_id} shares neither alignment key columns "
                f"{left_key_columns} nor {right_key_columns} with the candidate rows"
            )
        index_groups = _project_rows_to_alignment(
            aligned_rows,
            rows_table,
            alignment_key_columns=left_key_columns,
            candidate_key_columns=candidate_key_columns,
            label=f"export block {block.block_id}",
        )
        matrix = np.vstack(
            [aggregate_rows(matrix, indices, mode=block.alignment_aggregation) for indices in index_groups]
        ).astype(np.float32, copy=False)
        matrix = np.ascontiguousarray(matrix)
        rows_table = aligned_rows
    _assert_row_alignment(basis_table, rows_table, label=f"export block {block.block_id}")
    feature_rows = [
        {
            "feature_name": f"{block.feature_prefix}_pc_{feature_order + 1:03d}",
            "block_id": block.block_id,
            "block_order": block_order,
            "feature_order": feature_order + 1,
            "source_artifact_id": block.source,
            "source_column": f"component_{feature_order + 1:03d}",
        }
        for feature_order in range(matrix.shape[1])
    ]
    return ExportBlockPayload(
        matrix=matrix,
        feature_rows=feature_rows,
        block_row=[
            {
                "block_id": block.block_id,
                "source_artifact_id": block.source,
                "source_path": source_path.as_posix(),
                "rows_path": rows_path.as_posix(),
                "alignment_id": block.alignment,
                "alignment_path": None if alignment_path is None else alignment_path.as_posix(),
                "alignment_aggregation": block.alignment_aggregation if block.alignment is not None else None,
            }
        ],
    )


def _table_columns_block_payload(
    context: WorkspaceContext,
    *,
    block: TableColumnsExportBlockConfig,
    block_order: int,
    basis_table: pa.Table,
) -> ExportBlockPayload:
    source_path, table = _resolve_table_artifact(context, block.source)
    _require_columns(table, block.columns, label=f"export block {block.block_id}")
    matrix = np.column_stack([np.asarray(table[column].to_pylist(), dtype=np.float32) for column in block.columns])
    rows_table = table
    alignment_path = None
    if block.alignment is not None:
        alignment_path, aligned_rows, left_key_columns, right_key_columns = _load_alignment_projection(
            context,
            alignment_id=block.alignment,
        )
        candidate_key_columns = (
            left_key_columns
            if set(left_key_columns).issubset(set(rows_table.column_names))
            else right_key_columns
            if set(right_key_columns).issubset(set(rows_table.column_names))
            else []
        )
        if not candidate_key_columns:
            raise ContractViolationError(
                f"export block {block.block_id} shares neither alignment key columns "
                f"{left_key_columns} nor {right_key_columns} with the candidate rows"
            )
        index_groups = _project_rows_to_alignment(
            aligned_rows,
            rows_table,
            alignment_key_columns=left_key_columns,
            candidate_key_columns=candidate_key_columns,
            label=f"export block {block.block_id}",
        )
        matrix = np.vstack(
            [aggregate_rows(matrix, indices, mode=block.alignment_aggregation) for indices in index_groups]
        )
        rows_table = aligned_rows
    _assert_row_alignment(basis_table, rows_table, label=f"export block {block.block_id}")
    feature_rows = []
    for feature_order, column in enumerate(block.columns, start=1):
        feature_name = column if block.feature_prefix is None else f"{block.feature_prefix}_{column}"
        feature_rows.append(
            {
                "feature_name": feature_name,
                "block_id": block.block_id,
                "block_order": block_order,
                "feature_order": feature_order,
                "source_artifact_id": block.source,
                "source_column": column,
            }
        )
    return ExportBlockPayload(
        matrix=np.ascontiguousarray(matrix, dtype=np.float32),
        feature_rows=feature_rows,
        block_row=[
            {
                "block_id": block.block_id,
                "source_artifact_id": block.source,
                "source_path": source_path.as_posix(),
                "rows_path": source_path.as_posix(),
                "alignment_id": block.alignment,
                "alignment_path": None if alignment_path is None else alignment_path.as_posix(),
                "alignment_aggregation": block.alignment_aggregation if block.alignment is not None else None,
            }
        ],
    )


def resolve_export_blocks(
    context: WorkspaceContext,
    *,
    export_id: str,
) -> tuple[Path, pa.Table, list[ExportBlockPayload]]:
    export = context.require_export(export_id)
    basis_path, basis_table = resolve_export_basis(context, export_id=export_id)

    blocks: list[ExportBlockPayload] = []
    for block_order, block in enumerate(export.blocks, start=1):
        if isinstance(block, ReducedViewExportBlockConfig):
            blocks.append(
                _reduced_view_block_payload(
                    context,
                    block=block,
                    block_order=block_order,
                    basis_table=basis_table,
                )
            )
        elif isinstance(block, TableColumnsExportBlockConfig):
            blocks.append(
                _table_columns_block_payload(
                    context,
                    block=block,
                    block_order=block_order,
                    basis_table=basis_table,
                )
            )
        else:  # pragma: no cover - constrained by config model
            raise ContractViolationError(f"unsupported export block kind: {block.kind}")
    return basis_path, basis_table, blocks


def build_export_matrix_artifact(
    context: WorkspaceContext,
    *,
    export_id: str,
) -> tuple[Path, Path, int, int, list[dict[str, object]], list[dict[str, object]]]:
    basis_path, basis_table, blocks = resolve_export_blocks(context, export_id=export_id)
    matrices = [block.matrix for block in blocks]
    feature_rows = [row for block in blocks for row in block.feature_rows]
    block_rows = [row for block in blocks for row in block.block_row]
    feature_names = [str(row["feature_name"]) for row in feature_rows]
    duplicates = sorted(name for name, count in Counter(feature_names).items() if count > 1)
    if duplicates:
        raise ContractViolationError(f"export {export_id} defines duplicate feature names: {duplicates[:5]}")
    export = context.require_export(export_id)
    export_matrix = np.ascontiguousarray(np.column_stack(matrices), dtype=export.matrix_dtype or context.analysis_dtype)
    export_dir = context.output_root / "exports" / export_id
    write_matrix(export_dir / "matrix.npy", export_matrix)
    write_table(basis_table, export_dir / "rows.parquet")
    write_table(pa.Table.from_pylist(feature_rows), export_dir / "features.parquet")
    return export_dir, basis_path, export_matrix.shape[0], export_matrix.shape[1], feature_rows, block_rows
