"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/snapshots/build.py

Snapshot builders for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from ..io.parquet_io import read_schema, read_table, write_table
from ..sources.resolver import ensure_unique_keys, inspect_source_schema, read_records_table, resolve_source
from ..workspaces.loader import WorkspaceContext


def build_snapshot_artifact(
    context: WorkspaceContext,
    *,
    snapshot_id: str,
    source_id: str,
) -> tuple[Path, Path, int, list[str], list[str]]:
    source = context.require_source(source_id)
    resolved = resolve_source(source_id, source, workspace_dir=context.workspace_dir)
    row_columns = list(dict.fromkeys([source.record_key, source.subject_key, source.context_key]))
    if resolved.records_path is not None:
        available_columns = set(inspect_source_schema(resolved)["columns"])
        row_columns = [column for column in row_columns if column is not None and column in available_columns]
        metadata_columns = list(
            dict.fromkeys(
                [
                    *row_columns,
                    *(context.config.metadata.include or []),
                    *(source.metadata_include or []),
                ]
            )
        )
        metadata_columns = [column for column in metadata_columns if column in available_columns]
        metadata_table = read_records_table(resolved, columns=metadata_columns)
    else:
        assert resolved.rows_path is not None
        available_columns = set(read_schema(resolved.rows_path).names)
        row_columns = [column for column in row_columns if column is not None and column in available_columns]
        metadata_columns = list(
            dict.fromkeys(
                [
                    *row_columns,
                    *(context.config.metadata.include or []),
                    *(source.metadata_include or []),
                ]
            )
        )
        metadata_columns = [column for column in metadata_columns if column in available_columns]
        metadata_table = read_table(resolved.rows_path, columns=metadata_columns)

    row_table = metadata_table.select(row_columns)
    ensure_unique_keys(row_table, key_names=[source.record_key], label=f"source {source_id} record_key")

    artifact_dir = context.output_root / "snapshots" / snapshot_id
    write_table(
        row_table if isinstance(row_table, pa.Table) else pa.Table.from_pylist(row_table.to_pylist()),
        artifact_dir / "rows.parquet",
    )
    write_table(
        metadata_table if isinstance(metadata_table, pa.Table) else pa.Table.from_pylist(metadata_table.to_pylist()),
        artifact_dir / "metadata.parquet",
    )
    return artifact_dir, resolved.records_path or resolved.rows_path, row_table.num_rows, row_columns, metadata_columns
