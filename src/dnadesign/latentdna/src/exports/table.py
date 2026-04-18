"""
Tabular export builders for latentdna.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from ..io.parquet_io import write_table
from ..workspaces.loader import WorkspaceContext
from .matrix import _append_metadata_columns, resolve_export_blocks


def build_export_table_artifact(
    context: WorkspaceContext,
    *,
    export_id: str,
) -> tuple[Path, Path, int, int, list[dict[str, object]], list[dict[str, object]]]:
    basis_path, basis_table, blocks = resolve_export_blocks(context, export_id=export_id)
    export = context.require_export(export_id)
    basis_table = _append_metadata_columns(basis_table, blocks=blocks, required_columns=list(export.metadata_columns))

    arrays = [basis_table[name] for name in basis_table.column_names]
    names = list(basis_table.column_names)
    feature_rows = []
    block_rows = []
    for block in blocks:
        block_rows.extend(block.block_row)
        feature_rows.extend(block.feature_rows)
        for feature_index, feature_row in enumerate(block.feature_rows):
            names.append(str(feature_row["feature_name"]))
            arrays.append(pa.array(block.matrix[:, feature_index].tolist()))

    export_table = pa.Table.from_arrays(arrays, names=names)
    export_dir = context.output_root / "exports" / export_id
    write_table(basis_table, export_dir / "rows.parquet")
    write_table(export_table, export_dir / "table.parquet")
    write_table(pa.Table.from_pylist(feature_rows), export_dir / "features.parquet")
    return export_dir, basis_path, export_table.num_rows, len(feature_rows), feature_rows, block_rows
