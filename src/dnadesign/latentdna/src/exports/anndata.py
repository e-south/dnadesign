"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/exports/anndata.py

AnnData export builders for latentdna.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import sparse

from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..io.matrix_io import read_matrix
from ..io.parquet_io import read_table, write_table
from ..workspaces.loader import WorkspaceContext
from .matrix import _append_metadata_columns, _assert_row_alignment, resolve_export_blocks


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def _json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=_json_default)


def _safe_key(value: str) -> str:
    candidate = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
    candidate = re.sub(r"_+", "_", candidate).strip("_")
    return candidate or "artifact"


def _is_nested_metadata(value: object) -> bool:
    return isinstance(value, (list, tuple, dict, set))


def _is_missing_scalar(value: object) -> bool:
    if _is_nested_metadata(value):
        return False
    return value is None or bool(pd.isna(value))


def _sanitize_frame_for_h5ad(frame: pd.DataFrame) -> pd.DataFrame:
    sanitized = frame.copy()
    for column in sanitized.columns:
        series = sanitized[column]
        if series.map(_is_nested_metadata).any():
            sanitized[column] = series.map(
                lambda value: None
                if _is_missing_scalar(value)
                else _json_text(value)
                if _is_nested_metadata(value)
                else value
            ).astype(object)
            continue
        if pd.api.types.is_string_dtype(series.dtype):
            sanitized[column] = series.map(lambda value: None if _is_missing_scalar(value) else str(value)).astype(
                object
            )
        elif isinstance(sanitized[column].dtype, pd.ArrowDtype):
            sanitized[column] = sanitized[column].astype(object)
    return sanitized


def _table_to_obs(table: pa.Table) -> pd.DataFrame:
    frame = _sanitize_frame_for_h5ad(table.to_pandas())
    for column in ["sequence_id", "record_id", "record_key", "id", "anchor_id"]:
        if column not in frame.columns:
            continue
        values = frame[column].astype(str)
        if values.is_unique and not frame[column].isna().any():
            frame.index = pd.Index([str(value) for value in frame[column].tolist()], dtype=object, name=None)
            return frame
    frame.index = pd.Index([f"row_{index + 1:08d}" for index in range(len(frame))], dtype=object, name=None)
    return frame


def _feature_frame(feature_rows: list[dict[str, object]]) -> pd.DataFrame:
    feature_names = [str(row["feature_name"]) for row in feature_rows]
    duplicates = sorted(name for name, count in Counter(feature_names).items() if count > 1)
    if duplicates:
        raise ContractViolationError(f"AnnData export defines duplicate feature names: {duplicates[:5]}")
    frame = _sanitize_frame_for_h5ad(pd.DataFrame.from_records(feature_rows))
    frame.index = pd.Index(feature_names, dtype=object, name="feature_name")
    return frame


def _projection_key(projection_id: str, manifest: dict[str, Any]) -> str:
    params = manifest.get("params", {})
    method = params.get("method", "projection") if isinstance(params, dict) else "projection"
    return f"X_{_safe_key(str(method))}_{_safe_key(projection_id)}"


def _load_projection(
    context: WorkspaceContext,
    *,
    projection_id: str,
    basis_table: pa.Table,
) -> tuple[str, np.ndarray, Path]:
    projection_dir = context.output_root / "projections" / projection_id
    coords_path = projection_dir / "coords.parquet"
    manifest_path = projection_dir / "manifest.json"
    if not coords_path.exists() or not manifest_path.exists():
        raise MissingArtifactError(f"projection artifact not found for AnnData export: {projection_id}")
    coords_table = read_table(coords_path)
    _assert_row_alignment(basis_table, coords_table, label=f"AnnData projection {projection_id}")
    for column in ["x", "y"]:
        if column not in coords_table.column_names:
            raise ContractViolationError(f"AnnData projection {projection_id} is missing coordinate column: {column}")
    coords = np.column_stack(
        [
            np.asarray(coords_table["x"].to_pylist(), dtype=np.float32),
            np.asarray(coords_table["y"].to_pylist(), dtype=np.float32),
        ]
    )
    manifest = context.read_manifest(manifest_path)
    return _projection_key(projection_id, manifest), np.ascontiguousarray(coords), coords_path


def _load_neighbor_distances(
    context: WorkspaceContext,
    *,
    neighbor_id: str,
    basis_table: pa.Table,
) -> tuple[str, sparse.csr_matrix, Path, Path, Path]:
    neighbor_dir = context.output_root / "neighbors" / neighbor_id
    rows_path = neighbor_dir / "rows.parquet"
    indices_path = neighbor_dir / "indices.npy"
    distances_path = neighbor_dir / "distances.npy"
    manifest_path = neighbor_dir / "manifest.json"
    if not rows_path.exists() or not indices_path.exists() or not distances_path.exists() or not manifest_path.exists():
        raise MissingArtifactError(f"neighbor artifact not found for AnnData export: {neighbor_id}")
    rows_table = read_table(rows_path)
    _assert_row_alignment(basis_table, rows_table, label=f"AnnData neighbors {neighbor_id}")
    indices = np.asarray(read_matrix(indices_path), dtype=np.int64)
    distances = np.asarray(read_matrix(distances_path), dtype=np.float32)
    if indices.shape != distances.shape:
        raise ContractViolationError(
            f"AnnData neighbors {neighbor_id} indices shape {indices.shape} does not match distances {distances.shape}"
        )
    if indices.ndim != 2 or indices.shape[0] != basis_table.num_rows:
        raise ContractViolationError(
            f"AnnData neighbors {neighbor_id} must be a two-dimensional row-aligned neighbor matrix"
        )
    row_count, k = indices.shape
    if np.any(indices < 0) or np.any(indices >= row_count):
        raise ContractViolationError(f"AnnData neighbors {neighbor_id} contains out-of-range neighbor indices")
    row_indices = np.repeat(np.arange(row_count, dtype=np.int64), k)
    graph = sparse.csr_matrix(
        (distances.reshape(-1), (row_indices, indices.reshape(-1))),
        shape=(row_count, row_count),
        dtype=np.float32,
    )
    return f"neighbors_{_safe_key(neighbor_id)}_distances", graph, rows_path, indices_path, distances_path


def build_export_anndata_artifact(
    context: WorkspaceContext,
    *,
    export_id: str,
    projection_ids: list[str] | None = None,
    neighbor_ids: list[str] | None = None,
) -> tuple[Path, Path, Path, int, int, list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    basis_path, basis_table, blocks = resolve_export_blocks(context, export_id=export_id)
    export = context.require_export(export_id)
    basis_table = _append_metadata_columns(basis_table, blocks=blocks, required_columns=list(export.metadata_columns))
    matrices = [block.matrix for block in blocks]
    feature_rows = [row for block in blocks for row in block.feature_rows]
    block_rows = [row for block in blocks for row in block.block_row]
    export_matrix = np.ascontiguousarray(np.column_stack(matrices), dtype=export.matrix_dtype or context.analysis_dtype)

    obs = _table_to_obs(basis_table)
    var = _feature_frame(feature_rows)
    adata = ad.AnnData(X=export_matrix, obs=obs, var=var)
    adata.uns["latentdna_export"] = {
        "schema_version": "latentdna.anndata_export.v1",
        "workspace_id": context.workspace_id,
        "export_id": export_id,
        "row_basis": export.row_basis,
        "basis_path": basis_path.as_posix(),
        "block_count": len(export.blocks),
        "blocks_json": _json_text(block_rows),
        "matrix_dtype": str(export_matrix.dtype),
        "source_of_truth": "LatentDNA rows.parquet/features.parquet/export block artifacts",
    }

    supplemental_inputs: list[dict[str, object]] = []
    for projection_id in projection_ids or []:
        key, coords, coords_path = _load_projection(context, projection_id=projection_id, basis_table=basis_table)
        adata.obsm[key] = coords
        supplemental_inputs.append(
            {
                "kind": "projection",
                "id": projection_id,
                "path": coords_path.as_posix(),
                "anndata_slot": "obsm",
                "anndata_key": key,
            }
        )
    for neighbor_id in neighbor_ids or []:
        key, graph, rows_path, indices_path, distances_path = _load_neighbor_distances(
            context,
            neighbor_id=neighbor_id,
            basis_table=basis_table,
        )
        adata.obsp[key] = graph
        supplemental_inputs.append(
            {
                "kind": "neighbor_set",
                "id": neighbor_id,
                "rows_path": rows_path.as_posix(),
                "indices_path": indices_path.as_posix(),
                "distances_path": distances_path.as_posix(),
                "anndata_slot": "obsp",
                "anndata_key": key,
            }
        )
    if supplemental_inputs:
        adata.uns["latentdna_export"]["supplemental_inputs_json"] = _json_text(supplemental_inputs)

    export_dir = context.output_root / "exports" / export_id
    bundle_path = export_dir / "bundle.h5ad"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    write_table(basis_table, export_dir / "rows.parquet")
    write_table(pa.Table.from_pylist(feature_rows), export_dir / "features.parquet")
    adata.write_h5ad(bundle_path)
    return (
        export_dir,
        basis_path,
        bundle_path,
        adata.n_obs,
        adata.n_vars,
        feature_rows,
        block_rows,
        supplemental_inputs,
    )
