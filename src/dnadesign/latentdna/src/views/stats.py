"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/src/views/stats.py

Descriptive stats over persisted view and reduced-view artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..contracts.errors import MissingArtifactError
from ..io.json_io import read_json
from ..io.matrix_io import read_matrix
from ..io.parquet_io import read_table
from ..workspaces.loader import WorkspaceContext


def _artifact_paths(context: WorkspaceContext, view_id: str) -> tuple[str, Path, Path, dict[str, Any]]:
    for artifact_kind, relative_dir in [("view", "views"), ("reduced_view", "reduced_views")]:
        artifact_dir = context.output_root / relative_dir / view_id
        matrix_path = artifact_dir / "matrix.npy"
        rows_path = artifact_dir / "rows.parquet"
        manifest_path = artifact_dir / "manifest.json"
        if matrix_path.exists() and rows_path.exists() and manifest_path.exists():
            return artifact_kind, matrix_path, rows_path, context.read_manifest(manifest_path)
    raise MissingArtifactError(f"view stats artifact not found: {view_id}")


def _reducer_summary(context: WorkspaceContext, reducer_id: str | None) -> dict[str, Any] | None:
    if reducer_id is None:
        return None
    summary_path = context.output_root / "reducers" / reducer_id / "summary.json"
    if not summary_path.exists():
        return None
    return read_json(summary_path)


def compute_view_stats(context: WorkspaceContext, *, view_id: str) -> dict[str, Any]:
    artifact_kind, matrix_path, rows_path, manifest = _artifact_paths(context, view_id)
    matrix = np.asarray(read_matrix(matrix_path), dtype=np.float32)
    rows_table = read_table(rows_path)
    norms = np.linalg.norm(matrix, axis=1) if matrix.size else np.asarray([], dtype=np.float32)
    missing_values = int(np.isnan(matrix).sum())

    params = manifest.get("params", {})
    reducer_id = None
    if artifact_kind == "reduced_view":
        reducer_id = params.get("reducer_id")
    elif params.get("derive_kind") == "apply_reducer":
        reducer_id = params.get("reducer")
    reducer_summary = _reducer_summary(context, reducer_id if isinstance(reducer_id, str) else None)

    payload: dict[str, Any] = {
        "schema_version": "latentdna.view_stats.v1",
        "workspace_id": context.workspace_id,
        "artifact_kind": artifact_kind,
        "artifact_id": view_id,
        "rows": int(matrix.shape[0]),
        "dims": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "dtype": str(matrix.dtype),
        "row_columns": list(rows_table.column_names),
        "coordinate_space_id": params.get("coordinate_space_id"),
        "missing_values": missing_values,
        "missing_fraction": float(missing_values / matrix.size) if matrix.size else 0.0,
        "mean_norm": float(norms.mean()) if norms.size else 0.0,
        "min_norm": float(norms.min()) if norms.size else 0.0,
        "max_norm": float(norms.max()) if norms.size else 0.0,
    }
    if artifact_kind == "reduced_view" and "source_view_id" in params:
        payload["source_view_id"] = params["source_view_id"]
    if reducer_summary is not None:
        payload["explained_variance_ratio"] = reducer_summary.get("explained_variance_ratio", [])
        payload["fit_scope_kind"] = reducer_summary.get("scope_kind")
        payload["fit_scope_id"] = reducer_summary.get("scope_id")
    return payload
