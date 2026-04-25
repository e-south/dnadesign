"""
Projection fitting helpers for latentdna.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

from ..contracts.errors import BackendUnavailableError, ContractViolationError
from ..io.matrix_io import read_matrix
from ..io.parquet_io import read_row_count, read_table, write_table
from ..workspaces.loader import WorkspaceContext


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


def _fit_projection_artifact(
    context: WorkspaceContext,
    *,
    view_id: str,
    projection_id: str,
    sample_id: str,
    metric: str,
    seed: int,
    artifact_dir: Path | None = None,
) -> tuple[Path, int]:
    try:
        import umap
    except Exception as exc:  # pragma: no cover - dependency controlled by env
        raise BackendUnavailableError(f"UMAP backend is unavailable: {exc}") from exc

    view_manifest = context.read_manifest(context.output_root / "views" / view_id / "manifest.json")
    sample_manifest = context.read_manifest(context.output_root / "samples" / sample_id / "manifest.json")
    record_key = str(view_manifest["params"]["record_key"])
    sample_params = sample_manifest.get("params", {}) if isinstance(sample_manifest.get("params"), dict) else {}
    sample_strategy = str(sample_params.get("strategy", "unknown"))
    sample_rows_path = context.output_root / "samples" / sample_id / "rows.parquet"

    matrix = read_matrix(context.output_root / "views" / view_id / "matrix.npy")
    view_rows = read_table(context.output_root / "views" / view_id / "rows.parquet").to_pylist()
    use_full_view_directly = sample_strategy == "all" and read_row_count(sample_rows_path) == len(view_rows)
    if use_full_view_directly:
        sample_rows = view_rows
        subset = matrix if matrix.dtype == np.float32 else np.asarray(matrix, dtype=np.float32)
    else:
        sample_rows = read_table(sample_rows_path).to_pylist()
        indices = _ordered_indices(view_rows, sample_rows, record_key=record_key)
        subset = np.asarray(matrix[indices], dtype=np.float32)
    if len(sample_rows) < 3:
        raise ContractViolationError("projection fitting requires at least 3 sampled rows")
    n_neighbors = max(2, min(15, subset.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, metric=metric, n_neighbors=n_neighbors, random_state=seed)
    coords = reducer.fit_transform(subset)

    table = pa.Table.from_pylist(
        [{**row, "x": float(coord[0]), "y": float(coord[1])} for row, coord in zip(sample_rows, coords, strict=True)]
    )
    target_dir = artifact_dir or (context.output_root / "projections" / projection_id)
    write_table(table, target_dir / "coords.parquet")
    return target_dir, table.num_rows
