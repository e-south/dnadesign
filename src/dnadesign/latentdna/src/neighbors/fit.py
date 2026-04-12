"""
Neighbor artifact builders for latentdna.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..contracts.errors import BackendUnavailableError, ContractViolationError
from ..io.matrix_io import write_matrix
from ..io.parquet_io import write_table
from ..views.scopes import resolve_view_scope
from ..workspaces.loader import WorkspaceContext
from .backends import fit_neighbors_with_backend
from .backends.approximate import approximate_backend_available

_SUPPORTED_METRICS = {"euclidean", "cosine"}
_FULL_VIEW_EXACT_ROW_LIMIT = 5000


def fit_neighbor_artifact(
    context: WorkspaceContext,
    *,
    neighbor_id: str,
    view_id: str,
    k: int,
    metric: str,
    backend: str,
    seed: int,
    sample_id: str | None,
    alignment_id: str | None,
) -> tuple[Path, int, str, bool, str, str | None]:
    matrix, rows, scope_kind, scope_id = resolve_view_scope(
        context,
        view_id=view_id,
        sample_id=sample_id,
        alignment_id=alignment_id,
    )

    row_count = int(rows.num_rows)
    if row_count < 2:
        raise ContractViolationError("neighbor fitting requires at least 2 rows")
    if k < 1 or k >= row_count:
        raise ContractViolationError(f"neighbor count k must be between 1 and {row_count - 1}, got {k}")
    if metric not in _SUPPORTED_METRICS:
        raise ContractViolationError(f"unsupported neighbor metric: {metric!r}")
    if backend == "exact" and scope_kind == "full_view" and row_count > _FULL_VIEW_EXACT_ROW_LIMIT:
        raise ContractViolationError(
            "exact full-view neighbors are disabled above 5000 rows; use --sample, --alignment, or backend=approximate"
        )
    if backend == "auto" and scope_kind == "full_view" and row_count > _FULL_VIEW_EXACT_ROW_LIMIT:
        if not approximate_backend_available():
            raise BackendUnavailableError(
                "approximate neighbor backend is unavailable and exact full-view neighbors are disabled above 5000 rows"
            )

    indices, distances, resolved_backend, approximate = fit_neighbors_with_backend(
        np.ascontiguousarray(np.asarray(matrix, dtype=np.float32)),
        k=k,
        metric=metric,
        backend=backend,
        seed=seed,
    )

    artifact_dir = context.output_root / "neighbors" / neighbor_id
    write_matrix(artifact_dir / "indices.npy", indices)
    write_matrix(artifact_dir / "distances.npy", distances)
    write_table(rows, artifact_dir / "rows.parquet")
    return artifact_dir, row_count, resolved_backend, approximate, scope_kind, scope_id
