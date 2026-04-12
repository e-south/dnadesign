"""
Neighborhood enrichment artifact builders for latentdna.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..io.json_io import write_json
from ..io.matrix_io import read_matrix
from ..io.parquet_io import read_table, write_table
from ..workspaces.loader import WorkspaceContext


def _select_indices(rows: list[dict[str, Any]], where: dict[str, Any]) -> list[int]:
    column = where.get("column")
    if not isinstance(column, str):
        raise ContractViolationError("landmark where clause requires a 'column' field")
    if "equals" in where:
        target = where["equals"]
        return [index for index, row in enumerate(rows) if row.get(column) == target]
    if "in" in where:
        targets = set(where["in"])
        return [index for index, row in enumerate(rows) if row.get(column) in targets]
    raise ContractViolationError("landmark where clause requires either 'equals' or 'in'")


def score_enrichment_artifact(
    context: WorkspaceContext,
    *,
    enrichment_id: str,
    neighbors_id: str,
    cohort_id: str,
    landmark_ids: list[str],
) -> tuple[Path, int, dict[str, Any]]:
    if not landmark_ids:
        raise ContractViolationError("enrichment scoring requires at least one --landmark")

    neighbor_dir = context.output_root / "neighbors" / neighbors_id
    indices_path = neighbor_dir / "indices.npy"
    rows_path = neighbor_dir / "rows.parquet"
    manifest_path = neighbor_dir / "manifest.json"
    for required in [indices_path, rows_path, manifest_path]:
        if not required.exists():
            raise MissingArtifactError(f"neighbor artifact is missing for enrichment scoring: {required}")

    neighbor_manifest = context.read_manifest(manifest_path)
    view_id = str(neighbor_manifest["params"]["view_id"])
    view = context.require_source_view(view_id)
    cohort = context.require_cohort(cohort_id)
    if cohort.source != view.source:
        raise ContractViolationError(
            f"cohort {cohort_id} uses source {cohort.source!r} but neighbor view {view_id} uses {view.source!r}"
        )

    rows_table = read_table(rows_path)
    if cohort.column not in rows_table.column_names:
        raise ContractViolationError(
            f"cohort column {cohort.column!r} is missing from neighbor rows for {neighbors_id!r}"
        )
    rows = rows_table.to_pylist()
    if not rows:
        raise ContractViolationError("enrichment scoring requires at least one neighbor row")

    neighbor_indices = np.asarray(read_matrix(indices_path, mmap_mode=None), dtype=np.int64)
    if neighbor_indices.ndim != 2 or neighbor_indices.shape[0] != len(rows):
        raise ContractViolationError("neighbor artifact rows and indices are misaligned")

    cohort_counts = Counter(row[cohort.column] for row in rows)
    ordered_cohort_values = sorted(cohort_counts, key=str)
    background_total = len(rows)

    output_rows: list[dict[str, Any]] = []
    landmark_modes: dict[str, str] = {}
    landmark_seed_counts: dict[str, int] = {}
    for landmark_id in landmark_ids:
        landmark = context.require_landmark(landmark_id)
        if landmark.source != view.source:
            raise ContractViolationError(
                "landmark "
                f"{landmark_id} uses source {landmark.source!r} "
                f"but neighbor view {view_id} uses {view.source!r}"
            )
        selector_column = landmark.where.get("column")
        if selector_column not in rows_table.column_names:
            raise ContractViolationError(
                f"landmark selector column {selector_column!r} is missing from neighbor rows for {neighbors_id!r}"
            )
        seed_indices = _select_indices(rows, landmark.where)
        if not seed_indices:
            raise ContractViolationError(f"landmark {landmark_id} matched no rows in neighbor scope {neighbors_id!r}")

        landmark_modes[landmark_id] = landmark.representation.mode
        landmark_seed_counts[landmark_id] = len(seed_indices)

        neighbor_hits = Counter(rows[int(index)][cohort.column] for index in neighbor_indices[seed_indices].ravel())
        neighbor_total = int(len(seed_indices) * neighbor_indices.shape[1])
        for cohort_value in ordered_cohort_values:
            hits = int(neighbor_hits.get(cohort_value, 0))
            background_count = int(cohort_counts[cohort_value])
            neighbor_fraction = 0.0 if neighbor_total == 0 else hits / neighbor_total
            background_fraction = background_count / background_total
            enrichment_ratio = 0.0 if background_fraction == 0 else neighbor_fraction / background_fraction
            output_rows.append(
                {
                    "landmark_id": landmark_id,
                    "cohort_id": cohort_id,
                    "cohort_value": cohort_value,
                    "seed_count": len(seed_indices),
                    "neighbor_total": neighbor_total,
                    "neighbor_hits": hits,
                    "neighbor_fraction": float(neighbor_fraction),
                    "background_count": background_count,
                    "background_fraction": float(background_fraction),
                    "enrichment_delta": float(neighbor_fraction - background_fraction),
                    "enrichment_ratio": float(enrichment_ratio),
                }
            )

    artifact_dir = context.output_root / "enrichments" / enrichment_id
    write_table(pa.Table.from_pylist(output_rows), artifact_dir / "table.parquet")
    summary = {
        "method": "landmark_neighbor_cohort_delta",
        "neighbors_id": neighbors_id,
        "view_id": view_id,
        "cohort_id": cohort_id,
        "cohort_column": cohort.column,
        "rows": len(output_rows),
        "k": int(neighbor_indices.shape[1]),
        "landmarks": landmark_ids,
        "landmark_modes": landmark_modes,
        "landmark_seed_counts": landmark_seed_counts,
        "cohort_values": ordered_cohort_values,
    }
    write_json(artifact_dir / "summary.json", summary)
    return artifact_dir, len(output_rows), summary
