"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/atlas_semantic_profile/resume.py

Resume cache helpers for Eco1 ESM Atlas semantic-profile materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


@dataclass(frozen=True)
class ExistingAtlasRows:
    """Reusable Atlas rows keyed by candidate id."""

    profile_rows_by_candidate: dict[str, dict[str, object]]
    protein_activation_rows_by_candidate: dict[str, list[dict[str, object]]]
    residue_activation_rows_by_candidate: dict[str, list[dict[str, object]]]
    structure_prediction_rows_by_candidate: dict[str, list[dict[str, object]]]
    feature_catalog_rows: list[dict[str, object]]


def load_existing_rows(output_root: Path, *, structure_prediction_root: Path | None = None) -> ExistingAtlasRows | None:
    """Load compatible existing Atlas rows for exact per-sequence resume."""

    profile_path = output_root / "atlas_semantic_profile.parquet"
    protein_path = output_root / "atlas_protein_activations.parquet"
    residue_path = output_root / "atlas_residue_activations.parquet"
    feature_path = output_root / "atlas_feature_catalog.parquet"
    registry_path = None
    if structure_prediction_root is not None:
        registry_path = structure_prediction_root / "structure_prediction_registry.parquet"
    if not profile_path.exists():
        return None
    profile_table = pq.read_table(profile_path)
    if "atlas_query_hash" not in profile_table.column_names:
        return None
    protein_rows = pq.read_table(protein_path).to_pylist() if protein_path.exists() else []
    residue_rows = pq.read_table(residue_path).to_pylist() if residue_path.exists() else []
    feature_rows = pq.read_table(feature_path).to_pylist() if feature_path.exists() else []
    structure_rows = (
        pq.read_table(registry_path).to_pylist() if registry_path is not None and registry_path.exists() else []
    )
    return ExistingAtlasRows(
        profile_rows_by_candidate={str(row["candidate_id"]): dict(row) for row in profile_table.to_pylist()},
        protein_activation_rows_by_candidate=_group_rows_by_candidate(protein_rows),
        residue_activation_rows_by_candidate=_group_rows_by_candidate(residue_rows),
        structure_prediction_rows_by_candidate=_group_rows_by_candidate(structure_rows),
        feature_catalog_rows=[dict(row) for row in feature_rows],
    )


def cached_profile_row(
    *,
    existing_rows: ExistingAtlasRows | None,
    candidate_id: str,
    sequence_hash: str,
    atlas_query_hash: str,
    atlas_request_hash: str,
    source_request_hash: str,
) -> dict[str, object] | None:
    """Return a reusable profile row if the per-sequence query is identical."""

    if existing_rows is None:
        return None
    row = existing_rows.profile_rows_by_candidate.get(candidate_id)
    if row is None:
        return None
    if str(row.get("sequence_hash", "")) != sequence_hash:
        return None
    if str(row.get("atlas_query_hash", "")) != atlas_query_hash:
        return None
    if str(row.get("failure_reason", "")) == "atlas_request_not_attempted_due_to_max_new_requests":
        return None
    copied = dict(row)
    copied["atlas_request_hash"] = atlas_request_hash
    copied["source_request_hash"] = source_request_hash
    return copied


def cached_structure_prediction_rows(
    *,
    existing_rows: ExistingAtlasRows | None,
    candidate_id: str,
    prediction_set_id: str,
    atlas_request_hash: str,
) -> list[dict[str, object]]:
    """Return reusable structure-prediction rows for the current on-demand Atlas request."""

    if existing_rows is None:
        return []
    rows = existing_rows.structure_prediction_rows_by_candidate.get(candidate_id, [])
    return [
        dict(row)
        for row in rows
        if str(row.get("prediction_set_id", "")) == prediction_set_id
        and str(row.get("request_hash", "")) == atlas_request_hash
    ]


def _group_rows_by_candidate(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(dict(row))
    return grouped
