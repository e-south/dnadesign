"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/structure/contact_geometry.py

Contact-geometry artifact validators for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import (
    _REQUIRED_CONTACT_GEOMETRY_PROFILE_COLUMNS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.structure.provenance import (
    json_metadata_mapping,
    validate_upstream_artifact_hashes,
)


def validate_contact_geometry_profile_content(
    path: Path,
    *,
    residue_map_path: Path | None = None,
    upstream_artifact_paths: Mapping[str, Path] | None = None,
) -> list[ContractIssue]:
    """Validate materialized atom-class contact geometry as structure evidence."""

    issues: list[ContractIssue] = []
    table = pq.read_table(path)
    column_names = set(table.column_names)
    missing_columns = sorted(_REQUIRED_CONTACT_GEOMETRY_PROFILE_COLUMNS - column_names)
    if missing_columns:
        return [
            ContractIssue(
                check_id="eco1_rt.structure.contact_geometry_missing_columns",
                message=f"contact_geometry_profile.parquet is missing required columns: {missing_columns}",
                path=str(path),
            )
        ]

    metadata = table.schema.metadata or {}
    expected_metadata = {
        b"schema_id": b"eco1_rt_repack.contact_geometry_profile",
        b"status": b"materialized",
        b"geometry_backend_id": b"biopython_mmcif_atom_geometry_v1",
    }
    for required_key in (b"schema_version", b"artifact_id", b"created_by", b"created_at", b"upstream_artifact_hashes"):
        if not metadata.get(required_key):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.contact_geometry_missing_lifecycle_metadata",
                    message=f"contact_geometry_profile.parquet metadata {required_key.decode()} must be present",
                    path=str(path),
                )
            )
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.contact_geometry_metadata_mismatch",
                    message=f"contact_geometry_profile.parquet metadata {key.decode()} must equal {expected.decode()}",
                    path=str(path),
                )
            )
    if upstream_artifact_paths is not None:
        issues.extend(
            validate_upstream_artifact_hashes(
                json_metadata_mapping(metadata.get(b"upstream_artifact_hashes")),
                upstream_artifact_paths,
                path=path,
                check_id="eco1_rt.structure.contact_geometry_upstream_hash_mismatch",
                artifact_label="contact_geometry_profile.parquet",
            )
        )

    rows = table.to_pylist()
    observed_positions = [row.get("canonical_position") for row in rows]
    if observed_positions != sorted(observed_positions):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.contact_geometry_position_order_mismatch",
                message="contact_geometry_profile.parquet rows must be sorted by canonical position",
                path=str(path),
            )
        )
    _validate_geometry_rows(issues, rows=rows, path=path)
    if residue_map_path is not None:
        _validate_residue_map_join(issues, rows=rows, residue_map_path=residue_map_path, path=path)
    return issues


def _validate_geometry_rows(issues: list[ContractIssue], *, rows: list[dict[str, object]], path: Path) -> None:
    bad_unresolved: list[int] = []
    bad_mapped: list[int] = []
    for row in rows:
        position = int(row.get("canonical_position", 0))
        mapping_status = row.get("mapping_status")
        if mapping_status == "unresolved_structure":
            if (
                row.get("nearest_context_atom_distance_angstrom") is not None
                or row.get("sidechain_atom_status") != "unresolved_structure"
            ):
                bad_unresolved.append(position)
            continue
        if mapping_status == "mapped":
            if row.get("nearest_context_atom_distance_angstrom") is None or not isinstance(
                row.get("contact_atom_count_within_20a"), int
            ):
                bad_mapped.append(position)
            continue
        bad_mapped.append(position)
    if bad_unresolved:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.contact_geometry_unresolved_has_geometry",
                message=f"unresolved contact-geometry rows must have null distances: {bad_unresolved[:20]}",
                path=str(path),
            )
        )
    if bad_mapped:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.contact_geometry_mapped_row_invalid",
                message=f"mapped contact-geometry rows must have atom-class geometry: {bad_mapped[:20]}",
                path=str(path),
            )
        )


def _validate_residue_map_join(
    issues: list[ContractIssue],
    *,
    rows: list[dict[str, object]],
    residue_map_path: Path,
    path: Path,
) -> None:
    residue_rows = pq.read_table(residue_map_path).to_pylist()
    if len(rows) != len(residue_rows):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.contact_geometry_residue_count_mismatch",
                message="contact_geometry_profile.parquet must include one row per residue_map.parquet row",
                path=str(path),
            )
        )
        return
    residue_by_position: dict[int, Mapping[str, object]] = {
        int(row["canonical_position"]): row for row in residue_rows if isinstance(row, Mapping)
    }
    mismatches: list[int] = []
    for row in rows:
        position = int(row["canonical_position"])
        residue = residue_by_position.get(position)
        if not isinstance(residue, Mapping):
            mismatches.append(position)
            continue
        if row.get("wt_aa") != residue.get("wt_aa") or row.get("mapping_status") != residue.get("mapping_status"):
            mismatches.append(position)
    if mismatches:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.contact_geometry_residue_map_mismatch",
                message=f"contact_geometry_profile.parquet disagrees with residue_map.parquet: {mismatches[:20]}",
                path=str(path),
            )
        )
