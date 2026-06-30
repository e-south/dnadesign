"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/evidence_artifacts.py

Materialized evidence and mask artifact validators for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.conservation import (
    validate_conservation_profile_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import (
    _DOCS_ROOT,
    _REQUIRED_CONTACT_PROFILE_COLUMNS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.masks import validate_mask_set_content
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.structure.contact_geometry import (
    validate_contact_geometry_profile_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.structure.preprocessing import (
    contact_geometry_upstream_artifact_paths,
    validate_structure_preprocessing_manifest_content,
)


def _validate_materialized_evidence_and_mask_artifacts(
    repo_root: Path,
    structure_root: Path,
    *,
    numbering_policy: Mapping[str, Any],
    conservation_sources: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    conservation_profile = structure_root / "conservation_profile.parquet"
    contact_profile = structure_root / "contact_profile.parquet"
    structure_preprocessing_manifest = structure_root / "structure_preprocessing_manifest.yaml"
    contact_geometry_profile = structure_root / "contact_geometry_profile.parquet"
    manual_mask_authority = structure_root / "manual_mask_authority.yaml"
    mask_set = structure_root / "mask_set.yaml"

    if not conservation_profile.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.conservation_profile_not_materialized",
                message="Phase 1 mask construction requires a materialized conservation_profile.parquet",
                path=str(conservation_profile),
            )
        )
    else:
        issues.extend(
            validate_conservation_profile_content(
                conservation_profile,
                residue_map_path=structure_root / "residue_map.parquet",
                conservation_sources=conservation_sources,
                source_contract_path=repo_root / _DOCS_ROOT / "workbench/provenance/conservation-sources.yaml",
            )
        )
    if not contact_profile.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.contact_profile_not_materialized",
                message=(
                    "Phase 1 diagnostic contact review requires a materialized contact_profile.parquet; "
                    "the active 5 A mask input is contact_geometry_profile.parquet"
                ),
                path=str(contact_profile),
            )
        )
    else:
        issues.extend(
            _validate_contact_profile_content(
                contact_profile,
                residue_map_path=structure_root / "residue_map.parquet",
                numbering_policy=numbering_policy,
                profile=profile,
            )
        )
    if not structure_preprocessing_manifest.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.structure_preprocessing_manifest_not_materialized",
                message="Phase 1 mask evidence requires a materialized structure_preprocessing_manifest.yaml",
                path=str(structure_preprocessing_manifest),
            )
        )
    else:
        issues.extend(
            validate_structure_preprocessing_manifest_content(
                structure_preprocessing_manifest,
                repo_root=repo_root,
                structure_root=structure_root,
            )
        )
    if not contact_geometry_profile.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.contact_geometry_profile_not_materialized",
                message="Phase 1 mask evidence requires a materialized contact_geometry_profile.parquet",
                path=str(contact_geometry_profile),
            )
        )
    else:
        issues.extend(
            validate_contact_geometry_profile_content(
                contact_geometry_profile,
                residue_map_path=structure_root / "residue_map.parquet",
                upstream_artifact_paths=contact_geometry_upstream_artifact_paths(repo_root, structure_root),
            )
        )
    if not manual_mask_authority.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.manual_mask_authority_not_materialized",
                message="Phase 1 sampling requires a materialized manual_mask_authority.yaml",
                path=str(manual_mask_authority),
            )
        )
    if not mask_set.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.mask_set_not_materialized",
                message="Phase 1 sampling requires a materialized mask_set.yaml",
                path=str(mask_set),
            )
        )
    elif conservation_profile.exists() and contact_geometry_profile.exists() and manual_mask_authority.exists():
        issues.extend(
            validate_mask_set_content(
                mask_set,
                repo_root=repo_root,
                residue_map_path=structure_root / "residue_map.parquet",
                contact_geometry_profile_path=contact_geometry_profile,
                conservation_profile_path=conservation_profile,
                manual_mask_authority_path=manual_mask_authority,
            )
        )
    return issues


def _validate_contact_profile_content(
    path: Path,
    *,
    residue_map_path: Path,
    numbering_policy: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    contact_table = pq.read_table(path)
    column_names = set(contact_table.column_names)
    missing_columns = sorted(_REQUIRED_CONTACT_PROFILE_COLUMNS - column_names)
    if missing_columns:
        return [
            ContractIssue(
                check_id="eco1_rt.evidence.contact_profile_missing_columns",
                message=f"contact_profile.parquet is missing required columns: {missing_columns}",
                path=str(path),
            )
        ]

    source_hash = "sha256:" + str(numbering_policy.get("source_distance_profile_sha256", "")).removeprefix("sha256:")
    threshold = _contact_threshold_angstrom(profile)
    metadata = contact_table.schema.metadata or {}
    expected_metadata = {
        b"schema_id": b"thread.contact_profile",
        b"status": b"materialized",
        b"reference_sequence_hash": str(numbering_policy.get("reference_sequence_hash")).encode("utf-8"),
        b"selected_structure_source_id": str(numbering_policy.get("selected_structure_source_id")).encode("utf-8"),
        b"source_hash": source_hash.encode("utf-8"),
        b"contact_threshold_angstrom": str(threshold).encode("utf-8"),
    }
    for required_key in (b"schema_version", b"artifact_id", b"created_by", b"created_at", b"upstream_artifact_hashes"):
        if not metadata.get(required_key):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.evidence.contact_profile_missing_lifecycle_metadata",
                    message=f"contact_profile.parquet metadata {required_key.decode()} must be present",
                    path=str(path),
                )
            )
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.evidence.contact_profile_metadata_mismatch",
                    message=f"contact_profile.parquet metadata {key.decode()} must equal {expected.decode()}",
                    path=str(path),
                )
            )

    coverage = numbering_policy.get("coverage")
    if not isinstance(coverage, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.invalid_numbering_coverage",
                message="cannot validate contact_profile.parquet without numbering coverage",
                path=str(path),
            )
        )
        return issues

    contact_rows = contact_table.to_pylist()
    residue_rows = pq.read_table(residue_map_path).to_pylist()
    reference_length = int(coverage.get("reference_sequence_length", 0))
    expected_positions = list(range(1, reference_length + 1))
    observed_positions = [row.get("canonical_position") for row in contact_rows]
    if observed_positions != expected_positions:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.contact_profile_position_mismatch",
                message="contact_profile.parquet must include one ordered row per canonical reference position",
                path=str(path),
            )
        )

    residue_by_position = {row.get("canonical_position"): row for row in residue_rows}
    source_hashes = {row.get("source_hash") for row in contact_rows}
    if source_hashes != {source_hash}:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.contact_profile_source_hash_mismatch",
                message="contact_profile.parquet source_hash column must match source_distance_profile_sha256",
                path=str(path),
            )
        )

    thresholds = {row.get("contact_threshold_angstrom") for row in contact_rows}
    if thresholds != {threshold}:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.contact_profile_threshold_mismatch",
                message="contact_profile.parquet contact_threshold_angstrom must match the Eco1 profile threshold",
                path=str(path),
            )
        )

    bad_rows: list[int] = []
    unresolved_bad: list[int] = []
    mask_mismatches: list[int] = []
    for row in contact_rows:
        position = int(row.get("canonical_position", 0))
        residue = residue_by_position.get(position)
        if not isinstance(residue, Mapping):
            bad_rows.append(position)
            continue
        if row.get("wt_aa") != residue.get("wt_aa") or row.get("mapping_status") != residue.get("mapping_status"):
            bad_rows.append(position)
            continue
        if residue.get("mapping_status") == "mapped":
            nearest_distance = row.get("nearest_context_atom_distance_angstrom")
            if nearest_distance is None or row.get("structure_residue_id") != residue.get("structure_residue_id"):
                bad_rows.append(position)
                continue
            expected_mask = float(nearest_distance) <= threshold
            if row.get("passes_contact_mask") is not expected_mask:
                mask_mismatches.append(position)
        else:
            has_unresolved_distance = row.get("nearest_context_atom_distance_angstrom") is not None
            has_unresolved_contact_pass = row.get("passes_contact_mask") is not False
            if has_unresolved_distance or has_unresolved_contact_pass:
                unresolved_bad.append(position)

    if bad_rows:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.contact_profile_residue_map_mismatch",
                message=f"contact_profile.parquet rows disagree with residue_map.parquet: {bad_rows[:20]}",
                path=str(path),
            )
        )
    if unresolved_bad:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.contact_profile_missing_backbone_has_contact_evidence",
                message=(
                    "missing-backbone contact-profile rows must have no retained-context distance "
                    f"and must fail the contact mask: {unresolved_bad}"
                ),
                path=str(path),
            )
        )
    if mask_mismatches:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.contact_profile_mask_value_mismatch",
                message=f"contact-profile mask values disagree with threshold: {mask_mismatches[:20]}",
                path=str(path),
            )
        )
    return issues


def _contact_threshold_angstrom(profile: Mapping[str, Any]) -> float:
    policy = profile.get("conservative_policy")
    if not isinstance(policy, Mapping):
        return 0.0
    value = policy.get("direct_contact_threshold_angstrom")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0
