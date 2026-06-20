"""Materialized structure artifact validators for Eco1 RT repack."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.common import (
    _is_pending_value,
    _load_yaml,
    _phase_rank,
    _resolve_output_root,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import (
    _PLANNED_THREAD_ROOT,
    _REQUIRED_RESIDUE_MAP_COLUMNS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.evidence_artifacts import (
    _validate_materialized_evidence_and_mask_artifacts,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import (
    ContractIssue,
    ContractReport,
)


def validate_materialized_structure_artifacts(
    *,
    repo_root: Path,
    structure_sources: Mapping[str, Any],
    numbering_policy: Mapping[str, Any],
    conservation_sources: Mapping[str, Any],
    profile: Mapping[str, Any],
    phase: str,
    output_root: Path | None = None,
) -> ContractReport:
    """Validate materialized structure artifacts and expose the next honest blockers."""

    if _phase_rank(phase) < _phase_rank("phase1_thread_contract"):
        return ContractReport(phase=phase)

    issues: list[ContractIssue] = []
    structure_root = _resolve_output_root(repo_root, output_root)
    backbone_bundle = structure_root / "backbone_bundle.yaml"
    residue_map = structure_root / "residue_map.parquet"
    if not backbone_bundle.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.backbone_bundle_not_materialized",
                message=(
                    "Phase 1 structure authority requires materialized backbone_bundle.yaml before downstream gates"
                ),
                path=str(_PLANNED_THREAD_ROOT / "backbone_bundle.yaml"),
            )
        )
    else:
        issues.extend(
            _validate_backbone_bundle_content(
                backbone_bundle,
                structure_sources=structure_sources,
                numbering_policy=numbering_policy,
            )
        )
    if not residue_map.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.residue_map_not_materialized",
                message=(
                    "Phase 1 structure authority requires materialized residue_map.parquet before masks or sampling"
                ),
                path=str(_PLANNED_THREAD_ROOT / "residue_map.parquet"),
            )
        )
    else:
        issues.extend(
            _validate_residue_map_content(
                residue_map,
                structure_sources=structure_sources,
                numbering_policy=numbering_policy,
            )
        )
    if issues:
        return ContractReport(phase=phase, issues=tuple(issues))

    issues.extend(
        _validate_materialized_evidence_and_mask_artifacts(
            repo_root,
            structure_root,
            numbering_policy=numbering_policy,
            conservation_sources=conservation_sources,
            profile=profile,
        )
    )
    return ContractReport(phase=phase, issues=tuple(issues))


def _validate_backbone_bundle_content(
    path: Path,
    *,
    structure_sources: Mapping[str, Any],
    numbering_policy: Mapping[str, Any],
) -> list[ContractIssue]:
    issues: list[ContractIssue] = []
    bundle = _load_yaml(path)
    selected_source = structure_sources.get("selected_source")
    if not isinstance(selected_source, Mapping):
        return [
            ContractIssue(
                check_id="eco1_rt.structure.missing_selected_source",
                message="cannot validate backbone bundle without selected_source",
                path=str(path),
            )
        ]

    expected_pairs = {
        "schema_id": "thread.backbone_bundle",
        "status": "materialized",
        "structure_source_id": selected_source.get("source_id"),
        "source_ref": selected_source.get("source_ref"),
        "source_format": selected_source.get("structure_format"),
        "reference_sequence_hash": selected_source.get("reference_sequence_hash"),
        "rt_chain_id": selected_source.get("rt_chain_id"),
        "retained_context_policy": selected_source.get("retained_context_policy"),
        "residue_numbering_origin": numbering_policy.get("residue_numbering_origin"),
    }
    required_lifecycle_fields = (
        "schema_version",
        "artifact_id",
        "created_by",
        "created_at",
        "upstream_artifact_hashes",
    )
    for field in required_lifecycle_fields:
        if field not in bundle or _is_pending_value(bundle.get(field)):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.backbone_bundle_missing_lifecycle_field",
                    message=f"backbone_bundle.yaml must declare non-pending lifecycle field {field!r}",
                    path=f"{path}:{field}",
                )
            )
    for field, expected in expected_pairs.items():
        if bundle.get(field) != expected:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.backbone_bundle_field_mismatch",
                    message=f"backbone_bundle.yaml field {field!r} must equal {expected!r}",
                    path=f"{path}:{field}",
                )
            )

    chain_inventory = bundle.get("chain_inventory")
    if not isinstance(chain_inventory, list):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.backbone_bundle_missing_chain_inventory",
                message="backbone_bundle.yaml must declare a typed chain_inventory list",
                path=f"{path}:chain_inventory",
            )
        )
        return issues

    chain_roles = {
        str(row.get("chain_id")): row
        for row in chain_inventory
        if isinstance(row, Mapping) and isinstance(row.get("chain_id"), str)
    }
    expected_roles = {
        str(selected_source.get("rt_chain_id")): ("protein", "design_backbone"),
        "D": ("dna", "retained_context"),
        "E": ("rna", "retained_context"),
        "F": ("rna", "retained_context"),
    }
    for chain_id, (molecule_type, thread_role) in expected_roles.items():
        row = chain_roles.get(chain_id)
        if (
            not isinstance(row, Mapping)
            or row.get("molecule_type") != molecule_type
            or row.get("thread_role") != thread_role
        ):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.backbone_bundle_chain_inventory_mismatch",
                    message=(
                        f"backbone_bundle.yaml must type retained chains as {chain_id}={molecule_type}/{thread_role}"
                    ),
                    path=f"{path}:chain_inventory",
                )
            )
    return issues


def _validate_residue_map_content(
    path: Path,
    *,
    structure_sources: Mapping[str, Any],
    numbering_policy: Mapping[str, Any],
) -> list[ContractIssue]:
    selected_source = structure_sources.get("selected_source")
    if not isinstance(selected_source, Mapping):
        return [
            ContractIssue(
                check_id="eco1_rt.structure.missing_selected_source",
                message="cannot validate residue map without selected_source",
                path=str(path),
            )
        ]

    issues: list[ContractIssue] = []
    table = pq.read_table(path)
    column_names = set(table.column_names)
    missing_columns = sorted(_REQUIRED_RESIDUE_MAP_COLUMNS - column_names)
    if missing_columns:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.residue_map_missing_columns",
                message=f"residue_map.parquet is missing required columns: {missing_columns}",
                path=str(path),
            )
        )
        return issues

    metadata = table.schema.metadata or {}
    expected_metadata = {
        b"schema_id": b"thread.residue_map",
        b"status": b"materialized",
        b"selected_structure_source_id": str(selected_source.get("source_id")).encode("utf-8"),
        b"reference_sequence_hash": str(selected_source.get("reference_sequence_hash")).encode("utf-8"),
        b"residue_numbering_origin": str(numbering_policy.get("residue_numbering_origin")).encode("utf-8"),
    }
    for required_key in (b"schema_version", b"artifact_id", b"created_by", b"created_at", b"upstream_artifact_hashes"):
        if not metadata.get(required_key):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.residue_map_missing_lifecycle_metadata",
                    message=f"residue_map.parquet metadata {required_key.decode()} must be present",
                    path=str(path),
                )
            )
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.structure.residue_map_metadata_mismatch",
                    message=f"residue_map.parquet metadata {key.decode()} must equal {expected.decode()}",
                    path=str(path),
                )
            )

    coverage = numbering_policy.get("coverage")
    if not isinstance(coverage, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.invalid_numbering_coverage",
                message="cannot validate residue_map.parquet without numbering coverage",
                path=str(path),
            )
        )
        return issues

    rows = table.to_pylist()
    reference_length = int(coverage.get("reference_sequence_length", 0))
    if len(rows) != reference_length:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.residue_map_row_count_mismatch",
                message="residue_map.parquet must include one row per canonical reference position",
                path=str(path),
            )
        )

    mapped_rows = [row for row in rows if row.get("mapping_status") == "mapped"]
    if len(mapped_rows) != int(coverage.get("mapped_residue_count", -1)):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.residue_map_mapped_count_mismatch",
                message="residue_map.parquet mapped row count must match numbering-policy coverage",
                path=str(path),
            )
        )

    unresolved_positions = list(coverage.get("unresolved_canonical_positions", []))
    unresolved_rows = [row for row in rows if row.get("mapping_status") == "unresolved_structure"]
    observed_unresolved = [row.get("canonical_position") for row in unresolved_rows]
    if observed_unresolved != unresolved_positions:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.residue_map_unresolved_positions_mismatch",
                message="residue_map.parquet unresolved positions must match numbering-policy coverage exactly",
                path=str(path),
            )
        )
    bad_unresolved = [
        row.get("canonical_position")
        for row in unresolved_rows
        if row.get("unresolved_policy") != "fixed" or row.get("is_designable_initially") is not False
    ]
    if bad_unresolved:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.structure.residue_map_unresolved_not_fixed",
                message=f"unresolved residue-map rows must be fixed and non-designable: {bad_unresolved}",
                path=str(path),
            )
        )
    return issues
