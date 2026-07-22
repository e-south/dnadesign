"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/masks/set_artifacts.py

Materialized mask-set validators for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import (
    _CONTRACT_ROOT,
    _DOCS_ROOT,
    _REQUIRED_MASK_SET_COLUMNS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.masks.manual_artifacts import (
    validate_manual_mask_authority_content,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.masking import compose_mask_rows, summarize_mask_rows


def validate_mask_set_content(
    path: Path,
    *,
    repo_root: Path,
    residue_map_path: Path,
    contact_geometry_profile_path: Path,
    conservation_profile_path: Path,
    manual_mask_authority_path: Path,
) -> list[ContractIssue]:
    """Validate a materialized mask set against its upstream evidence artifacts."""

    issues: list[ContractIssue] = []
    mask_set = _load_yaml(path)
    manual_mask_authority = _load_yaml(manual_mask_authority_path)
    issues.extend(
        validate_manual_mask_authority_content(
            manual_mask_authority,
            repo_root=repo_root,
            residue_map_path=residue_map_path,
            path=manual_mask_authority_path,
        )
    )
    _validate_top_level_fields(issues, mask_set=mask_set, path=path)
    _validate_upstream_hashes(
        issues,
        mask_set=mask_set,
        path=path,
        repo_root=repo_root,
        residue_map_path=residue_map_path,
        contact_geometry_profile_path=contact_geometry_profile_path,
        conservation_profile_path=conservation_profile_path,
        manual_mask_authority_path=manual_mask_authority_path,
    )

    rows = mask_set.get("residues")
    if not isinstance(rows, list):
        return [
            ContractIssue(
                check_id="eco1_rt.mask.mask_set_missing_rows",
                message="mask_set.yaml must contain a residues list",
                path=str(path),
            )
        ]
    missing_columns = sorted(
        {
            column
            for column in _REQUIRED_MASK_SET_COLUMNS
            if any(not isinstance(row, Mapping) or column not in row for row in rows)
        }
    )
    if missing_columns:
        return [
            ContractIssue(
                check_id="eco1_rt.mask.mask_set_missing_columns",
                message=f"mask_set.yaml residue rows are missing required columns: {missing_columns}",
                path=str(path),
            )
        ]

    residue_rows = pq.read_table(residue_map_path).to_pylist()
    contact_geometry_rows = pq.read_table(contact_geometry_profile_path).to_pylist()
    conservation_rows = pq.read_table(conservation_profile_path).to_pylist()
    expected_rows = compose_mask_rows(
        residue_rows=residue_rows,
        contact_geometry_rows=contact_geometry_rows,
        conservation_rows=conservation_rows,
        manual_authority=manual_mask_authority,
    )
    _validate_rows(issues, rows=rows, expected_rows=expected_rows, path=path)
    _validate_summary(issues, mask_set=mask_set, rows=rows, path=path)
    return issues


def _validate_top_level_fields(issues: list[ContractIssue], *, mask_set: Mapping[str, Any], path: Path) -> None:
    expected = {
        "schema_id": "thread.mask_set",
        "schema_version": 1,
        "artifact_id": "eco1_rt_conservative_v1.mask_set",
        "status": "materialized",
        "mask_policy_id": "eco1_rt_clade9_plurality25_direct_contact5a_v1",
        "manual_mask_authority_status": "materialized_eco1_rt_manual_motif_wang_direct_contact_v1",
    }
    for key, value in expected.items():
        if mask_set.get(key) != value:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask.mask_set_metadata_mismatch",
                    message=f"mask_set.yaml field {key!r} must equal {value!r}",
                    path=str(path),
                )
            )
    for forbidden_field in ("selected_tier_id", "relaxed_tier_projections"):
        if forbidden_field in mask_set:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask.mask_set_legacy_field_present",
                    message=f"mask_set.yaml must not declare legacy field {forbidden_field!r}",
                    path=str(path),
                )
            )
    for field in ("created_by", "created_at", "upstream_artifact_hashes", "summary"):
        if field not in mask_set:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask.mask_set_missing_lifecycle_field",
                    message=f"mask_set.yaml must declare {field!r}",
                    path=str(path),
                )
            )


def _validate_upstream_hashes(
    issues: list[ContractIssue],
    *,
    mask_set: Mapping[str, Any],
    path: Path,
    repo_root: Path,
    residue_map_path: Path,
    contact_geometry_profile_path: Path,
    conservation_profile_path: Path,
    manual_mask_authority_path: Path,
) -> None:
    hashes = mask_set.get("upstream_artifact_hashes")
    if not isinstance(hashes, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.mask_set_missing_upstream_hashes",
                message="mask_set.yaml must declare upstream_artifact_hashes",
                path=str(path),
            )
        )
        return
    expected = {
        "profile": "sha256:" + _sha256(repo_root / _CONTRACT_ROOT / "fixtures/thread/eco1_rt_v1.profile.yaml"),
        "conservation_sources": "sha256:"
        + _sha256(repo_root / _DOCS_ROOT / "workbench/provenance/conservation-sources.yaml"),
        "residue_map": "sha256:" + _sha256(residue_map_path),
        "contact_geometry_profile": "sha256:" + _sha256(contact_geometry_profile_path),
        "conservation_profile": "sha256:" + _sha256(conservation_profile_path),
        "manual_mask_authority": "sha256:" + _sha256(manual_mask_authority_path),
    }
    for key, value in expected.items():
        if hashes.get(key) != value:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask.mask_set_upstream_hash_mismatch",
                    message=f"mask_set.yaml upstream hash {key!r} must match current artifact",
                    path=str(path),
                )
            )


def _validate_rows(
    issues: list[ContractIssue],
    *,
    rows: list[Any],
    expected_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    observed_positions = [row.get("canonical_position") for row in rows if isinstance(row, Mapping)]
    expected_positions = [row["canonical_position"] for row in expected_rows]
    if observed_positions != expected_positions:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.mask_set_position_mismatch",
                message="mask_set.yaml must include one ordered residue row per canonical position",
                path=str(path),
            )
        )
        return

    value_mismatches: list[int] = []
    protected_without_reason: list[int] = []
    unprotected_with_reason: list[int] = []
    legacy_rows: list[int] = []
    for row, expected in zip(rows, expected_rows, strict=True):
        if not isinstance(row, Mapping):
            continue
        position = int(expected["canonical_position"])
        for field in (
            "wt_aa",
            "design_position",
            "mapping_status",
            "has_backbone_coordinates",
            "min_distance_to_retained_dna_rna_angstrom",
            "direct_contact_threshold_angstrom",
            "direct_retained_dna_rna_contact_5a",
            "motif_protected",
            "wang_ec86_direct_contact_prior",
            "evolutionarily_conserved_clade9_25pct_plurality",
            "wt_plurality_frequency",
            "wt_plurality_aa",
            "conservation_profile_ids",
            "manual_mask_reason",
            "wang_ec86_direct_contact_reason",
            "rt_interval_review_label",
            "protected",
            "non_fixed",
            "non_fixed_missing_backbone",
            "protection_reasons",
            "conflict_status",
            "conflict_reason",
        ):
            if row.get(field) != expected[field]:
                value_mismatches.append(position)
                break
        if any(field in row for field in ("final_fixed", "proteinmpnn_designable", "mask_sources")):
            legacy_rows.append(position)
        protection_reasons = row.get("protection_reasons")
        if row.get("protected") is True and not protection_reasons:
            protected_without_reason.append(position)
        if row.get("protected") is False and protection_reasons:
            unprotected_with_reason.append(position)

    if value_mismatches:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.mask_set_value_mismatch",
                message=f"mask_set.yaml residue rows disagree with upstream evidence: {value_mismatches[:20]}",
                path=str(path),
            )
        )
    if legacy_rows:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.mask_set_legacy_row_field_present",
                message=(
                    "mask_set.yaml rows must not retain "
                    f"final_fixed/proteinmpnn_designable/mask_sources: {legacy_rows[:20]}"
                ),
                path=str(path),
            )
        )
    if protected_without_reason:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.mask_set_missing_protection_reason",
                message=(
                    f"protected mask rows must record at least one protection reason: {protected_without_reason[:20]}"
                ),
                path=str(path),
            )
        )
    if unprotected_with_reason:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.mask_set_unprotected_reason_mismatch",
                message=f"unprotected mask rows must not record protection reasons: {unprotected_with_reason[:20]}",
                path=str(path),
            )
        )


def _validate_summary(
    issues: list[ContractIssue],
    *,
    mask_set: Mapping[str, Any],
    rows: list[Any],
    path: Path,
) -> None:
    summary = mask_set.get("summary")
    if not isinstance(summary, Mapping):
        return
    expected_summary = summarize_mask_rows([dict(row) for row in rows if isinstance(row, Mapping)])
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask.mask_set_summary_mismatch",
                    message=f"mask_set.yaml summary {key!r} must equal {value}",
                    path=str(path),
                )
            )
    if expected_summary["non_fixed_mapped_position_count"] == 0:
        expected_status = "blocked_no_non_fixed_mapped_positions"
        expected_allowed = False
    else:
        expected_status = "pending_sampling_plan"
        expected_allowed = True
    if mask_set.get("sampling_status") != expected_status or mask_set.get("sampling_allowed") is not expected_allowed:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.mask_set_sampling_status_mismatch",
                message="mask_set.yaml sampling fields must match non-fixed mapped residue availability",
                path=str(path),
            )
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return loaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
