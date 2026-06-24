"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/masks/manual_artifacts.py

Manual mask-authority artifact validators for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import _CONTRACT_ROOT, _DOCS_ROOT
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.masks.rt_intervals import (
    validate_rt_interval_authority,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.masks.source import (
    candidate_prior_positions_from_source,
    load_manual_mask_authority_source,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue


def validate_manual_mask_authority_content(
    manual_mask_authority: Mapping[str, Any],
    *,
    repo_root: Path,
    residue_map_path: Path,
    path: Path,
) -> list[ContractIssue]:
    """Validate generated manual_mask_authority.yaml content."""

    issues: list[ContractIssue] = []
    _validate_metadata(issues, manual_mask_authority=manual_mask_authority, path=path)
    _validate_upstream_hashes(
        issues,
        manual_mask_authority=manual_mask_authority,
        repo_root=repo_root,
        residue_map_path=residue_map_path,
        path=path,
    )
    residue_rows = pq.read_table(residue_map_path).to_pylist()
    residue_by_position = {int(row["canonical_position"]): row for row in residue_rows}
    rows = manual_mask_authority.get("residues")
    if not isinstance(rows, list):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.manual_mask_authority_missing_rows",
                message="manual_mask_authority.yaml must contain a residues list",
                path=str(path),
            )
        )
        return issues
    _validate_manual_rows(issues, rows=rows, residue_by_position=residue_by_position, path=path)
    _validate_summary(issues, manual_mask_authority=manual_mask_authority, manual_rows=rows, path=path)
    _validate_wang_priors(
        issues,
        manual_mask_authority=manual_mask_authority,
        repo_root=repo_root,
        residue_by_position=residue_by_position,
        path=path,
    )
    return issues


def _validate_metadata(
    issues: list[ContractIssue],
    *,
    manual_mask_authority: Mapping[str, Any],
    path: Path,
) -> None:
    expected = {
        "schema_id": "eco1_rt_repack.manual_mask_authority",
        "schema_version": 1,
        "artifact_id": "eco1_rt_conservative_v1.manual_mask_authority",
        "status": "materialized",
        "mask_policy_id": "eco1_rt_manual_motif_wang_direct_contact_v1",
        "coordinate_space": "canonical_position",
    }
    for key, value in expected.items():
        if manual_mask_authority.get(key) != value:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask.manual_mask_authority_metadata_mismatch",
                    message=f"manual_mask_authority.yaml field {key!r} must equal {value!r}",
                    path=str(path),
                )
            )


def _validate_upstream_hashes(
    issues: list[ContractIssue],
    *,
    manual_mask_authority: Mapping[str, Any],
    repo_root: Path,
    residue_map_path: Path,
    path: Path,
) -> None:
    hashes = manual_mask_authority.get("upstream_artifact_hashes")
    if not isinstance(hashes, Mapping):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.manual_mask_authority_missing_upstream_hashes",
                message="manual_mask_authority.yaml must declare upstream_artifact_hashes",
                path=str(path),
            )
        )
        return
    expected_hashes = {
        "profile": "sha256:" + _sha256(repo_root / _CONTRACT_ROOT / "fixtures/thread/eco1_rt_v1.profile.yaml"),
        "manual_mask_authority_source": "sha256:"
        + _sha256(repo_root / _DOCS_ROOT / "workbench/ontology/manual-mask-authority.yaml"),
        "residue_map": "sha256:" + _sha256(residue_map_path),
    }
    for key, value in expected_hashes.items():
        if hashes.get(key) != value:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask.manual_mask_authority_upstream_hash_mismatch",
                    message=f"manual_mask_authority.yaml upstream hash {key!r} must match current artifact",
                    path=str(path),
                )
            )


def _validate_manual_rows(
    issues: list[ContractIssue],
    *,
    rows: list[Any],
    residue_by_position: Mapping[int, Mapping[str, Any]],
    path: Path,
) -> None:
    observed_positions: list[int] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        position = int(row.get("canonical_position", -1))
        observed_positions.append(position)
        residue = residue_by_position.get(position)
        if residue is None or residue.get("wt_aa") != row.get("wt_aa") or row.get("manual_mask") is not True:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask.manual_mask_authority_row_mismatch",
                    message=f"manual_mask_authority.yaml row for position {position} does not match residue map",
                    path=str(path),
                )
            )
    if observed_positions != sorted(set(observed_positions)):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.manual_mask_authority_position_order_mismatch",
                message="manual_mask_authority.yaml residues must be unique and sorted by canonical position",
                path=str(path),
            )
        )


def _validate_summary(
    issues: list[ContractIssue],
    *,
    manual_mask_authority: Mapping[str, Any],
    manual_rows: list[Any],
    path: Path,
) -> None:
    summary = manual_mask_authority.get("summary")
    if not isinstance(summary, Mapping):
        return
    if summary.get("manual_mask_position_count") != len(manual_rows):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.manual_mask_authority_summary_mismatch",
                message="manual_mask_authority.yaml summary manual_mask_position_count must match residue rows",
                path=str(path),
            )
        )
    features = manual_mask_authority.get("features")
    if isinstance(features, list):
        protected_features = [
            feature for feature in features if isinstance(feature, Mapping) and feature.get("policy") == "fixed"
        ]
        rt_interval_features = [
            feature
            for feature in features
            if isinstance(feature, Mapping) and feature.get("authority_type") == "rt_core_interval"
        ]
        if summary.get("protected_feature_count") != len(protected_features):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask.manual_mask_authority_summary_mismatch",
                    message="manual_mask_authority.yaml summary protected_feature_count must match fixed features",
                    path=str(path),
                )
            )
        if summary.get("rt_interval_feature_count") != len(rt_interval_features):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.mask.manual_mask_authority_summary_mismatch",
                    message=(
                        "manual_mask_authority.yaml summary rt_interval_feature_count must match RT interval labels"
                    ),
                    path=str(path),
                )
            )
    candidate_rows = manual_mask_authority.get("candidate_prior_residues")
    if isinstance(candidate_rows, list) and summary.get("candidate_prior_position_count") != len(candidate_rows):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.manual_mask_authority_summary_mismatch",
                message="manual_mask_authority.yaml summary candidate_prior_position_count must match prior rows",
                path=str(path),
            )
        )


def _validate_wang_priors(
    issues: list[ContractIssue],
    *,
    manual_mask_authority: Mapping[str, Any],
    repo_root: Path,
    residue_by_position: Mapping[int, Mapping[str, Any]],
    path: Path,
) -> None:
    _validate_wang_source_basis(issues, manual_mask_authority=manual_mask_authority, path=path)
    authority_source = load_manual_mask_authority_source(repo_root)
    expected_candidate_positions = candidate_prior_positions_from_source(authority_source)
    candidate_rows = manual_mask_authority.get("candidate_prior_residues")
    if not isinstance(candidate_rows, list) or not candidate_rows:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.manual_mask_authority_missing_candidate_priors",
                message="manual_mask_authority.yaml must retain Wang/Ec86 candidate prior rows",
                path=str(path),
            )
        )
        return
    _validate_wang_candidate_rows(
        issues,
        candidate_rows=candidate_rows,
        expected_positions=expected_candidate_positions,
        residue_by_position=residue_by_position,
        path=path,
    )
    issues.extend(
        validate_rt_interval_authority(
            manual_mask_authority,
            authority_source=authority_source,
            path=path,
        )
    )


def _validate_wang_source_basis(
    issues: list[ContractIssue],
    *,
    manual_mask_authority: Mapping[str, Any],
    path: Path,
) -> None:
    source_basis = manual_mask_authority.get("source_basis")
    if not isinstance(source_basis, list):
        source_basis = []
    source_basis_ids = {
        source.get("id") for source in source_basis if isinstance(source, Mapping) and isinstance(source.get("id"), str)
    }
    if "wang_et_al_2022_ec86_cryoem_structure_priors" not in source_basis_ids:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.manual_mask_authority_missing_wang_source_basis",
                message="manual_mask_authority.yaml must cite Wang et al. 2022 as an Ec86 structural mask prior",
                path=str(path),
            )
        )


def _validate_wang_candidate_rows(
    issues: list[ContractIssue],
    *,
    candidate_rows: list[Any],
    expected_positions: set[int],
    residue_by_position: Mapping[int, Mapping[str, Any]],
    path: Path,
) -> None:
    observed_positions: set[int] = set()
    malformed_positions: list[int] = []
    for row in candidate_rows:
        if not isinstance(row, Mapping):
            malformed_positions.append(-1)
            continue
        position = int(row.get("canonical_position", -1))
        observed_positions.add(position)
        residue = residue_by_position.get(position)
        if (
            residue is None
            or residue.get("wt_aa") != row.get("wt_aa")
            or residue.get("structure_chain_id") != row.get("structure_chain_id")
            or residue.get("structure_residue_id") != row.get("structure_residue_id")
            or row.get("policy") != "candidate_prior_not_mask_authoritative"
            or not row.get("source_locator")
            or not row.get("evidence_basis")
        ):
            malformed_positions.append(position)
    if observed_positions != expected_positions or malformed_positions:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.mask.manual_mask_authority_candidate_prior_mismatch",
                message=(
                    "manual_mask_authority.yaml Wang/Ec86 candidate priors must match audited "
                    f"ontology positions {sorted(expected_positions)}"
                ),
                path=str(path),
            )
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
