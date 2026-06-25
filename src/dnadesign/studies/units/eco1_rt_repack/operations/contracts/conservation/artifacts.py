"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/conservation/artifacts.py

Materialized conservation-profile validators for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.constants import (
    _REQUIRED_CONSERVATION_PROFILE_COLUMNS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue


def validate_conservation_profile_content(
    path: Path,
    *,
    residue_map_path: Path,
    conservation_sources: Mapping[str, Any],
    source_contract_path: Path,
) -> list[ContractIssue]:
    """Validate materialized conservation evidence against source and residue contracts."""

    issues: list[ContractIssue] = []
    table = pq.read_table(path)
    column_names = set(table.column_names)
    missing_columns = sorted(_REQUIRED_CONSERVATION_PROFILE_COLUMNS - column_names)
    if missing_columns:
        return [
            ContractIssue(
                check_id="eco1_rt.evidence.conservation_profile_missing_columns",
                message=f"conservation_profile.parquet is missing required columns: {missing_columns}",
                path=str(path),
            )
        ]

    target_hash = _require_nested_text(conservation_sources, ("target_sequence", "reference_sequence_hash"))
    source_contract_hash = "sha256:" + _sha256(source_contract_path)
    metadata = table.schema.metadata or {}
    expected_metadata = {
        b"schema_id": b"thread.conservation_profile",
        b"status": b"materialized",
        b"target_sequence_hash": target_hash.encode("utf-8"),
        b"source_contract_hash": source_contract_hash.encode("utf-8"),
    }
    for required_key in (b"schema_version", b"artifact_id", b"created_by", b"created_at", b"upstream_artifact_hashes"):
        if not metadata.get(required_key):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.evidence.conservation_profile_missing_lifecycle_metadata",
                    message=f"conservation_profile.parquet metadata {required_key.decode()} must be present",
                    path=str(path),
                )
            )
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.evidence.conservation_profile_metadata_mismatch",
                    message=f"conservation_profile.parquet metadata {key.decode()} must equal {expected.decode()}",
                    path=str(path),
                )
            )

    required_profile_ids = _required_profile_ids(conservation_sources)
    metadata_profile_ids = _json_metadata_list(metadata.get(b"profile_ids"))
    if metadata_profile_ids != required_profile_ids:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.conservation_profile_metadata_mismatch",
                message="conservation_profile.parquet metadata profile_ids must match source contract order",
                path=str(path),
            )
        )
    upstream_hashes = _json_metadata_mapping(metadata.get(b"upstream_artifact_hashes"))
    source_hash_by_profile = {
        profile_id: upstream_hashes.get(f"{profile_id}_aligned_fasta") for profile_id in required_profile_ids
    }

    residue_rows = pq.read_table(residue_map_path).to_pylist()
    reference_length = len(residue_rows)
    residue_by_position = {row.get("canonical_position"): row for row in residue_rows}
    table_rows = table.to_pylist()
    source_groups = _source_groups_by_id(conservation_sources)
    threshold = float(_require_mapping(conservation_sources.get("source_method"), "source_method")["threshold"])

    _validate_profile_coverage(
        issues,
        table_rows=table_rows,
        required_profile_ids=required_profile_ids,
        reference_length=reference_length,
        path=path,
    )
    _validate_profile_rows(
        issues,
        table_rows=table_rows,
        residue_by_position=residue_by_position,
        source_hash_by_profile=source_hash_by_profile,
        source_groups=source_groups,
        target_hash=target_hash,
        threshold=threshold,
        path=path,
    )
    return issues


def _validate_profile_coverage(
    issues: list[ContractIssue],
    *,
    table_rows: list[dict[str, Any]],
    required_profile_ids: list[str],
    reference_length: int,
    path: Path,
) -> None:
    expected_positions = list(range(1, reference_length + 1))
    for profile_id in required_profile_ids:
        positions = [row.get("canonical_position") for row in table_rows if row.get("profile_id") == profile_id]
        if positions != expected_positions:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.evidence.conservation_profile_position_mismatch",
                    message=f"conservation_profile.parquet must include ordered rows for {profile_id!r}",
                    path=str(path),
                )
            )


def _validate_profile_rows(
    issues: list[ContractIssue],
    *,
    table_rows: list[dict[str, Any]],
    residue_by_position: Mapping[Any, Mapping[str, Any]],
    source_hash_by_profile: Mapping[str, str | None],
    source_groups: Mapping[str, Mapping[str, Any]],
    target_hash: str,
    threshold: float,
    path: Path,
) -> None:
    residue_mismatches: list[tuple[str, int]] = []
    source_hash_mismatches: list[str] = []
    target_hash_mismatches: list[str] = []
    threshold_mismatches: list[str] = []
    mask_mismatches: list[tuple[str, int]] = []
    unresolved_bad: list[tuple[str, int]] = []
    for row in table_rows:
        profile_id = str(row.get("profile_id", ""))
        position = int(row.get("canonical_position", 0))
        residue = residue_by_position.get(position)
        if not isinstance(residue, Mapping):
            residue_mismatches.append((profile_id, position))
            continue
        if row.get("wt_aa") != residue.get("wt_aa") or row.get("mapping_status") != residue.get("mapping_status"):
            residue_mismatches.append((profile_id, position))
            continue

        if row.get("source_hash") != source_hash_by_profile.get(profile_id):
            source_hash_mismatches.append(profile_id)
        if row.get("target_sequence_hash") != target_hash:
            target_hash_mismatches.append(profile_id)
        if row.get("conservation_threshold") != threshold:
            threshold_mismatches.append(profile_id)

        min_non_gap_count = _min_non_gap_count(source_groups, profile_id)
        if row.get("min_non_gap_count") != min_non_gap_count:
            mask_mismatches.append((profile_id, position))
            continue
        expected_pass = (
            row.get("mapping_status") == "mapped"
            and int(row.get("non_gap_count", 0)) >= min_non_gap_count
            and row.get("wt_is_plurality") is True
            and float(row.get("wt_frequency", 0.0)) >= threshold
        )
        if row.get("passes_conservation_mask") is not expected_pass:
            mask_mismatches.append((profile_id, position))
        if row.get("mapping_status") != "mapped" and row.get("passes_conservation_mask") is not False:
            unresolved_bad.append((profile_id, position))

    _append_row_issues(
        issues,
        path=path,
        residue_mismatches=residue_mismatches,
        source_hash_mismatches=source_hash_mismatches,
        target_hash_mismatches=target_hash_mismatches,
        threshold_mismatches=threshold_mismatches,
        mask_mismatches=mask_mismatches,
        unresolved_bad=unresolved_bad,
    )


def _append_row_issues(
    issues: list[ContractIssue],
    *,
    path: Path,
    residue_mismatches: list[tuple[str, int]],
    source_hash_mismatches: list[str],
    target_hash_mismatches: list[str],
    threshold_mismatches: list[str],
    mask_mismatches: list[tuple[str, int]],
    unresolved_bad: list[tuple[str, int]],
) -> None:
    if residue_mismatches:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.conservation_profile_residue_map_mismatch",
                message=(
                    f"conservation_profile.parquet rows disagree with residue_map.parquet: {residue_mismatches[:20]}"
                ),
                path=str(path),
            )
        )
    if source_hash_mismatches:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.conservation_profile_source_hash_mismatch",
                message=(
                    "conservation_profile.parquet source_hash values must match upstream aligned FASTA hashes: "
                    f"{sorted(set(source_hash_mismatches))}"
                ),
                path=str(path),
            )
        )
    if target_hash_mismatches:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.conservation_profile_target_hash_mismatch",
                message="conservation_profile.parquet target_sequence_hash values must match source authority",
                path=str(path),
            )
        )
    if threshold_mismatches:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.conservation_profile_threshold_mismatch",
                message="conservation_profile.parquet threshold values must match source authority",
                path=str(path),
            )
        )
    if mask_mismatches:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.conservation_profile_mask_value_mismatch",
                message=f"conservation-profile mask values disagree with Tao rule: {mask_mismatches[:20]}",
                path=str(path),
            )
        )
    if unresolved_bad:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.evidence.conservation_profile_missing_backbone_has_conservation_mask",
                message=(
                    "missing-backbone conservation-profile rows must not create a conservation protection call: "
                    f"{unresolved_bad[:20]}"
                ),
                path=str(path),
            )
        )


def _required_profile_ids(conservation_sources: Mapping[str, Any]) -> list[str]:
    acceptance = _require_mapping(conservation_sources.get("phase1_acceptance"), "phase1_acceptance")
    profile_ids = acceptance.get("required_profile_ids")
    if not isinstance(profile_ids, list):
        return []
    return [str(item) for item in profile_ids if isinstance(item, str) and item.strip()]


def _source_groups_by_id(conservation_sources: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    groups = conservation_sources.get("source_groups")
    if not isinstance(groups, list):
        return {}
    grouped: dict[str, Mapping[str, Any]] = {}
    for group in groups:
        if isinstance(group, Mapping) and isinstance(group.get("profile_id"), str):
            grouped[str(group["profile_id"])] = group
    return grouped


def _min_non_gap_count(source_groups: Mapping[str, Mapping[str, Any]], profile_id: str) -> int:
    group = source_groups.get(profile_id)
    if not isinstance(group, Mapping):
        return 0
    value = group.get("min_non_gap_count")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _json_metadata_list(value: bytes | None) -> list[str]:
    if not value:
        return []
    loaded = json.loads(value.decode("utf-8"))
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded]


def _json_metadata_mapping(value: bytes | None) -> dict[str, str]:
    if not value:
        return {}
    loaded = json.loads(value.decode("utf-8"))
    if not isinstance(loaded, dict):
        return {}
    return {str(key): str(item) for key, item in loaded.items()}


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_nested_text(payload: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    current: Any = payload
    for field in fields:
        current = _require_mapping(current, ".".join(fields)).get(field)
    if not isinstance(current, str) or not current.strip():
        raise ValueError(f"{'.'.join(fields)} must be a non-empty string")
    return current.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
