"""Manifest-level sufficiency checks for Eco1 conservation source bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.contracts import (
    ConservationSourceContract,
    parse_conservation_source_contract,
    require_mapping,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.io import (
    load_yaml_mapping,
    resolve_path,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.issues import (
    append_field_mismatch,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.sufficiency.context import (
    SourceSequenceSufficiencyContext,
)

from .cache_hashes import validate_cache_hashes, validate_source_cache_presence
from .fasta import validate_source_fasta
from .provider_policy import validate_record_accessions

_INDEX_SCHEMA_ID = "eco1_rt_repack.conservation_source_sequence_bundle.index"
_PROFILE_SCHEMA_ID = "eco1_rt_repack.conservation_source_sequence_bundle.profile"


def collect_source_sequence_sufficiency_issues(
    context: SourceSequenceSufficiencyContext,
) -> tuple[ContractIssue, ...]:
    """Collect source-sequence sufficiency failures without running alignment."""

    contract = parse_conservation_source_contract(context.conservation_sources)
    source_records_path = context.source_cache_root / "source_records.yaml"
    provider_cache_root = context.source_cache_root / "provider_caches"
    issues: list[ContractIssue] = []
    validate_source_cache_presence(issues, source_cache_root=context.source_cache_root)

    index_path = context.bundle_root / "source_sequence_bundle_manifest.yaml"
    if not index_path.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.bundle_manifest_missing",
                message="source-sequence sufficiency requires source_sequence_bundle_manifest.yaml",
                path=str(index_path),
            )
        )
        return tuple(issues)

    index_manifest = load_yaml_mapping(index_path)
    _validate_index_manifest(
        issues,
        index_manifest=index_manifest,
        index_path=index_path,
        contract=contract,
        source_records_path=source_records_path,
        provider_cache_root=provider_cache_root,
    )
    profile_manifests = _profile_manifest_paths(index_manifest, root=context.repo_root)
    for profile_id in contract.profile_ids:
        manifest_path = profile_manifests.get(profile_id)
        if manifest_path is None or not manifest_path.exists():
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.source_sequences.profile_manifest_missing",
                    message=f"source-sequence profile manifest is missing for {profile_id!r}",
                    path=str(manifest_path or context.bundle_root / f"{profile_id}.source_manifest.yaml"),
                )
            )
            continue
        _validate_profile_manifest(
            issues,
            profile_manifest=load_yaml_mapping(manifest_path),
            manifest_path=manifest_path,
            source_group=require_mapping(contract.source_groups.get(profile_id), f"source group {profile_id}"),
            profile_id=profile_id,
            contract=contract,
            source_records_path=source_records_path,
            provider_cache_root=provider_cache_root,
            root=context.repo_root,
        )

    _validate_residue_map_presence(issues, output_root=context.output_root)
    return tuple(issues)


def _validate_index_manifest(
    issues: list[ContractIssue],
    *,
    index_manifest: Mapping[str, Any],
    index_path: Path,
    contract: ConservationSourceContract,
    source_records_path: Path,
    provider_cache_root: Path,
) -> None:
    append_field_mismatch(
        issues,
        payload=index_manifest,
        path=index_path,
        expected={
            "schema_id": _INDEX_SCHEMA_ID,
            "status": "materialized",
            "profile_ids": list(contract.profile_ids),
            "target_row_id": contract.target_row_id,
            "target_sequence_hash": contract.target_sequence_hash,
        },
    )
    upstream_hashes = require_mapping(index_manifest.get("upstream_hashes"), "index upstream_hashes")
    validate_cache_hashes(
        issues,
        upstream_hashes=upstream_hashes,
        source_records_path=source_records_path,
        provider_cache_root=provider_cache_root,
        provider_ids=contract.provider_ids,
        path=index_path,
    )


def _validate_profile_manifest(
    issues: list[ContractIssue],
    *,
    profile_manifest: Mapping[str, Any],
    manifest_path: Path,
    source_group: Mapping[str, Any],
    profile_id: str,
    contract: ConservationSourceContract,
    source_records_path: Path,
    provider_cache_root: Path,
    root: Path,
) -> None:
    append_field_mismatch(
        issues,
        payload=profile_manifest,
        path=manifest_path,
        expected={
            "schema_id": _PROFILE_SCHEMA_ID,
            "status": "materialized",
            "profile_id": profile_id,
            "target_row_id": contract.target_row_id,
            "target_sequence_hash": contract.target_sequence_hash,
        },
    )
    included_records = _require_list(profile_manifest.get("included_records"), "included_records")
    excluded_records = _require_list(profile_manifest.get("excluded_records"), "excluded_records")
    included_count = int(profile_manifest.get("included_record_count", -1))
    _validate_record_counts(
        issues,
        manifest_path=manifest_path,
        source_group=source_group,
        profile_id=profile_id,
        included_records=included_records,
        included_count=included_count,
    )
    upstream_hashes = require_mapping(profile_manifest.get("upstream_hashes"), "profile upstream_hashes")
    validate_cache_hashes(
        issues,
        upstream_hashes=upstream_hashes,
        source_records_path=source_records_path,
        provider_cache_root=provider_cache_root,
        provider_ids=contract.provider_ids,
        path=manifest_path,
    )
    validate_record_accessions(
        issues,
        records=included_records,
        profile_id=profile_id,
        manifest_path=manifest_path,
        provider_ids=contract.provider_ids,
        require_exclusion_reason=False,
    )
    validate_record_accessions(
        issues,
        records=excluded_records,
        profile_id=profile_id,
        manifest_path=manifest_path,
        provider_ids=contract.provider_ids,
        require_exclusion_reason=True,
    )
    validate_source_fasta(
        issues,
        profile_manifest=profile_manifest,
        manifest_path=manifest_path,
        profile_id=profile_id,
        target_row_id=contract.target_row_id,
        target_sequence_hash=contract.target_sequence_hash,
        included_count=included_count,
        root=root,
    )


def _validate_record_counts(
    issues: list[ContractIssue],
    *,
    manifest_path: Path,
    source_group: Mapping[str, Any],
    profile_id: str,
    included_records: Sequence[Any],
    included_count: int,
) -> None:
    if included_count != len(included_records):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.manifest_record_count_mismatch",
                message=f"{profile_id} included_record_count must match included_records length",
                path=str(manifest_path),
            )
        )
    min_included = _required_min_included_records(source_group)
    if included_count < min_included:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.insufficient_included_records",
                message=(
                    f"{profile_id} has {included_count} included source records; "
                    f"requires at least {min_included} before alignment"
                ),
                path=str(manifest_path),
            )
        )


def _validate_residue_map_presence(issues: list[ContractIssue], *, output_root: Path) -> None:
    if output_root.exists() and not (output_root / "residue_map.parquet").exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.residue_map_missing",
                message="source-sequence sufficiency expects residue_map.parquet in the output root",
                path=str(output_root / "residue_map.parquet"),
            )
        )


def _profile_manifest_paths(index_manifest: Mapping[str, Any], *, root: Path) -> dict[str, Path]:
    manifests = index_manifest.get("profile_manifests")
    if not isinstance(manifests, Mapping):
        return {}
    return {str(profile_id): resolve_path(root, Path(str(path))) for profile_id, path in manifests.items()}


def _required_min_included_records(source_group: Mapping[str, Any]) -> int:
    value = source_group.get("min_non_gap_count")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("source group min_non_gap_count must be a positive integer")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value
