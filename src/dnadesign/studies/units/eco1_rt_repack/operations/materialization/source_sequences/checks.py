"""Sufficiency checks for Eco1 conservation source FASTA bundles."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.io import (
    load_fasta_records_ordered,
    load_yaml_mapping,
    resolve_path,
    sha256_file,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.issues import (
    append_field_mismatch,
    invalid_manifest_record_issue,
)

_INDEX_SCHEMA_ID = "eco1_rt_repack.conservation_source_sequence_bundle.index"
_PROFILE_SCHEMA_ID = "eco1_rt_repack.conservation_source_sequence_bundle.profile"
_PROVIDER_ACCESSION_PATTERNS = {
    "ncbi_protein_efetch": re.compile(r"^WP_\d+\.\d+$"),
    "bv_brc_feature_protein_fasta": re.compile(r"^fig\|\d+\.\d+\.peg\.\d+$"),
}


@dataclass(frozen=True)
class SourceSequenceSufficiencyContext:
    """Resolved inputs for source-sequence bundle sufficiency validation."""

    repo_root: Path
    output_root: Path
    source_cache_root: Path
    bundle_root: Path
    conservation_sources_path: Path
    conservation_sources: Mapping[str, Any]


def collect_source_sequence_sufficiency_issues(
    context: SourceSequenceSufficiencyContext,
) -> tuple[ContractIssue, ...]:
    """Collect source-sequence sufficiency failures without running alignment."""

    sources = context.conservation_sources
    profile_ids = _required_profile_ids(sources)
    provider_ids = _required_provider_ids(sources)
    source_groups = _source_groups_by_id(sources)
    target_row_id = _require_nested_text(sources, ("alignment_policy", "target_row_id"))
    target_sequence_hash = _require_nested_text(sources, ("target_sequence", "reference_sequence_hash"))
    issues: list[ContractIssue] = []

    source_records_path = context.source_cache_root / "source_records.yaml"
    provider_cache_root = context.source_cache_root / "provider_caches"
    _validate_source_cache_presence(issues, source_cache_root=context.source_cache_root)

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
        profile_ids=profile_ids,
        provider_ids=provider_ids,
        target_row_id=target_row_id,
        target_sequence_hash=target_sequence_hash,
        source_records_path=source_records_path,
        provider_cache_root=provider_cache_root,
    )
    profile_manifests = _profile_manifest_paths(index_manifest, root=context.repo_root)
    for profile_id in profile_ids:
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
            source_group=_require_mapping(source_groups.get(profile_id), f"source group {profile_id}"),
            profile_id=profile_id,
            target_row_id=target_row_id,
            target_sequence_hash=target_sequence_hash,
            provider_ids=provider_ids,
            source_records_path=source_records_path,
            provider_cache_root=provider_cache_root,
            root=context.repo_root,
        )

    _validate_residue_map_presence(issues, output_root=context.output_root)
    return tuple(issues)


def _validate_source_cache_presence(issues: list[ContractIssue], *, source_cache_root: Path) -> None:
    if not source_cache_root.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.source_cache_root_missing",
                message="source-sequence sufficiency requires the real provider source cache root",
                path=str(source_cache_root),
            )
        )
    source_records_path = source_cache_root / "source_records.yaml"
    if not source_records_path.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.source_records_missing",
                message="source-sequence sufficiency requires source_records.yaml",
                path=str(source_records_path),
            )
        )


def _validate_index_manifest(
    issues: list[ContractIssue],
    *,
    index_manifest: Mapping[str, Any],
    index_path: Path,
    profile_ids: Sequence[str],
    provider_ids: Sequence[str],
    target_row_id: str,
    target_sequence_hash: str,
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
            "profile_ids": list(profile_ids),
            "target_row_id": target_row_id,
            "target_sequence_hash": target_sequence_hash,
        },
    )
    upstream_hashes = _require_mapping(index_manifest.get("upstream_hashes"), "index upstream_hashes")
    _validate_cache_hashes(
        issues,
        upstream_hashes=upstream_hashes,
        source_records_path=source_records_path,
        provider_cache_root=provider_cache_root,
        provider_ids=provider_ids,
        path=index_path,
    )


def _validate_profile_manifest(
    issues: list[ContractIssue],
    *,
    profile_manifest: Mapping[str, Any],
    manifest_path: Path,
    source_group: Mapping[str, Any],
    profile_id: str,
    target_row_id: str,
    target_sequence_hash: str,
    provider_ids: Sequence[str],
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
            "target_row_id": target_row_id,
            "target_sequence_hash": target_sequence_hash,
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
    upstream_hashes = _require_mapping(profile_manifest.get("upstream_hashes"), "profile upstream_hashes")
    _validate_cache_hashes(
        issues,
        upstream_hashes=upstream_hashes,
        source_records_path=source_records_path,
        provider_cache_root=provider_cache_root,
        provider_ids=provider_ids,
        path=manifest_path,
    )
    _validate_record_accessions(
        issues,
        records=included_records,
        profile_id=profile_id,
        manifest_path=manifest_path,
        provider_ids=provider_ids,
        require_exclusion_reason=False,
    )
    _validate_record_accessions(
        issues,
        records=excluded_records,
        profile_id=profile_id,
        manifest_path=manifest_path,
        provider_ids=provider_ids,
        require_exclusion_reason=True,
    )
    _validate_source_fasta(
        issues,
        profile_manifest=profile_manifest,
        manifest_path=manifest_path,
        profile_id=profile_id,
        target_row_id=target_row_id,
        target_sequence_hash=target_sequence_hash,
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


def _validate_cache_hashes(
    issues: list[ContractIssue],
    *,
    upstream_hashes: Mapping[str, Any],
    source_records_path: Path,
    provider_cache_root: Path,
    provider_ids: Sequence[str],
    path: Path,
) -> None:
    if source_records_path.exists() and upstream_hashes.get("source_records_yaml") != "sha256:" + sha256_file(
        source_records_path
    ):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.source_records_hash_mismatch",
                message="source_records.yaml hash must match source-sequence manifests",
                path=str(path),
            )
        )
    for provider_id in provider_ids:
        provider_path = provider_cache_root / f"{provider_id}.fasta"
        provider_key = f"provider_cache_{provider_id}"
        if not provider_path.exists():
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.source_sequences.provider_cache_missing",
                    message=f"provider cache {provider_id!r} is missing",
                    path=str(provider_path),
                )
            )
            continue
        if upstream_hashes.get(provider_key) != "sha256:" + sha256_file(provider_path):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.source_sequences.provider_cache_hash_mismatch",
                    message=f"provider cache hash for {provider_id!r} must match source-sequence manifests",
                    path=str(path),
                )
            )


def _validate_record_accessions(
    issues: list[ContractIssue],
    *,
    records: Sequence[Any],
    profile_id: str,
    manifest_path: Path,
    provider_ids: Sequence[str],
    require_exclusion_reason: bool,
) -> None:
    declared_provider_ids = set(provider_ids)
    for record in records:
        if not isinstance(record, Mapping):
            issues.append(invalid_manifest_record_issue(profile_id, manifest_path))
            continue
        provider_id = str(record.get("provider_id", ""))
        accession = str(record.get("accession", ""))
        if provider_id not in declared_provider_ids:
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.source_sequences.undeclared_provider",
                    message=f"{profile_id} uses undeclared provider_id {provider_id!r}",
                    path=str(manifest_path),
                )
            )
            continue
        if not _valid_provider_accession(provider_id, accession):
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.source_sequences.invalid_provider_accession",
                    message=f"{profile_id} accession {accession!r} is not a real-looking {provider_id} id",
                    path=str(manifest_path),
                )
            )
        if require_exclusion_reason and not str(record.get("exclusion_reason", "")).strip():
            issues.append(
                ContractIssue(
                    check_id="eco1_rt.source_sequences.exclusion_reason_missing",
                    message=f"{profile_id} excluded records must include exclusion_reason",
                    path=str(manifest_path),
                )
            )


def _validate_source_fasta(
    issues: list[ContractIssue],
    *,
    profile_manifest: Mapping[str, Any],
    manifest_path: Path,
    profile_id: str,
    target_row_id: str,
    target_sequence_hash: str,
    included_count: int,
    root: Path,
) -> None:
    fasta_path = resolve_path(root, Path(str(profile_manifest.get("fasta_path", ""))))
    if not fasta_path.exists():
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.source_fasta_missing",
                message=f"{profile_id} source FASTA is missing",
                path=str(fasta_path),
            )
        )
        return
    if profile_manifest.get("fasta_sha256") != "sha256:" + sha256_file(fasta_path):
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.source_fasta_hash_mismatch",
                message=f"{profile_id} source FASTA hash must match its profile manifest",
                path=str(manifest_path),
            )
        )
    records = load_fasta_records_ordered(fasta_path)
    if not records or records[0][0] != target_row_id:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.target_row_not_first",
                message=f"{profile_id} source FASTA must start with the ec86kit target row",
                path=str(fasta_path),
            )
        )
    elif "sha256:" + hashlib.sha256(records[0][1].encode("utf-8")).hexdigest() != target_sequence_hash:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.target_sequence_hash_mismatch",
                message=f"{profile_id} target FASTA row must match the ec86kit sequence hash",
                path=str(fasta_path),
            )
        )
    if len(records) != included_count + 1:
        issues.append(
            ContractIssue(
                check_id="eco1_rt.source_sequences.source_fasta_record_count_mismatch",
                message=f"{profile_id} source FASTA must contain target plus included records only",
                path=str(fasta_path),
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


def _valid_provider_accession(provider_id: str, accession: str) -> bool:
    pattern = _PROVIDER_ACCESSION_PATTERNS.get(provider_id)
    return pattern is not None and pattern.fullmatch(accession) is not None


def _required_min_included_records(source_group: Mapping[str, Any]) -> int:
    value = source_group.get("min_non_gap_count")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("source group min_non_gap_count must be a positive integer")
    return value


def _required_profile_ids(sources: Mapping[str, Any]) -> list[str]:
    acceptance = _require_mapping(sources.get("phase1_acceptance"), "phase1_acceptance")
    profile_ids = acceptance.get("required_profile_ids")
    if not isinstance(profile_ids, list) or not all(isinstance(item, str) and item for item in profile_ids):
        raise ValueError("phase1_acceptance.required_profile_ids must be a non-empty list of strings")
    return list(profile_ids)


def _required_provider_ids(sources: Mapping[str, Any]) -> list[str]:
    acceptance = _require_mapping(sources.get("phase1_acceptance"), "phase1_acceptance")
    provider_ids = acceptance.get("required_provider_ids")
    if not isinstance(provider_ids, list) or not all(isinstance(item, str) and item for item in provider_ids):
        raise ValueError("phase1_acceptance.required_provider_ids must be a non-empty list of strings")
    return list(provider_ids)


def _source_groups_by_id(sources: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    groups = sources.get("source_groups")
    if not isinstance(groups, list):
        raise ValueError("conservation-sources.yaml must declare source_groups")
    grouped: dict[str, Mapping[str, Any]] = {}
    for group in groups:
        mapping = _require_mapping(group, "source group")
        grouped[_require_text(mapping, "profile_id")] = mapping
    return grouped


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _require_nested_text(payload: Mapping[str, Any], fields: Sequence[str]) -> str:
    current: Any = payload
    for field in fields:
        current = _require_mapping(current, ".".join(fields)).get(field)
    if not isinstance(current, str) or not current.strip():
        raise ValueError(f"{'.'.join(fields)} must be a non-empty string")
    return current.strip()


def _require_text(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()
