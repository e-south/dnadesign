"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/pipeline.py

Materialize Eco1 RT conservation source caches from a hash-pinned roster table.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.contracts import (
    ProviderAccessionPolicy,
    load_conservation_source_contract,
    require_mapping,
    require_text,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.io import sha256_file
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.paths import (
    CONSERVATION_SOURCES,
    DEFAULT_CREATED_AT,
    DEFAULT_SOURCE_CACHE_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache.manifest import (
    write_roster_cache_manifest,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache.models import (
    MaterializedConservationRosterCache,
    RosterRow,
    SourceRecord,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache.providers import (
    load_provider_source_records,
    write_filtered_provider_caches,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache.roster import (
    load_roster_rows,
    select_profile_rows,
)

_CONSERVATION_SOURCES = CONSERVATION_SOURCES
_DEFAULT_SOURCE_CACHE_ROOT = DEFAULT_SOURCE_CACHE_ROOT
_DEFAULT_CREATED_AT = DEFAULT_CREATED_AT
_RECORDS_SCHEMA_ID = "eco1_rt_repack.conservation_source_sequence_cache.records"
_KNOWN_TARGET_EXCLUSION_REASON = "known_public_accession_mismatch_with_ec86kit_target"


def materialize_conservation_roster_cache(
    *,
    repo_root: Path | None = None,
    roster_table: Path,
    provider_source_root: Path,
    cache_root: Path | None = None,
    created_at: str = _DEFAULT_CREATED_AT,
    require_roster_source_hash: bool = True,
    provider_failure_ledger: Path | None = None,
) -> MaterializedConservationRosterCache:
    """Materialize source_records.yaml and filtered provider FASTA caches."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    sources_path = root / _CONSERVATION_SOURCES
    cache = _resolve_path(root, cache_root or _DEFAULT_SOURCE_CACHE_ROOT)
    roster_path = _resolve_path(root, roster_table)
    provider_root = _resolve_path(root, provider_source_root)

    roster_sha256 = "sha256:" + sha256_file(roster_path)
    source_contract = load_conservation_source_contract(sources_path)
    source_groups = source_contract.source_groups
    profile_ids = source_contract.profile_ids
    provider_ids = source_contract.provider_ids
    accession_policy = ProviderAccessionPolicy.from_contract(source_contract)
    _validate_roster_source_hashes(
        roster_sha256=roster_sha256,
        source_groups=source_groups,
        profile_ids=profile_ids,
        require_roster_source_hash=require_roster_source_hash,
    )

    roster_rows = load_roster_rows(roster_path, accession_field=source_contract.accession_field)
    provider_sources = load_provider_source_records(provider_root, provider_ids)
    provider_failure_reasons = _load_provider_failure_reasons(
        _resolve_path(root, provider_failure_ledger) if provider_failure_ledger else None
    )

    source_records, provider_accessions = _build_source_records(
        roster_rows=roster_rows,
        source_groups=source_groups,
        profile_ids=profile_ids,
        accession_policy=accession_policy,
        known_target_accession=source_contract.known_public_target_accession,
        provider_sources=provider_sources,
        provider_failure_reasons=provider_failure_reasons,
    )

    cache.mkdir(parents=True, exist_ok=True)
    provider_cache_paths = write_filtered_provider_caches(
        provider_caches=provider_sources,
        provider_accessions=provider_accessions,
        cache_root=cache,
    )
    source_records_path = cache / "source_records.yaml"
    _write_source_records(
        source_records_path,
        source_records=source_records,
        roster_sha256=roster_sha256,
        conservation_sources_sha256="sha256:" + sha256_file(sources_path),
        created_at=created_at,
    )
    provider_cache_hashes = {
        f"provider_cache_{provider_id}": "sha256:" + sha256_file(path)
        for provider_id, path in provider_cache_paths.items()
    }
    manifest_path = cache / "source_cache_manifest.yaml"
    write_roster_cache_manifest(
        manifest_path,
        roster_table=roster_path,
        roster_table_sha256=roster_sha256,
        conservation_sources_sha256="sha256:" + sha256_file(sources_path),
        source_records_path=source_records_path,
        provider_cache_hashes=provider_cache_hashes,
        profile_counts=_profile_counts(source_records, profile_ids=profile_ids),
        roster_hash_policy="required" if require_roster_source_hash else "fixture_uncontracted_hash_allowed",
        created_at=created_at,
    )
    return MaterializedConservationRosterCache(
        cache_root=cache,
        source_records_path=source_records_path,
        provider_cache_paths=provider_cache_paths,
        manifest_path=manifest_path,
    )


def _build_source_records(
    *,
    roster_rows: Sequence[RosterRow],
    source_groups: Mapping[str, Mapping[str, Any]],
    profile_ids: Sequence[str],
    accession_policy: ProviderAccessionPolicy,
    known_target_accession: str,
    provider_sources: Mapping[str, Any],
    provider_failure_reasons: Mapping[tuple[str, str], str],
) -> tuple[list[SourceRecord], dict[str, list[str]]]:
    source_records: list[SourceRecord] = []
    provider_accessions: dict[str, list[str]] = {provider_id: [] for provider_id in accession_policy.provider_ids}
    seen_records: set[tuple[str, str]] = set()

    for profile_id in profile_ids:
        selected_rows = select_profile_rows(
            roster_rows,
            profile_id=profile_id,
            source_group=require_mapping(source_groups.get(profile_id), f"source group {profile_id}"),
        )
        for row in selected_rows:
            provider_id = accession_policy.provider_for_accession(row.accession)
            record_id = _record_id(profile_id, row)
            key = (profile_id, row.accession)
            if key in seen_records:
                raise ValueError(f"duplicate accession {row.accession!r} in profile {profile_id!r}")
            seen_records.add(key)
            if row.status == "excluded":
                source_records.append(
                    SourceRecord(
                        profile_id=profile_id,
                        record_id=record_id,
                        provider_id=provider_id,
                        accession=row.accession,
                        status="excluded",
                        exclusion_reason=row.exclusion_reason,
                    )
                )
                continue
            if row.accession == known_target_accession:
                source_records.append(
                    SourceRecord(
                        profile_id=profile_id,
                        record_id=record_id,
                        provider_id=provider_id,
                        accession=row.accession,
                        status="excluded",
                        exclusion_reason=_KNOWN_TARGET_EXCLUSION_REASON,
                    )
                )
                continue
            if row.accession not in provider_sources[provider_id].records:
                failure_reason = provider_failure_reasons.get((provider_id, row.accession))
                if failure_reason:
                    source_records.append(
                        SourceRecord(
                            profile_id=profile_id,
                            record_id=record_id,
                            provider_id=provider_id,
                            accession=row.accession,
                            status="excluded",
                            exclusion_reason=failure_reason,
                        )
                    )
                    continue
                raise ValueError(
                    f"missing provider source sequence {row.accession!r} for provider {provider_id!r}; "
                    "exclude it explicitly before materializing source_records.yaml"
                )
            if row.accession not in provider_accessions[provider_id]:
                provider_accessions[provider_id].append(row.accession)
            source_records.append(
                SourceRecord(
                    profile_id=profile_id,
                    record_id=record_id,
                    provider_id=provider_id,
                    accession=row.accession,
                    status="included",
                )
            )
    return source_records, provider_accessions


def _load_provider_failure_reasons(path: Path | None) -> dict[tuple[str, str], str]:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"provider failure ledger must be a YAML mapping: {path}")
    failures = payload.get("failures")
    if not isinstance(failures, list):
        raise ValueError("provider failure ledger must declare failures as a list")
    reasons: dict[tuple[str, str], str] = {}
    for index, failure in enumerate(failures):
        if not isinstance(failure, Mapping):
            raise ValueError(f"provider failure ledger failures[{index}] must be a mapping")
        provider_id = require_text(failure, "provider_id")
        accession = require_text(failure, "accession")
        reason = require_text(failure, "exclusion_reason")
        reasons[(provider_id, accession)] = reason
    return reasons


def _write_source_records(
    path: Path,
    *,
    source_records: Sequence[SourceRecord],
    roster_sha256: str,
    conservation_sources_sha256: str,
    created_at: str,
) -> None:
    payload = {
        "schema_id": _RECORDS_SCHEMA_ID,
        "schema_version": 1,
        "version": 1,
        "study_id": "eco1_rt_repack",
        "status": "materialized",
        "upstream_hashes": {
            "conservation_sources_yaml": conservation_sources_sha256,
            "roster_table": roster_sha256,
        },
        "created_at": created_at,
        "records": [record.to_yaml_row() for record in source_records],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _profile_counts(source_records: Sequence[SourceRecord], *, profile_ids: Sequence[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {profile_id: {"included": 0, "excluded": 0} for profile_id in profile_ids}
    for record in source_records:
        counts[record.profile_id][record.status] += 1
    return counts


def _record_id(profile_id: str, row: RosterRow) -> str:
    node = re.sub(r"[^A-Za-z0-9_.-]+", "_", row.node_id).strip("_") or f"row_{row.row_index}"
    return f"{profile_id}__{node}__{row.row_index:04d}"


def _validate_roster_source_hashes(
    *,
    roster_sha256: str,
    source_groups: Mapping[str, Mapping[str, Any]],
    profile_ids: Sequence[str],
    require_roster_source_hash: bool,
) -> None:
    if not require_roster_source_hash:
        return
    for profile_id in profile_ids:
        roster_source = require_mapping(
            require_mapping(source_groups.get(profile_id), f"source group {profile_id}").get("roster_source"),
            f"source group {profile_id} roster_source",
        )
        expected = require_text(roster_source, "source_sha256")
        if _normalized_sha256(roster_sha256) != _normalized_sha256(expected):
            raise ValueError(
                f"roster source hash for {profile_id!r} must match conservation-sources.yaml: "
                f"expected {expected}, observed {roster_sha256}"
            )


def _normalized_sha256(value: str) -> str:
    return value.removeprefix("sha256:")


def _resolve_path(repo_root: Path, path: Path) -> Path:
    resolved = path.expanduser()
    return resolved if resolved.is_absolute() else (repo_root / resolved).resolve()


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")
