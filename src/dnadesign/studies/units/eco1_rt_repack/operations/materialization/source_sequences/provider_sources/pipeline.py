"""Materialize explicit provider FASTA sources for Eco1 conservation source sequences."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.contracts import (
    ProviderAccessionPolicy,
    load_conservation_source_contract,
    require_mapping,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.io import (
    resolve_path,
    sha256_file,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.paths import (
    CONSERVATION_SOURCES,
    DEFAULT_CREATED_AT,
    DEFAULT_PROVIDER_SOURCE_ROOT,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache.roster import (
    load_roster_rows,
    select_profile_rows,
)

from .bv_brc import fetch_bv_brc_feature_protein_fastas
from .fasta import write_provider_fasta
from .manifest import write_provider_failure_ledger, write_provider_source_manifest
from .ncbi import fetch_ncbi_protein_fastas

ProviderFetcher = Callable[[Sequence[str]], Mapping[str, str]]


@dataclass(frozen=True)
class MaterializedProviderSourceFastas:
    """Paths emitted by one provider-source materialization pass."""

    source_root: Path
    fasta_paths: dict[str, Path]
    manifest_path: Path
    failure_ledger_path: Path | None


def materialize_provider_source_fastas(
    *,
    repo_root: Path | None = None,
    roster_table: Path,
    source_root: Path | None = None,
    created_at: str = DEFAULT_CREATED_AT,
    fetchers: Mapping[str, ProviderFetcher] | None = None,
    require_roster_source_hash: bool = True,
    write_unresolved_ledger: bool = False,
) -> MaterializedProviderSourceFastas:
    """Fetch and write declared provider FASTA source files for the Mestre roster."""

    root = (repo_root or _find_repo_root(Path.cwd())).expanduser().resolve()
    roster_path = resolve_path(root, roster_table)
    sources_path = root / CONSERVATION_SOURCES
    destination = resolve_path(root, source_root or DEFAULT_PROVIDER_SOURCE_ROOT)
    destination.mkdir(parents=True, exist_ok=True)

    source_contract = load_conservation_source_contract(sources_path)
    roster_sha256 = "sha256:" + sha256_file(roster_path)
    _validate_roster_hash(
        roster_sha256=roster_sha256,
        source_contract=source_contract.source_groups,
        require_roster_source_hash=require_roster_source_hash,
    )

    roster_rows = load_roster_rows(roster_path, accession_field=source_contract.accession_field)
    accession_policy = ProviderAccessionPolicy.from_contract(source_contract)
    provider_accessions = _collect_provider_accessions(
        roster_rows=roster_rows,
        source_groups=source_contract.source_groups,
        profile_ids=source_contract.profile_ids,
        accession_policy=accession_policy,
        known_target_accession=source_contract.known_public_target_accession,
    )

    provider_fetchers = dict(_default_fetchers())
    provider_fetchers.update(fetchers or {})
    fasta_paths: dict[str, Path] = {}
    provider_source_hashes: dict[str, str] = {}
    provider_record_counts: dict[str, int] = {}
    provider_requested_counts: dict[str, int] = {}
    provider_missing_counts: dict[str, int] = {}
    provider_failures: dict[str, list[str]] = {}
    for provider_id in source_contract.provider_ids:
        accessions = provider_accessions[provider_id]
        provider_requested_counts[provider_id] = len(accessions)
        if not accessions:
            raise ValueError(f"provider {provider_id!r} has no requested source accessions")
        fetcher = provider_fetchers.get(provider_id)
        if fetcher is None:
            raise ValueError(f"no fetcher declared for provider {provider_id!r}")
        records = dict(fetcher(accessions))
        missing = _missing_records(requested=accessions, records=records)
        provider_missing_counts[provider_id] = len(missing)
        if missing and not write_unresolved_ledger:
            _raise_missing_records(provider_id=provider_id, missing=missing)
        provider_failures[provider_id] = missing
        fasta_path = destination / f"{provider_id}.fasta"
        write_provider_fasta(
            fasta_path,
            {accession: records[accession] for accession in accessions if accession in records},
        )
        fasta_paths[provider_id] = fasta_path
        provider_source_hashes[f"provider_source_{provider_id}"] = "sha256:" + sha256_file(fasta_path)
        provider_record_counts[provider_id] = len(accessions) - len(missing)

    failure_ledger_path = None
    if any(provider_failures.values()):
        failure_ledger_path = destination / "provider_source_failures.yaml"
        write_provider_failure_ledger(failure_ledger_path, failures=provider_failures, created_at=created_at)
    manifest_path = destination / "provider_source_manifest.yaml"
    write_provider_source_manifest(
        manifest_path,
        roster_table=roster_path,
        roster_table_sha256=roster_sha256,
        conservation_sources_sha256="sha256:" + sha256_file(sources_path),
        provider_source_hashes=provider_source_hashes,
        provider_record_counts=provider_record_counts,
        provider_requested_counts=provider_requested_counts,
        provider_missing_counts=provider_missing_counts,
        failure_ledger_path=failure_ledger_path,
        created_at=created_at,
    )
    return MaterializedProviderSourceFastas(
        source_root=destination,
        fasta_paths=fasta_paths,
        manifest_path=manifest_path,
        failure_ledger_path=failure_ledger_path,
    )


def _collect_provider_accessions(
    *,
    roster_rows: Sequence[object],
    source_groups: Mapping[str, Mapping[str, object]],
    profile_ids: Sequence[str],
    accession_policy: ProviderAccessionPolicy,
    known_target_accession: str,
) -> dict[str, list[str]]:
    provider_accessions: dict[str, list[str]] = {provider_id: [] for provider_id in accession_policy.provider_ids}
    seen: set[tuple[str, str]] = set()
    for profile_id in profile_ids:
        selected_rows = select_profile_rows(
            roster_rows,
            profile_id=profile_id,
            source_group=require_mapping(source_groups.get(profile_id), f"source group {profile_id}"),
        )
        for row in selected_rows:
            if row.status == "excluded" or row.accession == known_target_accession:
                continue
            provider_id = accession_policy.provider_for_accession(row.accession)
            key = (provider_id, row.accession)
            if key in seen:
                continue
            seen.add(key)
            provider_accessions[provider_id].append(row.accession)
    return provider_accessions


def _default_fetchers() -> dict[str, ProviderFetcher]:
    return {
        "ncbi_protein_efetch": fetch_ncbi_protein_fastas,
        "bv_brc_feature_protein_fasta": fetch_bv_brc_feature_protein_fastas,
    }


def _validate_roster_hash(
    *,
    roster_sha256: str,
    source_contract: Mapping[str, Mapping[str, object]],
    require_roster_source_hash: bool,
) -> None:
    if not require_roster_source_hash:
        return
    expected_hashes = {
        require_mapping(group.get("roster_source"), "roster_source").get("source_sha256")
        for group in source_contract.values()
    }
    normalized_expected_hashes = {_normalized_sha256(str(value)) for value in expected_hashes}
    if _normalized_sha256(roster_sha256) not in normalized_expected_hashes:
        raise ValueError(
            "roster table hash must match conservation-sources.yaml before provider-source acquisition: "
            f"observed {roster_sha256}"
        )


def _normalized_sha256(value: str) -> str:
    return value.removeprefix("sha256:")


def _missing_records(*, requested: Sequence[str], records: Mapping[str, str]) -> list[str]:
    return [accession for accession in requested if accession not in records]


def _raise_missing_records(*, provider_id: str, missing: Sequence[str]) -> None:
    sample = ", ".join(missing[:10])
    raise ValueError(f"provider {provider_id!r} did not return {len(missing)} requested records: {sample}")


def _find_repo_root(start: Path) -> Path:
    for parent in (start.resolve(), *start.resolve().parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("repo root with pyproject.toml not found")
