"""Roster-cache materialization tests for Eco1 conservation source sequences."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache import (
    materialize_conservation_roster_cache,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.source_sequences.roster_cache._fixtures import (
    roster_row,
    write_provider_sources,
    write_roster_table,
)


def test_roster_cache_materializer_writes_source_records_and_provider_fastas(tmp_path: Path) -> None:
    roster_path = write_roster_table(tmp_path)
    provider_root = write_provider_sources(tmp_path)

    result = materialize_conservation_roster_cache(
        repo_root=repo_root(),
        roster_table=roster_path,
        provider_source_root=provider_root,
        cache_root=tmp_path / "cache",
        require_roster_source_hash=False,
    )

    source_records = yaml.safe_load(result.source_records_path.read_text(encoding="utf-8"))
    records = source_records["records"]
    assert source_records["schema_id"] == "eco1_rt_repack.conservation_source_sequence_cache.records"
    assert {row["profile_id"] for row in records} == {"broad_retron_rt", "eco1_like_retron_rt"}
    assert _records_for(records, "broad_retron_rt", "included") == 41
    assert _records_for(records, "eco1_like_retron_rt", "included") == 20
    assert any(
        row["accession"] == "WP_099010551.1"
        and row["status"] == "excluded"
        and row["exclusion_reason"] == "known_public_accession_mismatch_with_ec86kit_target"
        for row in records
    )

    provider_cache_root = result.cache_root / "provider_caches"
    assert (provider_cache_root / "ncbi_protein_efetch.fasta").is_file()
    assert (provider_cache_root / "bv_brc_feature_protein_fasta.fasta").is_file()

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_id"] == "eco1_rt_repack.conservation_source_sequence_cache.index"
    assert manifest["profile_counts"]["broad_retron_rt"]["included"] == 41
    assert manifest["profile_counts"]["eco1_like_retron_rt"]["included"] == 20


def test_roster_cache_materializer_rejects_hash_drift_by_default(tmp_path: Path) -> None:
    roster_path = write_roster_table(tmp_path)
    provider_root = write_provider_sources(tmp_path)

    with pytest.raises(ValueError, match="roster source hash"):
        materialize_conservation_roster_cache(
            repo_root=repo_root(),
            roster_table=roster_path,
            provider_source_root=provider_root,
            cache_root=tmp_path / "cache",
        )


def test_roster_cache_materializer_rejects_unsupported_accession_provider(tmp_path: Path) -> None:
    roster_path = write_roster_table(tmp_path, extra_rows=[roster_row("bad_accession", "BAD_1", subtype="II-A3")])
    provider_root = write_provider_sources(tmp_path)

    with pytest.raises(ValueError, match="unsupported accession provider"):
        materialize_conservation_roster_cache(
            repo_root=repo_root(),
            roster_table=roster_path,
            provider_source_root=provider_root,
            cache_root=tmp_path / "cache",
            require_roster_source_hash=False,
        )


def test_roster_cache_materializer_rejects_missing_provider_sequence(tmp_path: Path) -> None:
    roster_path = write_roster_table(tmp_path)
    provider_root = write_provider_sources(tmp_path, omit_accessions={"WP_100000010.1"})

    with pytest.raises(ValueError, match="missing provider source sequence"):
        materialize_conservation_roster_cache(
            repo_root=repo_root(),
            roster_table=roster_path,
            provider_source_root=provider_root,
            cache_root=tmp_path / "cache",
            require_roster_source_hash=False,
        )


def test_roster_cache_materializer_accepts_explicitly_excluded_missing_provider_sequence(tmp_path: Path) -> None:
    roster_path = write_roster_table(
        tmp_path,
        extra_rows=[
            roster_row(
                "explicit_missing",
                "WP_300000001.1",
                status="excluded",
                exclusion_reason="provider_unresolved",
            )
        ],
    )
    provider_root = write_provider_sources(tmp_path)

    result = materialize_conservation_roster_cache(
        repo_root=repo_root(),
        roster_table=roster_path,
        provider_source_root=provider_root,
        cache_root=tmp_path / "cache",
        require_roster_source_hash=False,
    )

    source_records = yaml.safe_load(result.source_records_path.read_text(encoding="utf-8"))
    assert any(
        row["accession"] == "WP_300000001.1"
        and row["status"] == "excluded"
        and row["exclusion_reason"] == "provider_unresolved"
        for row in source_records["records"]
    )


def _records_for(records: list[dict[str, str]], profile_id: str, status: str) -> int:
    return sum(1 for row in records if row["profile_id"] == profile_id and row["status"] == status)
