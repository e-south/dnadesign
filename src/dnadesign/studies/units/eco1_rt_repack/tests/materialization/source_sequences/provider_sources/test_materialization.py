"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/source_sequences/provider_sources/test_materialization.py

Provider-source acquisition tests for Eco1 conservation source sequences.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.provider_sources import (
    materialize_provider_source_fastas,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.source_sequences.roster_cache._fixtures import (
    protein_sequence,
    roster_row,
    write_roster_table,
)


def test_provider_source_materializer_writes_provider_fastas_and_manifest(tmp_path: Path) -> None:
    roster_path = write_roster_table(tmp_path)

    result = materialize_provider_source_fastas(
        repo_root=repo_root(),
        roster_table=roster_path,
        source_root=tmp_path / "provider_sources",
        fetchers=_fetchers(),
        require_roster_source_hash=False,
    )

    assert (result.source_root / "ncbi_protein_efetch.fasta").is_file()
    assert (result.source_root / "bv_brc_feature_protein_fasta.fasta").is_file()
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_id"] == "eco1_rt_repack.conservation_provider_sources.index"
    assert manifest["status"] == "materialized"
    assert manifest["provider_record_counts"] == {
        "ncbi_protein_efetch": 21,
        "bv_brc_feature_protein_fasta": 20,
    }


def test_provider_source_materializer_rejects_hash_drift_by_default(tmp_path: Path) -> None:
    roster_path = write_roster_table(tmp_path)

    with pytest.raises(ValueError, match="roster table hash"):
        materialize_provider_source_fastas(
            repo_root=repo_root(),
            roster_table=roster_path,
            source_root=tmp_path / "provider_sources",
            fetchers=_fetchers(),
        )


def test_provider_source_materializer_rejects_missing_provider_records(tmp_path: Path) -> None:
    roster_path = write_roster_table(tmp_path)

    with pytest.raises(ValueError, match="did not return"):
        materialize_provider_source_fastas(
            repo_root=repo_root(),
            roster_table=roster_path,
            source_root=tmp_path / "provider_sources",
            fetchers={
                "ncbi_protein_efetch": lambda accessions: dict(_records_for(accessions[:-1])),
                "bv_brc_feature_protein_fasta": lambda accessions: dict(_records_for(accessions)),
            },
            require_roster_source_hash=False,
        )


def test_provider_source_materializer_can_write_unresolved_failure_ledger(tmp_path: Path) -> None:
    roster_path = write_roster_table(tmp_path)

    result = materialize_provider_source_fastas(
        repo_root=repo_root(),
        roster_table=roster_path,
        source_root=tmp_path / "provider_sources",
        fetchers={
            "ncbi_protein_efetch": lambda accessions: dict(_records_for(accessions[:-1])),
            "bv_brc_feature_protein_fasta": lambda accessions: dict(_records_for(accessions)),
        },
        require_roster_source_hash=False,
        write_unresolved_ledger=True,
    )

    assert result.failure_ledger_path is not None
    ledger = yaml.safe_load(result.failure_ledger_path.read_text(encoding="utf-8"))
    assert ledger["schema_id"] == "eco1_rt_repack.conservation_provider_sources.failure_ledger"
    assert ledger["failures"][0]["exclusion_reason"] == "provider_unresolved_in_declared_source"
    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["provider_missing_counts"]["ncbi_protein_efetch"] == 1


def test_provider_source_materializer_accepts_non_wp_ncbi_accessions(tmp_path: Path) -> None:
    roster_path = write_roster_table(
        tmp_path,
        extra_rows=[roster_row("genbank_ncbi", "EIJ70524.1", subtype="II-A3", cluster="42_1", clade="9")],
    )

    result = materialize_provider_source_fastas(
        repo_root=repo_root(),
        roster_table=roster_path,
        source_root=tmp_path / "provider_sources",
        fetchers=_fetchers(),
        require_roster_source_hash=False,
    )

    ncbi_fasta = (result.source_root / "ncbi_protein_efetch.fasta").read_text(encoding="utf-8")
    assert ">EIJ70524.1" in ncbi_fasta


def _fetchers() -> dict[str, object]:
    return {
        "ncbi_protein_efetch": lambda accessions: dict(_records_for(accessions)),
        "bv_brc_feature_protein_fasta": lambda accessions: dict(_records_for(accessions)),
    }


def _records_for(accessions: Sequence[str]) -> list[tuple[str, str]]:
    return [(accession, protein_sequence(index + 1)) for index, accession in enumerate(accessions)]
