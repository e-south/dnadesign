"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/source_sequences/roster_cache/test_materialization.py

Roster-cache materialization tests for Eco1 conservation source sequences.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache import (
    materialize_conservation_roster_cache,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache.roster import (
    load_roster_rows,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.source_sequences.roster_cache._fixtures import (
    ec86kit_target_sequence,
    fragment_sequence,
    no_catalytic_core_sequence,
    roster_row,
    write_provider_sources,
    write_roster_table,
)


def test_roster_cache_materializer_materializes_mestre_clade9_profile(tmp_path: Path) -> None:
    roster_path = write_roster_table(tmp_path)
    provider_root = write_provider_sources(tmp_path)

    result = materialize_conservation_roster_cache(
        repo_root=repo_root(),
        roster_table=roster_path,
        provider_source_root=provider_root,
        cache_root=tmp_path / "cache",
        require_roster_source_hash=False,
    )

    source_records = result.source_records_path.read_text(encoding="utf-8")
    assert "profile_id: ec86_clade9_conservation_v1" in source_records
    assert "profile_id: ec86_iia3_cluster42_1_conservation_v1" in source_records
    assert "known_public_accession_mismatch_with_ec86kit_target" in source_records


def test_roster_cache_materializer_records_sequence_qc_metadata(tmp_path: Path) -> None:
    roster_path = write_roster_table(tmp_path)
    provider_root = write_provider_sources(tmp_path)

    result = materialize_conservation_roster_cache(
        repo_root=repo_root(),
        roster_table=roster_path,
        provider_source_root=provider_root,
        cache_root=tmp_path / "cache",
        require_roster_source_hash=False,
    )

    payload = yaml.safe_load(result.source_records_path.read_text(encoding="utf-8"))
    included = next(record for record in payload["records"] if record["status"] == "included")

    assert included["sequence_qc"]["method_id"] == "eco1_roster_cache_sequence_qc_v1"
    assert included["sequence_qc"]["sequence_length_aa"] == 320
    assert included["sequence_qc"]["query_coverage"] == pytest.approx(1.0)
    assert 0.2 <= included["sequence_qc"]["pairwise_identity_to_target"] <= 0.9
    assert included["sequence_qc"]["identity_range_status"] == "within_declared_range"
    assert included["sequence_qc"]["length_status"] == "within_declared_range"
    assert included["sequence_qc"]["hard_reject_filters_triggered"] == []
    markers = included["sequence_qc"]["motif_qc_markers"]
    assert markers["rt_catalytic_dd_or_yadd_like_region"] == "present"
    assert markers["retron_x_naxxH_like_motif"] == "present"
    assert markers["retron_y_vtg_like_motif"] == "present"


def test_roster_cache_materializer_excludes_sequence_qc_failures_explicitly(tmp_path: Path) -> None:
    target = ec86kit_target_sequence()
    extra_rows = [
        roster_row("short_clade9", "WP_111111111.1", clade="9"),
        roster_row("no_core_clade9", "WP_111111112.1", clade="9"),
    ]
    roster_path = write_roster_table(tmp_path, extra_rows=extra_rows)
    provider_root = write_provider_sources(tmp_path, target_sequence=target)
    provider_fasta = provider_root / "ncbi_protein_efetch.fasta"
    provider_fasta.write_text(
        provider_fasta.read_text(encoding="utf-8")
        + f">WP_111111111.1\n{fragment_sequence(target)}\n"
        + f">WP_111111112.1\n{no_catalytic_core_sequence(target)}\n",
        encoding="utf-8",
    )

    result = materialize_conservation_roster_cache(
        repo_root=repo_root(),
        roster_table=roster_path,
        provider_source_root=provider_root,
        cache_root=tmp_path / "cache",
        require_roster_source_hash=False,
    )

    payload = yaml.safe_load(result.source_records_path.read_text(encoding="utf-8"))
    by_accession = {record["accession"]: record for record in payload["records"]}

    short_record = by_accession["WP_111111111.1"]
    assert short_record["status"] == "excluded"
    assert "outside_length_range" in short_record["sequence_qc"]["hard_reject_filters_triggered"]
    assert "below_query_coverage_minimum" in short_record["sequence_qc"]["hard_reject_filters_triggered"]
    assert short_record["exclusion_reason"].startswith("failed_sequence_qc:")

    no_core_record = by_accession["WP_111111112.1"]
    assert no_core_record["status"] == "excluded"
    assert no_core_record["sequence_qc"]["motif_qc_markers"]["rt_catalytic_dd_or_yadd_like_region"] == "absent"
    assert "missing_catalytic_rt_core" in no_core_record["sequence_qc"]["hard_reject_filters_triggered"]


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


def test_roster_loader_reads_mestre_workbook_preamble_header(tmp_path: Path) -> None:
    import pandas as pd

    workbook = tmp_path / "mestre_s1_like.xlsx"
    rows = [
        ["Table S1: List of Retrons (1928 RT dataset)", "", "", "", "", ""],
        ["aNodes and RT Clades described on Figure 2A", "", "", "", "", ""],
        ["Nodea", "RT/Cladea", "Retron (sub)b", "msr/msd familiyc", "Cluster/domaind", "Accesione"],
        ["1550", "9", "II-A3", "IIA3 (Proteobacteria)", "42_1", "WP_099010551.1"],
    ]
    pd.DataFrame(rows).to_excel(workbook, header=False, index=False)

    roster_rows = load_roster_rows(workbook, accession_field="Accesione")

    assert len(roster_rows) == 1
    assert roster_rows[0].node_id == "1550"
    assert roster_rows[0].rt_clade == "9"
    assert roster_rows[0].retron_subtype == "II-A3"
    assert roster_rows[0].cluster_domain == "42_1"
