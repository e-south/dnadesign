"""Roster-cache materialization tests for Eco1 conservation source sequences."""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache import (
    materialize_conservation_roster_cache,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache.roster import (
    load_roster_rows,
)
from dnadesign.studies.units.eco1_rt_repack.tests._helpers import repo_root
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.source_sequences.roster_cache._fixtures import (
    write_provider_sources,
    write_roster_table,
)


def test_roster_cache_materializer_rejects_bounded_broad_profile_until_selector_exists(tmp_path: Path) -> None:
    roster_path = write_roster_table(tmp_path)
    provider_root = write_provider_sources(tmp_path)

    with pytest.raises(ValueError, match="bounded-homolog selector slice"):
        materialize_conservation_roster_cache(
            repo_root=repo_root(),
            roster_table=roster_path,
            provider_source_root=provider_root,
            cache_root=tmp_path / "cache",
            require_roster_source_hash=False,
        )


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
