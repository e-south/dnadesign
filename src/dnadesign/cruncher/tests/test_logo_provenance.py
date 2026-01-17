"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_logo_provenance.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex
from dnadesign.cruncher.viz.logos import logo_subtitle, pwm_provenance_summary


def test_logo_subtitle_matrix_includes_source_and_origin() -> None:
    entry = CatalogEntry(
        source="meme_suite",
        motif_id="lexA_1",
        tf_name="lexA",
        kind="PFM",
        has_matrix=True,
        matrix_source="streme",
        tags={"discovery_nsites": "50"},
    )
    catalog = CatalogIndex(entries={entry.key: entry})
    subtitle = logo_subtitle(
        pwm_source="matrix",
        entry=entry,
        catalog=catalog,
        combine_sites=False,
        site_kinds=None,
    )
    assert subtitle == "meme_suite (streme, n=50)"
    provenance = pwm_provenance_summary(
        pwm_source="matrix",
        entry=entry,
        catalog=catalog,
        combine_sites=False,
        site_kinds=None,
    )
    assert provenance == "matrix (streme, n=50)"


def test_logo_subtitle_sites_combined_summary() -> None:
    entry_a = CatalogEntry(
        source="alpha",
        motif_id="A1",
        tf_name="lexA",
        kind="PFM",
        has_sites=True,
        site_kind="curated",
        site_count=12,
        site_total=20,
    )
    entry_b = CatalogEntry(
        source="beta",
        motif_id="B1",
        tf_name="lexA",
        kind="PFM",
        has_sites=True,
        site_kind="ht_chip_seq",
        site_count=8,
        site_total=8,
    )
    catalog = CatalogIndex(entries={entry_a.key: entry_a, entry_b.key: entry_b})
    subtitle = logo_subtitle(
        pwm_source="sites",
        entry=entry_a,
        catalog=catalog,
        combine_sites=True,
        site_kinds=None,
    )
    assert subtitle == "combined (n=20/28, sets=2, alpha+beta, curated+high-throughput (chip seq))"
    provenance = pwm_provenance_summary(
        pwm_source="sites",
        entry=entry_a,
        catalog=catalog,
        combine_sites=True,
        site_kinds=None,
    )
    assert provenance == "sites combined n_sets=2 n=20/28 sources=alpha+beta kinds=curated+high-throughput (chip seq)"
