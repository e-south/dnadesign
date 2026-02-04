"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_local_site_adapter.py

Tests for ingesting local binding-site FASTA sources.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.app.fetch_service import fetch_sites
from dnadesign.cruncher.ingest.adapters.local_sites import (
    LocalSiteAdapter,
    LocalSiteAdapterConfig,
)
from dnadesign.cruncher.ingest.models import SiteQuery
from dnadesign.cruncher.store.catalog_index import CatalogIndex


def _write_fasta(path: Path) -> None:
    path.write_text(
        ">BaeR|peak_0|NC_000913.3:1-5|strand=-|operon=intF|sn=5.48|type=promoter\n"
        "ACGTAC\n"
        ">BaeR|peak_1|NC_000913.3:10-16|strand=+|operon=spy|sn=8.23|type=promoter\n"
        "TGCAAA\n"
    )


def _make_adapter(path: Path) -> LocalSiteAdapter:
    cfg = LocalSiteAdapterConfig(
        source_id="local_sites",
        path=path,
        tf_name="BaeR",
        record_kind="chip_exo",
        citation="Choudhary et al. 2020 (DOI: 10.1128/mSystems.00980-20)",
        source_url="https://doi.org/10.1128/mSystems.00980-20",
        tags={"assay": "chip_exo"},
    )
    return LocalSiteAdapter(cfg)


def test_local_site_adapter_parses_fasta(tmp_path: Path) -> None:
    fasta_path = tmp_path / "sites.fasta"
    _write_fasta(fasta_path)
    adapter = _make_adapter(fasta_path)
    sites = list(adapter.list_sites(SiteQuery(tf_name="BaeR")))

    assert len(sites) == 2
    first = sites[0]
    assert first.motif_ref == "local_sites:BaeR"
    assert first.sequence == "ACGTAC"
    assert first.strand == "-"
    assert first.evidence["peak_id"] == "peak_0"
    assert first.evidence["coord"] == "NC_000913.3:1-5"
    assert first.evidence["operon"] == "intF"
    assert first.provenance.tags["record_kind"] == "chip_exo"


def test_fetch_sites_updates_catalog_for_local_site_source(tmp_path: Path) -> None:
    fasta_path = tmp_path / "sites.fasta"
    _write_fasta(fasta_path)
    adapter = _make_adapter(fasta_path)
    catalog_root = tmp_path / ".cruncher"

    written = fetch_sites(adapter, catalog_root, names=["BaeR"])

    assert written
    catalog = CatalogIndex.load(catalog_root)
    entry = catalog.entries.get("local_sites:BaeR")
    assert entry is not None
    assert entry.has_sites is True
    assert entry.site_total == 2
    assert entry.site_count == 2
    assert entry.site_kind == "chip_exo"
