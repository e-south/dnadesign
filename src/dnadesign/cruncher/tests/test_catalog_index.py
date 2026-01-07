"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_catalog_index.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime, timezone

from dnadesign.cruncher.ingest.models import OrganismRef
from dnadesign.cruncher.ingest.normalize import build_motif_record
from dnadesign.cruncher.store.catalog_index import CatalogIndex


def test_catalog_index_roundtrip(tmp_path):
    record = build_motif_record(
        source="regulondb",
        motif_id="RDB0001",
        tf_name="lexA",
        matrix=[[0.25, 0.25, 0.25, 0.25]],
        matrix_semantics="probabilities",
        organism=OrganismRef(taxon=562, name="E. coli"),
        raw_payload="{}",
        retrieved_at=datetime.now(timezone.utc),
        source_url="https://example",
        tags={"matrix_source": "alignment"},
    )

    index = CatalogIndex()
    index.upsert_from_record(record)
    index.upsert_sites(source="regulondb", motif_id="RDB0001", tf_name="lexA", site_count=5, site_total=5)
    index.save(tmp_path)

    loaded = CatalogIndex.load(tmp_path)
    entries = loaded.list(tf_name="lexA")
    assert len(entries) == 1
    entry = entries[0]
    assert entry.has_matrix is True
    assert entry.has_sites is True
    assert entry.site_count == 5
    assert entry.matrix_source == "alignment"
    assert entry.organism is not None
    assert entry.organism.get("name") == "E. coli"

    org_entries = loaded.list(organism={"name": "E. coli"})
    assert len(org_entries) == 1


def test_catalog_search_regex(tmp_path):
    index = CatalogIndex()
    record = build_motif_record(
        source="regulondb",
        motif_id="RDB0002",
        tf_name="SoxR",
        matrix=[[0.25, 0.25, 0.25, 0.25]],
        matrix_semantics="probabilities",
        organism=OrganismRef(name="E. coli"),
        raw_payload="{}",
    )
    index.upsert_from_record(record)
    index.save(tmp_path)

    loaded = CatalogIndex.load(tmp_path)
    matches = loaded.search(query="sox", regex=False)
    assert len(matches) == 1

    regex_matches = loaded.search(query="^S.*R$", regex=True, case_sensitive=True)
    assert len(regex_matches) == 1


def test_catalog_search_synonyms_and_fuzzy(tmp_path):
    index = CatalogIndex()
    record = build_motif_record(
        source="regulondb",
        motif_id="RDB0003",
        tf_name="LexA",
        matrix=[[0.25, 0.25, 0.25, 0.25]],
        matrix_semantics="probabilities",
        organism=OrganismRef(name="E. coli"),
        raw_payload="{}",
        tags={"synonyms": "SOSR;LEXA"},
    )
    index.upsert_from_record(record)
    index.save(tmp_path)

    loaded = CatalogIndex.load(tmp_path)
    synonym_matches = loaded.search(query="sosr")
    assert len(synonym_matches) == 1

    fuzzy_matches = loaded.search(query="lex", fuzzy=True, min_score=0.5)
    assert len(fuzzy_matches) == 1
