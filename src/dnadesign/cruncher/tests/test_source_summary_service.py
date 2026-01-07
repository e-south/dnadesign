from __future__ import annotations

from datetime import datetime, timezone

from dnadesign.cruncher.ingest.models import DatasetDescriptor, MotifDescriptor, OrganismRef
from dnadesign.cruncher.services.source_summary_service import (
    summarize_cache,
    summarize_combined,
    summarize_remote,
)
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex


def test_summarize_cache_counts(tmp_path):
    index = CatalogIndex()
    index.entries["regulondb:R1"] = CatalogEntry(
        source="regulondb",
        motif_id="R1",
        tf_name="LexA",
        kind="PFM",
        has_matrix=True,
        has_sites=True,
        site_count=5,
        site_total=7,
        dataset_id="D1",
        dataset_source="regulondb",
        dataset_method="ChIP",
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    index.entries["jaspar:J1"] = CatalogEntry(
        source="jaspar",
        motif_id="J1",
        tf_name="LexA",
        kind="PFM",
        has_matrix=True,
        has_sites=False,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    index.entries["regulondb:R2"] = CatalogEntry(
        source="regulondb",
        motif_id="R2",
        tf_name="CpxR",
        kind="PFM",
        has_matrix=True,
        has_sites=False,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    index.save(tmp_path)

    summary = summarize_cache(tmp_path)
    totals = summary["totals"]
    assert totals["entries"] == 3
    assert totals["tfs"] == 2
    assert totals["motifs"] == 3
    assert totals["site_sets"] == 1
    assert totals["sites_seq"] == 5
    assert totals["sites_total"] == 7
    assert totals["datasets"] == 1

    sources = summary["sources"]
    assert sources["regulondb"]["tfs"] == 2
    assert sources["regulondb"]["motifs"] == 2
    assert sources["regulondb"]["site_sets"] == 1
    assert sources["jaspar"]["tfs"] == 1

    regulators = {row["tf_name"]: row for row in summary["regulators"]}
    assert regulators["LexA"]["motifs"] == 2
    assert regulators["LexA"]["site_sets"] == 1
    assert regulators["LexA"]["sites_total"] == 7
    assert regulators["LexA"]["datasets"] == 1
    assert regulators["CpxR"]["motifs"] == 1


def test_summarize_remote_counts():
    class StubAdapter:
        source_id = "stub"

        def capabilities(self):
            return {"motifs:iter", "datasets:list"}

        def iter_motifs(self, query, *, page_size=200):
            org = OrganismRef(name="E. coli")
            yield MotifDescriptor(source="stub", motif_id="M1", tf_name="LexA", organism=org, length=0, kind="PFM")
            yield MotifDescriptor(source="stub", motif_id="M2", tf_name="LexA", organism=org, length=0, kind="PFM")
            yield MotifDescriptor(source="stub", motif_id="M3", tf_name="CpxR", organism=org, length=0, kind="PFM")

        def list_datasets(self, query):
            return [
                DatasetDescriptor(
                    source="stub",
                    dataset_id="D1",
                    dataset_source="stubsrc",
                    method="ChIP",
                    tf_names=("LexA",),
                )
            ]

    summary = summarize_remote(StubAdapter(), limit=None, page_size=10, include_datasets=True)
    totals = summary["totals"]
    assert totals["tfs"] == 2
    assert totals["motifs"] == 3
    assert totals["datasets"] == 1

    regulators = {row["tf_name"]: row for row in summary["regulators"]}
    assert regulators["LexA"]["motifs"] == 2
    assert regulators["LexA"]["datasets"] == 1
    assert regulators["CpxR"]["motifs"] == 1
    assert regulators["CpxR"]["datasets"] == 0


def test_summarize_combined_merges_cache_and_remote(tmp_path):
    index = CatalogIndex()
    index.entries["regulondb:R1"] = CatalogEntry(
        source="regulondb",
        motif_id="R1",
        tf_name="LexA",
        kind="PFM",
        has_matrix=True,
        has_sites=True,
        site_count=5,
        site_total=7,
        dataset_id="D1",
        dataset_source="regulondb",
        dataset_method="ChIP",
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    index.entries["regulondb:R2"] = CatalogEntry(
        source="regulondb",
        motif_id="R2",
        tf_name="CpxR",
        kind="PFM",
        has_matrix=True,
        has_sites=False,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    index.save(tmp_path)
    cache_summary = summarize_cache(tmp_path)

    class StubAdapter:
        source_id = "stub"

        def capabilities(self):
            return {"motifs:iter", "datasets:list"}

        def iter_motifs(self, query, *, page_size=200):
            org = OrganismRef(name="E. coli")
            yield MotifDescriptor(source="stub", motif_id="M1", tf_name="LexA", organism=org, length=0, kind="PFM")
            yield MotifDescriptor(source="stub", motif_id="M2", tf_name="SoxR", organism=org, length=0, kind="PFM")

        def list_datasets(self, query):
            return [
                DatasetDescriptor(
                    source="stub",
                    dataset_id="D2",
                    dataset_source="stubsrc",
                    method="ChIP",
                    tf_names=("LexA",),
                )
            ]

    remote_summary = summarize_remote(StubAdapter(), limit=None, page_size=10, include_datasets=True)
    combined = summarize_combined(cache_summary=cache_summary, remote_summaries={"stub": remote_summary})
    regulators = {row["tf_name"]: row for row in combined["regulators"]}
    lexa = regulators["LexA"]
    assert lexa["cache"]["motifs"] == 1
    assert lexa["cache"]["sites_total"] == 7
    assert lexa["remote"]["motifs"] == 1
    assert lexa["remote"]["datasets"] == 1
    assert regulators["SoxR"]["cache"]["motifs"] == 0
    assert regulators["SoxR"]["remote"]["motifs"] == 1
