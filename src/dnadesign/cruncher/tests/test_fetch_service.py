"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/test_fetch_service.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from dnadesign.cruncher.ingest.models import (
    GenomicInterval,
    MotifDescriptor,
    MotifQuery,
    OrganismRef,
    Provenance,
    SiteInstance,
    SiteQuery,
)
from dnadesign.cruncher.ingest.normalize import build_motif_record
from dnadesign.cruncher.services.fetch_service import fetch_motifs, fetch_sites
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex


class StubAdapter:
    source_id = "stub"

    def capabilities(self):
        return {"motifs:list", "motifs:get", "sites:list"}

    def list_motifs(self, query: MotifQuery):
        return [
            MotifDescriptor(
                source="stub",
                motif_id="M1",
                tf_name=query.tf_name or "tf",
                organism=OrganismRef(name="test"),
                length=4,
                kind="PFM",
            )
        ]

    def get_motif(self, motif_id: str):
        return build_motif_record(
            source="stub",
            motif_id=motif_id,
            tf_name="tf",
            matrix=[[0.25, 0.25, 0.25, 0.25]],
            matrix_semantics="probabilities",
            organism=None,
            raw_payload="{}",
        )

    def list_sites(self, query: SiteQuery):
        now = datetime.now(timezone.utc)
        provenance = Provenance(retrieved_at=now, source_url="stub://")
        yield SiteInstance(
            source="stub",
            site_id="s1",
            motif_ref="stub:M1",
            organism=None,
            coordinate=GenomicInterval(contig="chr", start=0, end=4),
            sequence="ACGT",
            strand="+",
            score=None,
            evidence={},
            provenance=provenance,
        )
        yield SiteInstance(
            source="stub",
            site_id="s2",
            motif_ref="stub:M1",
            organism=None,
            coordinate=None,
            sequence="ACGT",
            strand="+",
            score=None,
            evidence={},
            provenance=provenance,
        )

    def get_sites_for_motif(self, motif_id: str, query: SiteQuery):
        return self.list_sites(query)


class MultiMotifAdapter(StubAdapter):
    def list_motifs(self, query: MotifQuery):
        return [
            MotifDescriptor(
                source="stub",
                motif_id="M1",
                tf_name=query.tf_name or "tf",
                organism=None,
                length=4,
                kind="PFM",
            ),
            MotifDescriptor(
                source="stub",
                motif_id="M2",
                tf_name=query.tf_name or "tf",
                organism=None,
                length=4,
                kind="PFM",
            ),
        ]


class CoordOnlyAdapter(StubAdapter):
    def list_sites(self, query: SiteQuery):
        now = datetime.now(timezone.utc)
        provenance = Provenance(retrieved_at=now, source_url="stub://")
        yield SiteInstance(
            source="stub",
            site_id="s1",
            motif_ref="stub:M1",
            organism=None,
            coordinate=GenomicInterval(contig="chr", start=0, end=4),
            sequence=None,
            strand="+",
            score=None,
            evidence={},
            provenance=provenance,
        )


def test_fetch_sites_updates_catalog(tmp_path):
    adapter = StubAdapter()
    written = fetch_sites(adapter, tmp_path, names=["tf"])
    assert written
    site_path = tmp_path / "normalized" / "sites" / "stub" / "M1.jsonl"
    assert site_path.exists()
    catalog_path = tmp_path / "catalog.json"
    assert catalog_path.exists()
    payload = json.loads(catalog_path.read_text())
    entry = payload["entries"]["stub:M1"]
    assert entry["site_count"] == 2
    assert entry["site_length_mean"] == 4.0


def test_fetch_sites_by_motif_id(tmp_path):
    adapter = StubAdapter()
    written = fetch_sites(adapter, tmp_path, names=[], motif_ids=["M1"])
    assert written
    site_path = tmp_path / "normalized" / "sites" / "stub" / "M1.jsonl"
    assert site_path.exists()


def test_fetch_motifs_requires_disambiguation(tmp_path):
    adapter = MultiMotifAdapter()
    with pytest.raises(ValueError):
        fetch_motifs(adapter, tmp_path, names=["tf"], fetch_all=False)


def test_fetch_motifs_offline_uses_cache(tmp_path):
    adapter = StubAdapter()
    record = build_motif_record(
        source="stub",
        motif_id="M1",
        tf_name="tf",
        matrix=[[0.25, 0.25, 0.25, 0.25]],
        matrix_semantics="probabilities",
        organism=None,
        raw_payload="{}",
    )
    write_path = tmp_path / "normalized" / "motifs" / "stub"
    write_path.mkdir(parents=True, exist_ok=True)
    (write_path / "M1.json").write_text(
        json.dumps({"descriptor": {"source": "stub", "motif_id": "M1", "tf_name": "tf"}, "matrix": record.matrix})
    )
    catalog = CatalogIndex()
    catalog.upsert_from_record(record)
    catalog.save(tmp_path)

    paths = fetch_motifs(adapter, tmp_path, names=["tf"], offline=True)
    assert paths


def test_fetch_motifs_offline_ambiguous_requires_all(tmp_path):
    adapter = StubAdapter()
    record1 = build_motif_record(
        source="stub",
        motif_id="M1",
        tf_name="tf",
        matrix=[[0.25, 0.25, 0.25, 0.25]],
        matrix_semantics="probabilities",
        organism=None,
        raw_payload="{}",
    )
    record2 = build_motif_record(
        source="stub",
        motif_id="M2",
        tf_name="tf",
        matrix=[[0.25, 0.25, 0.25, 0.25]],
        matrix_semantics="probabilities",
        organism=None,
        raw_payload="{}",
    )
    write_path = tmp_path / "normalized" / "motifs" / "stub"
    write_path.mkdir(parents=True, exist_ok=True)
    (write_path / "M1.json").write_text(
        json.dumps({"descriptor": {"source": "stub", "motif_id": "M1", "tf_name": "tf"}, "matrix": record1.matrix})
    )
    (write_path / "M2.json").write_text(
        json.dumps({"descriptor": {"source": "stub", "motif_id": "M2", "tf_name": "tf"}, "matrix": record2.matrix})
    )
    catalog = CatalogIndex()
    catalog.upsert_from_record(record1)
    catalog.upsert_from_record(record2)
    catalog.save(tmp_path)

    with pytest.raises(ValueError):
        fetch_motifs(adapter, tmp_path, names=["tf"], offline=True, fetch_all=False)
    paths = fetch_motifs(adapter, tmp_path, names=["tf"], offline=True, fetch_all=True)
    assert len(paths) == 2


def test_fetch_sites_offline_missing_raises(tmp_path):
    adapter = StubAdapter()
    with pytest.raises(ValueError):
        fetch_sites(adapter, tmp_path, names=["tf"], offline=True)


def test_fetch_sites_offline_ambiguous_raises(tmp_path):
    adapter = StubAdapter()
    catalog = CatalogIndex(
        entries={
            "stub:M1": CatalogEntry(
                source="stub",
                motif_id="M1",
                tf_name="tf",
                kind="PFM",
                has_sites=True,
                site_count=1,
                site_total=1,
            ),
            "stub:M2": CatalogEntry(
                source="stub",
                motif_id="M2",
                tf_name="tf",
                kind="PFM",
                has_sites=True,
                site_count=1,
                site_total=1,
            ),
        }
    )
    catalog.save(tmp_path)
    sites_dir = tmp_path / "normalized" / "sites" / "stub"
    sites_dir.mkdir(parents=True, exist_ok=True)
    (sites_dir / "M1.jsonl").write_text(json.dumps({"sequence": "ACGT"}) + "\n")
    (sites_dir / "M2.jsonl").write_text(json.dumps({"sequence": "ACGT"}) + "\n")

    with pytest.raises(ValueError):
        fetch_sites(adapter, tmp_path, names=["tf"], offline=True)


def test_fetch_sites_requires_hydration_for_coord_only(tmp_path):
    adapter = CoordOnlyAdapter()
    with pytest.raises(ValueError, match="genome hydration"):
        fetch_sites(adapter, tmp_path, names=["tf"])
