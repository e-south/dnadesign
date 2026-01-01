from __future__ import annotations

import json
from datetime import datetime, timezone

from dnadesign.cruncher.ingest.models import GenomicInterval, Provenance, SiteInstance, SiteQuery
from dnadesign.cruncher.ingest.sequence_provider import FastaSequenceProvider, NCBISequenceProvider
from dnadesign.cruncher.services.fetch_service import fetch_sites


class CoordOnlyAdapter:
    source_id = "coord"

    def capabilities(self):
        return {"sites:list"}

    def list_sites(self, query: SiteQuery):
        provenance = Provenance(retrieved_at=datetime.now(timezone.utc))
        yield SiteInstance(
            source="coord",
            site_id="s1",
            motif_ref="coord:M1",
            organism=None,
            coordinate=GenomicInterval(contig="chr1", start=2, end=6),
            sequence=None,
            strand="+",
            score=None,
            evidence={},
            provenance=provenance,
        )

    def get_sites_for_motif(self, motif_id: str, query: SiteQuery):
        return self.list_sites(query)


def test_fetch_sites_hydrates_from_fasta(tmp_path):
    fasta = tmp_path / "genome.fasta"
    fasta.write_text(">chr1\nAACCGGTTAACCGGTT\n")
    adapter = CoordOnlyAdapter()
    provider = FastaSequenceProvider(fasta)
    written = fetch_sites(adapter, tmp_path, names=["tf"], sequence_provider=provider)
    assert written
    sites_path = tmp_path / "normalized" / "sites" / "coord" / "M1.jsonl"
    payload = json.loads(sites_path.read_text().strip())
    assert payload["sequence"] == "CCGG"
    provider.close()


def test_ncbi_sequence_provider_uses_cached_fasta(tmp_path):
    accession = "U00096.3"
    cache_dir = tmp_path / accession
    cache_dir.mkdir(parents=True)
    fasta_path = cache_dir / f"{accession}.fna"
    fasta_path.write_text(f">{accession}\nAACCGGTTAACCGGTT\n")
    provider = NCBISequenceProvider(cache_root=tmp_path, offline=True)
    seq = provider.fetch(GenomicInterval(contig=accession, start=2, end=6, assembly=accession))
    assert seq == "CCGG"
    provider.close()
