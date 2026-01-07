from __future__ import annotations

from dnadesign.cruncher.ingest.adapters.regulondb import RegulonDBAdapter, RegulonDBAdapterConfig
from dnadesign.cruncher.ingest.models import MotifQuery


def test_regulondb_iter_motifs_paginates():
    pages = {
        0: [
            {"_id": "R1", "regulator": {"abbreviatedName": "LexA"}},
            {"_id": "R2", "regulator": {"abbreviatedName": "CpxR"}},
        ],
        1: [
            {"_id": "R3", "regulator": {"abbreviatedName": "SoxR"}},
        ],
        2: [],
    }

    def transport(_query, variables):
        page = variables.get("page", 0)
        return {"getAllRegulon": {"data": pages.get(page, [])}}

    adapter = RegulonDBAdapter(RegulonDBAdapterConfig(), transport=transport)
    motifs = list(adapter.iter_motifs(MotifQuery(limit=None), page_size=2))

    assert [motif.motif_id for motif in motifs] == ["R1", "R2", "R3"]
    assert [motif.tf_name for motif in motifs] == ["LexA", "CpxR", "SoxR"]
