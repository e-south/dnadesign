from __future__ import annotations

import json
from pathlib import Path

from dnadesign.cruncher.services.fetch_service import hydrate_sites
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex


class StubProvider:
    source_id = "stub"

    def fetch(self, interval):
        return "ACGT"

    def close(self) -> None:
        return None


def test_hydrate_sites_updates_sequences(tmp_path: Path) -> None:
    catalog_root = tmp_path
    entry = CatalogEntry(
        source="stub",
        motif_id="M1",
        tf_name="tf",
        kind="PFM",
        has_sites=True,
        site_count=0,
        site_total=1,
    )
    CatalogIndex(entries={entry.key: entry}).save(catalog_root)

    sites_path = catalog_root / "normalized" / "sites" / "stub" / "M1.jsonl"
    sites_path.parent.mkdir(parents=True, exist_ok=True)
    sites_path.write_text(
        json.dumps(
            {
                "source": "stub",
                "site_id": "s1",
                "motif_ref": "stub:M1",
                "organism": None,
                "coordinate": {"contig": "chr", "start": 0, "end": 4, "assembly": "chr"},
                "sequence": None,
                "strand": "+",
                "score": None,
                "evidence": {},
                "provenance": {"retrieved_at": "2025-01-01T00:00:00Z", "tags": {}},
            }
        )
        + "\n"
    )

    written = hydrate_sites(
        catalog_root,
        names=["tf"],
        motif_ids=None,
        sequence_provider=StubProvider(),
    )
    assert sites_path in written
    payload = json.loads(sites_path.read_text().strip())
    assert payload["sequence"] == "ACGT"
    assert payload["provenance"]["tags"]["sequence_source"] == "stub"

    updated = CatalogIndex.load(catalog_root).entries["stub:M1"]
    assert updated.site_count == 1
    assert updated.site_length_mean == 4.0
