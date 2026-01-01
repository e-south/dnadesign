from __future__ import annotations

import json

from dnadesign.cruncher.services.catalog_service import verify_cache
from dnadesign.cruncher.store.catalog_index import CatalogEntry, CatalogIndex


def test_cache_verify_reports_missing_files(tmp_path):
    index = CatalogIndex()
    entry = CatalogEntry(
        source="regulondb",
        motif_id="RDB0004",
        tf_name="LexA",
        kind="PFM",
        has_matrix=True,
        has_sites=True,
        site_count=2,
        site_total=2,
    )
    index.entries[entry.key] = entry
    index.save(tmp_path)

    issues = verify_cache(tmp_path)
    assert any("motif" in issue for issue in issues)
    assert any("sites" in issue for issue in issues)

    motif_path = tmp_path / "normalized" / "motifs" / "regulondb"
    motif_path.mkdir(parents=True, exist_ok=True)
    (motif_path / "RDB0004.json").write_text(json.dumps({"matrix": [[0.25, 0.25, 0.25, 0.25]]}))

    sites_path = tmp_path / "normalized" / "sites" / "regulondb"
    sites_path.mkdir(parents=True, exist_ok=True)
    (sites_path / "RDB0004.jsonl").write_text("{}\n")

    assert verify_cache(tmp_path) == []
