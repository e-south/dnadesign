"""Filesystem-backed promoter source discovery."""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.ingest.promoter_filesystem import (
    iter_promoter_association_source_files,
    iter_promoter_source_files,
)


def test_explicit_root_discovers_parseable_promoter_and_association_sources(tmp_path: Path) -> None:
    current = tmp_path / "sources/databases/regulondb/13.0"
    historical = tmp_path / "sources/databases/regulondb/11.0"
    (current / "promoters").mkdir(parents=True)
    (current / "promoters/PromoterSet.tsv").write_text("header\n", encoding="utf-8")
    (current / "binding_sites").mkdir()
    (current / "binding_sites/TF-RISet.tsv").write_text("header\n", encoding="utf-8")
    (historical / "promoters").mkdir(parents=True)
    (historical / "promoters/PromoterSet.csv").write_text("header\n", encoding="utf-8")
    (historical / "network_associations").mkdir()
    (historical / "network_associations/network_tf_tu.txt").write_text("row\n", encoding="utf-8")

    sources = iter_promoter_source_files(tmp_path)
    associations = iter_promoter_association_source_files(tmp_path)

    assert [source.source_id for source in sources] == [
        "regulondb_13_promoter_set",
        "regulondb_11_promoter_set",
    ]
    assert [source.file_format for source in sources] == ["tsv", "csv"]
    assert [source.source_id for source in associations] == ["regulondb_13_tf_riset"]


def test_explicit_root_falls_back_to_historical_association_sources(tmp_path: Path) -> None:
    network = tmp_path / "sources/databases/regulondb/11.0/network_associations/network_tf_tu.txt"
    network.parent.mkdir(parents=True)
    network.write_text("row\n", encoding="utf-8")

    associations = iter_promoter_association_source_files(tmp_path)

    assert [source.source_id for source in associations] == ["regulondb_11_network_tf_tu"]
