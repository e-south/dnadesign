from pathlib import Path

from dnadesign.cruncher.cli.commands.fetch import _render_sites_summary


def test_render_sites_summary_handles_missing_catalog_entry(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    paths = [tmp_path / "normalized" / "sites" / "regulondb" / "M1.jsonl"]
    _render_sites_summary(catalog_root, paths)
