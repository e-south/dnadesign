"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_fetch_cli.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from pathlib import Path

import pytest
import typer

from dnadesign.cruncher.cli.commands.fetch import (
    _render_sites_summary,
    _resolve_fetch_source,
)


def test_render_sites_summary_handles_missing_catalog_entry(tmp_path: Path) -> None:
    catalog_root = tmp_path / "catalog"
    paths = [tmp_path / "normalized" / "sites" / "regulondb" / "M1.jsonl"]
    _render_sites_summary(catalog_root, paths)


def test_fetch_source_prefers_explicit_source() -> None:
    source = _resolve_fetch_source(
        "demo_local_meme",
        ["regulondb"],
        available_sources={"demo_local_meme", "regulondb"},
    )
    assert source == "demo_local_meme"


def test_fetch_source_uses_source_preference() -> None:
    source = _resolve_fetch_source(
        None,
        ["demo_local_meme", "regulondb"],
        available_sources={"demo_local_meme", "regulondb"},
    )
    assert source == "demo_local_meme"


def test_fetch_source_skips_unavailable_preferences() -> None:
    source = _resolve_fetch_source(
        None,
        ["meme_suite_meme", "demo_local_meme"],
        available_sources={"demo_local_meme"},
    )
    assert source == "demo_local_meme"


def test_fetch_source_requires_explicit_or_preference() -> None:
    with pytest.raises(typer.BadParameter, match="--source is required"):
        _resolve_fetch_source(None, [], available_sources={"regulondb"})


def test_fetch_source_rejects_unavailable_preference() -> None:
    with pytest.raises(typer.BadParameter, match="source_preference"):
        _resolve_fetch_source(None, ["meme_suite_meme"], available_sources={"regulondb"})
