"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_banner_assets.py

Tests banner inventory, accessibility, SVG validity, and generation drift.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from dnadesign.devtools.docs.banners.catalog import BANNERS, REPOSITORY_BANNER_PATH
from dnadesign.devtools.docs.banners.render import check_banners, expected_banners


def test_checked_in_banners_match_their_source() -> None:
    repo_root = Path(__file__).resolve().parents[5]

    assert check_banners(repo_root) == ()


def test_banner_catalog_matches_readme_references() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    referenced: set[str] = set()
    for readme in (repo_root / "src" / "dnadesign").rglob("README.md"):
        for line in readme.read_text(encoding="utf-8").splitlines()[:5]:
            if "banner" not in line.lower() or not line.rstrip().endswith(".svg)"):
                continue
            link = line.rsplit("(", 1)[1][:-1]
            referenced.add(str((readme.parent / link).relative_to(repo_root)))

    assert referenced == {spec.path for spec in BANNERS}


def test_banners_are_accessible_valid_svg_without_transient_copy() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    expected = expected_banners(repo_root)

    assert REPOSITORY_BANNER_PATH in {str(path.relative_to(repo_root)) for path in expected}
    for path, content in expected.items():
        root = ElementTree.fromstring(content)
        namespace = "{http://www.w3.org/2000/svg}"
        assert root.tag == f"{namespace}svg"
        assert root.find(f"{namespace}title") is not None
        assert root.find(f"{namespace}desc") is not None
        lowered = content.lower()
        assert "schema v" not in lowered
        assert "generated on" not in lowered
        assert "agent" not in lowered
        assert path.suffix == ".svg"
