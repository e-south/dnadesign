"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_banner_assets.py

Tests banner inventory, accessibility, SVG validity, and generation drift.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree

import pytest

from dnadesign.devtools.docs.banners import render as banner_render
from dnadesign.devtools.docs.banners.catalog import BANNERS, REPOSITORY_BANNER_PATH
from dnadesign.devtools.docs.banners.render import check_banners, expected_banners, render_banners


def _write_repo_markers(repo_root: Path) -> None:
    (repo_root / "src" / "dnadesign").mkdir(parents=True)
    (repo_root / "src" / "dnadesign" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "pyproject.toml").write_text('[project]\nname = "dnadesign"\n', encoding="utf-8")


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
        assert re.search(r"(?<!/)\b(?:19|20)\d{2}\b", content) is None
        assert (
            re.search(
                r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
                lowered,
            )
            is None
        )
        assert re.search(r"\b(?:version|release|schema)\b", lowered) is None
        assert re.search(r"\b(?:version|release|schema)\s*(?:v\s*)?\d+", lowered) is None
        assert re.search(r"\bv\d+(?:\.\d+)*\b", lowered) is None
        assert re.search(r"\b(?:stage|step|phase|node)\s*0?\d+\b", lowered) is None
        assert re.search(r">\s*0[1-9]\s*<", content) is None
        assert path.suffix == ".svg"


def test_banners_do_not_place_ornaments_on_horizontal_rails() -> None:
    repo_root = Path(__file__).resolve().parents[5]

    for content in expected_banners(repo_root).values():
        rails: list[tuple[float, float, float]] = []
        ornaments: list[tuple[float, float]] = []

        def collect_geometry(element: ElementTree.Element, offset_x: float = 0, offset_y: float = 0) -> None:
            transform = element.attrib.get("transform", "")
            translated = re.fullmatch(r"translate\(([-\d.]+)(?:[ ,]+)([-\d.]+)\)", transform)
            if translated is not None:
                offset_x += float(translated.group(1))
                offset_y += float(translated.group(2))

            tag = element.tag.rsplit("}", 1)[-1]
            if tag == "path":
                rail = re.fullmatch(r"M([-\d.]+) ([-\d.]+)H([-\d.]+)", element.attrib.get("d", ""))
                if rail is not None:
                    rails.append(
                        (
                            offset_x + float(rail.group(1)),
                            offset_x + float(rail.group(3)),
                            offset_y + float(rail.group(2)),
                        )
                    )
            elif tag == "rect":
                ornaments.append(
                    (
                        offset_x + float(element.attrib.get("x", 0)) + float(element.attrib["width"]) / 2,
                        offset_y + float(element.attrib.get("y", 0)) + float(element.attrib["height"]) / 2,
                    )
                )
            elif tag == "circle":
                ornaments.append(
                    (
                        offset_x + float(element.attrib["cx"]),
                        offset_y + float(element.attrib["cy"]),
                    )
                )

            for child in element:
                collect_geometry(child, offset_x, offset_y)

        collect_geometry(ElementTree.fromstring(content))

        assert all(not (start <= x <= end and y == rail_y) for start, end, rail_y in rails for x, y in ornaments)


def test_render_rejects_wrong_repo_root_without_mutation(tmp_path: Path) -> None:
    wrong_root = tmp_path / "not-dnadesign"

    with pytest.raises(ValueError, match="dnadesign repository root"):
        render_banners(wrong_root)

    assert not wrong_root.exists()


def test_render_preflights_every_output_before_mutation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    outside = tmp_path / "escaped.svg"
    malicious = banner_render.BannerSpec(
        path="../escaped.svg",
        name="escaped",
        capability="ESCAPE",
        description="Must not be written.",
        glyph="align",
    )
    monkeypatch.setattr(banner_render, "BANNERS", (*BANNERS, malicious))

    with pytest.raises(ValueError, match="escapes repository root"):
        render_banners(repo_root)

    assert not outside.exists()
    assert not (repo_root / REPOSITORY_BANNER_PATH).exists()


def test_render_rejects_symlinked_output_parent_without_mutation(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo_root / "assets").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink component"):
        render_banners(repo_root)

    assert list(outside.iterdir()) == []


def test_render_rejects_in_repo_symlinked_output_parent_without_mutation(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    alternate_assets = repo_root / "alternate-assets"
    alternate_assets.mkdir()
    (repo_root / "assets").symlink_to(alternate_assets, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink component"):
        render_banners(repo_root)

    assert list(alternate_assets.iterdir()) == []


def test_render_rejects_symlinked_output_file_without_mutation(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    _write_repo_markers(repo_root)
    unrelated = repo_root / "unrelated.svg"
    unrelated.write_text("preserve me", encoding="utf-8")
    output = repo_root / REPOSITORY_BANNER_PATH
    output.parent.mkdir(parents=True)
    output.symlink_to(unrelated)

    with pytest.raises(ValueError, match="symlink component"):
        render_banners(repo_root)

    assert output.is_symlink()
    assert unrelated.read_text(encoding="utf-8") == "preserve me"
