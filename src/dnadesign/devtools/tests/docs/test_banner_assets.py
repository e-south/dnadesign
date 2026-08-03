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
        visible_text = "\n".join("".join(element.itertext()) for element in root.findall(f".//{namespace}text"))
        assert re.search(r"(?m)^\s*0?\d+(?:\s|[.:_-])", visible_text) is None
        assert path.suffix == ".svg"


def test_banners_do_not_place_ornaments_on_horizontal_rails() -> None:
    repo_root = Path(__file__).resolve().parents[5]

    for content in expected_banners(repo_root).values():
        rails: list[tuple[float, float, float]] = []
        ornaments: list[tuple[float, float, float, float]] = []
        canvas_width = float(ElementTree.fromstring(content).attrib["width"])
        canvas_height = float(ElementTree.fromstring(content).attrib["height"])

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
                x = offset_x + float(element.attrib.get("x", 0))
                y = offset_y + float(element.attrib.get("y", 0))
                width = float(element.attrib["width"])
                height = float(element.attrib["height"])
                if not (x == 0 and y == 0 and width == canvas_width and height == canvas_height):
                    ornaments.append((x, x + width, y, y + height))
            elif tag == "circle":
                center_x = offset_x + float(element.attrib["cx"])
                center_y = offset_y + float(element.attrib["cy"])
                radius = float(element.attrib["r"])
                ornaments.append(
                    (
                        center_x - radius,
                        center_x + radius,
                        center_y - radius,
                        center_y + radius,
                    )
                )

            for child in element:
                collect_geometry(child, offset_x, offset_y)

        collect_geometry(ElementTree.fromstring(content))

        assert all(
            not (max(start, left) <= min(end, right) and top <= rail_y <= bottom)
            for start, end, rail_y in rails
            for left, right, top, bottom in ornaments
        )
