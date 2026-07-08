"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/structure_svg.py

Parse retron-hairpin ViennaRNA SVG geometry for the Reader SPOP plot.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class OrientedStructureLine:
    points: tuple[tuple[float, float], ...]
    color: str
    width: float


@dataclass(frozen=True, slots=True)
class OrientedStructureText:
    point: tuple[float, float]
    text: str


@dataclass(frozen=True, slots=True)
class OrientedStructureGeometry:
    lines: tuple[OrientedStructureLine, ...]
    texts: tuple[OrientedStructureText, ...]
    bounds: tuple[float, float, float, float]


@lru_cache(maxsize=128)
def oriented_structure_geometry(svg_path: str) -> OrientedStructureGeometry:
    """Return cap-right ViennaRNA structure geometry with unrotated text positions."""

    elements = _parse_structure_svg(Path(svg_path))
    oriented_lines = tuple(
        OrientedStructureLine(
            points=tuple(_orient_point(point) for point in line.points),
            color=line.color,
            width=line.width,
        )
        for line in elements.lines
    )
    oriented_texts = tuple(
        OrientedStructureText(point=_orient_point(text.point), text=text.text) for text in elements.texts
    )
    bounds = _bounds(oriented_lines=oriented_lines, oriented_texts=oriented_texts)
    return OrientedStructureGeometry(lines=oriented_lines, texts=oriented_texts, bounds=bounds)


@dataclass(frozen=True, slots=True)
class _SourceLineElement:
    points: tuple[tuple[float, float], ...]
    color: str
    width: float


@dataclass(frozen=True, slots=True)
class _SourceTextElement:
    point: tuple[float, float]
    text: str


@dataclass(frozen=True, slots=True)
class _SourceStructureElements:
    lines: tuple[_SourceLineElement, ...]
    texts: tuple[_SourceTextElement, ...]


def _parse_structure_svg(path: Path) -> _SourceStructureElements:
    tree = ET.parse(path)
    root = tree.getroot()
    lines: list[_SourceLineElement] = []
    texts: list[_SourceTextElement] = []

    def visit(element: ET.Element, transforms: list[tuple[str, tuple[float, ...]]]) -> None:
        tag = _svg_tag(element)
        next_transforms = [*transforms, *_parse_svg_transform(element.get("transform", ""))]
        if tag == "polyline":
            points = tuple(_apply_svg_transform(point, next_transforms) for point in _svg_points(element))
            if len(points) >= 2:
                lines.append(
                    _SourceLineElement(points=points, color=_svg_stroke(element), width=_svg_stroke_width(element))
                )
        elif tag == "line":
            start = _apply_svg_transform(
                (_float_attr(element, "x1"), _float_attr(element, "y1")),
                next_transforms,
            )
            end = _apply_svg_transform(
                (_float_attr(element, "x2"), _float_attr(element, "y2")),
                next_transforms,
            )
            lines.append(
                _SourceLineElement(points=(start, end), color=_svg_stroke(element), width=_svg_stroke_width(element))
            )
        elif tag == "text":
            point = _apply_svg_transform(
                (_float_attr(element, "x"), _float_attr(element, "y")),
                next_transforms,
            )
            texts.append(_SourceTextElement(point=point, text=element.text or ""))

        for child in element:
            visit(child, next_transforms)

    visit(root, [])
    return _SourceStructureElements(lines=tuple(lines), texts=tuple(texts))


def _orient_point(point: tuple[float, float]) -> tuple[float, float]:
    x, y = point
    return y, x


def _bounds(
    *,
    oriented_lines: Sequence[OrientedStructureLine],
    oriented_texts: Sequence[OrientedStructureText],
) -> tuple[float, float, float, float]:
    points = [point for line in oriented_lines for point in line.points]
    points.extend(text.point for text in oriented_texts)
    if not points:
        return 0.0, 1.0, 0.0, 1.0
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def _svg_tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", maxsplit=1)[-1]


def _parse_svg_transform(value: str) -> list[tuple[str, tuple[float, ...]]]:
    transforms: list[tuple[str, tuple[float, ...]]] = []
    for chunk in value.replace(",", " ").split(")"):
        chunk = chunk.strip()
        if not chunk or "(" not in chunk:
            continue
        name, raw_args = chunk.split("(", 1)
        args = tuple(float(part) for part in raw_args.split())
        transform_name = name.strip()
        if transform_name in {"translate", "scale"}:
            transforms.append((transform_name, args))
    return transforms


def _apply_svg_transform(
    point: tuple[float, float],
    transforms: Sequence[tuple[str, tuple[float, ...]]],
) -> tuple[float, float]:
    x, y = point
    for name, args in reversed(transforms):
        if name == "translate":
            x += args[0] if args else 0.0
            y += args[1] if len(args) > 1 else 0.0
        elif name == "scale":
            sx = args[0] if args else 1.0
            sy = args[1] if len(args) > 1 else sx
            x *= sx
            y *= sy
    return x, y


def _svg_points(element: ET.Element) -> list[tuple[float, float]]:
    raw_points = str(element.get("points") or "").replace(",", " ").split()
    numbers = [float(value) for value in raw_points]
    return list(zip(numbers[0::2], numbers[1::2], strict=False))


def _svg_stroke(element: ET.Element) -> str:
    class_name = str(element.get("class") or "")
    if class_name == "basepairs":
        return "#d55e00"
    if class_name == "backbone":
        return "#737373"
    style = _style_map(element)
    return style.get("stroke") or "#111827"


def _svg_stroke_width(element: ET.Element) -> float:
    class_name = str(element.get("class") or "")
    if class_name == "basepairs":
        return 2.3
    if class_name == "backbone":
        return 1.4
    return _optional_svg_float(_style_map(element).get("stroke-width")) or 1.0


def _style_map(element: ET.Element) -> dict[str, str]:
    style: dict[str, str] = {}
    for part in str(element.get("style") or "").split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        style[key.strip()] = value.strip()
    return style


def _float_attr(element: ET.Element, name: str) -> float:
    value = _optional_svg_float(element.get(name))
    if value is None:
        raise ValueError(f"ViennaRNA native SVG element is missing numeric '{name}' attribute.")
    return value


def _optional_svg_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    for suffix in ("px", "pt"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    try:
        return float(text)
    except ValueError:
        return None
