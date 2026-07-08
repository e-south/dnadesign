"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reader_spop_composite/structure_svg.py

Parse retron-hairpin ViennaRNA SVG geometry for the Reader SPOP plot.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
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
    semantic: str = ""
    kind: str = ""


@dataclass(frozen=True, slots=True)
class OrientedStructureText:
    point: tuple[float, float]
    text: str
    color: str = ""
    semantic: str = ""


@dataclass(frozen=True, slots=True)
class OrientedStructureGeometry:
    lines: tuple[OrientedStructureLine, ...]
    texts: tuple[OrientedStructureText, ...]
    bounds: tuple[float, float, float, float]


@lru_cache(maxsize=128)
def oriented_structure_geometry(
    svg_path: str, annotation_manifest_path: str | None = None
) -> OrientedStructureGeometry:
    """Return cap-right ViennaRNA structure geometry with unrotated text positions."""

    annotation = _load_structure_annotation_manifest(annotation_manifest_path)
    elements = _parse_structure_svg(Path(svg_path), annotation=annotation)
    oriented_lines = tuple(
        OrientedStructureLine(
            points=tuple(_orient_point(point) for point in line.points),
            color=line.color,
            width=line.width,
            semantic=line.semantic,
            kind=line.kind,
        )
        for line in elements.lines
    )
    oriented_texts = tuple(
        OrientedStructureText(
            point=_orient_point(text.point),
            text=text.text,
            color=text.color,
            semantic=text.semantic,
        )
        for text in elements.texts
    )
    bounds = _bounds(oriented_lines=oriented_lines, oriented_texts=oriented_texts)
    return OrientedStructureGeometry(lines=oriented_lines, texts=oriented_texts, bounds=bounds)


@dataclass(frozen=True, slots=True)
class _SourceLineElement:
    points: tuple[tuple[float, float], ...]
    color: str
    width: float
    semantic: str = ""
    kind: str = ""


@dataclass(frozen=True, slots=True)
class _SourceTextElement:
    point: tuple[float, float]
    text: str
    color: str = ""
    semantic: str = ""


@dataclass(frozen=True, slots=True)
class _SourceStructureElements:
    lines: tuple[_SourceLineElement, ...]
    texts: tuple[_SourceTextElement, ...]


@dataclass(frozen=True, slots=True)
class _StructureAnnotation:
    nucleotide_roles: tuple[str, ...]
    nucleotide_colors: tuple[str, ...]
    palette: dict[str, str]


def _parse_structure_svg(path: Path, *, annotation: _StructureAnnotation | None) -> _SourceStructureElements:
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
                lines.extend(_polyline_elements(element, points=points, annotation=annotation))
        elif tag == "line":
            start = _apply_svg_transform(
                (_float_attr(element, "x1"), _float_attr(element, "y1")),
                next_transforms,
            )
            end = _apply_svg_transform(
                (_float_attr(element, "x2"), _float_attr(element, "y2")),
                next_transforms,
            )
            semantic, color = _line_semantic_color(element, annotation=annotation)
            lines.append(
                _SourceLineElement(
                    points=(start, end),
                    color=color,
                    width=_svg_stroke_width(element),
                    semantic=semantic,
                    kind=_line_kind(element),
                )
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
    return _SourceStructureElements(lines=tuple(lines), texts=_annotated_texts(texts, annotation=annotation))


def _annotated_texts(
    texts: Sequence[_SourceTextElement],
    *,
    annotation: _StructureAnnotation | None,
) -> tuple[_SourceTextElement, ...]:
    if annotation is None or len(texts) != len(annotation.nucleotide_roles):
        return tuple(texts)
    annotated: list[_SourceTextElement] = []
    for index, text in enumerate(texts):
        semantic = annotation.nucleotide_roles[index]
        annotated.append(
            _SourceTextElement(
                point=text.point,
                text=text.text,
                color=annotation.nucleotide_colors[index],
                semantic=semantic,
            )
        )
    return tuple(annotated)


def _polyline_elements(
    element: ET.Element,
    *,
    points: tuple[tuple[float, float], ...],
    annotation: _StructureAnnotation | None,
) -> tuple[_SourceLineElement, ...]:
    if annotation is None or "backbone" not in set(str(element.get("class") or "").split()):
        return (
            _SourceLineElement(
                points=points,
                color=_svg_stroke(element),
                width=_svg_stroke_width(element),
                kind=_line_kind(element),
            ),
        )
    if len(points) != len(annotation.nucleotide_roles):
        return (
            _SourceLineElement(
                points=points,
                color=_svg_stroke(element),
                width=_svg_stroke_width(element),
                kind=_line_kind(element),
            ),
        )
    lines: list[_SourceLineElement] = []
    for index, (start, end) in enumerate(zip(points, points[1:], strict=False)):
        semantic = annotation.nucleotide_roles[index]
        color = annotation.nucleotide_colors[index] or annotation.palette.get(semantic) or _svg_stroke(element)
        lines.append(
            _SourceLineElement(
                points=(start, end),
                color=color,
                width=_svg_stroke_width(element),
                semantic=semantic,
                kind="backbone",
            )
        )
    return tuple(lines)


def _line_semantic_color(
    element: ET.Element,
    *,
    annotation: _StructureAnnotation | None,
) -> tuple[str, str]:
    source_color = _svg_stroke(element)
    if annotation is None or "basepairs" not in set(str(element.get("class") or "").split()):
        return "", source_color
    pair_id = str(element.get("id") or "")
    try:
        left_1, right_1 = (int(part) for part in pair_id.split(",", maxsplit=1))
    except ValueError:
        return "", source_color
    left_index = left_1 - 1
    right_index = right_1 - 1
    if left_index < 0 or right_index < 0:
        return "", source_color
    if left_index >= len(annotation.nucleotide_roles) or right_index >= len(annotation.nucleotide_roles):
        return "", source_color
    left_semantic = annotation.nucleotide_roles[left_index]
    right_semantic = annotation.nucleotide_roles[right_index]
    semantic = _paired_segment_semantic(left_semantic=left_semantic, right_semantic=right_semantic)
    color = annotation.palette.get(semantic) or annotation.nucleotide_colors[left_index] or source_color
    return semantic, color


def _paired_segment_semantic(*, left_semantic: str, right_semantic: str) -> str:
    pair = {left_semantic, right_semantic}
    if pair == {"payload_primary", "payload_complement"}:
        return "payload_primary"
    if pair == {"snapback_retained_stem", "snapback_foldback_return"}:
        return "snapback_retained_stem"
    if "stem_base_left" in pair and "stem_base_right" in pair:
        return "stem_base_left"
    return left_semantic or right_semantic


def _load_structure_annotation_manifest(path: str | None) -> _StructureAnnotation | None:
    if path is None or not str(path).strip():
        return None
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Structure annotation manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    palette = _manifest_palette(payload)
    nucleotide_count = len(payload.get("nucleotides", [])) if isinstance(payload.get("nucleotides"), list) else 0
    roles = [""] * nucleotide_count
    colors = [""] * nucleotide_count
    sections = payload.get("section_annotations", [])
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            role = _section_role(section)
            if not role:
                continue
            try:
                start = int(section["start"])
                end = int(section["end"])
            except (KeyError, TypeError, ValueError):
                continue
            color = palette.get(role) or str(section.get("hue") or "")
            for index in range(max(0, start), min(nucleotide_count, end)):
                if not roles[index] or _role_priority(role) >= _role_priority(roles[index]):
                    roles[index] = role
                    colors[index] = color
    nucleotides = payload.get("nucleotides", [])
    if isinstance(nucleotides, list):
        for index, nucleotide in enumerate(nucleotides[:nucleotide_count]):
            if not isinstance(nucleotide, dict):
                continue
            if not roles[index]:
                role = _nucleotide_role(nucleotide, palette=palette)
                roles[index] = role
                colors[index] = palette.get(role) or str(nucleotide.get("hue") or "")
            elif not colors[index]:
                colors[index] = palette.get(roles[index]) or str(nucleotide.get("hue") or "")
    return _StructureAnnotation(nucleotide_roles=tuple(roles), nucleotide_colors=tuple(colors), palette=palette)


def _manifest_palette(payload: dict[str, object]) -> dict[str, str]:
    raw_palette = payload.get("palette")
    if not isinstance(raw_palette, dict):
        return {}
    return {str(key): str(value) for key, value in raw_palette.items()}


def _section_role(section: dict[str, object]) -> str:
    label = str(section.get("label") or "").strip().lower()
    if str(section.get("section_kind") or "") == "stem_base":
        if "left" in label:
            return "stem_base_left"
        if "right" in label:
            return "stem_base_right"
    semantic = str(section.get("section_semantic") or "").strip()
    if semantic:
        return semantic
    tokens = section.get("semantic_tokens")
    if isinstance(tokens, list) and tokens:
        return str(tokens[0])
    return ""


def _nucleotide_role(nucleotide: dict[str, object], *, palette: dict[str, str]) -> str:
    effect_tags = nucleotide.get("effect_tags")
    if isinstance(effect_tags, list):
        for raw_tag in effect_tags:
            tag = str(raw_tag)
            if tag in palette:
                return tag
    css_class = str(nucleotide.get("css_class") or "")
    prefix = "dnadesign-component-"
    if css_class.startswith(prefix):
        return css_class.removeprefix(prefix)
    return ""


def _role_priority(role: str) -> int:
    if role in {"stem_base_left", "stem_base_right", "snapback_cap"}:
        return 80
    if role in {"snapback_retained_stem", "snapback_foldback_return"}:
        return 70
    if role in {"stem_extension_left", "stem_extension_right"}:
        return 60
    if role in {"payload_primary", "payload_complement"}:
        return 50
    if role == "snapback_foldback_geometry":
        return 40
    return 10


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


def _line_kind(element: ET.Element) -> str:
    class_names = set(str(element.get("class") or "").split())
    if "basepairs" in class_names:
        return "basepair"
    if "backbone" in class_names:
        return "backbone"
    return ""


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
