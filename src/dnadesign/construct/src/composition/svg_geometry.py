"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition/svg_geometry.py

SVG geometry and style parsing helpers for construct composition reviews.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from ..contracts.errors import ValidationError


def contains_fill_color(element: ET.Element, color: str) -> bool:
    normalized_color = normalize_css_text(color)
    return any(normalized_color in normalize_css_text(str(node.attrib.get("style", ""))) for node in element.iter())


def translate_y_from_transform(transform: str) -> float | None:
    match = re.search(r"translate\(([^)]*)\)", transform)
    if match is None:
        return None
    values = parse_float_list(match.group(1))
    if len(values) < 2:
        return 0.0 if values else None
    return values[1]


def normalize_css_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text).strip().lower())


def viewbox(root: ET.Element) -> tuple[float, float, float, float]:
    raw = str(root.attrib.get("viewBox", "")).strip()
    if raw:
        values = parse_float_list(raw)
        if len(values) == 4:
            return values[0], values[1], values[2], values[3]
    width = numeric_length(root.attrib.get("width"))
    height = numeric_length(root.attrib.get("height"))
    if width is not None and height is not None:
        return 0.0, 0.0, width, height
    raise ValidationError("SVG must provide either viewBox or width/height.")


def extract_nucleotide_font_size_px(root: ET.Element, *, default: float) -> float:
    for node in root.iter():
        if local_name(node.tag) != "text":
            continue
        classes = set(str(node.attrib.get("class", "")).split())
        if "nucleotide" not in classes:
            continue
        font_size = font_size_from_style(str(node.attrib.get("style", "")))
        if font_size is not None:
            return font_size
        font_size = numeric_length(node.attrib.get("font-size"))
        if font_size is not None:
            return font_size
    style_text = "\n".join(str(node.text or "") for node in root.iter() if local_name(node.tag) == "style")
    match = re.search(r"\.nucleotide\s*\{[^}]*font-size\s*:\s*([0-9.]+)", style_text, flags=re.IGNORECASE)
    if match is not None:
        return float(match.group(1))
    return default


def font_size_from_style(style: str) -> float | None:
    match = re.search(r"(?:^|;)\s*font-size\s*:\s*([0-9.]+)", style, flags=re.IGNORECASE)
    if match is None:
        return None
    return float(match.group(1))


def append_style_declarations(style: str, declarations: tuple[str, ...]) -> str:
    existing = style.strip()
    suffix = "; ".join(item.strip().rstrip(";") for item in declarations if item.strip())
    if not suffix:
        return existing
    if existing and not existing.endswith(";"):
        existing = f"{existing};"
    return f"{existing} {suffix};".strip() if existing else f"{suffix};"


def numeric_length(value: object) -> float | None:
    if value is None:
        return None
    match = re.match(r"\s*([-+]?\d+(?:\.\d+)?)", str(value))
    if match is None:
        return None
    return float(match.group(1))


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in re.split(r"[\s,]+", raw.strip()):
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values


def local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", maxsplit=1)[-1]
    return tag


__all__ = [
    "append_style_declarations",
    "contains_fill_color",
    "extract_nucleotide_font_size_px",
    "font_size_from_style",
    "local_name",
    "normalize_css_text",
    "numeric_length",
    "parse_float_list",
    "translate_y_from_transform",
    "viewbox",
]
