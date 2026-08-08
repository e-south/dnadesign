"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/composition/review_svg.py

Composition-review SVG layout and panel normalization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from dnadesign.contracts.visual import SequenceEvidenceMapV1

from ..contracts.errors import ValidationError
from .review_assets import (
    COMPONENT_PANEL_EMPHASIS,
    COMPONENT_SOURCE_TITLE_COLOR,
    COMPONENT_SOURCE_TITLE_POLICY,
    STRUCTURE_FIT_POLICY,
    SVG_NS,
    SvgAsset,
    composition_id_from_visual_contract,
)
from .svg_geometry import (
    append_style_declarations,
    contains_fill_color,
    local_name,
    normalize_css_text,
    numeric_length,
    translate_y_from_transform,
)


@dataclass(frozen=True)
class ReviewSvgBuild:
    root: ET.Element
    component_source_title_omitted_count: int


def compose_review_svg(
    *,
    structure: SvgAsset,
    component: SvgAsset,
    visual_contract: SequenceEvidenceMapV1,
    target_nucleotide_font_size_px: float,
    structure_scale: float,
    component_scale: float,
    structure_effective_font_size: float,
    component_effective_font_size: float,
) -> ReviewSvgBuild:
    pad = 18.0
    gap = 18.0
    structure_width = structure.width * structure_scale
    structure_height = structure.height * structure_scale
    component_width = component.width * component_scale
    component_height = component.height * component_scale
    structure_caption = _structure_caption_for_review(visual_contract, structure)
    caption_title, caption_subtitles = structure_caption
    caption_height = _structure_caption_height(structure_caption)
    content_width = max(structure_width, component_width)
    outer_width = content_width + 2 * pad
    outer_height = caption_height + structure_height + gap + component_height + 2 * pad
    composition_id = composition_id_from_visual_contract(visual_contract)
    root = ET.Element(
        f"{{{SVG_NS}}}svg",
        {
            "width": f"{outer_width:.0f}",
            "height": f"{outer_height:.0f}",
            "viewBox": f"0 0 {outer_width:.3f} {outer_height:.3f}",
            "data-dnadesign-contract-kind": "composition_review_svg_v1",
            "data-dnadesign-composition-id": composition_id,
            "data-dnadesign-component-nucleotide-font-size-px": f"{target_nucleotide_font_size_px:.3f}",
            "data-dnadesign-structure-effective-nucleotide-font-size-px": f"{structure_effective_font_size:.3f}",
            "data-dnadesign-component-effective-nucleotide-font-size-px": f"{component_effective_font_size:.3f}",
            "data-dnadesign-component-panel-emphasis": COMPONENT_PANEL_EMPHASIS,
            "data-dnadesign-structure-fit-policy": STRUCTURE_FIT_POLICY,
            "data-dnadesign-structure-source-policy": "oriented_annotated_geometry_with_review_caption",
        },
    )
    title = ET.SubElement(root, f"{{{SVG_NS}}}title")
    title.text = f"{composition_id} composition review"
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}rect",
        {
            "x": "0",
            "y": "0",
            "width": f"{outer_width:.3f}",
            "height": f"{outer_height:.3f}",
            "style": "fill: #FFFFFF; stroke: none;",
        },
    )
    if caption_title:
        _append_structure_caption(
            root,
            x=pad + content_width / 2.0,
            y=pad,
            title=caption_title,
            subtitles=caption_subtitles,
        )
    _append_panel(
        root,
        asset=structure,
        panel="secondary_structure",
        row=1,
        x=pad + (content_width - structure_width) / 2.0,
        y=pad + caption_height,
        scale=structure_scale,
        effective_font_size=structure_effective_font_size,
    )
    component_source_title_omitted_count = _append_panel(
        root,
        asset=component,
        panel="component_span",
        row=2,
        x=pad + (content_width - component_width) / 2.0,
        y=pad + caption_height + structure_height + gap,
        scale=component_scale,
        effective_font_size=component_effective_font_size,
    )
    return ReviewSvgBuild(
        root=root,
        component_source_title_omitted_count=component_source_title_omitted_count,
    )


def validate_structure_nucleotide_text(
    root: ET.Element,
    *,
    expected_count: int,
    source_nucleotide_font_size_px: float,
) -> None:
    panel = _find_panel(root, "secondary_structure")
    if panel is None:
        raise ValidationError("Composition review SVG is missing the secondary_structure panel.")
    nucleotide_nodes = [
        node
        for node in panel.iter()
        if local_name(node.tag) == "text" and "nucleotide" in set(str(node.attrib.get("class", "")).split())
    ]
    if len(nucleotide_nodes) != expected_count:
        raise ValidationError(
            "Composition review secondary_structure panel has "
            f"{len(nucleotide_nodes)} nucleotide text nodes; expected {expected_count}."
        )
    missing_style_count = sum(
        1
        for node in nucleotide_nodes
        if numeric_length(node.attrib.get("font-size")) != source_nucleotide_font_size_px
        or "dejavusans" not in normalize_css_text(str(node.attrib.get("font-family", "")))
        or "font-size" not in normalize_css_text(str(node.attrib.get("style", "")))
        or "font-family" not in normalize_css_text(str(node.attrib.get("style", "")))
    )
    if missing_style_count:
        raise ValidationError(
            "Composition review secondary_structure panel contains "
            f"{missing_style_count} nucleotide text nodes without explicit renderer-safe font styling."
        )
    panel.set("data-dnadesign-structure-nucleotide-text-count", str(len(nucleotide_nodes)))
    panel.set("data-dnadesign-structure-nucleotide-text-font-policy", "explicit_renderer_safe")


def _structure_caption(visual_contract: SequenceEvidenceMapV1) -> tuple[str, list[str]]:
    title = str(
        visual_contract.meta.get("structure_title") or visual_contract.display.title or visual_contract.state_id
    )
    title = re.sub(r"\s+", " ", title.strip())
    subtitles: list[str] = []
    facts = visual_contract.meta.get("facts")
    if isinstance(facts, list):
        subtitle_parts = [
            f"{str(fact.get('label') or '').strip()} {str(fact.get('value') or '').strip()}".strip()
            for fact in facts
            if isinstance(fact, dict) and str(fact.get("label") or "").strip() and str(fact.get("value") or "").strip()
        ]
        if subtitle_parts:
            subtitles.append(" | ".join(subtitle_parts))
    return title, subtitles


def _structure_caption_for_review(
    visual_contract: SequenceEvidenceMapV1,
    structure: SvgAsset,
) -> tuple[str, list[str]]:
    if _has_element_id(structure.root, "dnadesign-secondary-structure-title"):
        return "", []
    return _structure_caption(visual_contract)


def _has_element_id(root: ET.Element, element_id: str) -> bool:
    return any(node.attrib.get("id") == element_id for node in root.iter())


def _structure_caption_height(structure_caption: tuple[str, list[str]]) -> float:
    title, subtitles = structure_caption
    if not title:
        return 0.0
    return 24.0 + 12.0 * len(subtitles)


def _append_structure_caption(
    root: ET.Element,
    *,
    x: float,
    y: float,
    title: str,
    subtitles: list[str],
) -> None:
    layer = ET.SubElement(
        root,
        f"{{{SVG_NS}}}g",
        {
            "id": "dnadesign-composition-review-structure-caption",
            "data-dnadesign-caption-layer": "secondary_structure",
        },
    )
    title_node = ET.SubElement(
        layer,
        f"{{{SVG_NS}}}text",
        {
            "class": "dnadesign-structure-title",
            "x": f"{x:.3f}",
            "y": f"{y + 8.0:.3f}",
            "text-anchor": "middle",
            "dominant-baseline": "middle",
            "style": "font-family: DejaVu Sans, Arial, sans-serif; font-size: 12px; font-weight: 700; fill: #111827;",
        },
    )
    title_node.text = title
    for index, subtitle in enumerate(subtitles):
        subtitle_node = ET.SubElement(
            layer,
            f"{{{SVG_NS}}}text",
            {
                "class": "dnadesign-structure-subtitle",
                "data-dnadesign-subtitle-line-index": str(index),
                "x": f"{x:.3f}",
                "y": f"{y + 21.0 + index * 12.0:.3f}",
                "text-anchor": "middle",
                "dominant-baseline": "middle",
                "style": (
                    "font-family: DejaVu Sans, Arial, sans-serif; font-size: 9px; font-weight: 500; fill: #475569;"
                ),
            },
        )
        subtitle_node.text = subtitle


def _append_panel(
    root: ET.Element,
    *,
    asset: SvgAsset,
    panel: str,
    row: int,
    x: float,
    y: float,
    scale: float,
    effective_font_size: float,
) -> int:
    nested = ET.SubElement(
        root,
        f"{{{SVG_NS}}}svg",
        {
            "x": f"{x:.3f}",
            "y": f"{y:.3f}",
            "width": f"{asset.width * scale:.3f}",
            "height": f"{asset.height * scale:.3f}",
            "viewBox": " ".join(f"{part:.3f}" for part in asset.viewbox),
            "data-dnadesign-panel": panel,
            "data-dnadesign-panel-row": str(row),
            "data-dnadesign-source-svg": asset.path.name,
            "data-dnadesign-source-nucleotide-font-size-px": f"{asset.source_nucleotide_font_size_px:.3f}",
            "data-dnadesign-effective-nucleotide-font-size-px": f"{effective_font_size:.3f}",
            "overflow": "visible",
        },
    )
    if panel == "secondary_structure":
        orientation = str(asset.root.attrib.get("data-dnadesign-orientation", "")).strip()
        orientation_angle = str(asset.root.attrib.get("data-dnadesign-orientation-angle-deg", "")).strip()
        if orientation:
            nested.set("data-dnadesign-source-orientation", orientation)
        if orientation_angle:
            nested.set("data-dnadesign-source-orientation-angle-deg", orientation_angle)
    omitted_count = 0
    if panel == "component_span":
        nested.set("data-dnadesign-component-panel-emphasis", COMPONENT_PANEL_EMPHASIS)
        nested.set("data-dnadesign-component-source-title-policy", COMPONENT_SOURCE_TITLE_POLICY)
    for child in list(asset.root):
        child_copy = copy.deepcopy(child)
        if panel == "secondary_structure":
            _normalize_structure_nucleotide_text(
                child_copy,
                source_nucleotide_font_size_px=asset.source_nucleotide_font_size_px,
            )
        if panel == "component_span":
            omitted_count += _remove_component_source_title_groups(child_copy, source_height=asset.height)
        nested.append(child_copy)
    if panel == "component_span":
        nested.set("data-dnadesign-component-source-title-omitted-count", str(omitted_count))
    return omitted_count


def _normalize_structure_nucleotide_text(element: ET.Element, *, source_nucleotide_font_size_px: float) -> None:
    for node in element.iter():
        if local_name(node.tag) != "text":
            continue
        if "nucleotide" not in set(str(node.attrib.get("class", "")).split()):
            continue
        node.set("font-size", f"{source_nucleotide_font_size_px:.3f}px")
        node.set("font-family", "DejaVu Sans, Arial, sans-serif")
        style = str(node.attrib.get("style", "")).strip()
        node.set(
            "style",
            append_style_declarations(
                style,
                (
                    "font-family: DejaVu Sans, Arial, sans-serif",
                    f"font-size: {source_nucleotide_font_size_px:.3f}px",
                ),
            ),
        )


def _find_panel(root: ET.Element, panel: str) -> ET.Element | None:
    for node in root.iter():
        if node.attrib.get("data-dnadesign-panel") == panel:
            return node
    return None


def _remove_component_source_title_groups(element: ET.Element, *, source_height: float) -> int:
    removed_count = 0
    for child in list(element):
        if _is_component_source_title_group(child, source_height=source_height):
            element.remove(child)
            removed_count += 1
            continue
        removed_count += _remove_component_source_title_groups(child, source_height=source_height)
    return removed_count


def _is_component_source_title_group(element: ET.Element, *, source_height: float) -> bool:
    if local_name(element.tag) != "g":
        return False
    if not str(element.attrib.get("id", "")).startswith("text_"):
        return False
    if not contains_fill_color(element, COMPONENT_SOURCE_TITLE_COLOR):
        return False
    y_values = [
        y
        for node in element.iter()
        if (y := translate_y_from_transform(str(node.attrib.get("transform", "")))) is not None
    ]
    if not y_values:
        return False
    return min(y_values) <= max(12.0, source_height * 0.24)


__all__ = [
    "ReviewSvgBuild",
    "compose_review_svg",
    "validate_structure_nucleotide_text",
]
