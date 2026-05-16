"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/folding/src/viennarna_svg.py

SVG DOM annotation and orientation helpers for ViennaRNA-native plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from dnadesign.contracts.visual import SequenceEvidenceMapV1

from .errors import FoldingExecutionError
from .pairing_qa import copy_for_index, pair_key
from .viennarna_ontology import component_token, hue_for_owners, slug_token
from .viennarna_summary import structure_subtitle_lines, structure_title

SVG_NS = "http://www.w3.org/2000/svg"
_TRANSFORM_RE = re.compile(r"([A-Za-z]+)\(([^)]*)\)")
_GRAPHICS_WRAP_EXCLUDE = {"script", "style", "defs", "title", "desc"}
_DEFAULT_LABEL_OFFSETS = (34.0, 46.0, 58.0, 70.0, 84.0, 100.0, 118.0, 136.0, 156.0, 178.0)
_DEFAULT_LABEL_LATERALS = (0.0, 18.0, -18.0, 36.0, -36.0, 58.0, -58.0, 82.0, -82.0)
_STEM_BASE_LABEL_OFFSETS = (18.0, 24.0, 30.0, 38.0, 48.0, 56.0, 68.0, 84.0)
_STEM_BASE_LABEL_LATERALS = (0.0, 12.0, -12.0, 24.0, -24.0, 38.0, -38.0, 50.0, -50.0)
_ANNOTATION_FONT_SIZE_PX = 9.0


@dataclass(frozen=True)
class SvgSurface:
    tree: ET.ElementTree
    nucleotide_nodes: tuple[ET.Element, ...]
    basepair_nodes: tuple[ET.Element, ...]
    parent_map: dict[ET.Element, ET.Element]


@dataclass(frozen=True)
class SvgAnnotationResult:
    basepairs: list[dict[str, object]]
    section_annotations: list[dict[str, object]]
    layout_normalization: dict[str, object]


def load_svg_surface(path: Path) -> SvgSurface:
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise FoldingExecutionError(f"ViennaRNA SVG is not parseable XML: {path}") from exc
    root = tree.getroot()
    nucleotide_nodes: list[ET.Element] = []
    basepair_nodes: list[ET.Element] = []
    for element in root.iter():
        classes = set(str(element.attrib.get("class", "")).split())
        if _local_name(element.tag) == "text" and "nucleotide" in classes:
            nucleotide_nodes.append(element)
        if _local_name(element.tag) == "line" and "basepairs" in classes:
            basepair_nodes.append(element)
    parent_map = {child: parent for parent in root.iter() for child in list(parent)}
    return SvgSurface(
        tree=tree,
        nucleotide_nodes=tuple(nucleotide_nodes),
        basepair_nodes=tuple(basepair_nodes),
        parent_map=parent_map,
    )


def annotate_svg_surface(
    surface: SvgSurface,
    *,
    nucleotide_annotations: list[dict[str, object]],
    unit_copy_spans: tuple[dict[str, int | str], ...],
    intended_pair_lookup: dict[tuple[int, int], tuple[str, ...]],
    visual_contract: SequenceEvidenceMapV1 | None,
    emphasize_stem_base_nucleotides: bool = True,
) -> SvgAnnotationResult:
    basepair_annotations = _annotate_nucleotides_and_basepairs(
        surface,
        nucleotide_annotations=nucleotide_annotations,
        unit_copy_spans=unit_copy_spans,
        intended_pair_lookup=intended_pair_lookup,
        emphasize_stem_base_nucleotides=emphasize_stem_base_nucleotides,
    )
    section_annotations = _section_annotations(surface, visual_contract=visual_contract)
    layout_normalization = _normalize_structure_orientation(surface, section_annotations)
    highlight_boxes = _add_section_highlight_layer(
        surface,
        section_annotations=section_annotations,
        layout_normalization=layout_normalization,
    )
    section_annotations = _add_section_label_layer(
        surface,
        section_annotations=section_annotations,
        layout_normalization=layout_normalization,
        visual_contract=visual_contract,
        extra_view_boxes=highlight_boxes,
    )
    return SvgAnnotationResult(
        basepairs=basepair_annotations,
        section_annotations=section_annotations,
        layout_normalization=layout_normalization,
    )


def _annotate_nucleotides_and_basepairs(
    surface: SvgSurface,
    *,
    nucleotide_annotations: list[dict[str, object]],
    unit_copy_spans: tuple[dict[str, int | str], ...],
    intended_pair_lookup: dict[tuple[int, int], tuple[str, ...]],
    emphasize_stem_base_nucleotides: bool,
) -> list[dict[str, object]]:
    for element, annotation in zip(surface.nucleotide_nodes, nucleotide_annotations, strict=True):
        owner_ids = ",".join(str(item) for item in annotation["owner_ids"])
        effect_tag_values = tuple(str(item) for item in annotation["effect_tags"])
        effect_tags = ",".join(effect_tag_values)
        css_class = str(annotation["css_class"])
        element.set("data-dnadesign-index0", str(annotation["index_0"]))
        element.set("data-dnadesign-index1", str(annotation["display_index_1"]))
        element.set("data-dnadesign-owner-ids", owner_ids)
        element.set("data-dnadesign-effect-tags", effect_tags)
        element.set("data-dnadesign-hue", str(annotation["hue"]))
        classes = [item for item in str(element.attrib.get("class", "")).split() if item]
        if css_class not in classes:
            classes.append(css_class)
        is_stem_base = any(_is_stem_base_tag(tag) for tag in effect_tag_values)
        if emphasize_stem_base_nucleotides and is_stem_base:
            element.set("data-dnadesign-stem-base-emphasis", "true")
            if "dnadesign-stem-base-nucleotide" not in classes:
                classes.append("dnadesign-stem-base-nucleotide")
        element.set("class", " ".join(classes))
        style = str(element.attrib.get("style", "")).strip()
        hue_style = f"fill: {annotation['hue']};"
        next_style = f"{style} {hue_style}".strip() if style else hue_style
        if emphasize_stem_base_nucleotides and is_stem_base:
            next_style = _append_style_declarations(
                next_style,
                (
                    "font-weight: 700",
                    "stroke: #111827",
                    "stroke-width: 0.35px",
                    "paint-order: stroke fill",
                ),
            )
        element.set("style", next_style)

    basepair_annotations: list[dict[str, object]] = []
    for element in surface.basepair_nodes:
        pair_id = str(element.attrib.get("id", ""))
        try:
            left_1, right_1 = (int(part) for part in pair_id.split(",", maxsplit=1))
        except ValueError:
            continue
        left_0 = left_1 - 1
        right_0 = right_1 - 1
        intended_pairing_ids = intended_pair_lookup.get(pair_key(left_0, right_0), ())
        left_copy = copy_for_index(left_0, unit_copy_spans)
        right_copy = copy_for_index(right_0, unit_copy_spans)
        is_cross_copy = (
            left_copy is not None
            and right_copy is not None
            and (left_copy["unit_id"], left_copy["copy_index"]) != (right_copy["unit_id"], right_copy["copy_index"])
        )
        element.set("data-dnadesign-left-index0", str(left_0))
        element.set("data-dnadesign-right-index0", str(right_0))
        element.set("data-dnadesign-pair-kind", "predicted_secondary_structure")
        element.set("data-dnadesign-cross-copy", "true" if is_cross_copy else "false")
        element.set("data-dnadesign-intended-match", "true" if intended_pairing_ids else "false")
        if intended_pairing_ids:
            element.set("data-dnadesign-intended-pairing-ids", ",".join(intended_pairing_ids))
        if left_copy is not None:
            element.set("data-dnadesign-left-copy-index", str(left_copy["copy_index"]))
        if right_copy is not None:
            element.set("data-dnadesign-right-copy-index", str(right_copy["copy_index"]))
        basepair_annotations.append(
            {
                "left_index_0": left_0,
                "right_index_0": right_0,
                "left_display_index_1": left_1,
                "right_display_index_1": right_1,
                "pair_kind": "predicted_secondary_structure",
                "is_cross_copy": is_cross_copy,
                "intended_match": bool(intended_pairing_ids),
                "intended_pairing_ids": list(intended_pairing_ids),
                "left_unit_id": None if left_copy is None else left_copy["unit_id"],
                "left_copy_index": None if left_copy is None else left_copy["copy_index"],
                "right_unit_id": None if right_copy is None else right_copy["unit_id"],
                "right_copy_index": None if right_copy is None else right_copy["copy_index"],
            }
        )
    return basepair_annotations


def _section_annotations(
    surface: SvgSurface,
    *,
    visual_contract: SequenceEvidenceMapV1 | None,
) -> list[dict[str, object]]:
    if visual_contract is None:
        return []
    component_palette = _component_palette(visual_contract)
    raw_labels = visual_contract.meta.get("segment_labels")
    if not isinstance(raw_labels, list):
        raw_labels = []
    annotations: list[dict[str, object]] = []
    for index, raw in enumerate(raw_labels):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        try:
            start = int(raw.get("start"))
            end = int(raw.get("end"))
        except (TypeError, ValueError):
            continue
        if start < 0 or end > len(surface.nucleotide_nodes) or end <= start:
            continue
        section_nodes = surface.nucleotide_nodes[start:end]
        anchor_x, anchor_y = _centroid([_element_world_point(surface, node) for node in section_nodes])
        owner_ids = _owner_ids_for_span(visual_contract, start=start, end=end)
        semantic_tokens = tuple(dict.fromkeys(component_token((owner_id,)) for owner_id in owner_ids))
        annotations.append(
            {
                "section_id": f"section_{index:02d}_{slug_token(text)}",
                "label": text,
                "section_kind": _section_kind(text),
                "start": start,
                "end": end,
                "anchor_x": anchor_x,
                "anchor_y": anchor_y,
                "owner_ids": list(owner_ids),
                "semantic_tokens": list(semantic_tokens),
                "hue": hue_for_owners(owner_ids, palette=component_palette),
            }
        )
    return annotations


def _normalize_structure_orientation(
    surface: SvgSurface,
    section_annotations: list[dict[str, object]],
) -> dict[str, object]:
    requested_orientation = "cap_right"
    normalization: dict[str, object] = {
        "requested_orientation": requested_orientation,
        "applied": False,
        "angle_degrees": 0.0,
        "anchor": None,
    }
    if not section_annotations:
        normalization["reason"] = "no_section_annotations"
        return normalization
    cap_section = _cap_section(section_annotations)
    if cap_section is None:
        normalization["reason"] = "no_cap_section"
        return normalization
    non_cap_points = [
        (float(section["anchor_x"]), float(section["anchor_y"]))
        for section in section_annotations
        if section is not cap_section
    ]
    if not non_cap_points:
        normalization["reason"] = "no_non_cap_sections"
        return normalization
    cap_point = (float(cap_section["anchor_x"]), float(cap_section["anchor_y"]))
    stem_center = _centroid(non_cap_points)
    dx = cap_point[0] - stem_center[0]
    dy = cap_point[1] - stem_center[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        normalization["reason"] = "cap_vector_degenerate"
        return normalization
    angle_degrees = -math.degrees(math.atan2(dy, dx))
    normalization.update(
        {
            "angle_degrees": round(angle_degrees, 6),
            "anchor": cap_section["section_id"],
            "center_x": stem_center[0],
            "center_y": stem_center[1],
        }
    )
    original_points = [_element_world_point(surface, node) for node in surface.nucleotide_nodes]
    if abs(angle_degrees) < 1.0:
        normalization["reason"] = "already_cap_right"
        normalization["geometry_bbox"] = _bbox(original_points)
        surface.tree.getroot().set("data-dnadesign-orientation", requested_orientation)
        return normalization
    if not _wrap_svg_graphics_for_orientation(surface, angle_degrees=angle_degrees, center=stem_center):
        normalization["reason"] = "no_graphics_to_wrap"
        normalization["geometry_bbox"] = _bbox(original_points)
        return normalization
    _counter_rotate_nucleotide_text(surface, angle_degrees=angle_degrees)
    rotated_points = [
        _rotate_point(point, center=stem_center, angle_degrees=angle_degrees) for point in original_points
    ]
    normalization["geometry_bbox"] = _bbox(rotated_points)
    surface.tree.getroot().set("data-dnadesign-orientation", requested_orientation)
    surface.tree.getroot().set("data-dnadesign-orientation-angle-deg", f"{angle_degrees:.3f}")
    normalization["nucleotide_text_orientation"] = "upright_counter_rotated"
    normalization["applied"] = True
    return normalization


def _counter_rotate_nucleotide_text(surface: SvgSurface, *, angle_degrees: float) -> None:
    counter_angle = -angle_degrees
    for node in surface.nucleotide_nodes:
        x = _float_attr(node, "x", fallback=0.0)
        y = _float_attr(node, "y", fallback=0.0)
        existing = str(node.attrib.get("transform", "")).strip()
        counter = f"rotate({counter_angle:.3f} {x:.3f} {y:.3f})"
        node.set("transform", f"{counter} {existing}".strip() if existing else counter)
        node.set("data-dnadesign-upright-text", "true")


def _add_section_highlight_layer(
    surface: SvgSurface,
    *,
    section_annotations: list[dict[str, object]],
    layout_normalization: dict[str, object],
) -> list[tuple[float, float, float, float]]:
    stem_sections = [section for section in section_annotations if str(section.get("section_kind", "")) == "stem_base"]
    if not stem_sections:
        return []
    root = surface.tree.getroot()
    layer = ET.Element(
        f"{{{SVG_NS}}}g",
        {
            "id": "dnadesign-secondary-structure-highlights",
            "data-dnadesign-highlight-layer": "section_annotations",
        },
    )
    boxes: list[tuple[float, float, float, float]] = []
    for section in stem_sections:
        points = _section_display_points(surface, section, layout_normalization)
        if not points:
            continue
        box = _padded_box(_points_bbox(points), pad=9.0)
        boxes.append(box)
        ET.SubElement(
            layer,
            f"{{{SVG_NS}}}rect",
            {
                "class": "dnadesign-section-highlight dnadesign-section-highlight-stem-base",
                "data-dnadesign-section-id": str(section["section_id"]),
                "data-dnadesign-section-label": str(section["label"]),
                "data-dnadesign-section-kind": "stem_base",
                "x": f"{box[0]:.3f}",
                "y": f"{box[1]:.3f}",
                "width": f"{box[2] - box[0]:.3f}",
                "height": f"{box[3] - box[1]:.3f}",
                "rx": "4.000",
                "ry": "4.000",
                "style": (
                    "fill: #FDE68A; fill-opacity: 0.28; stroke: #B45309; stroke-width: 0.9px; stroke-opacity: 0.72;"
                ),
            },
        )
    if len(layer):
        root.append(layer)
    return boxes


def _add_section_label_layer(
    surface: SvgSurface,
    *,
    section_annotations: list[dict[str, object]],
    layout_normalization: dict[str, object],
    visual_contract: SequenceEvidenceMapV1 | None,
    extra_view_boxes: list[tuple[float, float, float, float]],
) -> list[dict[str, object]]:
    if not section_annotations:
        return []
    root = surface.tree.getroot()
    layer = ET.Element(
        f"{{{SVG_NS}}}g",
        {
            "id": "dnadesign-secondary-structure-labels",
            "data-dnadesign-label-layer": "section_annotations",
        },
    )
    center = _label_center(section_annotations, layout_normalization)
    nucleotide_boxes = _nucleotide_obstacle_boxes(surface, layout_normalization)
    title_layout = _title_layout(
        visual_contract=visual_contract,
        section_annotations=section_annotations,
        layout_normalization=layout_normalization,
    )
    title_boxes = [] if title_layout is None else title_layout["boxes"]
    placed_boxes: list[tuple[float, float, float, float]] = []
    placed_annotations: list[dict[str, object]] = []
    for section in section_annotations:
        is_stem_base = _is_stem_base_section(section)
        reserved_boxes = [*title_boxes] if is_stem_base else [*title_boxes, *extra_view_boxes]
        anchor = (float(section["anchor_x"]), float(section["anchor_y"]))
        if bool(layout_normalization.get("applied")):
            anchor = _rotate_point(
                anchor,
                center=(
                    float(layout_normalization.get("center_x", center[0])),
                    float(layout_normalization.get("center_y", center[1])),
                ),
                angle_degrees=float(layout_normalization.get("angle_degrees", 0.0)),
            )
        label_width = max(24.0, len(str(section["label"])) * 5.4)
        label_height = 11.0
        label_x, label_y, box, attempts = _place_label_without_collisions(
            anchor,
            center=center,
            label_width=label_width,
            label_height=label_height,
            occupied_boxes=[*nucleotide_boxes, *reserved_boxes, *placed_boxes],
            offset_distances=_STEM_BASE_LABEL_OFFSETS if is_stem_base else _DEFAULT_LABEL_OFFSETS,
            lateral_distances=_STEM_BASE_LABEL_LATERALS if is_stem_base else _DEFAULT_LABEL_LATERALS,
        )
        placed_boxes.append(box)
        label_nucleotide_overlap_count = sum(1 for obstacle in nucleotide_boxes if _boxes_overlap(box, obstacle))
        label_reserved_overlap_count = sum(1 for obstacle in reserved_boxes if _boxes_overlap(box, obstacle))
        label_peer_overlap_count = sum(1 for obstacle in placed_boxes[:-1] if _boxes_overlap(box, obstacle))
        color = str(section.get("hue") or "#334155")
        leader = ET.SubElement(
            layer,
            f"{{{SVG_NS}}}line",
            {
                "class": "dnadesign-section-label-leader",
                "x1": f"{anchor[0]:.3f}",
                "y1": f"{anchor[1]:.3f}",
                "x2": f"{label_x:.3f}",
                "y2": f"{label_y:.3f}",
                "style": "stroke: #94A3B8; stroke-width: 0.7px; stroke-opacity: 0.72;",
            },
        )
        leader.set("data-dnadesign-section-label", str(section["label"]))
        text = ET.SubElement(
            layer,
            f"{{{SVG_NS}}}text",
            {
                "class": "dnadesign-section-label",
                "x": f"{label_x:.3f}",
                "y": f"{label_y:.3f}",
                "text-anchor": "middle",
                "dominant-baseline": "middle",
                "data-dnadesign-section-id": str(section["section_id"]),
                "data-dnadesign-section-label": str(section["label"]),
                "style": (
                    f"font-family: DejaVu Sans, Arial, sans-serif; font-size: {_format_px(_ANNOTATION_FONT_SIZE_PX)}; "
                    f"font-weight: 600; fill: {color}; stroke: #FFFFFF; stroke-width: 2px; "
                    "paint-order: stroke fill; stroke-linejoin: round;"
                ),
            },
        )
        text.text = str(section["label"])
        output = dict(section)
        output.update(
            {
                "anchor_x": round(anchor[0], 6),
                "anchor_y": round(anchor[1], 6),
                "label_x": round(label_x, 6),
                "label_y": round(label_y, 6),
                "collision_adjustments": attempts,
                "label_nucleotide_overlap_count": label_nucleotide_overlap_count,
                "label_reserved_overlap_count": label_reserved_overlap_count,
                "label_peer_overlap_count": label_peer_overlap_count,
            }
        )
        placed_annotations.append(output)
    root.append(layer)
    title_boxes = _add_title_layer(
        root,
        visual_contract=visual_contract,
        section_annotations=section_annotations,
        layout_normalization=layout_normalization,
        title_layout=title_layout,
    )
    _expand_root_viewbox(
        root,
        layout_normalization=layout_normalization,
        label_boxes=[*extra_view_boxes, *placed_boxes, *title_boxes],
    )
    return placed_annotations


def _owner_ids_for_span(
    visual_contract: SequenceEvidenceMapV1,
    *,
    start: int,
    end: int,
) -> tuple[str, ...]:
    owner_ids: list[str] = []
    for owner in visual_contract.owners:
        if owner.row_id != "primary":
            continue
        if int(owner.start) < end and start < int(owner.end):
            owner_ids.append(owner.owner_id)
    return tuple(dict.fromkeys(owner_ids))


def _section_kind(label: str) -> str:
    lowered = str(label).strip().lower()
    if "stem base" in lowered or lowered in {"left base", "right base"}:
        return "stem_base"
    if "cap" in lowered:
        return "cap"
    return "component"


def _is_stem_base_section(section: dict[str, object]) -> bool:
    return str(section.get("section_kind", "")) == "stem_base"


def _is_stem_base_tag(tag: str) -> bool:
    token = str(tag).strip().lower()
    if token.startswith("effect:"):
        token = token.removeprefix("effect:")
    return token in {"stem_base", "stem_base_left", "stem_base_right"}


def _append_style_declarations(style: str, declarations: tuple[str, ...]) -> str:
    existing = style.strip()
    suffix = "; ".join(item.strip().rstrip(";") for item in declarations if item.strip())
    if not suffix:
        return existing
    if existing and not existing.endswith(";"):
        existing = f"{existing};"
    return f"{existing} {suffix};".strip() if existing else f"{suffix};"


def _cap_section(section_annotations: list[dict[str, object]]) -> dict[str, object] | None:
    for section in section_annotations:
        if str(section.get("section_kind", "")) == "cap":
            return section
    return None


def _component_palette(visual_contract: SequenceEvidenceMapV1) -> dict[str, str]:
    raw_palette = visual_contract.meta.get("component_palette")
    if not isinstance(raw_palette, dict):
        return {}
    return {str(key): str(value) for key, value in raw_palette.items() if str(key).strip() and str(value).strip()}


def _label_center(
    section_annotations: list[dict[str, object]],
    layout_normalization: dict[str, object],
) -> tuple[float, float]:
    if "center_x" in layout_normalization and "center_y" in layout_normalization:
        return float(layout_normalization["center_x"]), float(layout_normalization["center_y"])
    return _centroid([(float(section["anchor_x"]), float(section["anchor_y"])) for section in section_annotations])


def _place_label_without_collisions(
    anchor: tuple[float, float],
    *,
    center: tuple[float, float],
    label_width: float,
    label_height: float,
    occupied_boxes: list[tuple[float, float, float, float]],
    offset_distances: tuple[float, ...] = _DEFAULT_LABEL_OFFSETS,
    lateral_distances: tuple[float, ...] = _DEFAULT_LABEL_LATERALS,
) -> tuple[float, float, tuple[float, float, float, float], int]:
    dx = anchor[0] - center[0]
    dy = anchor[1] - center[1]
    distance = math.hypot(dx, dy)
    if distance < 1e-6:
        dx, dy = 0.0, -1.0
        distance = 1.0
    ux = dx / distance
    uy = dy / distance
    px = -uy
    py = ux
    best: tuple[int, float, float, tuple[float, float, float, float], int] | None = None
    attempts = 0
    for offset in offset_distances:
        for lateral in lateral_distances:
            label_x = anchor[0] + ux * offset + px * lateral
            label_y = anchor[1] + uy * offset + py * lateral
            box = (
                label_x - label_width / 2.0,
                label_y - label_height / 2.0,
                label_x + label_width / 2.0,
                label_y + label_height / 2.0,
            )
            overlap_count = sum(1 for existing in occupied_boxes if _boxes_overlap(box, existing))
            if best is None or overlap_count < best[0]:
                best = (overlap_count, label_x, label_y, box, attempts)
            if overlap_count == 0:
                return label_x, label_y, box, attempts
            attempts += 1
    assert best is not None
    return best[1], best[2], best[3], best[4]


def _nucleotide_obstacle_boxes(
    surface: SvgSurface,
    layout_normalization: dict[str, object],
) -> list[tuple[float, float, float, float]]:
    points = _display_points(surface, surface.nucleotide_nodes, layout_normalization)
    return [_padded_box((point[0], point[1], point[0], point[1]), pad=8.0) for point in points]


def _section_display_points(
    surface: SvgSurface,
    section: dict[str, object],
    layout_normalization: dict[str, object],
) -> list[tuple[float, float]]:
    start = int(section["start"])
    end = int(section["end"])
    return _display_points(surface, surface.nucleotide_nodes[start:end], layout_normalization)


def _display_points(
    surface: SvgSurface,
    nodes: tuple[ET.Element, ...],
    layout_normalization: dict[str, object],
) -> list[tuple[float, float]]:
    points = [_element_world_point(surface, node) for node in nodes]
    if not bool(layout_normalization.get("applied")):
        return points
    center = (
        float(layout_normalization.get("center_x", 0.0)),
        float(layout_normalization.get("center_y", 0.0)),
    )
    angle = float(layout_normalization.get("angle_degrees", 0.0))
    return [_rotate_point(point, center=center, angle_degrees=angle) for point in points]


def _points_bbox(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    if not points:
        return 0.0, 0.0, 0.0, 0.0
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _padded_box(box: tuple[float, float, float, float], *, pad: float) -> tuple[float, float, float, float]:
    return box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad


def _add_title_layer(
    root: ET.Element,
    *,
    visual_contract: SequenceEvidenceMapV1 | None,
    section_annotations: list[dict[str, object]],
    layout_normalization: dict[str, object],
    title_layout: dict[str, object] | None = None,
) -> list[tuple[float, float, float, float]]:
    if title_layout is None:
        title_layout = _title_layout(
            visual_contract=visual_contract,
            section_annotations=section_annotations,
            layout_normalization=layout_normalization,
        )
    if title_layout is None:
        return []
    title = str(title_layout["title"])
    subtitle_lines = [str(line) for line in title_layout.get("subtitle_lines", [])]
    x = float(title_layout["x"])
    title_y = float(title_layout["title_y"])
    layer = ET.Element(
        f"{{{SVG_NS}}}g",
        {
            "id": "dnadesign-secondary-structure-title",
            "data-dnadesign-title-layer": "structure_summary",
            "data-dnadesign-title-align": "content_center",
        },
    )
    title_node = ET.SubElement(
        layer,
        f"{{{SVG_NS}}}text",
        {
            "class": "dnadesign-structure-title",
            "x": f"{x:.3f}",
            "y": f"{title_y:.3f}",
            "text-anchor": "middle",
            "dominant-baseline": "middle",
            "style": ("font-family: DejaVu Sans, Arial, sans-serif; font-size: 11px; font-weight: 700; fill: #111827;"),
        },
    )
    title_node.text = title
    boxes = list(title_layout["boxes"])
    for index, subtitle in enumerate(subtitle_lines):
        subtitle_y = float(title_layout["subtitle_y"]) + index * float(title_layout["subtitle_line_height"])
        subtitle_node = ET.SubElement(
            layer,
            f"{{{SVG_NS}}}text",
            {
                "class": "dnadesign-structure-subtitle",
                "data-dnadesign-subtitle-line-index": str(index),
                "x": f"{x:.3f}",
                "y": f"{subtitle_y:.3f}",
                "text-anchor": "middle",
                "dominant-baseline": "middle",
                "style": (
                    "font-family: DejaVu Sans, Arial, sans-serif; "
                    f"font-size: {_format_px(_ANNOTATION_FONT_SIZE_PX)}; font-weight: 500; fill: #475569;"
                ),
            },
        )
        subtitle_node.text = subtitle
    root.append(layer)
    return boxes


def _title_layout(
    *,
    visual_contract: SequenceEvidenceMapV1 | None,
    section_annotations: list[dict[str, object]],
    layout_normalization: dict[str, object],
) -> dict[str, object] | None:
    if visual_contract is None:
        return None
    title = structure_title(visual_contract)
    if not title:
        return None
    geometry_bbox = layout_normalization.get("geometry_bbox")
    if isinstance(geometry_bbox, dict):
        try:
            left = float(geometry_bbox["min_x"])
            top = float(geometry_bbox["min_y"])
            right = float(geometry_bbox["max_x"])
        except (KeyError, TypeError, ValueError):
            left, top, right = 0.0, 0.0, 0.0
    else:
        left, top, right = 0.0, 0.0, 0.0
    x = (left + right) / 2.0
    subtitle_lines = structure_subtitle_lines(section_annotations, visual_contract)
    subtitle_line_height = 11.5
    title_y = top - (24.0 + len(subtitle_lines) * subtitle_line_height)
    subtitle_y = title_y + 13.0
    boxes: list[tuple[float, float, float, float]] = [_centered_text_box(x, title_y, max(60.0, len(title) * 6.2), 16.0)]
    for index, subtitle in enumerate(subtitle_lines):
        line_y = subtitle_y + index * subtitle_line_height
        boxes.append(_centered_text_box(x, line_y, max(80.0, len(subtitle) * 5.0), 13.0))
    return {
        "title": title,
        "subtitle_lines": subtitle_lines,
        "x": x,
        "title_y": title_y,
        "subtitle_y": subtitle_y,
        "subtitle_line_height": subtitle_line_height,
        "boxes": boxes,
    }


def _centered_text_box(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    return (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )


def _format_px(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}px"
    return f"{value:g}px"


def _wrap_svg_graphics_for_orientation(
    surface: SvgSurface,
    *,
    angle_degrees: float,
    center: tuple[float, float],
) -> bool:
    root = surface.tree.getroot()
    children_to_wrap: list[ET.Element] = []
    for child in list(root):
        local = _local_name(child.tag)
        if local in _GRAPHICS_WRAP_EXCLUDE:
            continue
        if local == "rect":
            continue
        if str(child.attrib.get("id", "")) == "dnadesign-secondary-structure-labels":
            continue
        children_to_wrap.append(child)
    if not children_to_wrap:
        return False
    wrapper = ET.Element(
        f"{{{SVG_NS}}}g",
        {
            "id": "dnadesign-viennarna-normalized-layout",
            "data-dnadesign-orientation": "cap_right",
            "transform": f"rotate({angle_degrees:.3f} {center[0]:.3f} {center[1]:.3f})",
        },
    )
    for child in children_to_wrap:
        root.remove(child)
        wrapper.append(child)
    root.append(wrapper)
    return True


def _element_world_point(surface: SvgSurface, element: ET.Element) -> tuple[float, float]:
    x = _float_attr(element, "x", fallback=0.0)
    y = _float_attr(element, "y", fallback=0.0)
    matrix = _identity_matrix()
    ancestors: list[ET.Element] = []
    current = element
    while current in surface.parent_map:
        current = surface.parent_map[current]
        ancestors.append(current)
    for ancestor in reversed(ancestors):
        matrix = _matrix_multiply(matrix, _transform_matrix(str(ancestor.attrib.get("transform", ""))))
    matrix = _matrix_multiply(matrix, _transform_matrix(str(element.attrib.get("transform", ""))))
    return _apply_matrix(matrix, x, y)


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return 0.0, 0.0
    return sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points)


def _bbox(points: list[tuple[float, float]]) -> dict[str, float]:
    if not points:
        return {"min_x": 0.0, "min_y": 0.0, "max_x": 0.0, "max_y": 0.0}
    return {
        "min_x": min(point[0] for point in points),
        "min_y": min(point[1] for point in points),
        "max_x": max(point[0] for point in points),
        "max_y": max(point[1] for point in points),
    }


def _expand_root_viewbox(
    root: ET.Element,
    *,
    layout_normalization: dict[str, object],
    label_boxes: list[tuple[float, float, float, float]],
) -> None:
    root.set("overflow", "visible")
    native_background_onclick = _remove_native_background_rects(root)
    width = _float_text(root.attrib.get("width"))
    height = _float_text(root.attrib.get("height"))
    viewbox = _root_viewbox(root, width=width, height=height)
    candidate_boxes: list[tuple[float, float, float, float]] = []
    geometry_box: tuple[float, float, float, float] | None = None
    geometry_bbox = layout_normalization.get("geometry_bbox")
    if isinstance(geometry_bbox, dict):
        try:
            geometry_box = (
                float(geometry_bbox["min_x"]),
                float(geometry_bbox["min_y"]),
                float(geometry_bbox["max_x"]),
                float(geometry_bbox["max_y"]),
            )
            candidate_boxes.append(geometry_box)
        except (KeyError, TypeError, ValueError):
            pass
    candidate_boxes.extend(label_boxes)
    if candidate_boxes:
        min_x = min(box[0] for box in candidate_boxes)
        min_y = min(box[1] for box in candidate_boxes)
        max_x = max(box[2] for box in candidate_boxes)
        max_y = max(box[3] for box in candidate_boxes)
    elif viewbox is not None:
        min_x, min_y, view_width, view_height = viewbox
        max_x = min_x + view_width
        max_y = min_y + view_height
    else:
        return
    if geometry_box is not None and geometry_box[2] > geometry_box[0]:
        geometry_margin = max(
            geometry_box[0] - min_x,
            max_x - geometry_box[2],
            0.0,
        )
        min_x = geometry_box[0] - geometry_margin
        max_x = geometry_box[2] + geometry_margin
    pad = 12.0
    new_width = max_x - min_x + 2 * pad
    new_height = max_y - min_y + 2 * pad
    root.set(
        "viewBox",
        f"{min_x - pad:.3f} {min_y - pad:.3f} {new_width:.3f} {new_height:.3f}",
    )
    root.set("width", f"{new_width:.0f}")
    root.set("height", f"{new_height:.0f}")
    _upsert_svg_background(
        root,
        x=min_x - pad,
        y=min_y - pad,
        width=new_width,
        height=new_height,
        onclick=native_background_onclick,
    )


def _remove_native_background_rects(root: ET.Element) -> str | None:
    onclick: str | None = None
    for element in list(root):
        if _local_name(element.tag) != "rect":
            continue
        if str(element.attrib.get("id", "")) == "dnadesign-secondary-structure-background":
            continue
        if not _is_native_background_rect(element):
            continue
        if onclick is None and "onclick" in element.attrib:
            onclick = str(element.attrib["onclick"])
        root.remove(element)
    return onclick


def _is_native_background_rect(element: ET.Element) -> bool:
    fill = str(element.attrib.get("fill", "")).strip().lower()
    style = re.sub(r"\s+", "", str(element.attrib.get("style", "")).lower())
    return fill in {"white", "#fff", "#ffffff"} or "fill:white" in style or "fill:#fff" in style


def _upsert_svg_background(
    root: ET.Element,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    onclick: str | None,
) -> None:
    background_id = "dnadesign-secondary-structure-background"
    background = next((element for element in root if str(element.attrib.get("id", "")) == background_id), None)
    if background is None:
        background = ET.Element(
            f"{{{SVG_NS}}}rect",
            {
                "id": background_id,
                "data-dnadesign-background-layer": "annotated_viewbox",
            },
        )
        root.insert(0, background)
    background.set("x", f"{x:.3f}")
    background.set("y", f"{y:.3f}")
    background.set("width", f"{width:.3f}")
    background.set("height", f"{height:.3f}")
    background.set("style", "fill: #FFFFFF; stroke: none;")
    if onclick:
        background.set("onclick", onclick)
    elif "onclick" in background.attrib:
        del background.attrib["onclick"]


def _root_viewbox(
    root: ET.Element,
    *,
    width: float | None,
    height: float | None,
) -> tuple[float, float, float, float] | None:
    raw = str(root.attrib.get("viewBox", "")).strip()
    if raw:
        parts = _parse_transform_values(raw)
        if len(parts) == 4:
            return parts[0], parts[1], parts[2], parts[3]
    if width is None or height is None:
        return None
    return 0.0, 0.0, width, height


def _float_text(raw: object) -> float | None:
    if raw is None:
        return None
    match = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)", str(raw))
    if match is None:
        return None
    return float(match.group(1))


def _float_attr(element: ET.Element, key: str, *, fallback: float) -> float:
    try:
        return float(str(element.attrib.get(key, fallback)))
    except ValueError:
        return fallback


def _identity_matrix() -> tuple[float, float, float, float, float, float]:
    return 1.0, 0.0, 0.0, 1.0, 0.0, 0.0


def _matrix_multiply(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    a1, b1, c1, d1, e1, f1 = left
    a2, b2, c2, d2, e2, f2 = right
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _apply_matrix(
    matrix: tuple[float, float, float, float, float, float],
    x: float,
    y: float,
) -> tuple[float, float]:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _transform_matrix(transform: str) -> tuple[float, float, float, float, float, float]:
    matrix = _identity_matrix()
    for match in _TRANSFORM_RE.finditer(transform):
        name = match.group(1).lower()
        values = _parse_transform_values(match.group(2))
        transform_matrix = _identity_matrix()
        if name == "matrix" and len(values) >= 6:
            transform_matrix = tuple(values[:6])  # type: ignore[assignment]
        elif name == "translate" and values:
            tx = values[0]
            ty = values[1] if len(values) > 1 else 0.0
            transform_matrix = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale" and values:
            sx = values[0]
            sy = values[1] if len(values) > 1 else sx
            transform_matrix = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "rotate" and values:
            angle = math.radians(values[0])
            cos_v = math.cos(angle)
            sin_v = math.sin(angle)
            rotate_matrix = (cos_v, sin_v, -sin_v, cos_v, 0.0, 0.0)
            if len(values) >= 3:
                cx, cy = values[1], values[2]
                transform_matrix = _matrix_multiply(
                    _matrix_multiply((1.0, 0.0, 0.0, 1.0, cx, cy), rotate_matrix),
                    (1.0, 0.0, 0.0, 1.0, -cx, -cy),
                )
            else:
                transform_matrix = rotate_matrix
        matrix = _matrix_multiply(matrix, transform_matrix)
    return matrix


def _parse_transform_values(raw: str) -> list[float]:
    values: list[float] = []
    for part in re.split(r"[\s,]+", raw.strip()):
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            continue
    return values


def _rotate_point(
    point: tuple[float, float],
    *,
    center: tuple[float, float],
    angle_degrees: float,
) -> tuple[float, float]:
    angle = math.radians(angle_degrees)
    cos_v = math.cos(angle)
    sin_v = math.sin(angle)
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return center[0] + dx * cos_v - dy * sin_v, center[1] + dx * sin_v + dy * cos_v


def _boxes_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return not (left[2] <= right[0] or right[2] <= left[0] or left[3] <= right[1] or right[3] <= left[1])


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", maxsplit=1)[1]
    return tag


__all__ = ["SVG_NS", "SvgAnnotationResult", "SvgSurface", "annotate_svg_surface", "load_svg_surface"]
