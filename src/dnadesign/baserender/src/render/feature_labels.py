"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/feature_labels.py

Plan producer-neutral feature annotations before allocating drawing resources.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import matplotlib.colors as mcolors

from ..config import Style
from ..core import Feature, LayoutError, Record, RenderingError
from .effects.anchored_illustration import anchored_illustration_occupied_boxes
from .effects.span_link import SpanLinkGeometry, compute_span_link_geometry
from .layout import LayoutContext, measure_annotation_label, measure_text_width_px
from .palette import Palette, darken_rgb


@dataclass(frozen=True)
class PlannedLabel:
    text: str
    x: float
    y: float
    horizontal_alignment: str
    size_pt: float
    color: str | tuple[float, float, float]
    bounds: tuple[float, float, float, float]


def plan_feature_labels(
    record: Record,
    layout: LayoutContext,
    palette: Palette,
    style: Style,
    *,
    reserved_boxes: Sequence[tuple[float, float, float, float]] = (),
    span_link_geometries: Mapping[int, SpanLinkGeometry] | None = None,
) -> tuple[PlannedLabel, ...]:
    """Resolve required labels and collisions before any figure is allocated."""
    span_boxes = _span_link_label_boxes(record, layout, style, geometries=span_link_geometries)
    reserved_boxes = (*reserved_boxes, *span_boxes)
    near = _plan_near_feature_labels(record, layout, palette, style, reserved_boxes=reserved_boxes)
    if style.legend and str(style.legend_mode).lower() == "inline":
        inline = _plan_inline_feature_labels(
            record,
            layout,
            palette,
            style,
            reserved_boxes=(*reserved_boxes, *(label.bounds for label in near)),
        )
    else:
        inline = ()
    return (*near, *inline)


def draw_feature_labels(ax, labels: tuple[PlannedLabel, ...], style: Style) -> None:
    for label in labels:
        ax.text(
            label.x,
            label.y,
            label.text,
            ha=label.horizontal_alignment,
            va="center",
            fontsize=label.size_pt,
            family=style.font_label,
            color=label.color,
            zorder=6.2,
            clip_on=False,
        )


def uses_near_feature_annotation_label(feature: Feature) -> bool:
    annotation = feature.render.get("annotation_label")
    return annotation is not None and annotation["placement"] == "near_feature"


def _capitalize_first(text: str) -> str:
    t = text.strip()
    if not t:
        return t
    return t[0].upper() + t[1:]


def boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return max(ax0, bx0) < min(ax1, bx1) and max(ay0, by0) < min(ay1, by1)


def _span_link_label_boxes(
    record: Record,
    layout: LayoutContext,
    style: Style,
    *,
    geometries: Mapping[int, SpanLinkGeometry] | None = None,
) -> list[tuple[float, float, float, float]]:
    if geometries is None:
        geometries = {
            index: compute_span_link_geometry(effect, layout, style, layout.feature_boxes)
            for index, effect in enumerate(record.effects)
            if effect.kind == "span_link"
        }
    return [box for geometry in geometries.values() for box in geometry.occupied_boxes]


def _sequence_strand_occupied_boxes(layout: LayoutContext, style: Style) -> list[tuple[float, float, float, float]]:
    pad = max(3.0, float(style.font_size_seq) / 72.0 * float(style.dpi) * 0.12)
    x0 = float(style.padding_x)
    x1 = float(layout.width - style.padding_x)
    boxes = [
        (
            x0,
            float(layout.y_forward) - float(layout.sequence_extent_down) - pad,
            x1,
            float(layout.y_forward) + float(layout.sequence_extent_up) + pad,
        )
    ]
    if abs(float(layout.y_reverse) - float(layout.y_forward)) > 1.0e-6:
        boxes.append(
            (
                x0,
                float(layout.y_reverse) - float(layout.sequence_extent_down) - pad,
                x1,
                float(layout.y_reverse) + float(layout.sequence_extent_up) + pad,
            )
        )
    return boxes


def _plan_near_feature_labels(
    record: Record, layout: LayoutContext, palette: Palette, style: Style, *, reserved_boxes=()
) -> tuple[PlannedLabel, ...]:
    if not layout.placements:
        return ()

    margin = max(float(style.legend_inline_margin_cells) * layout.cw, float(style.kmer.pad_x_px) + 4.0)
    x_min = float(style.padding_x)
    x_max = float(layout.width - style.padding_x)

    feature_box_pad = float(style.kmer.pad_x_px)
    coordinate_scale = max(1.0e-6, float(style.figure_scale))
    annotation_font_height_px = measure_annotation_label("", style).height_px * coordinate_scale
    feature_label_clearance = max(
        max(0.0, margin - feature_box_pad),
        max(8.0, annotation_font_height_px * 0.2) / coordinate_scale,
    )
    occupied_boxes: list[tuple[float, float, float, float]] = []
    for placement in layout.placements:
        occupied_boxes.append(
            (
                placement.x - feature_box_pad - feature_label_clearance,
                placement.y - placement.h / 2.0,
                placement.x + placement.w + feature_box_pad + feature_label_clearance,
                placement.y + placement.h / 2.0,
            )
        )
    occupied_boxes.extend(_sequence_strand_occupied_boxes(layout, style))
    for effect in record.effects:
        if effect.kind == "anchored_illustration":
            occupied_boxes.extend(
                anchored_illustration_occupied_boxes(
                    effect,
                    record,
                    layout,
                    style,
                    layout.feature_boxes,
                )
            )
    occupied_boxes.extend(reserved_boxes)
    planned: list[PlannedLabel] = []
    placed_label_boxes: list[tuple[float, float, float, float]] = []

    def _candidate_box(
        *,
        x_anchor: float,
        y_anchor: float,
        ha: str,
        text_w: float,
        text_h: float,
    ) -> tuple[float, float, float, float]:
        if ha == "left":
            x0 = float(x_anchor)
            x1 = float(x_anchor) + float(text_w)
        elif ha == "right":
            x0 = float(x_anchor) - float(text_w)
            x1 = float(x_anchor)
        else:
            x0 = float(x_anchor) - float(text_w) / 2.0
            x1 = float(x_anchor) + float(text_w) / 2.0
        y0 = float(y_anchor) - float(text_h) / 2.0
        y1 = float(y_anchor) + float(text_h) / 2.0
        return (x0, y0, x1, y1)

    placement_by_id = {placement.feature_id: placement for placement in layout.placements}
    force_baseline_label_ids: set[str] = set()

    def _annotation_label_geometry(
        placement,
    ) -> tuple[str, float, float, float, float, float, float, float] | None:
        feature = record.features[placement.feature_index]
        if not uses_near_feature_annotation_label(feature):
            return None
        text = feature.render["annotation_label"]["text"]
        metrics = measure_annotation_label(text, style)
        text_size, text_w, text_h = metrics.size_pt, metrics.width_px, metrics.height_px
        center_x = placement.x + placement.w / 2.0
        label_gap = metrics.gap_px
        if placement.above:
            away_y = placement.y + placement.h / 2.0 + label_gap + text_h / 2.0
        else:
            away_y = placement.y - placement.h / 2.0 - label_gap - text_h / 2.0
        right_x = placement.x + placement.w + feature_box_pad + feature_label_clearance
        left_x = placement.x - feature_box_pad - feature_label_clearance
        return text, text_size, text_w, text_h, center_x, away_y, right_x, left_x

    def _candidate_fits(
        bbox: tuple[float, float, float, float],
        *,
        occupied: Sequence[tuple[float, float, float, float]],
    ) -> bool:
        if bbox[0] < x_min or bbox[2] > x_max:
            return False
        if bbox[1] < 0.0 or bbox[3] > float(layout.height):
            return False
        return not any(boxes_overlap(bbox, occupied_box) for occupied_box in occupied)

    for effect in record.effects:
        if effect.kind != "span_link":
            continue
        from_id = effect.target.get("from_feature_id")
        to_id = effect.target.get("to_feature_id")
        if not isinstance(from_id, str) or not isinstance(to_id, str):
            continue
        from_placement = placement_by_id.get(from_id)
        to_placement = placement_by_id.get(to_id)
        if from_placement is None or to_placement is None:
            continue
        endpoint_bboxes: list[tuple[float, float, float, float]] = []
        can_place_both_above = True
        for endpoint in (from_placement, to_placement):
            geometry = _annotation_label_geometry(endpoint)
            if geometry is None:
                can_place_both_above = False
                break
            _text, _text_size, text_w, text_h, center_x, top_y, _right_x, _left_x = geometry
            bbox = _candidate_box(
                x_anchor=center_x,
                y_anchor=top_y,
                ha="center",
                text_w=text_w,
                text_h=text_h,
            )
            if not _candidate_fits(bbox, occupied=occupied_boxes):
                can_place_both_above = False
                break
            if any(boxes_overlap(bbox, existing) for existing in endpoint_bboxes):
                can_place_both_above = False
                break
            endpoint_bboxes.append(bbox)
        if not can_place_both_above:
            force_baseline_label_ids.update({from_id, to_id})

    for placement in layout.placements:
        feature = record.features[placement.feature_index]
        if not uses_near_feature_annotation_label(feature):
            continue

        tag = feature.tags[0] if feature.tags else feature.kind
        geometry = _annotation_label_geometry(placement)
        if geometry is None:
            continue
        text, text_size, text_w, text_h, center_x, top_y, right_x, left_x = geometry

        lateral_step = max(8.0, feature_box_pad + text_h * 0.75)
        max_lateral_steps = 6
        side_candidates: list[tuple[float, float, str]] = []
        top_candidates: list[tuple[float, float, str]] = [(center_x, top_y, "center")]
        for step in range(max_lateral_steps + 1):
            delta = lateral_step * float(step)
            side_candidates.extend(
                (
                    (right_x + delta, placement.y, "left"),
                    (left_x - delta, placement.y, "right"),
                )
            )
            if step > 0:
                top_candidates.extend(
                    (
                        (center_x + delta, top_y, "center"),
                        (center_x - delta, top_y, "center"),
                    )
                )
        if placement.feature_id in force_baseline_label_ids:
            candidates = (*side_candidates, *top_candidates)
        else:
            candidates = (*top_candidates, *side_candidates)

        selected: tuple[float, float, str, tuple[float, float, float, float]] | None = None
        for x_anchor, y_anchor, ha in candidates:
            bbox = _candidate_box(x_anchor=x_anchor, y_anchor=y_anchor, ha=ha, text_w=text_w, text_h=text_h)
            if bbox[0] < x_min or bbox[2] > x_max:
                continue
            if bbox[1] < 0.0 or bbox[3] > float(layout.height):
                continue
            if any(boxes_overlap(bbox, occupied) for occupied in occupied_boxes):
                continue
            if any(boxes_overlap(bbox, occupied) for occupied in placed_label_boxes):
                continue
            selected = (x_anchor, y_anchor, ha, bbox)
            break

        if selected is None:
            raise LayoutError(
                f"Feature {feature.id!r} annotation cannot fit in the {layout.width:g} x {layout.height:g} px canvas; "
                "increase the canvas or label clearance, or revise the declared annotation"
            )

        x_anchor, y_anchor, ha, bbox = selected
        annotation_color_raw = feature.attrs.get("annotation_color")
        if annotation_color_raw is None:
            annotation_color = darken_rgb(palette.color_for(tag), factor=0.6)
        else:
            annotation_color = str(annotation_color_raw)
            if not mcolors.is_color_like(annotation_color):
                raise RenderingError(f"Feature {feature.id!r} annotation_color must be a valid color")
        planned.append(PlannedLabel(text, x_anchor, y_anchor, ha, text_size, annotation_color, bbox))
        placed_label_boxes.append(bbox)

    return tuple(planned)


def _plan_inline_feature_labels(
    record: Record, layout: LayoutContext, palette: Palette, style: Style, *, reserved_boxes=()
) -> tuple[PlannedLabel, ...]:
    if not layout.placements:
        return ()

    labels = dict(record.display.tag_labels)
    side_pref = str(style.legend_inline_side).lower()
    margin = float(style.legend_inline_margin_cells) * layout.cw
    box_pad = float(style.kmer.pad_x_px)
    x_min = style.padding_x
    x_max = layout.width - style.padding_x

    text_h = max(8.0, (float(style.legend_font_size) / 72.0) * float(style.dpi))
    lateral_step = max(8.0, box_pad + text_h * 0.75)
    max_lateral_steps = max(4, int(math.ceil((x_max - x_min) / max(1.0, lateral_step))))

    occupied_boxes: list[tuple[float, float, float, float]] = []
    for placement in layout.placements:
        occupied_boxes.append(
            (
                placement.x - box_pad,
                placement.y - placement.h / 2.0,
                placement.x + placement.w + box_pad,
                placement.y + placement.h / 2.0,
            )
        )
    sequence_x0 = float(layout.x_left)
    sequence_x1 = float(layout.x_left + len(record.sequence) * layout.cw)
    occupied_boxes.append(
        (
            sequence_x0,
            float(layout.y_forward - layout.sequence_extent_down),
            sequence_x1,
            float(layout.y_forward + layout.sequence_extent_up),
        )
    )
    if bool(style.show_reverse_complement and record.alphabet in {"DNA", "IUPAC_DNA"}):
        occupied_boxes.append(
            (
                sequence_x0,
                float(layout.y_reverse - layout.sequence_extent_down),
                sequence_x1,
                float(layout.y_reverse + layout.sequence_extent_up),
            )
        )

    for effect_index, effect in enumerate(record.effects):
        if effect.kind != "motif_logo":
            continue
        feature_id_raw = effect.target.get("feature_id")
        if not isinstance(feature_id_raw, str) or not feature_id_raw:
            continue
        feature_box = layout.feature_boxes.get(feature_id_raw)
        if feature_box is None:
            continue
        motif_y0 = layout.motif_logo_y0_by_effect.get(effect_index)
        if motif_y0 is None:
            continue
        occupied_boxes.append(
            (
                float(feature_box[0]),
                float(motif_y0),
                float(feature_box[2]),
                float(motif_y0 + layout.motif_logo_height),
            )
        )

    occupied_boxes.extend(reserved_boxes)
    planned: list[PlannedLabel] = []
    placed_label_boxes: list[tuple[float, float, float, float]] = []

    def _candidate_box(
        *,
        x_anchor: float,
        y_anchor: float,
        ha: str,
        text_w: float,
    ) -> tuple[float, float, float, float]:
        if ha == "left":
            x0 = float(x_anchor)
            x1 = float(x_anchor) + float(text_w)
        elif ha == "right":
            x0 = float(x_anchor) - float(text_w)
            x1 = float(x_anchor)
        else:
            x0 = float(x_anchor) - float(text_w) / 2.0
            x1 = float(x_anchor) + float(text_w) / 2.0
        y0 = float(y_anchor) - text_h / 2.0
        y1 = float(y_anchor) + text_h / 2.0
        return (x0, y0, x1, y1)

    for placement in layout.placements:
        feature = record.features[placement.feature_index]
        if uses_near_feature_annotation_label(feature):
            continue
        tag = feature.tags[0] if feature.tags else feature.kind
        fallback = tag.split(":")[-1] if ":" in tag else tag
        annotation = feature.render.get("annotation_label")
        raw_label = annotation["text"] if annotation is not None else str(labels.get(tag, fallback))
        text = raw_label if annotation is not None else _capitalize_first(raw_label)
        if not text:
            continue

        text_w = measure_text_width_px(text, style.font_label, style.legend_font_size, style.dpi)
        x_left = placement.x - margin
        x_right = placement.x + placement.w + margin
        left_room = (x_left - text_w) - x_min
        right_room = x_max - (x_right + text_w)

        if side_pref == "left":
            side_order = ("left", "right")
        elif side_pref == "right":
            side_order = ("right", "left")
        else:
            side_order = ("right", "left") if right_room >= left_room else ("left", "right")

        candidates: list[tuple[float, float, str]] = []
        for step in range(0, max_lateral_steps + 1):
            delta = lateral_step * float(step)
            for side in side_order:
                if side == "right":
                    candidates.append((x_right + delta, placement.y, "left"))
                else:
                    candidates.append((x_left - delta, placement.y, "right"))

        selected: tuple[float, float, str, tuple[float, float, float, float]] | None = None
        for x_anchor, y_anchor, ha in candidates:
            bbox = _candidate_box(x_anchor=x_anchor, y_anchor=y_anchor, ha=ha, text_w=text_w)
            if bbox[0] < x_min or bbox[2] > x_max:
                continue
            if bbox[1] < 0.0 or bbox[3] > float(layout.height):
                continue
            if any(boxes_overlap(bbox, occupied) for occupied in occupied_boxes):
                continue
            if any(boxes_overlap(bbox, occupied) for occupied in placed_label_boxes):
                continue
            selected = (x_anchor, y_anchor, ha, bbox)
            break
        if selected is None:
            if annotation is not None:
                raise LayoutError(
                    f"Feature {feature.id!r} inline annotation cannot fit in the "
                    f"{layout.width:g} x {layout.height:g} px canvas"
                )
            continue
        x_text, y_text, ha, bbox = selected

        planned.append(PlannedLabel(text, x_text, y_text, ha, style.legend_font_size, palette.color_for(tag), bbox))
        placed_label_boxes.append(bbox)

    return tuple(planned)
