"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/layout.py

Layout math and generic track assignment for sequence-row rendering.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from ..config import Style
from ..core import BoundsError, Effect, Feature, Record

_DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def comp(seq: str) -> str:
    return seq.translate(_DNA_COMP)


def revcomp(seq: str) -> str:
    return comp(seq)[::-1]


def grid_x_left(layout: LayoutContext, col_idx: int) -> float:
    return layout.grid_x_left(col_idx)


def grid_x_right(layout: LayoutContext, col_idx: int) -> float:
    return layout.grid_x_right(col_idx)


def span_to_x(layout: LayoutContext, start: int, end: int) -> tuple[float, float]:
    return layout.span_to_x(start, end)


@dataclass(frozen=True)
class CharCell:
    width: float
    height: float


@dataclass(frozen=True)
class FeaturePlacement:
    feature_index: int
    feature_id: str
    track: int
    above: bool
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class LayoutContext:
    cw: float
    ch: float
    width: float
    height: float
    y_forward: float
    y_reverse: float
    x_left: float
    n_up_tracks: int
    n_dn_tracks: int
    placements: tuple[FeaturePlacement, ...]
    feature_boxes: Mapping[str, tuple[float, float, float, float]]
    motif_logo_height: float
    motif_logo_gap: float
    motif_logo_lane_by_effect: Mapping[int, int]
    motif_logo_above_by_effect: Mapping[int, bool]
    motif_logo_lanes_up: int
    motif_logo_lanes_down: int

    def grid_x_left(self, col_idx: int) -> float:
        return self.x_left + col_idx * self.cw

    def grid_x_right(self, col_idx: int) -> float:
        return self.x_left + (col_idx + 1) * self.cw

    def span_to_x(self, start: int, end: int) -> tuple[float, float]:
        return self.grid_x_left(start), self.x_left + end * self.cw


@dataclass(frozen=True)
class _IntervalItem:
    index: int
    priority: int
    start: int
    end: int
    fixed_lane: int | None = None


@lru_cache(maxsize=64)
def measure_char_cell(font_family: str, font_size: int, dpi: int) -> CharCell:
    prop = FontProperties(family=font_family, size=font_size)
    run = "M" * 64
    bbox_w = TextPath((0, 0), run, prop=prop).get_extents()
    cw_pt = bbox_w.width / 64
    ch_pt = TextPath((0, 0), "Ag", prop=prop).get_extents().height
    return CharCell(width=cw_pt / 72.0 * dpi, height=ch_pt / 72.0 * dpi)


def _feature_priority(feature: Feature) -> int:
    raw = feature.render.get("priority", 10)
    try:
        return int(raw)
    except Exception:
        return 10


def _feature_track(feature: Feature) -> int | None:
    raw = feature.render.get("track")
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception as exc:
        raise BoundsError(f"feature.render.track must be int (feature id={feature.id})") from exc


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return not (a_end <= b_start or b_end <= a_start)


def _assign_interval_lanes(items: Sequence[_IntervalItem], *, ctx: str) -> dict[int, int]:
    lanes: dict[int, int] = {}
    per_lane_spans: dict[int, list[tuple[int, int, int]]] = {}

    for item in items:
        if item.fixed_lane is None:
            continue
        spans = per_lane_spans.setdefault(item.fixed_lane, [])
        for start, end, other_idx in spans:
            if _overlap(item.start, item.end, start, end):
                msg = (
                    "Fixed-track overlap detected: "
                    f"{ctx}[{item.index}] overlaps {ctx}[{other_idx}] on track {item.fixed_lane}"
                )
                raise BoundsError(msg)
        spans.append((item.start, item.end, item.index))
        lanes[item.index] = item.fixed_lane

    pending = sorted(
        [item for item in items if item.index not in lanes],
        key=lambda item: (item.priority, item.start, -(item.end - item.start), item.index),
    )
    for item in pending:
        placed = False
        for lane, spans in sorted(per_lane_spans.items(), key=lambda kv: kv[0]):
            if all(not _overlap(item.start, item.end, s, e) for s, e, _ in spans):
                spans.append((item.start, item.end, item.index))
                lanes[item.index] = lane
                placed = True
                break
        if placed:
            continue
        next_lane = 0
        while next_lane in per_lane_spans:
            next_lane += 1
        per_lane_spans[next_lane] = [(item.start, item.end, item.index)]
        lanes[item.index] = next_lane
    return lanes


def assign_tracks(features: Sequence[Feature]) -> list[int]:
    lane_map = _assign_interval_lanes(
        [
            _IntervalItem(
                index=idx,
                priority=_feature_priority(feat),
                start=feat.span.start,
                end=feat.span.end,
                fixed_lane=_feature_track(feat),
            )
            for idx, feat in enumerate(features)
        ],
        ctx="feature",
    )
    return [lane_map[idx] for idx in range(len(features))]


def _effect_priority(effect: Effect) -> int:
    raw = effect.render.get("priority", 10)
    try:
        return int(raw)
    except Exception as exc:
        raise BoundsError("motif_logo effect.render.priority must be int") from exc


def _effect_track(effect: Effect) -> int | None:
    raw = effect.render.get("track")
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception as exc:
        raise BoundsError("motif_logo effect.render.track must be int") from exc


def _resolve_motif_logo_feature(
    record: Record,
    effect: Effect,
    *,
    effect_index: int,
) -> Feature:
    feature_id_raw = effect.target.get("feature_id")
    if not isinstance(feature_id_raw, str) or feature_id_raw.strip() == "":
        raise BoundsError(f"motif_logo effect[{effect_index}] target.feature_id must be a non-empty string")
    feature = next((f for f in record.features if f.id == feature_id_raw), None)
    if feature is None:
        raise BoundsError(f"motif_logo effect[{effect_index}] references unknown feature_id '{feature_id_raw}'")
    return feature


def assign_motif_logo_lanes(
    record: Record,
    *,
    layout_mode: str,
) -> tuple[dict[int, int], dict[int, bool], int, int]:
    mode = str(layout_mode).lower()
    if mode not in {"stack", "overlay"}:
        raise BoundsError(f"motif_logo layout must be 'stack' or 'overlay', got {layout_mode!r}")

    up_items: list[_IntervalItem] = []
    dn_items: list[_IntervalItem] = []
    above_by_effect: dict[int, bool] = {}

    for effect_index, effect in enumerate(record.effects):
        if effect.kind != "motif_logo":
            continue
        feature = _resolve_motif_logo_feature(record, effect, effect_index=effect_index)
        above = feature.span.strand != "rev"
        above_by_effect[effect_index] = above
        interval = _IntervalItem(
            index=effect_index,
            priority=_effect_priority(effect),
            start=feature.span.start,
            end=feature.span.end,
            fixed_lane=_effect_track(effect),
        )
        if above:
            up_items.append(interval)
        else:
            dn_items.append(interval)

    if mode == "overlay":
        lane_by_effect = {effect_index: 0 for effect_index in above_by_effect}
        up_count = 1 if any(above_by_effect.values()) else 0
        dn_count = 1 if any(not above for above in above_by_effect.values()) else 0
        return lane_by_effect, above_by_effect, up_count, dn_count

    up_lanes = _assign_interval_lanes(up_items, ctx="motif_logo.effect")
    dn_lanes = _assign_interval_lanes(dn_items, ctx="motif_logo.effect")
    lane_by_effect = {**up_lanes, **dn_lanes}
    up_count = (max(up_lanes.values()) + 1) if up_lanes else 0
    dn_count = (max(dn_lanes.values()) + 1) if dn_lanes else 0
    return lane_by_effect, above_by_effect, up_count, dn_count


def compute_layout(record: Record, style: Style, *, fixed_n: int | None = None) -> LayoutContext:
    cell = measure_char_cell(style.font_mono, style.font_size_seq, style.dpi)
    cw, ch = cell.width, cell.height
    h = ch * style.kmer.height_factor
    n = fixed_n if fixed_n is not None else len(record.sequence)

    label_pad_x = style.font_size_label / 72.0 * style.dpi * 1.6
    x_left = style.padding_x + label_pad_x

    up_indices = [i for i, feat in enumerate(record.features) if feat.span.strand == "fwd"]
    dn_indices = [i for i, feat in enumerate(record.features) if feat.span.strand == "rev"]

    up_features = [record.features[i] for i in up_indices]
    dn_features = [record.features[i] for i in dn_indices]
    up_tracks = assign_tracks(up_features)
    dn_tracks = assign_tracks(dn_features)
    motif_lane_by_effect, motif_above_by_effect, motif_lanes_up, motif_lanes_down = assign_motif_logo_lanes(
        record,
        layout_mode=style.motif_logo.layout,
    )

    n_up_tracks = (max(up_tracks) + 1) if up_tracks else 0
    n_dn_tracks = (max(dn_tracks) + 1) if dn_tracks else 0

    y_forward = style.padding_y + n_up_tracks * style.track_spacing + ch
    y_reverse = (
        y_forward - style.baseline_spacing
        if (style.show_reverse_complement and record.alphabet == "DNA")
        else y_forward
    )

    label_pad_y = style.font_size_label / 72.0 * style.dpi * 1.2
    legend_space = (style.legend_height_px + style.legend_pad_px) if style.legend else 0.0

    top = y_forward + n_up_tracks * style.track_spacing + h + label_pad_y
    bottom = y_reverse - n_dn_tracks * style.track_spacing - h - label_pad_y
    motif_logo_height = style.motif_logo.height_cells * ch
    motif_logo_gap = style.motif_logo.y_pad_cells * ch
    if motif_lanes_up > 0:
        top += motif_lanes_up * (motif_logo_height + motif_logo_gap)
    if motif_lanes_down > 0:
        bottom -= motif_lanes_down * (motif_logo_height + motif_logo_gap)
    margin = max(2.0, 0.5 * style.kmer.round_px)
    top += margin
    bottom -= margin

    if legend_space > 0 and bottom < legend_space:
        delta = legend_space - bottom
        y_forward += delta
        y_reverse += delta
        top += delta
        bottom += delta

    content_height = top - bottom + style.padding_y
    height = content_height + legend_space
    width = x_left + n * cw + style.padding_x + label_pad_x

    placements: list[FeaturePlacement] = []
    feature_boxes: dict[str, tuple[float, float, float, float]] = {}

    feature_ids = [feat.id if feat.id is not None else f"f{idx}" for idx, feat in enumerate(record.features)]

    for local_idx, feat_idx in enumerate(up_indices):
        feat = record.features[feat_idx]
        track = up_tracks[local_idx]
        x = x_left + feat.span.start * cw
        x_end = x_left + feat.span.end * cw
        y = y_forward + (track + 1) * style.track_spacing
        w = x_end - x
        placement = FeaturePlacement(
            feature_index=feat_idx,
            feature_id=feature_ids[feat_idx],
            track=track,
            above=True,
            x=x,
            y=y,
            w=w,
            h=h,
        )
        placements.append(placement)
        feature_boxes[placement.feature_id] = (x, y - h / 2.0, x + w, y + h / 2.0)

    for local_idx, feat_idx in enumerate(dn_indices):
        feat = record.features[feat_idx]
        track = dn_tracks[local_idx]
        x = x_left + feat.span.start * cw
        x_end = x_left + feat.span.end * cw
        y = y_reverse - (track + 1) * style.track_spacing
        w = x_end - x
        placement = FeaturePlacement(
            feature_index=feat_idx,
            feature_id=feature_ids[feat_idx],
            track=track,
            above=False,
            x=x,
            y=y,
            w=w,
            h=h,
        )
        placements.append(placement)
        feature_boxes[placement.feature_id] = (x, y - h / 2.0, x + w, y + h / 2.0)

    return LayoutContext(
        cw=cw,
        ch=ch,
        width=width,
        height=height,
        y_forward=y_forward,
        y_reverse=y_reverse,
        x_left=x_left,
        n_up_tracks=n_up_tracks,
        n_dn_tracks=n_dn_tracks,
        placements=tuple(placements),
        feature_boxes=feature_boxes,
        motif_logo_height=motif_logo_height,
        motif_logo_gap=motif_logo_gap,
        motif_logo_lane_by_effect=motif_lane_by_effect,
        motif_logo_above_by_effect=motif_above_by_effect,
        motif_logo_lanes_up=motif_lanes_up,
        motif_logo_lanes_down=motif_lanes_down,
    )
