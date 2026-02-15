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
    sequence_half_height: float
    sequence_extent_up: float
    sequence_extent_down: float
    x_left: float
    n_up_tracks: int
    n_dn_tracks: int
    kmer_box_height: float
    feature_track_step: float
    feature_track_base_offset: float
    feature_track_base_offset_up: float
    feature_track_base_offset_down: float
    placements: tuple[FeaturePlacement, ...]
    feature_boxes: Mapping[str, tuple[float, float, float, float]]
    feature_track_by_id: Mapping[str, int]
    feature_above_by_id: Mapping[str, bool]
    motif_logo_height: float
    motif_logo_gap: float
    motif_logo_lane_by_effect: Mapping[int, int]
    motif_logo_above_by_effect: Mapping[int, bool]
    motif_logo_y0_by_effect: Mapping[int, float]
    motif_logo_lanes_up: int
    motif_logo_lanes_down: int
    content_top: float
    content_bottom: float

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


@lru_cache(maxsize=64)
def measure_sequence_extents(font_family: str, font_size: int, dpi: int) -> tuple[float, float]:
    prop = FontProperties(family=font_family, size=font_size)
    px_per_pt = dpi / 72.0
    ag_bbox = TextPath((0, 0), "Ag", prop=prop).get_extents()
    y_mid = ((ag_bbox.y0 + ag_bbox.y1) / 2.0) * px_per_pt
    max_up = 0.0
    max_down = 0.0
    for base in ("A", "C", "G", "T", "N"):
        bbox = TextPath((0, 0), base, prop=prop).get_extents()
        y0 = bbox.y0 * px_per_pt
        y1 = bbox.y1 * px_per_pt
        max_up = max(max_up, y1 - y_mid)
        max_down = max(max_down, y_mid - y0)
    max_half = max(max_up, max_down)
    if max_half <= 0:
        raise BoundsError("Failed to measure sequence glyph half-height")
    return max_up, max_down


@lru_cache(maxsize=64)
def measure_sequence_half_height(font_family: str, font_size: int, dpi: int) -> float:
    max_up, max_down = measure_sequence_extents(font_family, font_size, dpi)
    return max(max_up, max_down)


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


def _feature_index_for_id(record: Record, feature_id: str) -> int:
    for idx, feature in enumerate(record.features):
        if feature.id == feature_id:
            return idx
    raise BoundsError(f"Unknown feature_id '{feature_id}'")


def assign_motif_logo_lanes(
    record: Record,
    *,
    layout_mode: str,
    lane_mode: str,
    feature_track_by_index: Mapping[int, int],
    feature_above_by_index: Mapping[int, bool],
) -> tuple[dict[int, int], dict[int, bool], int, int]:
    mode = str(layout_mode).lower()
    if mode not in {"stack", "overlay"}:
        raise BoundsError(f"motif_logo layout must be 'stack' or 'overlay', got {layout_mode!r}")

    lane_mode_norm = str(lane_mode).lower()
    if lane_mode_norm not in {"follow_feature_track", "independent"}:
        raise BoundsError(f"motif_logo lane mode must be 'follow_feature_track' or 'independent', got {lane_mode!r}")

    if lane_mode_norm == "follow_feature_track":
        lane_by_effect: dict[int, int] = {}
        above_by_effect: dict[int, bool] = {}
        up_max = -1
        dn_max = -1
        for effect_index, effect in enumerate(record.effects):
            if effect.kind != "motif_logo":
                continue
            feature = _resolve_motif_logo_feature(record, effect, effect_index=effect_index)
            if feature.id is None:
                raise BoundsError(f"motif_logo effect[{effect_index}] target feature must have a stable id")
            feature_index = _feature_index_for_id(record, feature.id)
            if feature_index not in feature_track_by_index:
                raise BoundsError(
                    f"motif_logo effect[{effect_index}] target feature '{feature.id}' has no assigned feature track"
                )
            lane = int(feature_track_by_index[feature_index])
            fixed_lane = _effect_track(effect)
            if fixed_lane is not None and fixed_lane != lane:
                raise BoundsError(
                    f"motif_logo effect[{effect_index}] render.track={fixed_lane} "
                    f"does not match target feature track={lane} in follow_feature_track mode"
                )
            above = bool(feature_above_by_index.get(feature_index, feature.span.strand != "rev"))
            lane_by_effect[effect_index] = lane
            above_by_effect[effect_index] = above
            if above:
                up_max = max(up_max, lane)
            else:
                dn_max = max(dn_max, lane)
        return lane_by_effect, above_by_effect, up_max + 1, dn_max + 1

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
    n = len(record.sequence) if fixed_n is None else max(len(record.sequence), int(fixed_n))

    show_two = bool(style.show_reverse_complement and record.alphabet == "DNA")
    sequence_extent_up, sequence_extent_down = measure_sequence_extents(style.font_mono, style.font_size_seq, style.dpi)
    sequence_half_height = max(sequence_extent_up, sequence_extent_down)
    strand_gap = style.sequence.strand_gap_cells * ch
    baseline_spacing = style.baseline_spacing
    min_baseline = sequence_extent_down + strand_gap + sequence_extent_up
    if show_two:
        baseline_spacing = max(baseline_spacing, min_baseline)
    y_reverse_base = 0.0
    y_forward_base = baseline_spacing if show_two else 0.0

    kmer_box_height = ch * style.kmer.box_height_cells
    motif_logo_height = style.motif_logo.height_bits * style.motif_logo.bits_to_cells * ch
    motif_logo_gap = style.kmer.to_logo_gap_cells * ch
    sequence_to_kmer_gap = style.sequence.to_kmer_gap_cells * ch
    has_motif_logo = any(effect.kind == "motif_logo" for effect in record.effects)
    if has_motif_logo and str(style.motif_logo.lane_mode).lower() == "follow_feature_track":
        min_track_step = kmer_box_height + motif_logo_height + (2.0 * motif_logo_gap)
    else:
        min_track_step = kmer_box_height + ch * 0.2
    feature_track_step = max(style.track_spacing, min_track_step)
    feature_track_base_offset_up = sequence_extent_up + sequence_to_kmer_gap + (0.5 * kmer_box_height)
    feature_track_base_offset_down = sequence_extent_down + sequence_to_kmer_gap + (0.5 * kmer_box_height)
    feature_track_base_offset = feature_track_base_offset_up

    label_pad_x = style.font_size_label / 72.0 * style.dpi * 1.6
    x_left = style.padding_x + label_pad_x
    width = x_left + n * cw + style.padding_x + label_pad_x

    up_indices = [i for i, feat in enumerate(record.features) if feat.span.strand == "fwd"]
    dn_indices = [i for i, feat in enumerate(record.features) if feat.span.strand == "rev"]

    up_features = [record.features[i] for i in up_indices]
    dn_features = [record.features[i] for i in dn_indices]
    up_tracks = assign_tracks(up_features)
    dn_tracks = assign_tracks(dn_features)

    n_up_tracks = (max(up_tracks) + 1) if up_tracks else 0
    n_dn_tracks = (max(dn_tracks) + 1) if dn_tracks else 0

    feature_ids = [feat.id if feat.id is not None else f"f{idx}" for idx, feat in enumerate(record.features)]
    placements: list[FeaturePlacement] = []
    feature_boxes: dict[str, tuple[float, float, float, float]] = {}
    feature_track_by_index: dict[int, int] = {}
    feature_above_by_index: dict[int, bool] = {}
    feature_track_by_id: dict[str, int] = {}
    feature_above_by_id: dict[str, bool] = {}

    for local_idx, feat_idx in enumerate(up_indices):
        feat = record.features[feat_idx]
        track = up_tracks[local_idx]
        x = x_left + feat.span.start * cw
        x_end = x_left + feat.span.end * cw
        w = x_end - x
        y = y_forward_base + feature_track_base_offset_up + track * feature_track_step
        feature_id = feature_ids[feat_idx]
        placement = FeaturePlacement(
            feature_index=feat_idx,
            feature_id=feature_id,
            track=track,
            above=True,
            x=x,
            y=y,
            w=w,
            h=kmer_box_height,
        )
        placements.append(placement)
        feature_boxes[feature_id] = (x, y - kmer_box_height / 2.0, x + w, y + kmer_box_height / 2.0)
        feature_track_by_index[feat_idx] = track
        feature_above_by_index[feat_idx] = True
        feature_track_by_id[feature_id] = track
        feature_above_by_id[feature_id] = True

    for local_idx, feat_idx in enumerate(dn_indices):
        feat = record.features[feat_idx]
        track = dn_tracks[local_idx]
        x = x_left + feat.span.start * cw
        x_end = x_left + feat.span.end * cw
        w = x_end - x
        y = y_reverse_base - feature_track_base_offset_down - track * feature_track_step
        feature_id = feature_ids[feat_idx]
        placement = FeaturePlacement(
            feature_index=feat_idx,
            feature_id=feature_id,
            track=track,
            above=False,
            x=x,
            y=y,
            w=w,
            h=kmer_box_height,
        )
        placements.append(placement)
        feature_boxes[feature_id] = (x, y - kmer_box_height / 2.0, x + w, y + kmer_box_height / 2.0)
        feature_track_by_index[feat_idx] = track
        feature_above_by_index[feat_idx] = False
        feature_track_by_id[feature_id] = track
        feature_above_by_id[feature_id] = False

    motif_logo_lane_by_effect, motif_above_by_effect, motif_lanes_up, motif_lanes_down = assign_motif_logo_lanes(
        record,
        layout_mode=style.motif_logo.layout,
        lane_mode=style.motif_logo.lane_mode,
        feature_track_by_index=feature_track_by_index,
        feature_above_by_index=feature_above_by_index,
    )

    motif_stride = motif_logo_height + motif_logo_gap
    motif_logo_y0_by_effect: dict[int, float] = {}
    lane_mode = str(style.motif_logo.lane_mode).lower()
    for effect_index, lane in motif_logo_lane_by_effect.items():
        effect = record.effects[effect_index]
        feature = _resolve_motif_logo_feature(record, effect, effect_index=effect_index)
        if feature.id is None:
            raise BoundsError(f"motif_logo effect[{effect_index}] target feature must have a stable id")
        box = feature_boxes.get(feature.id)
        if box is None:
            raise BoundsError(f"motif_logo effect[{effect_index}] target feature '{feature.id}' has no placement box")
        above = bool(motif_above_by_effect.get(effect_index, feature.span.strand != "rev"))
        if lane_mode == "follow_feature_track":
            y0 = box[3] + motif_logo_gap if above else box[1] - motif_logo_gap - motif_logo_height
        else:
            lane_offset = lane * motif_stride
            y0 = (
                box[3] + motif_logo_gap + lane_offset
                if above
                else box[1] - motif_logo_gap - motif_logo_height - lane_offset
            )
        motif_logo_y0_by_effect[effect_index] = y0

    y_mins = [y_forward_base - sequence_extent_down]
    y_maxs = [y_forward_base + sequence_extent_up]
    if show_two:
        y_mins.append(y_reverse_base - sequence_extent_down)
        y_maxs.append(y_reverse_base + sequence_extent_up)
    for x0, y0, x1, y1 in feature_boxes.values():
        _ = (x0, x1)
        y_mins.append(y0)
        y_maxs.append(y1)
    for y0 in motif_logo_y0_by_effect.values():
        y_mins.append(y0)
        y_maxs.append(y0 + motif_logo_height)

    content_bottom_raw = min(y_mins)
    content_top_raw = max(y_maxs)
    outer_pad = style.layout.outer_pad_cells * ch
    legend_mode = str(style.legend_mode).lower()
    draw_bottom_legend = bool(style.legend) and legend_mode == "bottom"
    legend_space = (style.legend_height_px + style.legend_pad_px) if draw_bottom_legend else 0.0
    desired_bottom = style.padding_y + outer_pad + legend_space
    shift = desired_bottom - content_bottom_raw

    shifted_placements: list[FeaturePlacement] = []
    for placement in placements:
        shifted_placements.append(
            FeaturePlacement(
                feature_index=placement.feature_index,
                feature_id=placement.feature_id,
                track=placement.track,
                above=placement.above,
                x=placement.x,
                y=placement.y + shift,
                w=placement.w,
                h=placement.h,
            )
        )

    shifted_boxes: dict[str, tuple[float, float, float, float]] = {}
    for fid, (x0, y0, x1, y1) in feature_boxes.items():
        shifted_boxes[fid] = (x0, y0 + shift, x1, y1 + shift)

    shifted_motif_y0 = {idx: y0 + shift for idx, y0 in motif_logo_y0_by_effect.items()}
    y_forward = y_forward_base + shift
    y_reverse = y_reverse_base + shift
    content_bottom = content_bottom_raw + shift
    content_top = content_top_raw + shift
    title_space = max((style.font_size_label / 72.0 * style.dpi) * 1.6, ch * 0.8)
    height = content_top + style.padding_y + outer_pad + title_space

    return LayoutContext(
        cw=cw,
        ch=ch,
        width=width,
        height=height,
        y_forward=y_forward,
        y_reverse=y_reverse,
        sequence_half_height=sequence_half_height,
        sequence_extent_up=sequence_extent_up,
        sequence_extent_down=sequence_extent_down,
        x_left=x_left,
        n_up_tracks=n_up_tracks,
        n_dn_tracks=n_dn_tracks,
        kmer_box_height=kmer_box_height,
        feature_track_step=feature_track_step,
        feature_track_base_offset=feature_track_base_offset,
        feature_track_base_offset_up=feature_track_base_offset_up,
        feature_track_base_offset_down=feature_track_base_offset_down,
        placements=tuple(shifted_placements),
        feature_boxes=shifted_boxes,
        feature_track_by_id=feature_track_by_id,
        feature_above_by_id=feature_above_by_id,
        motif_logo_height=motif_logo_height,
        motif_logo_gap=motif_logo_gap,
        motif_logo_lane_by_effect=motif_logo_lane_by_effect,
        motif_logo_above_by_effect=motif_above_by_effect,
        motif_logo_y0_by_effect=shifted_motif_y0,
        motif_logo_lanes_up=motif_lanes_up,
        motif_logo_lanes_down=motif_lanes_down,
        content_top=content_top,
        content_bottom=content_bottom,
    )
