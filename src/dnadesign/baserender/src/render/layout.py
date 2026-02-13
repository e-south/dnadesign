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
from ..core import BoundsError, Feature, Record

_DNA_COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def comp(seq: str) -> str:
    return seq.translate(_DNA_COMP)


def revcomp(seq: str) -> str:
    return comp(seq)[::-1]


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


def assign_tracks(features: Sequence[Feature]) -> list[int]:
    tracks: list[int] = [-1] * len(features)
    per_track_spans: dict[int, list[tuple[int, int, int]]] = {}

    # Fixed tracks first, hard-fail on overlap.
    for idx, feat in enumerate(features):
        track = _feature_track(feat)
        if track is None:
            continue
        spans = per_track_spans.setdefault(track, [])
        for start, end, other_idx in spans:
            if _overlap(feat.span.start, feat.span.end, start, end):
                raise BoundsError(
                    f"Fixed-track overlap detected: feature[{idx}] overlaps feature[{other_idx}] on track {track}"
                )
        spans.append((feat.span.start, feat.span.end, idx))
        tracks[idx] = track

    # Remaining features use greedy interval coloring by priority.
    order = sorted(
        [
            (idx, _feature_priority(feat), feat.span.start, feat.span.end)
            for idx, feat in enumerate(features)
            if tracks[idx] == -1
        ],
        key=lambda item: (item[1], item[2], -(item[3] - item[2])),
    )

    for idx, _prio, start, end in order:
        placed = False
        for track, spans in sorted(per_track_spans.items(), key=lambda kv: kv[0]):
            if all(not _overlap(start, end, s, e) for s, e, _ in spans):
                spans.append((start, end, idx))
                tracks[idx] = track
                placed = True
                break
        if not placed:
            next_track = 0
            while next_track in per_track_spans:
                next_track += 1
            per_track_spans[next_track] = [(start, end, idx)]
            tracks[idx] = next_track

    return tracks


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
    if any(effect.kind == "motif_logo" for effect in record.effects):
        top += h
        bottom -= h
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
        y = y_forward + (track + 1) * style.track_spacing
        w = feat.span.length() * cw
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
        y = y_reverse - (track + 1) * style.track_spacing
        w = feat.span.length() * cw
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
    )
