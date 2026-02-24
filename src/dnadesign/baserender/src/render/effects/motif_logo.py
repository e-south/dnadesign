"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/effects/motif_logo.py

Motif-logo effect drawer using stacked vector glyphs aligned to the base grid.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

from ...core import Effect, Record, RenderingError
from ..layout import LayoutContext, span_to_x
from ..palette import Palette

_DNA_BASES: tuple[str, str, str, str] = ("A", "C", "G", "T")

_DEFAULT_BASE_COLORS: Mapping[str, str] = {
    "A": "#1f77b4",
    "C": "#2ca02c",
    "G": "#ff7f0e",
    "T": "#d62728",
}


@dataclass(frozen=True)
class MotifLogoGeometry:
    feature_id: str
    x0: float
    x1: float
    columns: tuple[tuple[float, float], ...]
    y0: float
    height: float
    lane: int
    above: bool
    baseline: str
    observed: str
    matrix: tuple[tuple[float, float, float, float], ...]


def _normalize_row(row: Sequence[float]) -> tuple[float, float, float, float]:
    if len(row) < 4:
        raise RenderingError("motif_logo matrix rows must contain at least 4 probabilities [A,C,G,T]")
    vals = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
    total = sum(vals)
    if total <= 0:
        raise RenderingError("motif_logo matrix rows must have positive sum")
    return (
        vals[0] / total,
        vals[1] / total,
        vals[2] / total,
        vals[3] / total,
    )


def reverse_complement_matrix(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    rc: list[list[float]] = []
    for row in reversed(matrix):
        if len(row) < 4:
            raise RenderingError("motif_logo matrix rows must contain at least 4 probabilities [A,C,G,T]")
        a, c, g, t = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        rc.append([t, g, c, a])
    return rc


def reverse_matrix_rows(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for row in reversed(matrix):
        if len(row) < 4:
            raise RenderingError("motif_logo matrix rows must contain at least 4 probabilities [A,C,G,T]")
        out.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
    return out


def _effect_index(record: Record, effect: Effect) -> int:
    for idx, candidate in enumerate(record.effects):
        if candidate is effect:
            return idx
    raise RenderingError("motif_logo effect object not present in record.effects")


def _resolve_feature(record: Record, feature_id: str):
    feature = next((f for f in record.features if f.id == feature_id), None)
    if feature is None:
        raise RenderingError(f"motif_logo target feature '{feature_id}' not found")
    if feature.kind not in {"kmer", "regulator_window"}:
        raise RenderingError("motif_logo target feature must be kind='kmer' or 'regulator_window'")
    if feature.label is None:
        raise RenderingError("motif_logo target feature must have label")
    return feature


def _coerce_matrix_rows(matrix_raw: object) -> list[list[float]]:
    if not isinstance(matrix_raw, list) or not matrix_raw:
        raise RenderingError("motif_logo params.matrix must be a non-empty list at draw time")
    matrix_rows: list[list[float]] = []
    for row in matrix_raw:
        if not isinstance(row, (list, tuple)):
            raise RenderingError("motif_logo matrix rows must be lists or tuples")
        matrix_rows.append([float(v) for v in row])
    return matrix_rows


def _style_base_colors(style) -> Mapping[str, str]:
    raw = style.motif_logo.colors
    if not isinstance(raw, Mapping):
        raise RenderingError("style.motif_logo.colors must be a mapping")
    colors = {str(k).upper(): str(v) for k, v in raw.items()}
    for base in _DNA_BASES:
        if base not in colors:
            return _DEFAULT_BASE_COLORS
    return colors


@lru_cache(maxsize=16)
def _glyph_path(base: str, font_family: str):
    prop = FontProperties(family=font_family, size=1.0)
    raw = TextPath((0, 0), base, prop=prop, usetex=False)
    bbox = raw.get_extents()
    if bbox.width <= 0 or bbox.height <= 0:
        raise RenderingError(f"Failed to build motif glyph for base {base!r}")
    normalized = Affine2D().translate(-bbox.x0, -bbox.y0).transform_path(raw)
    nb = normalized.get_extents()
    return normalized, nb


def _entropy_bits(row: tuple[float, float, float, float]) -> float:
    entropy = 0.0
    for prob in row:
        if prob > 0:
            entropy += -prob * math.log2(prob)
    return entropy


def _logo_stack_bits(row: tuple[float, float, float, float], *, max_bits: float) -> list[tuple[str, float]]:
    info_bits = max(0.0, max_bits - _entropy_bits(row))
    return [(base, row[idx] * info_bits) for idx, base in enumerate(_DNA_BASES)]


def _logo_stack_probs(row: tuple[float, float, float, float]) -> list[tuple[str, float]]:
    return [(base, row[idx]) for idx, base in enumerate(_DNA_BASES)]


def compute_motif_logo_geometry(
    *,
    record: Record,
    effect_index: int,
    layout: LayoutContext,
    style,
    feature_boxes: Mapping[str, tuple[float, float, float, float]] | None = None,
) -> MotifLogoGeometry:
    _ = feature_boxes
    if effect_index < 0 or effect_index >= len(record.effects):
        raise RenderingError(f"motif_logo effect index out of bounds: {effect_index}")
    effect = record.effects[effect_index]
    if effect.kind != "motif_logo":
        raise RenderingError(f"effect[{effect_index}] is not motif_logo")

    feature_id_raw = effect.target.get("feature_id")
    if not isinstance(feature_id_raw, str) or feature_id_raw.strip() == "":
        raise RenderingError("motif_logo target.feature_id is required")
    feature_id = feature_id_raw
    feature = _resolve_feature(record, feature_id)

    matrix_rows = _coerce_matrix_rows(effect.params.get("matrix"))
    if len(matrix_rows) != feature.span.length():
        raise RenderingError("motif_logo matrix length must match target kmer length")

    observed = feature.label.upper()
    if feature.span.strand == "rev":
        matrix_rows = reverse_matrix_rows(matrix_rows)
        observed = observed[::-1]

    row_count = len(matrix_rows)
    x0, x1 = span_to_x(layout, feature.span.start, feature.span.end)
    columns = tuple(
        span_to_x(layout, feature.span.start + offset, feature.span.start + offset + 1) for offset in range(row_count)
    )

    lane = int(layout.motif_logo_lane_by_effect.get(effect_index, 0))
    above = bool(layout.motif_logo_above_by_effect.get(effect_index, feature.span.strand != "rev"))
    y0 = layout.motif_logo_y0_by_effect.get(effect_index)
    if y0 is None:
        raise RenderingError(f"motif_logo effect[{effect_index}] is missing precomputed y-placement in layout context")

    return MotifLogoGeometry(
        feature_id=feature_id,
        x0=x0,
        x1=x1,
        columns=columns,
        y0=y0,
        height=layout.motif_logo_height,
        lane=lane,
        above=above,
        baseline="bottom" if above else "top",
        observed=observed,
        matrix=tuple(tuple(_normalize_row(row)) for row in matrix_rows),
    )


def _draw_letter(
    ax,
    *,
    base: str,
    x0: float,
    x1: float,
    y0: float,
    height: float,
    style,
    color: str,
    alpha: float,
    observed: bool,
    gid: str,
) -> None:
    if height <= 0:
        return

    glyph, glyph_bbox = _glyph_path(base, style.font_mono)
    col_w = x1 - x0
    pad_frac = float(style.motif_logo.letter_x_pad_frac)
    target_w = max(0.5, col_w * (1.0 - pad_frac))
    x_left = x0 + (col_w - target_w) / 2.0

    sx = target_w / glyph_bbox.width
    sy = height / glyph_bbox.height
    transform = Affine2D().scale(sx, sy).translate(x_left - glyph_bbox.x0 * sx, y0 - glyph_bbox.y0 * sy) + ax.transData

    patch = PathPatch(
        glyph,
        transform=transform,
        facecolor=color,
        edgecolor=color if observed else "none",
        linewidth=0.6 if observed else 0.0,
        alpha=alpha,
        zorder=4.8,
        clip_on=False,
    )
    patch.set_gid(gid)
    ax.add_patch(patch)


def draw_motif_logo(
    ax,
    effect: Effect,
    record: Record,
    layout: LayoutContext,
    style,
    palette: Palette,
    feature_boxes: dict[str, tuple[float, float, float, float]],
) -> None:
    effect_index = _effect_index(record, effect)
    geometry = compute_motif_logo_geometry(
        record=record,
        effect_index=effect_index,
        layout=layout,
        style=style,
        feature_boxes=feature_boxes,
    )

    base_colors = _style_base_colors(style)
    target_feature = _resolve_feature(record, geometry.feature_id)
    feature_tag = target_feature.tags[0] if target_feature.tags else target_feature.kind
    feature_fill_color = palette.color_for(feature_tag)
    letter_coloring_mode = str(style.motif_logo.letter_coloring.mode).lower()
    if letter_coloring_mode not in {"classic", "match_window_seq"}:
        raise RenderingError(f"Unknown motif_logo letter coloring mode: {letter_coloring_mode!r}")
    observed_color_source = str(style.motif_logo.letter_coloring.observed_color_source).lower()
    if observed_color_source not in {"nucleotide_palette", "feature_fill"}:
        raise RenderingError(
            f"Unknown motif_logo observed color source: {style.motif_logo.letter_coloring.observed_color_source!r}"
        )
    other_color = str(style.motif_logo.letter_coloring.other_color)
    display_mode = str(style.motif_logo.display_mode).lower()
    if display_mode == "information":
        unit_to_px = geometry.height / float(style.motif_logo.height_bits)
    elif display_mode == "probability":
        unit_to_px = geometry.height
    else:
        raise RenderingError(f"Unknown motif_logo display mode: {display_mode!r}")
    alpha_other = float(style.motif_logo.alpha_other)
    alpha_observed = float(style.motif_logo.alpha_observed)

    for col_index, row in enumerate(geometry.matrix):
        if display_mode == "information":
            stack = _logo_stack_bits(row, max_bits=float(style.motif_logo.height_bits))
        else:
            stack = _logo_stack_probs(row)
        stack.sort(key=lambda item: item[1])

        col_x0, col_x1 = geometry.columns[col_index]
        y_cursor = geometry.y0 if geometry.baseline == "bottom" else (geometry.y0 + geometry.height)
        observed_base = geometry.observed[col_index] if col_index < len(geometry.observed) else ""
        if letter_coloring_mode == "match_window_seq" and observed_base in _DNA_BASES:
            observed_color = (
                base_colors[observed_base] if observed_color_source == "nucleotide_palette" else feature_fill_color
            )
        else:
            observed_color = None

        for base, bits in stack:
            letter_h = bits * unit_to_px
            if letter_h <= 0:
                continue
            is_observed = observed_base == base
            if geometry.baseline == "bottom":
                y_draw = y_cursor
                y_cursor += letter_h
            else:
                y_cursor -= letter_h
                y_draw = y_cursor
            if letter_coloring_mode == "classic":
                letter_color = base_colors[base]
            elif observed_color is None:
                letter_color = other_color
            elif is_observed:
                letter_color = observed_color
            else:
                letter_color = other_color
            _draw_letter(
                ax,
                base=base,
                x0=col_x0,
                x1=col_x1,
                y0=y_draw,
                height=letter_h,
                style=style,
                color=letter_color,
                alpha=alpha_observed if is_observed else alpha_other,
                observed=is_observed,
                gid=f"motif_logo:{geometry.feature_id}:{col_index}:{base}",
            )

    if bool(style.motif_logo.debug_bounds):
        ax.add_patch(
            Rectangle(
                (geometry.x0, geometry.y0),
                geometry.x1 - geometry.x0,
                geometry.height,
                facecolor="none",
                edgecolor="#9ca3af",
                linewidth=0.75,
                alpha=0.7,
                zorder=4.2,
                clip_on=False,
            )
        )
