"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/snapback/released_plot_common.py

Shared constants and drawing helpers for released-product snapback hit plots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Literal

from dnadesign.cruncher.snapback.released_plot_models import PlotSpan
from dnadesign.cruncher.snapback.released_route_policy import ReleasedActiveStrand

_TEXT = "#334155"
_TITLE = "#475569"
_BOUNDARY = "#0F172A"
_NICK = "#2563EB"
_RELEASE = "#D97706"
_RETAINED = "#059669"
_ACTIVE = "#0F766E"
_TAIL = "#94A3B8"
_STEM = "#DC2626"
_CAP = "#DB2777"
_FOLDBACK = "#7C3AED"
_OVERHANG = "#64748B"
_FIGURE_FACE = "#FFFFFF"
_NICK_SITE_FILL = "#DBEAFE"
_RELEASE_SITE_FILL = "#FFEDD5"
_ASSIGNABLE_BASE_FILL = "#E0F2FE"
_ASSIGNABLE_BASE_EDGE = "#93C5FD"
_TITLE_FONT = "DejaVu Sans"
_LABEL_FONT = "DejaVu Sans"
_SEQUENCE_FONT = "DejaVu Sans Mono"
_TITLE_SIZE = 15.8
_LABEL_SIZE = 12.6
_SEQUENCE_SIZE = 16.4
_ANNOTATION_SIZE = 12.6
_BOUNDARY_LABEL_SIZE = 11.2
_BASE_ADVANCE = 0.82
_ROW_TOP_Y = 0.77
_ROW_BOTTOM_Y = 0.61
_ROW_CONNECT_TOP = 0.71
_ROW_CONNECT_BOTTOM = 0.67
_PRIMARY_SPAN_Y = 1.22
_SECONDARY_SPAN_Y = 1.50
_LOWER_PRIMARY_SPAN_Y = 0.34
_LOWER_SECONDARY_SPAN_Y = 0.16
_STATUS_Y = -0.36
_AXIS_YMIN = 0.12
_AXIS_YMAX = 1.34
_TITLE_Y = 1.24
_STRUCTURE_LABEL_Y = 1.04
_TITLE_LEFT_MARGIN = 3.2
_TITLE_RIGHT_MARGIN = 1.2
_ROW_LABEL_X_GAP = 1.78
_BOUNDARY_MARK_HALF_HEIGHT = 0.08
_BOUNDARY_CONNECTOR_GAP = 0.04
_BOUNDARY_LABEL_MARGIN = 0.02
_SPAN_ENDCAP_HALF_HEIGHT = 0.032
_SITE_FOOTPRINT_VERTICAL_PAD = 0.038
_SITE_LABEL_GAP = 0.022
PANEL_TITLE_PRECURSOR_SITES = "Precursor Sites"
PANEL_TITLE_POST_RELEASE_FRAGMENTS = "Post-Release Fragments"
PANEL_TITLE_FOLDBACK = "Foldback"
ROW_LABEL_TOP = "Top"
ROW_LABEL_BOTTOM = "Bottom"
ROW_LABEL_STEM = "Stem"
ROW_LABEL_FOLDBACK_STEM = "Foldback Stem"
LABEL_NICK = "Nick"
LABEL_TOP_CUT = "Top Cut"
LABEL_BOTTOM_CUT = "Bottom Cut"
LABEL_STEM = "Stem"
LABEL_CAP = "Cap"
LABEL_FOLDBACK = "Foldback"


def strand_row_label(strand: ReleasedActiveStrand) -> str:
    return ROW_LABEL_TOP if strand == "top" else ROW_LABEL_BOTTOM


def retained_partner_row_label(strand: ReleasedActiveStrand) -> str:
    return f"Retained {strand_row_label(strand)}"


def exposed_product_row_label(strand: ReleasedActiveStrand) -> str:
    return f"Exposed {strand_row_label(strand)}"


def retained_strand_for_nick(nicked_strand: ReleasedActiveStrand) -> ReleasedActiveStrand:
    return "bottom" if nicked_strand == "top" else "top"


def post_release_physical_row_label(
    *,
    strand: ReleasedActiveStrand,
    nicked_strand: ReleasedActiveStrand,
) -> str:
    if strand == nicked_strand:
        return exposed_product_row_label(strand)
    return retained_partner_row_label(strand)


def estimated_label_half_width_nt(label: str) -> float:
    return max(0.80, len(label) * 0.18)


def label_interval_nt(*, start: float, end: float, label: str) -> tuple[float, float]:
    center = (start + end) / 2.0
    half_width = estimated_label_half_width_nt(label)
    return (center - half_width, center + half_width)


def boundary_label_interval_nt(*, boundary: float, label: str) -> tuple[float, float]:
    half_width = estimated_label_half_width_nt(label)
    return (boundary - half_width, boundary + half_width)


def staggered_label_y(
    *,
    index: int,
    labeled_spans: list[tuple[float, float, str]],
    base_y: float,
    stagger_y: float,
    obstacles: list[tuple[float, float]] | None = None,
) -> float:
    direction = 1.0 if stagger_y >= base_y else -1.0
    step = max(abs(stagger_y - base_y), 0.06)
    placed: list[tuple[float, float, int]] = []
    obstacle_intervals = list(obstacles or [])
    selected_level = 0
    for current_index, (start, end, label) in enumerate(labeled_spans[: index + 1]):
        left, right = label_interval_nt(start=start, end=end, label=label)
        level = 0
        while True:
            collides_with_obstacle = level == 0 and any(
                left <= previous_right and right >= previous_left
                for previous_left, previous_right in obstacle_intervals
            )
            collides_with_previous_label = any(
                previous_level == level and left <= previous_right and right >= previous_left
                for previous_left, previous_right, previous_level in placed
            )
            if not collides_with_obstacle and not collides_with_previous_label:
                break
            level += 1
        placed.append((left, right, level))
        if current_index == index:
            selected_level = level
            break
    return base_y + (direction * step * selected_level)


def axis_center(x_min: float, x_max: float) -> float:
    return ((x_for_boundary(x_min) - _TITLE_LEFT_MARGIN) + (x_for_boundary(x_max) + _TITLE_RIGHT_MARGIN)) / 2.0


def x_for_boundary(boundary: float, *, x_start: float = 0.0) -> float:
    return (x_start + boundary) * _BASE_ADVANCE


def x_for_base(index: int, *, x_start: float = 0.0) -> float:
    return x_for_boundary(index + 0.5, x_start=x_start)


def wrap_annotation_label(label: str, *, start: int, end: int) -> str:
    if "\n" in label or " " not in label:
        return label
    span_width = max(end - start, 1)
    if len(label) <= span_width + 4 and len(label) <= 12:
        return label
    words = label.split()
    if len(words) == 2:
        return "\n".join(words)
    midpoint = len(label) / 2.0
    best_split = 1
    best_distance = abs(len(words[0]) - midpoint)
    running = len(words[0])
    for index in range(1, len(words)):
        distance = abs(running - midpoint)
        if distance < best_distance:
            best_split = index
            best_distance = distance
        running += 1 + len(words[index])
    return "\n".join((" ".join(words[:best_split]), " ".join(words[best_split:])))


def soft_fill(color: str, *, mix_with_figure: float = 0.80) -> tuple[float, float, float, float]:
    from matplotlib.colors import to_rgba

    red, green, blue, _ = to_rgba(color)
    figure_red, figure_green, figure_blue, _ = to_rgba(_FIGURE_FACE)
    mixed_red = (mix_with_figure * figure_red) + ((1.0 - mix_with_figure) * red)
    mixed_green = (mix_with_figure * figure_green) + ((1.0 - mix_with_figure) * green)
    mixed_blue = (mix_with_figure * figure_blue) + ((1.0 - mix_with_figure) * blue)
    return (mixed_red, mixed_green, mixed_blue, 1.0)


def site_label_y(site_top_y: float) -> float:
    return site_top_y + _SITE_LABEL_GAP


def strand_y(strand: ReleasedActiveStrand) -> float:
    return _ROW_TOP_Y if strand == "top" else _ROW_BOTTOM_Y


def boundary_label_y(strand: ReleasedActiveStrand, *, label_above: bool) -> float:
    lane_y = strand_y(strand)
    if label_above:
        return lane_y + _BOUNDARY_MARK_HALF_HEIGHT + _BOUNDARY_LABEL_MARGIN
    return lane_y - _BOUNDARY_MARK_HALF_HEIGHT - _BOUNDARY_LABEL_MARGIN


def site_footprint_bounds() -> tuple[float, float]:
    return (
        _ROW_BOTTOM_Y - _SITE_FOOTPRINT_VERTICAL_PAD,
        _ROW_TOP_Y + _SITE_FOOTPRINT_VERTICAL_PAD,
    )


def span_contains_boundary(span: PlotSpan, boundary: int) -> bool:
    return span.start <= boundary <= span.end


def draw_sequence_pairing(
    ax,
    *,
    start: int,
    end: int,
    mismatch_positions: set[int],
    linewidth: float,
) -> None:
    for position in range(start, end):
        x = x_for_base(position)
        ax.plot(
            [x, x],
            [_ROW_CONNECT_BOTTOM, _ROW_CONNECT_TOP],
            color="#CBD5E1" if position not in mismatch_positions else _STEM,
            linewidth=linewidth,
        )


def draw_assignable_base_zstack(ax, *, x: float, y: float) -> None:
    from matplotlib.patches import FancyBboxPatch

    width = _BASE_ADVANCE * 0.66
    height = 0.058
    ax.add_patch(
        FancyBboxPatch(
            (x - (width / 2.0), y - (height / 2.0)),
            width,
            height,
            boxstyle="round,pad=0.004,rounding_size=0.022",
            linewidth=0.55,
            edgecolor=_ASSIGNABLE_BASE_EDGE,
            facecolor=_ASSIGNABLE_BASE_FILL,
            alpha=0.92,
            zorder=1.5,
        )
    )


def _position_in_segments(position: int, segments: list[tuple[int, int]] | None) -> bool:
    if segments is None:
        return False
    return any(start <= position < end for start, end in segments)


def draw_sequence(
    ax,
    *,
    sequence: str,
    y: float,
    row_label: str,
    start_terminal: str | None,
    end_terminal: str | None,
    x_start: float = 0.0,
    color_segments: list[tuple[int, int, str]] | None = None,
    assignable_base_positions: list[int] | None = None,
    emphasis_segments: list[tuple[int, int]] | None = None,
) -> None:
    ax.text(
        x_for_boundary(x_start) - _ROW_LABEL_X_GAP,
        y,
        row_label,
        ha="right",
        va="center",
        fontsize=_LABEL_SIZE,
        fontweight="semibold",
        family=_LABEL_FONT,
        color=_TEXT,
    )
    if start_terminal:
        ax.text(
            x_for_boundary(x_start) + 0.06,
            y,
            start_terminal,
            ha="right",
            va="center",
            fontsize=_SEQUENCE_SIZE,
            family=_SEQUENCE_FONT,
            color=_TITLE,
        )
    if end_terminal:
        ax.text(
            x_for_boundary(x_start + len(sequence)) + 0.34,
            y,
            end_terminal,
            ha="left",
            va="center",
            fontsize=_SEQUENCE_SIZE,
            family=_SEQUENCE_FONT,
            color=_TITLE,
        )
    assignable_positions = set(assignable_base_positions or [])
    for index, base in enumerate(sequence):
        position = int(x_start) + index
        x = x_for_base(index, x_start=x_start)
        if position in assignable_positions:
            draw_assignable_base_zstack(ax, x=x, y=y)
        base_color = _TEXT
        if color_segments is not None:
            for start, end, color in color_segments:
                if start <= position < end:
                    base_color = color
                    break
        ax.text(
            x,
            y,
            base,
            ha="center",
            va="center",
            fontsize=_SEQUENCE_SIZE,
            family=_SEQUENCE_FONT,
            color=base_color,
            fontweight="bold" if _position_in_segments(position, emphasis_segments) else "normal",
            zorder=3.0,
        )


def draw_span(ax, *, start: int, end: int, y: float, label: str, color: str) -> None:
    if end <= start:
        return
    x0 = x_for_boundary(start) + 0.06
    x1 = x_for_boundary(end) - 0.06
    wrapped_label = wrap_annotation_label(label, start=start, end=end)
    ax.plot([x0, x1], [y, y], color=color, linewidth=2.2, solid_capstyle="round", zorder=1)
    ax.plot(
        [x0, x0],
        [y - _SPAN_ENDCAP_HALF_HEIGHT, y + _SPAN_ENDCAP_HALF_HEIGHT],
        color=color,
        linewidth=1.2,
        zorder=1,
    )
    ax.plot(
        [x1, x1],
        [y - _SPAN_ENDCAP_HALF_HEIGHT, y + _SPAN_ENDCAP_HALF_HEIGHT],
        color=color,
        linewidth=1.2,
        zorder=1,
    )
    ax.text(
        (x0 + x1) / 2.0,
        y,
        wrapped_label,
        ha="center",
        va="center",
        fontsize=_ANNOTATION_SIZE,
        fontweight="semibold",
        family=_LABEL_FONT,
        color=color,
        multialignment="center",
        linespacing=0.95,
        bbox={
            "facecolor": soft_fill(color),
            "edgecolor": "none",
            "boxstyle": "round,pad=0.22,rounding_size=0.14",
        },
        zorder=3,
    )


def draw_region_label(ax, *, start: int, end: int, y: float, label: str, color: str) -> None:
    if end <= start:
        return
    ax.text(
        (x_for_boundary(start) + x_for_boundary(end)) / 2.0,
        y,
        wrap_annotation_label(label, start=start, end=end),
        ha="center",
        va="center",
        fontsize=_ANNOTATION_SIZE,
        fontweight="semibold",
        family=_LABEL_FONT,
        color=color,
        multialignment="center",
        linespacing=0.95,
        bbox={
            "facecolor": soft_fill(color),
            "edgecolor": "none",
            "boxstyle": "round,pad=0.22,rounding_size=0.14",
        },
        zorder=3,
    )


def draw_strand_boundary(
    ax,
    *,
    boundary: int,
    strand: ReleasedActiveStrand,
    label: str,
    color: str,
    label_y: float,
    label_above: bool = True,
    dashed: bool = False,
) -> None:
    boundary_x = x_for_boundary(boundary)
    lane_y = strand_y(strand)
    strand_marker = ax.plot(
        [boundary_x, boundary_x],
        [lane_y - _BOUNDARY_MARK_HALF_HEIGHT, lane_y + _BOUNDARY_MARK_HALF_HEIGHT],
        color=color,
        linewidth=1.4,
        zorder=2,
    )[0]
    if label_above:
        connector = ax.plot(
            [boundary_x, boundary_x],
            [lane_y + _BOUNDARY_MARK_HALF_HEIGHT, label_y - _BOUNDARY_CONNECTOR_GAP],
            color=color,
            linewidth=1.2,
            zorder=2,
        )[0]
        text_va = "bottom"
    else:
        connector = ax.plot(
            [boundary_x, boundary_x],
            [label_y + _BOUNDARY_CONNECTOR_GAP, lane_y - _BOUNDARY_MARK_HALF_HEIGHT],
            color=color,
            linewidth=1.2,
            zorder=2,
        )[0]
        text_va = "top"
    if dashed:
        strand_marker.set_dashes((2.5, 2.0))
        connector.set_dashes((2.5, 2.0))
    ax.text(
        boundary_x,
        label_y,
        label,
        ha="center",
        va=text_va,
        fontsize=_BOUNDARY_LABEL_SIZE,
        fontweight="semibold",
        family=_LABEL_FONT,
        color=color,
        bbox={"facecolor": _FIGURE_FACE, "edgecolor": "none", "pad": 0.16},
        zorder=3,
    )


def draw_site_footprint(
    ax,
    *,
    start: int,
    end: int,
    label: str,
    fill_color: str,
    text_color: str,
    label_placement: Literal["above", "below"] = "above",
) -> None:
    if end <= start:
        return
    from matplotlib.patches import FancyBboxPatch

    fill_y0, fill_y1 = site_footprint_bounds()
    x0 = x_for_boundary(start) - 0.02
    x1 = x_for_boundary(end) + 0.02
    ax.add_patch(
        FancyBboxPatch(
            (x0, fill_y0),
            x1 - x0,
            fill_y1 - fill_y0,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor=fill_color,
            edgecolor="none",
            alpha=0.85,
            zorder=-10,
        )
    )
    if label_placement == "above":
        label_y = site_label_y(fill_y1)
        label_va = "bottom"
    elif label_placement == "below":
        label_y = fill_y0 - _SITE_LABEL_GAP
        label_va = "top"
    else:
        raise ValueError("site footprint label_placement must be 'above' or 'below'.")
    ax.text(
        (x0 + x1) / 2.0,
        label_y,
        label,
        ha="center",
        va=label_va,
        fontsize=_ANNOTATION_SIZE + 0.2,
        fontweight="semibold",
        family=_LABEL_FONT,
        color=text_color,
        bbox={"facecolor": _FIGURE_FACE, "edgecolor": "none", "pad": 0.18},
        zorder=3,
    )


def configure_axis(ax, *, x_min: float, x_max: float, title: str) -> None:
    ax.set_axis_off()
    ax.set_facecolor(_FIGURE_FACE)
    ax.set_xlim(x_for_boundary(x_min) - _TITLE_LEFT_MARGIN, x_for_boundary(x_max) + _TITLE_RIGHT_MARGIN)
    ax.set_ylim(_AXIS_YMIN, _AXIS_YMAX)
    ax.text(
        axis_center(x_min, x_max),
        _TITLE_Y,
        title,
        ha="center",
        va="top",
        fontsize=_TITLE_SIZE,
        fontweight="semibold",
        family=_TITLE_FONT,
        color=_TITLE,
    )


__all__ = [
    "_ACTIVE",
    "_ANNOTATION_SIZE",
    "_CAP",
    "_FOLDBACK",
    "_FIGURE_FACE",
    "_LOWER_PRIMARY_SPAN_Y",
    "_NICK",
    "_NICK_SITE_FILL",
    "_OVERHANG",
    "_RELEASE",
    "_RELEASE_SITE_FILL",
    "_ROW_BOTTOM_Y",
    "_ROW_TOP_Y",
    "_SEQUENCE_SIZE",
    "_SITE_FOOTPRINT_VERTICAL_PAD",
    "_STEM",
    "_TAIL",
    "_STRUCTURE_LABEL_Y",
    "LABEL_BOTTOM_CUT",
    "LABEL_CAP",
    "LABEL_FOLDBACK",
    "LABEL_NICK",
    "LABEL_STEM",
    "LABEL_TOP_CUT",
    "PANEL_TITLE_FOLDBACK",
    "PANEL_TITLE_POST_RELEASE_FRAGMENTS",
    "PANEL_TITLE_PRECURSOR_SITES",
    "ROW_LABEL_BOTTOM",
    "ROW_LABEL_FOLDBACK_STEM",
    "ROW_LABEL_STEM",
    "ROW_LABEL_TOP",
    "boundary_label_interval_nt",
    "boundary_label_y",
    "configure_axis",
    "draw_region_label",
    "draw_sequence",
    "draw_assignable_base_zstack",
    "draw_sequence_pairing",
    "draw_site_footprint",
    "draw_span",
    "draw_strand_boundary",
    "estimated_label_half_width_nt",
    "exposed_product_row_label",
    "label_interval_nt",
    "post_release_physical_row_label",
    "retained_partner_row_label",
    "retained_strand_for_nick",
    "site_footprint_bounds",
    "span_contains_boundary",
    "staggered_label_y",
    "strand_row_label",
]
