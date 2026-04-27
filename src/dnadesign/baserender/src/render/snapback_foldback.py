"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/snapback_foldback.py

Foldback-specific snapback rendering helpers for compact corner-triloop views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from dnadesign.contracts.visual import SnapbackVisualV1

from ..config import Style
from ..core import RenderingError

_CAP_SHOULDER_OFFSET_X = 0.64
_CAP_APEX_OFFSET_X = 1.28
_CAP_SHOULDER_VERTICAL_OVERSHOOT = 0.06
_RIGHT_TERMINAL_PAD_X = 0.56


@dataclass(frozen=True)
class FoldbackCornerLayout:
    primary_indices: list[int]
    complement_indices: list[int]
    cap_indices: list[int]
    stem_x_positions: list[float]
    cap_x_positions: list[float]
    cap_y_positions: list[float]
    top_y: float
    bottom_y: float
    left_pad: float
    right_pad: float
    bottom_extent: float
    top_extent: float
    title_y: float
    figure_width: float
    figure_height: float
    row_label_x: float
    left_terminal_x: float
    right_terminal_x: float

    @property
    def right_x(self) -> float:
        return self.stem_x_positions[-1]

    @property
    def max_cap_x(self) -> float:
        return max(self.cap_x_positions)


def _build_layout(contract: SnapbackVisualV1, style: Style) -> FoldbackCornerLayout:
    loop = contract.loop_geometry
    if loop is None or contract.cap_span is None:
        raise RenderingError("foldback corner rendering requires loop_geometry and cap_span")
    cap_indices = list(range(contract.cap_span.start, contract.cap_span.end))
    if loop.kind != "hairpin_corner_triloop_v1" or len(cap_indices) != 3:
        raise RenderingError("foldback corner renderer requires a triloop cap of exactly 3 nt")

    primary_indices = list(range(loop.display_primary_span.start, loop.display_primary_span.end))
    complement_indices = list(range(loop.display_complement_span.end - 1, loop.display_complement_span.start - 1, -1))
    if len(primary_indices) != len(complement_indices):
        raise RenderingError("foldback corner renderer requires equal retained/foldback display spans")

    top_y = 1.55
    bottom_y = 0.55
    stem_x_positions = [offset + 0.5 for offset in range(len(primary_indices))]
    right_x = stem_x_positions[-1]
    cap_x_positions = [
        right_x + _CAP_SHOULDER_OFFSET_X,
        right_x + _CAP_APEX_OFFSET_X,
        right_x + _CAP_SHOULDER_OFFSET_X,
    ]
    cap_y_positions = [
        top_y + _CAP_SHOULDER_VERTICAL_OVERSHOOT,
        (top_y + bottom_y) / 2.0,
        bottom_y - _CAP_SHOULDER_VERTICAL_OVERSHOOT,
    ]

    left_x = min(stem_x_positions)
    left_pad = 5.8
    right_pad = 3.05
    top_extent = top_y + 0.82
    bottom_extent = bottom_y - 0.88
    title_y = top_extent + 0.36
    figure_width = max(6.6, (len(primary_indices) + left_pad + right_pad) * 0.46 * float(style.figure_scale))
    figure_height = max(3.9, (title_y - bottom_extent + 0.34) * 0.95 * float(style.figure_scale))
    row_label_x = left_x - 1.65
    left_terminal_x = left_x - 0.72
    right_terminal_x = max(right_x, max(cap_x_positions)) + _RIGHT_TERMINAL_PAD_X
    return FoldbackCornerLayout(
        primary_indices=primary_indices,
        complement_indices=complement_indices,
        cap_indices=cap_indices,
        stem_x_positions=stem_x_positions,
        cap_x_positions=cap_x_positions,
        cap_y_positions=cap_y_positions,
        top_y=top_y,
        bottom_y=bottom_y,
        left_pad=left_pad,
        right_pad=right_pad,
        bottom_extent=bottom_extent,
        top_extent=top_extent,
        title_y=title_y,
        figure_width=figure_width,
        figure_height=figure_height,
        row_label_x=row_label_x,
        left_terminal_x=left_terminal_x,
        right_terminal_x=right_terminal_x,
    )


def _draw_boundary(
    ax,
    *,
    x: float,
    y0: float,
    y1: float,
    label: str,
    color: str,
    dashed: bool = False,
    label_y: float | None = None,
    text_x: float | None = None,
    gid: str | None = None,
) -> None:
    line = ax.plot([x, x], [y0, y1], color=color, linewidth=1.2, zorder=3.2)[0]
    if dashed:
        line.set_dashes((2.5, 2.0))
    else:
        ax.plot([x - 0.08, x + 0.08], [y1, y1], color=color, linewidth=1.0, zorder=3.2)
    text = ax.text(
        x if text_x is None else text_x,
        label_y if label_y is not None else y1 + 0.12,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        family="DejaVu Sans",
        color=color,
        zorder=3.3,
    )
    if gid is not None:
        text.set_gid(gid)


def _draw_custom_sequence_row(
    ax,
    *,
    sequence: str,
    indices: list[int],
    x_positions: list[float],
    y: float,
    row_label: str,
    row_label_x: float,
    start_terminal: str,
    start_terminal_x: float,
    end_terminal: str,
    end_terminal_x: float,
    mismatch_indices: set[int],
    font_size: float,
    text_color: str,
    mismatch_color: str,
    terminal_color: str,
    row_tag: str,
) -> None:
    if len(indices) != len(x_positions):
        raise RenderingError("custom snapback row indices must match x_positions length")
    if not indices:
        raise RenderingError("custom snapback rows must include at least one base")
    ax.text(
        row_label_x,
        y,
        row_label,
        ha="right",
        va="center",
        fontsize=15,
        family="DejaVu Sans",
        color=text_color,
        zorder=3.0,
    )
    start_text = ax.text(
        start_terminal_x,
        y,
        start_terminal,
        ha="right",
        va="center",
        fontsize=13,
        family="DejaVu Sans",
        color=terminal_color,
        zorder=3.0,
    )
    start_text.set_gid(f"{row_tag}-start-terminal")
    end_text = ax.text(
        end_terminal_x,
        y,
        end_terminal,
        ha="left",
        va="center",
        fontsize=13,
        family="DejaVu Sans",
        color=terminal_color,
        zorder=3.0,
    )
    end_text.set_gid(f"{row_tag}-end-terminal")
    for index, x in zip(indices, x_positions, strict=True):
        color = mismatch_color if index in mismatch_indices else text_color
        base_text = ax.text(
            x,
            y,
            sequence[index],
            ha="center",
            va="center",
            fontsize=font_size,
            family="DejaVu Sans Mono",
            color=color,
            zorder=3.1,
        )
        base_text.set_gid(f"{row_tag}-base-{index}")


def _draw_loop_backbone(ax, *, layout: FoldbackCornerLayout, color: str) -> None:
    path = Path(
        [
            (layout.right_x + 0.12, layout.top_y),
            (layout.right_x + 0.34, layout.top_y - 0.02),
            (layout.cap_x_positions[0] - 0.06, layout.cap_y_positions[0] + 0.02),
            (layout.cap_x_positions[1] - 0.04, layout.cap_y_positions[1]),
            (layout.cap_x_positions[2] - 0.06, layout.cap_y_positions[2] - 0.02),
            (layout.right_x + 0.34, layout.bottom_y + 0.02),
            (layout.right_x + 0.12, layout.bottom_y),
        ],
        [
            Path.MOVETO,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
            Path.CURVE3,
        ],
    )
    patch = PathPatch(
        path,
        fill=False,
        edgecolor=color,
        linewidth=1.5,
        capstyle="round",
        joinstyle="round",
        zorder=1.2,
    )
    patch.set_gid("foldback-loop-backbone")
    ax.add_patch(patch)


def _draw_protected_rail(ax, *, contract: SnapbackVisualV1, layout: FoldbackCornerLayout, color: str) -> None:
    loop = contract.loop_geometry
    if contract.protected_region_span is None or loop is None:
        return
    protected_start = max(contract.protected_region_span.start, loop.display_primary_span.start)
    protected_end = min(contract.protected_region_span.end, loop.display_primary_span.end)
    if protected_end <= protected_start:
        return
    rail_x0 = protected_start - loop.display_primary_span.start + 0.12
    rail_x1 = protected_end - loop.display_primary_span.start - 0.12
    rail_y = layout.bottom_y - 0.72
    ax.plot([rail_x0, rail_x1], [rail_y, rail_y], color=color, linewidth=2.0, zorder=2.0)
    ax.plot([rail_x0, rail_x0], [rail_y - 0.05, rail_y + 0.05], color=color, linewidth=1.2, zorder=2.0)
    ax.plot([rail_x1, rail_x1], [rail_y - 0.05, rail_y + 0.05], color=color, linewidth=1.2, zorder=2.0)
    ax.text(
        (rail_x0 + rail_x1) / 2.0,
        rail_y - 0.16,
        "protected overlap",
        ha="center",
        va="top",
        fontsize=9.5,
        family="DejaVu Sans",
        color=color,
        zorder=2.1,
    )


def _draw_cap_partition_rail(
    ax,
    *,
    x_positions: list[float],
    y: float,
    label: str,
    color: str,
) -> None:
    if not x_positions:
        return
    x0 = min(x_positions) - 0.10
    x1 = max(x_positions) + 0.10
    ax.plot([x0, x1], [y, y], color=color, linewidth=1.8, zorder=2.0)
    ax.text(
        (x0 + x1) / 2.0,
        y - 0.14,
        label,
        ha="center",
        va="top",
        fontsize=9.5,
        family="DejaVu Sans",
        color=color,
        zorder=2.1,
    )


def _draw_cap_partition_rails(
    ax,
    *,
    contract: SnapbackVisualV1,
    layout: FoldbackCornerLayout,
    colors: Mapping[str, str],
) -> None:
    loop = contract.loop_geometry
    if loop is None:
        return
    cap_x_by_index = dict(zip(layout.cap_indices, layout.cap_x_positions, strict=True))
    source_cap_positions = [
        cap_x_by_index[index]
        for index in layout.cap_indices
        if loop.source_cap_span.start <= index < loop.source_cap_span.end
    ]
    cap_extension_positions = [
        cap_x_by_index[index]
        for index in layout.cap_indices
        if loop.cap_extension_span.start <= index < loop.cap_extension_span.end
    ]
    _draw_cap_partition_rail(
        ax,
        x_positions=source_cap_positions,
        y=layout.bottom_y - 0.24,
        label="source cap",
        color=colors["source_cap"],
    )
    _draw_cap_partition_rail(
        ax,
        x_positions=cap_extension_positions,
        y=layout.bottom_y - (0.44 if source_cap_positions else 0.24),
        label="extension",
        color=colors["cap_extension"],
    )


def render_foldback_corner_triloop(contract: SnapbackVisualV1, style: Style, *, colors: Mapping[str, str]):
    layout = _build_layout(contract, style)
    primary_x_by_index = dict(zip(layout.primary_indices, layout.stem_x_positions, strict=True))
    complement_x_by_index = dict(zip(layout.complement_indices, layout.stem_x_positions, strict=True))

    fig, ax = plt.subplots(figsize=(layout.figure_width, layout.figure_height), dpi=style.dpi)
    ax.set_axis_off()

    title = str(contract.title or "").strip()
    if title:
        ax.text(
            -layout.left_pad + 0.25,
            layout.title_y,
            title,
            ha="left",
            va="top",
            fontsize=17,
            family="DejaVu Sans",
            color=colors["title"],
            zorder=4.0,
        )

    for pair in contract.pairings:
        x_primary = primary_x_by_index.get(pair.left_index)
        x_complement = complement_x_by_index.get(pair.right_index)
        if x_primary is None or x_complement is None:
            raise RenderingError("foldback corner renderer received out-of-display pairing indices")
        ax.plot(
            [x_primary, x_complement],
            [layout.top_y - 0.15, layout.bottom_y + 0.15],
            color=colors["pair"],
            linewidth=1.1,
            alpha=0.95,
            zorder=1.0,
        )

    _draw_custom_sequence_row(
        ax,
        sequence=contract.primary_sequence,
        indices=layout.primary_indices,
        x_positions=layout.stem_x_positions,
        y=layout.top_y,
        row_label=contract.primary_row_label,
        row_label_x=layout.row_label_x,
        start_terminal="5'",
        start_terminal_x=layout.left_terminal_x,
        end_terminal="3'",
        end_terminal_x=layout.right_terminal_x,
        mismatch_indices=set(contract.primary_mismatch_positions),
        font_size=max(22.0, float(style.font_size_seq) * 1.55),
        text_color=colors["text"],
        mismatch_color=colors["mismatch"],
        terminal_color=colors["terminal"],
        row_tag="primary",
    )
    _draw_custom_sequence_row(
        ax,
        sequence=contract.complement_sequence,
        indices=layout.complement_indices,
        x_positions=layout.stem_x_positions,
        y=layout.bottom_y,
        row_label=contract.complement_row_label,
        row_label_x=layout.row_label_x,
        start_terminal="3'",
        start_terminal_x=layout.left_terminal_x,
        end_terminal="5'",
        end_terminal_x=layout.right_terminal_x,
        mismatch_indices=set(contract.complement_mismatch_positions),
        font_size=max(22.0, float(style.font_size_seq) * 1.55),
        text_color=colors["text"],
        mismatch_color=colors["mismatch"],
        terminal_color=colors["terminal"],
        row_tag="complement",
    )

    _draw_loop_backbone(ax, layout=layout, color=colors["loop_backbone"])
    loop = contract.loop_geometry
    assert loop is not None
    for index, x, y in zip(layout.cap_indices, layout.cap_x_positions, layout.cap_y_positions, strict=True):
        color = colors["source_cap"] if index < loop.source_cap_span.end else colors["cap_extension"]
        cap_text = ax.text(
            x,
            y,
            contract.primary_sequence[index],
            ha="center",
            va="center",
            fontsize=max(20.0, float(style.font_size_seq) * 1.44),
            family="DejaVu Sans Mono",
            color=color,
            zorder=3.2,
        )
        cap_text.set_gid(f"cap-base-{index}")
    _draw_cap_partition_rails(ax, contract=contract, layout=layout, colors=colors)

    _draw_boundary(
        ax,
        x=0.0,
        y0=layout.bottom_y - 0.18,
        y1=layout.top_y + 0.18,
        label="Origin",
        color=colors["boundary"],
        dashed=False,
        label_y=layout.top_y + 0.40,
        text_x=-0.12,
        gid="origin-label",
    )
    _draw_protected_rail(ax, contract=contract, layout=layout, color=colors["protected"])

    ax.set_xlim(-layout.left_pad, layout.right_x + layout.right_pad)
    ax.set_ylim(layout.bottom_extent, layout.title_y + 0.12)
    return fig
