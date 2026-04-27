"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/snapback_map.py

Snapback-specific nucleotide-resolution renderer for Cruncher QA views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dnadesign.contracts.visual import SnapbackVisualV1

from ..config import Style
from ..core import Record, RenderingError
from .palette import Palette
from .snapback_foldback import render_foldback_corner_triloop

_TITLE_COLOR = "#475569"
_TEXT_COLOR = "#334155"
_DIM_COLOR = "#CBD5E1"
_MISMATCH_COLOR = "#B91C1C"
_PAIR_COLOR = "#94A3B8"
_BOUNDARY_COLOR = "#0F172A"

_RELEASED_COLOR = "#64748B"
_STEM_COLOR = "#DC2626"
_CAP_COLOR = "#DB2777"
_SOURCE_CAP_COLOR = "#BE185D"
_CAP_EXTENSION_COLOR = "#EC4899"
_FOLDBACK_COLOR = "#7C3AED"
_WINDOW_COLOR = "#94A3B8"
_SITE_COLOR = "#D97706"
_PROTECTED_COLOR = "#2563EB"
_ANCHORED_COLOR = "#059669"
_EXPOSED_COLOR = "#D97706"
_FOLDBACK_RENDER_COLORS = {
    "boundary": _BOUNDARY_COLOR,
    "cap_extension": _CAP_EXTENSION_COLOR,
    "loop_backbone": "#CBD5E1",
    "mismatch": _MISMATCH_COLOR,
    "pair": _PAIR_COLOR,
    "protected": _PROTECTED_COLOR,
    "source_cap": _SOURCE_CAP_COLOR,
    "terminal": "#475569",
    "text": _TEXT_COLOR,
    "title": _TITLE_COLOR,
}


@dataclass(frozen=True)
class _Rail:
    label: str
    start: int
    end: int
    color: str


def _contract_from_record(record: Record) -> SnapbackVisualV1:
    meta = record.meta if isinstance(record.meta, Mapping) else None
    if meta is None:
        raise RenderingError("snapback_map requires record.meta.contract")
    raw = meta.get("contract")
    if not isinstance(raw, Mapping):
        raise RenderingError("snapback_map requires record.meta.contract")
    try:
        return SnapbackVisualV1.model_validate(raw)
    except Exception as exc:
        raise RenderingError(f"snapback_map received invalid snapback contract: {exc}") from exc


def _assign_tiers(rails: Iterable[_Rail]) -> list[tuple[_Rail, int]]:
    placements: list[tuple[_Rail, int]] = []
    occupied: list[list[tuple[float, float]]] = []
    for rail in sorted(rails, key=lambda item: (item.start, item.end, item.label)):
        center = (rail.start + rail.end) / 2.0
        label_half_width = max(0.8, len(rail.label) * 0.28)
        left = center - label_half_width
        right = center + label_half_width
        tier = len(occupied)
        for index, spans in enumerate(occupied):
            if all(right <= other_left - 0.4 or left >= other_right + 0.4 for other_left, other_right in spans):
                tier = index
                spans.append((left, right))
                break
        else:
            occupied.append([(left, right)])
        placements.append((rail, tier))
    return placements


def _structural_rails(contract: SnapbackVisualV1) -> list[_Rail]:
    rails: list[_Rail] = []
    if (
        contract.released_prefix_span is not None
        and contract.released_prefix_span.end > contract.released_prefix_span.start
    ):
        rails.append(
            _Rail(
                label="released",
                start=contract.released_prefix_span.start,
                end=contract.released_prefix_span.end,
                color=_RELEASED_COLOR,
            )
        )
    rails.append(
        _Rail(
            label="stem",
            start=contract.retained_stem_span.start,
            end=contract.retained_stem_span.end,
            color=_STEM_COLOR,
        )
    )
    if (
        contract.released_suffix_span is not None
        and contract.released_suffix_span.end > contract.released_suffix_span.start
    ):
        rails.append(
            _Rail(
                label="suffix",
                start=contract.released_suffix_span.start,
                end=contract.released_suffix_span.end,
                color=_RELEASED_COLOR,
            )
        )
    if contract.cap_span is not None and contract.cap_span.end > contract.cap_span.start:
        rails.append(
            _Rail(
                label="cap",
                start=contract.cap_span.start,
                end=contract.cap_span.end,
                color=_CAP_COLOR,
            )
        )
    rails.append(
        _Rail(
            label="foldback",
            start=contract.foldback_revcomp_span.start,
            end=contract.foldback_revcomp_span.end,
            color=_FOLDBACK_COLOR,
        )
    )
    return rails


def _context_rails(contract: SnapbackVisualV1) -> list[_Rail]:
    rails: list[_Rail] = []
    if contract.state_kind == "pre_nick_duplex":
        if contract.pre_nick_duplex_window_span is not None:
            rails.append(
                _Rail(
                    label="window",
                    start=contract.pre_nick_duplex_window_span.start,
                    end=contract.pre_nick_duplex_window_span.end,
                    color=_WINDOW_COLOR,
                )
            )
        if contract.intended_site_span is not None:
            rails.append(
                _Rail(
                    label="site",
                    start=contract.intended_site_span.start,
                    end=contract.intended_site_span.end,
                    color=_SITE_COLOR,
                )
            )
        if contract.protected_region_span is not None:
            rails.append(
                _Rail(
                    label="protected",
                    start=contract.protected_region_span.start,
                    end=contract.protected_region_span.end,
                    color=_PROTECTED_COLOR,
                )
            )
    elif contract.state_kind == "post_nick_exposed":
        if (
            contract.anchored_duplex_span is not None
            and contract.anchored_duplex_span.end > contract.anchored_duplex_span.start
        ):
            rails.append(
                _Rail(
                    label="anchored",
                    start=contract.anchored_duplex_span.start,
                    end=contract.anchored_duplex_span.end,
                    color=_ANCHORED_COLOR,
                )
            )
        if (
            contract.exposed_complement_span is not None
            and contract.exposed_complement_span.end > contract.exposed_complement_span.start
        ):
            rails.append(
                _Rail(
                    label="exposed",
                    start=contract.exposed_complement_span.start,
                    end=contract.exposed_complement_span.end,
                    color=_EXPOSED_COLOR,
                )
            )
        if contract.protected_region_span is not None:
            rails.append(
                _Rail(
                    label="protected",
                    start=contract.protected_region_span.start,
                    end=contract.protected_region_span.end,
                    color=_PROTECTED_COLOR,
                )
            )
    elif contract.state_kind == "post_nick_foldback" and contract.protected_region_span is not None:
        rails.append(
            _Rail(
                label="protected",
                start=contract.protected_region_span.start,
                end=contract.protected_region_span.end,
                color=_PROTECTED_COLOR,
            )
        )
    return rails


def _draw_rail(ax, rail: _Rail, *, y: float, label_y: float, color: str, font_size: float) -> None:
    x0 = rail.start + 0.08
    x1 = rail.end - 0.08
    if x1 <= x0:
        center = (rail.start + rail.end) / 2.0
        x0 = center - 0.12
        x1 = center + 0.12
    ax.plot([x0, x1], [y, y], color=color, linewidth=2.2, solid_capstyle="round", zorder=2.0)
    ax.plot([x0, x0], [y - 0.05, y + 0.05], color=color, linewidth=1.4, zorder=2.0)
    ax.plot([x1, x1], [y - 0.05, y + 0.05], color=color, linewidth=1.4, zorder=2.0)
    ax.text(
        (rail.start + rail.end) / 2.0,
        label_y,
        rail.label,
        ha="center",
        va="center",
        fontsize=font_size,
        family="DejaVu Sans",
        color=color,
        zorder=2.1,
    )


def _draw_boundary(
    ax,
    *,
    x: float,
    y0: float,
    y1: float,
    label: str,
    dashed: bool = False,
    label_y: float | None = None,
) -> None:
    line = ax.plot([x, x], [y0, y1], color=_BOUNDARY_COLOR, linewidth=1.2, zorder=3.2)[0]
    if dashed:
        line.set_dashes((2.5, 2.0))
    else:
        ax.plot([x - 0.08, x + 0.08], [y1, y1], color=_BOUNDARY_COLOR, linewidth=1.0, zorder=3.2)
    ax.text(
        x,
        label_y if label_y is not None else y1 + 0.12,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        family="DejaVu Sans",
        color=_BOUNDARY_COLOR,
        zorder=3.3,
    )


def _draw_connectors(ax, contract: SnapbackVisualV1, *, primary_y: float, complement_y: float) -> None:
    if contract.state_kind == "pre_nick_duplex":
        paired_indices = range(len(contract.primary_sequence))
        for index in paired_indices:
            x = index + 0.5
            ax.plot([x, x], [complement_y + 0.12, primary_y - 0.12], color="#E2E8F0", linewidth=0.8, zorder=1.0)
        return
    if contract.state_kind == "post_nick_exposed":
        if contract.anchored_duplex_span is None:
            return
        for index in range(contract.anchored_duplex_span.start, contract.anchored_duplex_span.end):
            x = index + 0.5
            ax.plot([x, x], [complement_y + 0.12, primary_y - 0.12], color="#CBD5E1", linewidth=0.9, zorder=1.0)
        return
    for pair in contract.pairings:
        ax.plot(
            [pair.left_index + 0.5, pair.right_index + 0.5],
            [primary_y - 0.14, complement_y + 0.14],
            color=_PAIR_COLOR,
            linewidth=1.0,
            alpha=0.9,
            zorder=1.0,
        )


def _dim_sets(contract: SnapbackVisualV1) -> tuple[set[int], set[int]]:
    if contract.state_kind == "post_nick_exposed":
        return (
            set(range(contract.nick_boundary or 0, len(contract.primary_sequence))),
            set(),
        )
    if contract.state_kind == "post_nick_foldback":
        left = {pair.left_index for pair in contract.pairings}
        right = {pair.right_index for pair in contract.pairings}
        return (
            {index for index in range(len(contract.primary_sequence)) if index not in left},
            {index for index in range(len(contract.complement_sequence)) if index not in right},
        )
    return set(), set()


def _draw_sequence_row(
    ax,
    *,
    sequence: str,
    y: float,
    row_label: str,
    left_label_x: float,
    start_terminal: str,
    end_terminal: str,
    dim_indices: set[int],
    mismatch_indices: set[int],
    font_size: float,
) -> None:
    ax.text(
        left_label_x,
        y,
        row_label,
        ha="right",
        va="center",
        fontsize=16,
        family="DejaVu Sans",
        color=_TEXT_COLOR,
        zorder=3.0,
    )
    ax.text(
        0.06,
        y,
        start_terminal,
        ha="right",
        va="center",
        fontsize=13,
        family="DejaVu Sans",
        color="#475569",
    )
    ax.text(
        len(sequence) + 0.94,
        y,
        end_terminal,
        ha="left",
        va="center",
        fontsize=13,
        family="DejaVu Sans",
        color="#475569",
    )
    for index, base in enumerate(sequence):
        color = _MISMATCH_COLOR if index in mismatch_indices else (_DIM_COLOR if index in dim_indices else _TEXT_COLOR)
        ax.text(
            index + 0.5,
            y,
            base,
            ha="center",
            va="center",
            fontsize=font_size,
            family="DejaVu Sans Mono",
            color=color,
            zorder=3.1,
        )


@dataclass(frozen=True)
class SnapbackMapRenderer:
    def render(self, record: Record, style: Style, palette: Palette):
        _ = palette
        record = record.validate()
        contract = _contract_from_record(record)

        if contract.state_kind == "post_nick_foldback" and contract.loop_geometry is not None:
            return render_foldback_corner_triloop(contract, style, colors=_FOLDBACK_RENDER_COLORS)

        primary_y = 1.45
        complement_y = 0.45
        if contract.state_kind == "post_nick_exposed":
            top_rails = _assign_tiers(_context_rails(contract))
            bottom_rails = _assign_tiers(_structural_rails(contract))
        else:
            top_rails = _assign_tiers(_structural_rails(contract))
            bottom_rails = _assign_tiers(_context_rails(contract))
        top_step = 0.34
        bottom_step = 0.30
        top_base = primary_y + 0.55
        bottom_base = complement_y - 0.55
        top_extent = top_base + (len(top_rails) - 1) * top_step if top_rails else top_base - 0.18
        bottom_extent = bottom_base - (len(bottom_rails) - 1) * bottom_step if bottom_rails else bottom_base + 0.18
        title = str(contract.title or "").strip()
        title_y = top_extent + 0.52

        left_pad = 7.6
        right_pad = 1.6
        width_nt = len(contract.primary_sequence) + left_pad + right_pad
        figure_width = max(8.0, width_nt * 0.36 * float(style.figure_scale))
        figure_height = max(3.6, (title_y - bottom_extent + 0.55) * 0.95 * float(style.figure_scale))
        fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=style.dpi)
        ax.set_axis_off()

        if title:
            ax.text(
                -left_pad + 0.25,
                title_y,
                title,
                ha="left",
                va="top",
                fontsize=17,
                family="DejaVu Sans",
                color=_TITLE_COLOR,
                zorder=4.0,
            )

        _draw_connectors(ax, contract, primary_y=primary_y, complement_y=complement_y)

        primary_dim, complement_dim = _dim_sets(contract)
        _draw_sequence_row(
            ax,
            sequence=contract.primary_sequence,
            y=primary_y,
            row_label=contract.primary_row_label,
            left_label_x=-1.0,
            start_terminal="5'",
            end_terminal="3'",
            dim_indices=primary_dim,
            mismatch_indices=set(contract.primary_mismatch_positions),
            font_size=max(22.0, float(style.font_size_seq) * 1.55),
        )
        _draw_sequence_row(
            ax,
            sequence=contract.complement_sequence,
            y=complement_y,
            row_label=contract.complement_row_label,
            left_label_x=-1.0,
            start_terminal="3'",
            end_terminal="5'",
            dim_indices=complement_dim,
            mismatch_indices=set(contract.complement_mismatch_positions),
            font_size=max(22.0, float(style.font_size_seq) * 1.55),
        )

        boundary_y0 = (
            complement_y - 0.18 if contract.state_kind in {"pre_nick_duplex", "post_nick_exposed"} else primary_y - 0.18
        )
        boundary_y1 = primary_y + 0.18
        nick_label_y = primary_y + 0.28
        if contract.nick_boundary is not None and contract.nick_boundary == contract.ligation_junction_boundary:
            _draw_boundary(
                ax,
                x=contract.nick_boundary,
                y0=boundary_y0,
                y1=boundary_y1,
                label="Nick / origin",
                dashed=False,
                label_y=nick_label_y,
            )
        else:
            if contract.nick_boundary is not None:
                _draw_boundary(
                    ax,
                    x=contract.nick_boundary,
                    y0=boundary_y0,
                    y1=boundary_y1,
                    label="Nick",
                    dashed=False,
                    label_y=nick_label_y,
                )
            if contract.state_kind == "post_nick_foldback":
                _draw_boundary(
                    ax,
                    x=contract.ligation_junction_boundary,
                    y0=primary_y - 0.18,
                    y1=primary_y + 0.18,
                    label="Origin",
                    dashed=True,
                    label_y=primary_y + 0.30,
                )
            else:
                _draw_boundary(
                    ax,
                    x=contract.ligation_junction_boundary,
                    y0=boundary_y0,
                    y1=boundary_y1,
                    label="Origin",
                    dashed=True,
                    label_y=primary_y + 0.30,
                )

        for rail, tier in top_rails:
            rail_y = top_base + tier * top_step
            _draw_rail(
                ax,
                rail,
                y=rail_y,
                label_y=rail_y + 0.14,
                color=rail.color,
                font_size=10.5,
            )
        for rail, tier in bottom_rails:
            rail_y = bottom_base - tier * bottom_step
            _draw_rail(
                ax,
                rail,
                y=rail_y,
                label_y=rail_y - 0.16,
                color=rail.color,
                font_size=10.0,
            )

        ax.set_xlim(-left_pad, len(contract.primary_sequence) + right_pad)
        ax.set_ylim(bottom_extent - 0.36, title_y + 0.18)
        return fig
