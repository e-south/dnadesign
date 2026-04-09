"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/topology_cartoon.py

QA-oriented topology-cartoon renderer for YIU circular, branched, and retained
states.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle

from ..config import Style
from ..core import Record, RenderingError
from .palette import Palette

_CIRCULAR_TOPOLOGY_KINDS = {"circular_duplex", "circular_dsdna_candidate"}
_SUPPORTED_TOPOLOGY_KINDS = _CIRCULAR_TOPOLOGY_KINDS | {"branched_y", "fragment_pool"}


def _mapping_list(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _sorted_segments(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    segments = _mapping_list(payload.get("segments"))
    return sorted(
        segments,
        key=lambda item: (
            int(item.get("state_start", 0)),
            int(item.get("state_end", 0)),
            str(item.get("segment_id") or ""),
        ),
    )


def _segment_span(segment: Mapping[str, Any]) -> tuple[int, int]:
    start = int(segment.get("state_start", 0))
    end = int(segment.get("state_end", 0))
    return start, end


def _segment_color(segment: Mapping[str, Any], palette: Palette) -> tuple[float, float, float]:
    segment_id = str(segment.get("segment_id") or "segment")
    return palette.color_for(f"yiu:{segment_id}")


def _segment_label(segment: Mapping[str, Any]) -> str:
    return str(segment.get("segment_id") or "segment")


def _sequence_length(payload: Mapping[str, Any]) -> int:
    sequence = str(payload.get("sequence") or "")
    if sequence:
        return len(sequence)
    segments = _sorted_segments(payload)
    if not segments:
        return 1
    return max(int(segment.get("state_end", 0)) for segment in segments)


def _renderable_segments(payload: Mapping[str, Any], *, topology_kind: str) -> list[Mapping[str, Any]]:
    segments = _sorted_segments(payload)
    renderable_segments: list[Mapping[str, Any]] = []
    for segment in segments:
        start, end = _segment_span(segment)
        if end <= start:
            continue
        renderable_segments.append(segment)
    if not renderable_segments:
        raise RenderingError(
            f"topology_cartoon requires at least one positive-length segment for topology_kind '{topology_kind}'"
        )
    if topology_kind == "branched_y" and len(renderable_segments) != 3:
        raise RenderingError(
            "topology_cartoon requires exactly three positive-length segments for topology_kind 'branched_y'"
        )
    return renderable_segments


def _index_to_angle(index: int, total_length: int) -> float:
    if total_length <= 0:
        return -90.0
    return ((index / total_length) * 360.0) - 90.0


def _point_on_circle(center: tuple[float, float], radius: float, angle_deg: float) -> tuple[float, float]:
    angle_rad = math.radians(angle_deg)
    return (
        center[0] + math.cos(angle_rad) * radius,
        center[1] + math.sin(angle_rad) * radius,
    )


def _draw_legend(ax, *, entries: list[tuple[str, tuple[float, float, float]]], style: Style) -> None:
    x0 = 0.76
    y0 = 0.84
    ax.text(x0, y0 + 0.07, "Segments", ha="left", va="center", fontsize=style.font_size_label, weight="bold")
    for index, (label, color) in enumerate(entries):
        y = y0 - (index * 0.08)
        ax.add_line(Line2D([x0, x0 + 0.05], [y, y], linewidth=5.0, color=color, solid_capstyle="round"))
        ax.text(x0 + 0.07, y, label, ha="left", va="center", fontsize=style.font_size_label)


def _draw_circular_topology(
    ax,
    payload: Mapping[str, Any],
    *,
    segments: list[Mapping[str, Any]],
    style: Style,
    palette: Palette,
) -> None:
    center = (0.38, 0.52)
    radius = 0.24
    total_length = _sequence_length(payload)

    legend_entries: list[tuple[str, tuple[float, float, float]]] = []
    for segment in segments:
        start, end = _segment_span(segment)
        if end <= start:
            continue
        theta1 = _index_to_angle(start, total_length)
        theta2 = _index_to_angle(end, total_length)
        color = _segment_color(segment, palette)
        ax.add_patch(
            Arc(
                center,
                radius * 2.0,
                radius * 2.0,
                theta1=theta1,
                theta2=theta2,
                linewidth=10.0,
                color=color,
                capstyle="round",
            )
        )
        mid_angle = theta1 + ((theta2 - theta1) / 2.0)
        label_x, label_y = _point_on_circle(center, radius + 0.09, mid_angle)
        ax.text(label_x, label_y, _segment_label(segment), ha="center", va="center", fontsize=style.font_size_label)
        legend_entries.append((_segment_label(segment), color))

    ax.add_patch(Circle(center, radius - 0.07, fill=False, linewidth=1.0, color="#E5E7EB"))

    for junction in _mapping_list(payload.get("junctions")):
        join_index = int(junction.get("join_index", 0))
        angle = _index_to_angle(join_index, total_length)
        inner = _point_on_circle(center, radius - 0.045, angle)
        outer = _point_on_circle(center, radius + 0.045, angle)
        ax.add_line(Line2D([inner[0], outer[0]], [inner[1], outer[1]], linewidth=2.0, color="#111827"))
        ax.text(
            outer[0],
            outer[1] + 0.03,
            str(junction.get("id") or "junction"),
            ha="center",
            va="bottom",
            fontsize=style.font_size_label,
        )

    for cut in _mapping_list(payload.get("cuts")):
        for key in ("top_boundary", "bottom_boundary"):
            if key not in cut:
                continue
            angle = _index_to_angle(int(cut[key]), total_length)
            inner = _point_on_circle(center, radius - 0.025, angle)
            outer = _point_on_circle(center, radius + 0.065, angle)
            ax.add_line(Line2D([inner[0], outer[0]], [inner[1], outer[1]], linewidth=1.4, color="#B91C1C"))

    _draw_legend(ax, entries=legend_entries[:5], style=style)


def _draw_branched_y_topology(
    ax,
    payload: Mapping[str, Any],
    *,
    segments: list[Mapping[str, Any]],
    style: Style,
    palette: Palette,
) -> None:
    center = (0.38, 0.48)
    arm_specs = [
        (segments[0], (0.22, 0.80)),
        (segments[1], (0.54, 0.80)),
        (segments[2], (0.38, 0.16)),
    ]
    legend_entries: list[tuple[str, tuple[float, float, float]]] = []
    for segment, endpoint in arm_specs:
        color = _segment_color(segment, palette)
        ax.add_line(
            Line2D(
                [center[0], endpoint[0]],
                [center[1], endpoint[1]],
                linewidth=7.0,
                color=color,
                solid_capstyle="round",
            )
        )
        ax.text(
            endpoint[0],
            endpoint[1],
            _segment_label(segment),
            ha="center",
            va="center",
            fontsize=style.font_size_label,
        )
        legend_entries.append((_segment_label(segment), color))
    ax.add_patch(Circle(center, 0.018, color="#111827"))
    ax.text(center[0], center[1] + 0.05, "branch anchor", ha="center", va="bottom", fontsize=style.font_size_label)
    _draw_legend(ax, entries=legend_entries[:5], style=style)


def _draw_linear_topology(
    ax,
    payload: Mapping[str, Any],
    *,
    segments: list[Mapping[str, Any]],
    style: Style,
    palette: Palette,
) -> None:
    x_left = 0.10
    x_right = 0.66
    y = 0.52
    total_length = _sequence_length(payload)

    legend_entries: list[tuple[str, tuple[float, float, float]]] = []
    for segment in segments:
        start, end = _segment_span(segment)
        if end <= start:
            continue
        span_left = x_left + ((start / total_length) * (x_right - x_left))
        span_right = x_left + ((end / total_length) * (x_right - x_left))
        color = _segment_color(segment, palette)
        ax.add_line(Line2D([span_left, span_right], [y, y], linewidth=8.0, color=color, solid_capstyle="round"))
        ax.text(
            (span_left + span_right) / 2.0,
            y + 0.08,
            _segment_label(segment),
            ha="center",
            va="center",
            fontsize=style.font_size_label,
        )
        legend_entries.append((_segment_label(segment), color))

    for junction in _mapping_list(payload.get("junctions")):
        join_index = int(junction.get("join_index", 0))
        x = x_left + ((join_index / total_length) * (x_right - x_left))
        ax.add_line(Line2D([x, x], [y - 0.08, y + 0.08], linewidth=1.6, color="#111827"))
        ax.text(
            x,
            y - 0.12,
            str(junction.get("id") or "junction"),
            ha="center",
            va="top",
            fontsize=style.font_size_label,
        )

    fragment_text = ", ".join(str(fragment.get("length_nt")) for fragment in _mapping_list(payload.get("fragments")))
    if fragment_text:
        ax.text(
            x_left,
            y - 0.20,
            f"fragment lengths: {fragment_text}",
            ha="left",
            va="center",
            fontsize=style.font_size_label,
        )

    _draw_legend(ax, entries=legend_entries[:5], style=style)


@dataclass(frozen=True)
class TopologyCartoonRenderer:
    def render(self, record: Record, style: Style, palette: Palette):
        record = record.validate()
        payload = record.meta.get("topology_cartoon") if isinstance(record.meta, Mapping) else None
        if not isinstance(payload, Mapping):
            raise RenderingError("topology_cartoon requires record.meta.topology_cartoon")

        topology_kind = str(payload.get("topology_kind") or "")
        meta = payload.get("meta")
        evidence_mode = str(meta.get("evidence_mode") or "") if isinstance(meta, Mapping) else ""
        if topology_kind not in _SUPPORTED_TOPOLOGY_KINDS:
            raise RenderingError(
                "topology_cartoon does not support topology_kind "
                f"{topology_kind!r}; expected one of {sorted(_SUPPORTED_TOPOLOGY_KINDS)}"
            )
        segments = _renderable_segments(payload, topology_kind=topology_kind)
        fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=style.dpi)
        ax.set_axis_off()

        if topology_kind in _CIRCULAR_TOPOLOGY_KINDS:
            _draw_circular_topology(ax, payload, segments=segments, style=style, palette=palette)
        elif topology_kind == "branched_y":
            _draw_branched_y_topology(ax, payload, segments=segments, style=style, palette=palette)
        else:
            _draw_linear_topology(ax, payload, segments=segments, style=style, palette=palette)

        title = str(record.display.overlay_text or "")
        if title:
            ax.text(0.02, 0.98, title, ha="left", va="top", fontsize=style.font_size_label + 1, weight="bold")
        if evidence_mode:
            ax.text(
                0.02,
                0.06,
                f"evidence_mode: {evidence_mode}",
                ha="left",
                va="bottom",
                fontsize=style.font_size_label,
                color="#374151",
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return fig
