"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/src/render/effects/boundary_marker.py

Boundary-marker drawer for strand-local nick/cut annotations.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ...core import Effect, RenderingError


def draw_boundary_marker(ax, effect: Effect, record, layout, style, palette, feature_boxes) -> None:
    _ = (record, palette, feature_boxes)
    boundary = effect.target.get("boundary")
    lane = str(effect.target.get("lane", "")).strip().lower()
    if not isinstance(boundary, int):
        raise RenderingError("boundary_marker.target.boundary must be int")
    if boundary > len(record.sequence):
        raise RenderingError("boundary_marker.target.boundary must be within sequence boundaries")
    if lane not in {"primary", "complement"}:
        raise RenderingError("boundary_marker.target.lane must be primary|complement")
    x = layout.x_left + boundary * layout.cw
    y = layout.y_forward if lane == "primary" else layout.y_reverse
    height = max(12.0, layout.ch * 0.8)
    ax.plot([x, x], [y - height / 2.0, y + height / 2.0], color="#111827", linewidth=1.6, zorder=6.0)
    ax.plot([x - 2.0, x + 2.0], [y + height / 2.0, y + height / 2.0], color="#111827", linewidth=1.2, zorder=6.0)
    label = effect.params.get("label")
    if label:
        ax.text(
            x,
            y + height / 2.0 + 6.0,
            str(label),
            ha="center",
            va="bottom",
            fontsize=max(8, style.font_size_label - 1),
            color="#111827",
            zorder=6.1,
        )
