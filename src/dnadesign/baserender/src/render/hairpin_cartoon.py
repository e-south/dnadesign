"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/render/hairpin_cartoon.py

Folded ssDNA hairpin renderer for topology-first cassette QA views.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle

from ..config import Style
from ..core import Record, RenderingError
from .palette import Palette


def _feature_color(feature, palette: Palette):
    token = str(feature.attrs.get("style_token", "")).strip()
    if token:
        return palette.color_for(token)
    if feature.tags:
        return palette.color_for(feature.tags[0])
    return palette.color_for(feature.kind)


@dataclass(frozen=True)
class HairpinCartoonRenderer:
    def render(self, record: Record, style: Style, palette: Palette):
        record = record.validate()
        topology = record.meta.get("hairpin_topology") if isinstance(record.meta, Mapping) else None
        if not isinstance(topology, Mapping):
            raise RenderingError("hairpin_cartoon requires record.meta.hairpin_topology")

        stem5p = topology.get("stem5p_span")
        loop = topology.get("loop_span")
        stem3p = topology.get("stem3p_span")
        if not all(isinstance(item, Mapping) for item in (stem5p, loop, stem3p)):
            raise RenderingError("hairpin_cartoon topology spans must be mappings")

        fig, ax = plt.subplots(figsize=(max(6.0, len(record.sequence) * 0.22), 3.2), dpi=style.dpi)
        ax.set_axis_off()

        stem_y = 0.55
        stem_bottom_y = -0.55
        sequence_x = list(range(len(record.sequence)))

        for feature in record.features:
            if feature.kind != "interval_annotation":
                continue
            color = _feature_color(feature, palette)
            semantic = str(feature.attrs.get("semantic", "")).strip().lower()
            alpha = 0.30 if semantic in {"stem5p_arm", "stem3p_arm", "loop"} else 0.55
            if semantic == "stem3p_arm":
                y = stem_bottom_y - 0.22
            else:
                y = stem_y - 0.22
            if semantic == "loop":
                y = stem_y + 0.65
            rect = Rectangle(
                (feature.span.start, y),
                feature.span.end - feature.span.start,
                0.26,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                zorder=1.0,
            )
            ax.add_patch(rect)
            if feature.label and semantic in {"loop", "motif_projection"}:
                ax.text(
                    feature.span.start + (feature.span.end - feature.span.start) / 2.0,
                    y + 0.35,
                    feature.label,
                    ha="center",
                    va="bottom",
                    fontsize=max(8, style.font_size_label - 1),
                    color="#111827",
                    zorder=4.0,
                )

        ax.plot(
            [stem5p["start"], stem5p["end"]],
            [stem_y, stem_y],
            color="#111827",
            linewidth=2.2,
            zorder=2.0,
        )
        ax.plot(
            [stem3p["start"], stem3p["end"]],
            [stem_bottom_y, stem_bottom_y],
            color="#111827",
            linewidth=2.2,
            zorder=2.0,
        )
        loop_center_x = (loop["start"] + loop["end"]) / 2.0
        loop_width = max(1.6, float(loop["end"] - loop["start"]) + 1.0)
        arc = Arc(
            (loop_center_x, 0.05),
            width=loop_width,
            height=1.4,
            theta1=0,
            theta2=180,
            linewidth=2.2,
            color="#111827",
            zorder=2.0,
        )
        ax.add_patch(arc)

        if bool(style.show_pair_rungs):
            pair_effect = next((effect for effect in record.effects if effect.kind == "pair_map"), None)
            if pair_effect is None:
                raise RenderingError("hairpin_cartoon requires a pair_map effect")
            pairs = pair_effect.target.get("pairs")
            if not isinstance(pairs, list):
                raise RenderingError("pair_map effect requires target.pairs list")
            for pair in pairs:
                if not isinstance(pair, Mapping):
                    raise RenderingError("pair_map pairs must be mappings")
                left_x = float(pair["left_index"]) + 0.5
                right_x = float(pair["right_index"]) + 0.5
                ax.plot([left_x, right_x], [stem_y, stem_bottom_y], color="#9ca3af", linewidth=1.0, zorder=1.5)

        if bool(style.show_base_text) and len(record.sequence) <= 40:
            for idx, base in enumerate(record.sequence):
                y = stem_y + 0.12 if idx < int(loop["start"]) else stem_bottom_y - 0.18
                if int(loop["start"]) <= idx < int(loop["end"]):
                    y = 1.10
                ax.text(
                    idx + 0.5,
                    y,
                    base,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#111827",
                    zorder=3.5,
                )

        if bool(style.show_loop_label):
            ax.text(
                loop_center_x,
                1.05,
                "Loop",
                ha="center",
                va="bottom",
                fontsize=style.font_size_label,
                color="#111827",
                zorder=4.0,
            )

        ax.text(
            float(stem5p["start"]) - 0.4,
            stem_y,
            "5'",
            ha="right",
            va="center",
            fontsize=style.font_size_label,
            color="#111827",
        )
        ax.text(
            float(stem3p["end"]) + 0.4,
            stem_bottom_y,
            "3'",
            ha="left",
            va="center",
            fontsize=style.font_size_label,
            color="#111827",
        )

        notes = record.meta.get("hairpin_notes") if isinstance(record.meta, Mapping) else None
        if isinstance(notes, list) and notes:
            note_text = "\n".join(str(item.get("text", "")) for item in notes if isinstance(item, Mapping))
            if note_text.strip():
                ax.text(
                    0.0,
                    -1.35,
                    note_text,
                    ha="left",
                    va="top",
                    fontsize=max(8, style.font_size_label - 2),
                    color="#4b5563",
                    zorder=4.0,
                )

        if record.display.overlay_text:
            ax.text(
                0.0,
                1.55,
                record.display.overlay_text,
                ha="left",
                va="bottom",
                fontsize=style.font_size_label,
                color="#111827",
                zorder=4.5,
            )

        ax.set_xlim(-0.5, max(sequence_x) + 1.5)
        ax.set_ylim(-1.6, 1.8)
        return fig
