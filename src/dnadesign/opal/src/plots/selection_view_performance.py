"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/selection_view_performance.py

Renders observed objective distributions by selection view and round.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.selection_views.performance import selection_view_performance

_COLORS = ("#2A6F97", "#2F7F74", "#D97757", "#6F5B7E", "#767676")
_VIEW_LABEL_WIDTH = 26


def _require_single_line_header(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"selection-view performance {field} must be non-empty when provided")
    if "\n" in normalized or "\r" in normalized:
        raise ValueError(f"selection-view performance {field} must be one line")
    return normalized


def _display_view_label(view_id: str, labels: Mapping[str, str]) -> str:
    value = " ".join(str(labels.get(view_id, view_id)).split())
    if not value:
        raise ValueError(f"selection-view performance view label for {view_id!r} must be non-empty")
    return value


def _wrapped_view_label(view_id: str, labels: Mapping[str, str], *, prefix: str = "") -> str:
    return textwrap.fill(
        f"{prefix}{_display_view_label(view_id, labels)}",
        width=_VIEW_LABEL_WIDTH,
        break_long_words=True,
        break_on_hyphens=False,
    )


def _header_layout(*, round_count: int, title: str | None, subtitle: str | None) -> dict[str, float]:
    header_height = 2.0 if subtitle else (1.15 if title else 0.85)
    bottom_height = 1.35
    figure_height = header_height + bottom_height + 5.3 * round_count
    return {
        "figure_height": figure_height,
        "top": 1.0 - header_height / figure_height,
        "bottom": bottom_height / figure_height,
        "title_y": 1.0 - 0.25 / figure_height,
        "subtitle_y": 1.0 - 0.85 / figure_height,
    }


def _keep_y_tick_labels_inside_canvas(fig: object, axes: np.ndarray) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left = min(label.get_window_extent(renderer=renderer).x0 for axis in axes.flat for label in axis.get_yticklabels())
    padding = 8.0
    if left < padding:
        correction = (padding - left) / fig.bbox.width
        fig.subplots_adjust(left=min(fig.subplotpars.left + correction, 0.42))


def render_selection_view_performance(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    objective_value_label: str,
    title: str | None = None,
    subtitle: str | None = None,
    view_labels: Mapping[str, str] | None = None,
) -> None:
    """Render candidate points and medians within each objective-view panel."""

    performance = selection_view_performance(frame)
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() not in {".png", ".svg"}:
        raise ValueError("selection-view performance output must be PNG or SVG")
    if not objective_value_label.strip():
        raise ValueError("selection-view performance objective-value label must be non-empty")
    title = _require_single_line_header(title, field="title")
    subtitle = _require_single_line_header(subtitle, field="subtitle")
    if subtitle and not title:
        raise ValueError("selection-view performance subtitle requires a non-empty title")
    labels = dict(view_labels or {})
    rounds = sorted(performance.observations["observed_round"].unique().tolist())
    objectives = sorted(performance.observations["objective_view_id"].unique().tolist())
    selected_views = sorted(performance.observations["selected_for_view_id"].unique().tolist())

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    style = matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 19,
            "axes.titlesize": 21,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "path",
            "svg.hashsalt": "opal-selection-view-performance-v1",
        }
    )
    with style:
        layout = _header_layout(round_count=len(rounds), title=title, subtitle=subtitle)
        figure_width = 6.4 * len(objectives)
        fig, axes = plt.subplots(
            len(rounds),
            len(objectives),
            figsize=(figure_width, layout["figure_height"]),
            squeeze=False,
            sharey=True,
        )
        for row_index, observed_round in enumerate(rounds):
            for column_index, objective_view in enumerate(objectives):
                ax = axes[row_index, column_index]
                panel = performance.observations.loc[
                    performance.observations["observed_round"].eq(observed_round)
                    & performance.observations["objective_view_id"].eq(objective_view)
                ]
                for position, selected_view in enumerate(selected_views):
                    cohort = panel.loc[panel["selected_for_view_id"].eq(selected_view)].sort_values(
                        "candidate_id", kind="stable"
                    )
                    offsets = np.array([0.0]) if len(cohort) == 1 else np.linspace(-0.14, 0.14, len(cohort))
                    ax.scatter(
                        cohort["objective_value"],
                        np.full(len(cohort), position, dtype=float) + offsets,
                        s=70,
                        facecolor="white",
                        edgecolor=_COLORS[position % len(_COLORS)],
                        linewidth=1.8,
                        zorder=3,
                    )
                    ax.vlines(
                        float(cohort["objective_value"].median()),
                        position - 0.24,
                        position + 0.24,
                        color=_COLORS[position % len(_COLORS)],
                        linewidth=4.0,
                        zorder=4,
                    )
                ax.axvline(0.0, color="#999999", linewidth=1.2, linestyle="--", zorder=1)
                ax.grid(axis="x", color="#E6E6E6", linewidth=1.0)
                ax.set_axisbelow(True)
                ax.set_box_aspect(1)
                ax.set_anchor("N")
                ax.set_yticks(range(len(selected_views)))
                ax.set_yticklabels([_wrapped_view_label(view, labels, prefix="Selected: ") for view in selected_views])
                ax.set_xlabel(objective_value_label)
                round_label = f"Round {observed_round}"
                objective_label = _wrapped_view_label(objective_view, labels)
                ax.set_title(f"{objective_label} objective\n{round_label}", pad=18)
        if title:
            fig.suptitle(title, fontsize=25, fontweight="bold", x=0.5, y=layout["title_y"], ha="center")
        if subtitle:
            fig.text(
                0.5,
                layout["subtitle_y"],
                subtitle,
                ha="center",
                va="center",
                fontsize=16.5,
                color="#5E6A73",
                gid="figure-subtitle",
            )
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                markersize=9,
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="#666666",
            ),
            Line2D([], [], color="#666666", linewidth=4.0),
        ]
        fig.legend(
            handles,
            ["Measured candidate", "Cohort median"],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=2,
            frameon=False,
            fontsize=16,
        )
        left = min(2.0 / figure_width, 0.24)
        fig.subplots_adjust(
            left=left,
            right=1.0 - 0.5 / figure_width,
            top=layout["top"],
            bottom=layout["bottom"],
            wspace=0.18,
            hspace=0.40,
        )
        _keep_y_tick_labels_inside_canvas(fig, axes)
        fig.align_titles()
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"Date": None} if output.suffix.lower() == ".svg" else {"Software": "dnadesign.opal"}
        fig.savefig(output, dpi=240, facecolor="white", metadata=metadata)
        if output.suffix.lower() == ".svg":
            text = output.read_text(encoding="utf-8")
            output.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")
        plt.close(fig)


__all__ = ["render_selection_view_performance"]
