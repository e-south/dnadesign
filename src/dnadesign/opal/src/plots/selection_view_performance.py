"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/plots/selection_view_performance.py

Renders observed objective distributions by selection view and round.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from ..analysis.selection_views.performance import selection_view_performance

_COLORS = ("#2A6F97", "#2F7F74", "#D97757", "#6F5B7E", "#767676")
_ROUND_LABELS = {0: "Initial observations", 1: "First selected round"}


def render_selection_view_performance(
    frame: pd.DataFrame,
    *,
    output_path: Path,
    objective_value_label: str,
    title: str | None = None,
    view_labels: Mapping[str, str] | None = None,
) -> None:
    """Render candidate points and medians within each objective-view panel."""

    performance = selection_view_performance(frame)
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() not in {".png", ".svg"}:
        raise ValueError("selection-view performance output must be PNG or SVG")
    if not objective_value_label.strip():
        raise ValueError("selection-view performance objective-value label must be non-empty")
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
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 19,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "path",
            "svg.hashsalt": "opal-selection-view-performance-v1",
        }
    )
    with style:
        fig, axes = plt.subplots(
            len(rounds),
            len(objectives),
            figsize=(6.2 * len(objectives), 7.2 * len(rounds)),
            squeeze=False,
            sharey=True,
            layout="constrained",
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
                ax.set_yticks(range(len(selected_views)))
                ax.set_yticklabels([f"Selected: {labels.get(view, view)}" for view in selected_views])
                ax.set_xlabel(objective_value_label)
                round_label = _ROUND_LABELS.get(int(observed_round), f"Round {observed_round}")
                ax.set_title(f"{labels.get(objective_view, objective_view)} objective\n{round_label}", pad=14)
        if title:
            fig.suptitle(title, fontsize=23, fontweight="bold", x=0.5)
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
            ["Measured promoter", "Cohort median"],
            loc="outside lower center",
            ncol=2,
            frameon=False,
            fontsize=15,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"Date": None} if output.suffix.lower() == ".svg" else {"Software": "dnadesign.opal"}
        fig.savefig(output, dpi=240, facecolor="white", metadata=metadata)
        if output.suffix.lower() == ".svg":
            text = output.read_text(encoding="utf-8")
            output.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")
        plt.close(fig)


__all__ = ["render_selection_view_performance"]
