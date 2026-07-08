"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/selection_readiness/sankey_plot.py

Primary-panel selection flow plot for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plot_support import (
    plot_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.visual_inventory import (
    SELECTION_PLOT_PLAIN_TITLES,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import (
    OKABE_ITO,
    TITLE_SIZE,
    save_accessible_svg,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402


def write_primary_panel_sankey_plot(
    plot_root: Path,
    *,
    primary_panel_selection_trace_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    """Write the compact primary-panel selection funnel plot."""

    title = SELECTION_PLOT_PLAIN_TITLES["selection_primary_panel_sankey"]
    if not primary_panel_selection_trace_rows:
        raise ValueError("primary-panel Sankey plot requires selection trace rows")
    by_stage = {str(row["stage_id"]): row for row in primary_panel_selection_trace_rows}
    required = {
        "candidate_pool",
        "preservation_gate",
        "chemistry_support_gate",
        "global_conservative_diverse_selection",
    }
    missing = required - set(by_stage)
    if missing:
        raise ValueError(f"Primary-panel Sankey plot is missing trace stages: {', '.join(sorted(missing))}")
    counts = {stage_id: int(by_stage[stage_id]["remaining_count"]) for stage_id in required}
    other_primary = max(counts["chemistry_support_gate"] - counts["global_conservative_diverse_selection"], 0)
    max_flow = max(counts.values())
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = {
        "candidate_pool": (0.04, 0.47, "Accepted\ncandidates", counts["candidate_pool"], OKABE_ITO["blue"]),
        "preservation_gate": (
            0.28,
            0.47,
            "Preservation\ngate",
            counts["preservation_gate"],
            OKABE_ITO["green"],
        ),
        "chemistry_support_gate": (
            0.53,
            0.52,
            "Chemistry and\nsupport gate",
            counts["chemistry_support_gate"],
            OKABE_ITO["sky"],
        ),
        "global_conservative_diverse_selection": (
            0.79,
            0.66,
            "Selected primary\npanel",
            counts["global_conservative_diverse_selection"],
            OKABE_ITO["orange"],
        ),
        "other_primary": (
            0.79,
            0.41,
            "Other primary\ncandidates",
            other_primary,
            "#c9d1d9",
        ),
    }
    _draw_flow(
        ax,
        start=(0.22, 0.55),
        end=(0.28, 0.55),
        count=counts["preservation_gate"],
        max_count=max_flow,
        color=OKABE_ITO["green"],
    )
    _draw_flow(
        ax,
        start=(0.46, 0.62),
        end=(0.53, 0.6),
        count=counts["chemistry_support_gate"],
        max_count=max_flow,
        color=OKABE_ITO["sky"],
    )
    _draw_flow(
        ax,
        start=(0.71, 0.74),
        end=(0.79, 0.74),
        count=counts["global_conservative_diverse_selection"],
        max_count=max_flow,
        color=OKABE_ITO["orange"],
    )
    _draw_flow(
        ax,
        start=(0.71, 0.66),
        end=(0.79, 0.49),
        count=other_primary,
        max_count=max_flow,
        color="#c9d1d9",
    )
    for x, y, label, count, color in nodes.values():
        _draw_sankey_node(ax, x=x, y=y, label=label, count=count, color=color)
    ax.text(
        0.04,
        0.89,
        (
            f"{counts['candidate_pool']} accepted -> {counts['preservation_gate']} preservation-pass rows -> "
            f"{counts['chemistry_support_gate']} chemistry/support-pass rows -> "
            f"{counts['global_conservative_diverse_selection']} selected"
        ),
        ha="left",
        va="center",
        fontsize=10.8,
        color="#57606a",
    )
    ax.text(
        0.04,
        0.08,
        "The final step is a global conservative-diverse selection, not a design-class quota.",
        ha="left",
        va="center",
        fontsize=10.5,
        color="#57606a",
    )
    ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
    path = plot_root / "selection_primary_panel_sankey.svg"
    alt = (
        "Sankey-style flow showing accepted Eco1 RT candidates, preservation-pass rows, chemistry/support-pass "
        "rows, and the selected conservative-diverse primary-panel rows."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_primary_panel_sankey",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Shows how the selector moves from accepted candidates through the preservation and chemistry/support "
            "gates, then to the final conservative-diverse selected panel."
        ),
        interpretation_limit=(
            "The flow is a protein-level selection record. It does not measure RT activity, processivity, or strand "
            "displacement."
        ),
        render_mode="wide_visual",
    )


def _draw_flow(
    ax: plt.Axes,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    count: int,
    max_count: int,
    color: str,
) -> None:
    if count <= 0:
        return
    width = 4.0 + 30.0 * (count / max(max_count, 1)) ** 0.5
    control_dx = max((end[0] - start[0]) * 0.55, 0.02)
    path = MplPath(
        [
            start,
            (start[0] + control_dx, start[1]),
            (end[0] - control_dx, end[1]),
            end,
        ],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(
        PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            lw=width,
            alpha=0.34,
            capstyle="round",
            zorder=1,
        )
    )


def _draw_sankey_node(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    label: str,
    count: int,
    color: str,
) -> None:
    width = 0.18
    height = 0.14
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.016",
            linewidth=0.8,
            edgecolor="#d0d7de",
            facecolor="#ffffff",
            zorder=4,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            0.018,
            height,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            linewidth=0,
            facecolor=color,
            alpha=0.95,
            zorder=5,
        )
    )
    ax.text(x + 0.03, y + 0.088, label, ha="left", va="center", fontsize=10.2, color="#24292f", zorder=6)
    ax.text(x + 0.03, y + 0.038, str(count), ha="left", va="center", fontsize=13.0, weight="bold", zorder=6)
