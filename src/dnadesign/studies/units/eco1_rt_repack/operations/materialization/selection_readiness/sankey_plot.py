"""Quantitative candidate-flow plot for Eco1 RT panel selection."""

from __future__ import annotations

from dataclasses import dataclass
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
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

_STAGE_IDS = ("candidate_pool", "local_geometry_screen", "design_groups", "selected_panel")
_STAGE_SHORT_LABELS = {
    "candidate_pool": "ProteinMPNN\ncomplete sequences",
    "local_geometry_screen": "ColabFold\nlocal RMSD review",
    "design_groups": "Generation\npolicy",
    "selected_panel": "Within-group\nJaccard selection",
}
_REJECTION_LABELS = {
    "local_geometry_screen": "Local geometry not retained",
    "design_groups": "",
    "selected_panel": "Not selected",
}


@dataclass(frozen=True)
class FlowStage:
    """One ordered candidate-flow stage."""

    stage_id: str
    label: str
    count: int
    fraction_of_pool: float


@dataclass(frozen=True)
class FlowTransition:
    """One count-conserving transition between adjacent stages."""

    source_stage_id: str
    target_stage_id: str
    source_count: int
    retained_count: int
    removed_count: int
    rejection_label: str


def quantitative_flow(
    trace_rows: list[dict[str, object]],
) -> tuple[tuple[FlowStage, ...], tuple[FlowTransition, ...]]:
    """Validate the public trace and return count-conserving plot data."""

    by_stage = {str(row["stage_id"]): row for row in trace_rows}
    missing = set(_STAGE_IDS) - set(by_stage)
    if missing:
        raise ValueError(f"Protein-hypothesis flow plot is missing trace stages: {', '.join(sorted(missing))}")
    initial_count = int(by_stage[_STAGE_IDS[0]]["remaining_count"])
    if initial_count <= 0:
        raise ValueError("Protein-hypothesis flow plot requires a non-empty candidate pool")
    stages = tuple(
        FlowStage(
            stage_id=stage_id,
            label=_STAGE_SHORT_LABELS[stage_id],
            count=int(by_stage[stage_id]["remaining_count"]),
            fraction_of_pool=int(by_stage[stage_id]["remaining_count"]) / initial_count,
        )
        for stage_id in _STAGE_IDS
    )
    transitions: list[FlowTransition] = []
    for source, target in zip(stages[:-1], stages[1:], strict=True):
        target_input = int(by_stage[target.stage_id]["input_count"])
        removed_count = int(by_stage[target.stage_id]["removed_count"])
        if target_input != source.count:
            raise ValueError(
                f"Selection trace is discontinuous at {target.stage_id}: "
                f"input_count={target_input}, previous remaining_count={source.count}"
            )
        if source.count != target.count + removed_count:
            raise ValueError(
                f"Selection trace does not conserve rows at {target.stage_id}: "
                f"{source.count} != {target.count} + {removed_count}"
            )
        transitions.append(
            FlowTransition(
                source_stage_id=source.stage_id,
                target_stage_id=target.stage_id,
                source_count=source.count,
                retained_count=target.count,
                removed_count=removed_count,
                rejection_label=_REJECTION_LABELS[target.stage_id],
            )
        )
    return stages, tuple(transitions)


def write_hypothesis_panel_flow_plot(
    plot_root: Path,
    *,
    hypothesis_panel_selection_trace_rows: list[dict[str, object]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    """Write the quantitative flow from generated sequences to the selected panel."""

    title = SELECTION_PLOT_PLAIN_TITLES["selection_hypothesis_panel_flow"]
    stages, _transitions = quantitative_flow(hypothesis_panel_selection_trace_rows)
    stage_by_id = {stage.stage_id: stage for stage in stages}
    row_by_id = {str(row["stage_id"]): row for row in hypothesis_panel_selection_trace_rows}
    initial_count = stage_by_id["candidate_pool"].count
    geometry_count = stage_by_id["local_geometry_screen"].count
    selected_count = stage_by_id["selected_panel"].count
    group_row = row_by_id["design_groups"]
    selected_row = row_by_id["selected_panel"]
    groups = (
        (
            "Distal scaffold",
            int(group_row["distal_pool_count"]),
            int(selected_row["distal_selected_count"]),
            OKABE_ITO["blue"],
        ),
        (
            "Peripheral shell",
            int(group_row["peripheral_pool_count"]),
            int(selected_row["peripheral_selected_count"]),
            OKABE_ITO["green"],
        ),
        (
            "Combined",
            int(group_row["combined_pool_count"]),
            int(selected_row["combined_selected_count"]),
            OKABE_ITO["vermillion"],
        ),
    )
    if sum(pool_count for _label, pool_count, _selected, _color in groups) != geometry_count:
        raise ValueError("Design-group counts do not sum to the local-geometry-pass count")
    if sum(selected for _label, _pool, selected, _color in groups) != selected_count:
        raise ValueError("Selected counts do not sum to the selected-panel count")

    fig, ax = plt.subplots(figsize=(12.4, 6.35))
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 1)
    ax.axis("off")
    node_width = 0.014
    flow_top = 0.78
    scale = 0.56 / initial_count
    accepted_x, geometry_x, group_x, selected_x = (0.055, 0.245, 0.505, 0.82)
    accepted_height = initial_count * scale
    geometry_height = geometry_count * scale

    _draw_filter_transition(
        ax,
        source_x=accepted_x,
        target_x=geometry_x,
        node_width=node_width,
        flow_top=flow_top,
        source_height=accepted_height,
        retained_height=geometry_height,
        removed_label=f"{initial_count - geometry_count:,} local-geometry\nfailures",
        reject_y=0.11,
        color=OKABE_ITO["blue"],
    )
    ax.add_patch(
        Rectangle(
            (accepted_x, flow_top - accepted_height), node_width, accepted_height, color=OKABE_ITO["blue"], zorder=4
        )
    )
    ax.add_patch(
        Rectangle(
            (geometry_x, flow_top - geometry_height), node_width, geometry_height, color=OKABE_ITO["sky"], zorder=4
        )
    )

    cursor_top = flow_top
    for label, pool_count, group_selected_count, color in groups:
        pool_height = pool_count * scale
        pool_bottom = cursor_top - pool_height
        _draw_band(
            ax,
            start_x=geometry_x + node_width,
            end_x=group_x,
            start_top=cursor_top,
            start_bottom=pool_bottom,
            end_top=cursor_top,
            end_bottom=pool_bottom,
            color=color,
            alpha=0.40,
            zorder=2,
        )
        ax.add_patch(Rectangle((group_x, pool_bottom), node_width, pool_height, color=color, zorder=4))
        selected_height = group_selected_count * scale
        visible_selected_height = max(selected_height, 0.014)
        _draw_band(
            ax,
            start_x=group_x + node_width,
            end_x=selected_x,
            start_top=cursor_top,
            start_bottom=cursor_top - selected_height,
            end_top=cursor_top,
            end_bottom=cursor_top - selected_height,
            color=color,
            alpha=0.76,
            zorder=3,
        )
        ax.add_patch(
            Rectangle(
                (selected_x, cursor_top - visible_selected_height),
                node_width,
                visible_selected_height,
                color=color,
                zorder=5,
            )
        )
        midpoint = (cursor_top + pool_bottom) / 2
        ax.text(0.415, midpoint, f"{label}\n{pool_count:,}", ha="center", va="center", fontsize=10.8)
        ax.text(
            0.68,
            midpoint,
            f"{pool_count - group_selected_count:,} not selected",
            ha="center",
            va="center",
            fontsize=9.8,
            color="#57606A",
        )
        ax.text(
            selected_x + 0.025,
            cursor_top - visible_selected_height / 2,
            f"{group_selected_count} selected",
            ha="left",
            va="center",
            fontsize=10.2,
        )
        cursor_top = pool_bottom

    headers = (
        (accepted_x, "ProteinMPNN\ncomplete sequences", initial_count),
        (geometry_x, "ColabFold\nlocal Cα RMSD ≤2.5 Å", geometry_count),
        (group_x, "Generation\npolicy", geometry_count),
        (selected_x, "Within-group Jaccard\nselected panel", selected_count),
    )
    for x_position, label, count in headers:
        ax.text(x_position + node_width / 2, 0.925, label, ha="center", va="center", fontsize=11.0, color="#24292F")
        ax.text(
            x_position + node_width / 2,
            0.855,
            f"{count:,}",
            ha="center",
            va="center",
            fontsize=12.2,
            weight="semibold",
            color="#24292F",
        )

    ax.set_title(title, fontsize=TITLE_SIZE, pad=12)
    path = plot_root / "selection_hypothesis_panel_flow.svg"
    group_summary = ", ".join(
        f"{label}: {pool_count:,} passing and {selected} selected" for label, pool_count, selected, _color in groups
    )
    alt = (
        f"Candidate flow from {initial_count:,} complete sequences through {geometry_count:,} sequences retaining "
        f"local geometry to a {selected_count}-sequence panel. {group_summary}. Ribbon widths are proportional "
        "to sequence count."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return plot_row(
        plot_id="selection_hypothesis_panel_flow",
        title=title,
        path=path,
        input_hashes=input_hashes,
        alt_text=alt,
        description=(
            "Quantifies complete ProteinMPNN sequences, the ColabFold local-RMSD review, the three generation "
            "policies, and within-policy mutation-set selection."
        ),
        interpretation_limit=(
            "The flow records computational screening and experimental design. It does not measure RT activity, "
            "affinity, processivity, or strand displacement."
        ),
        render_mode="wide_visual",
    )


def _draw_filter_transition(
    ax: plt.Axes,
    *,
    source_x: float,
    target_x: float,
    node_width: float,
    flow_top: float,
    source_height: float,
    retained_height: float,
    removed_label: str,
    reject_y: float,
    color: str,
) -> None:
    """Draw one count-conserving filter and its rejection branch."""

    _draw_band(
        ax,
        start_x=source_x + node_width,
        end_x=target_x,
        start_top=flow_top,
        start_bottom=flow_top - retained_height,
        end_top=flow_top,
        end_bottom=flow_top - retained_height,
        color=color,
        alpha=0.42,
        zorder=2,
    )
    removed_height = source_height - retained_height
    if removed_height <= 0:
        return
    reject_x = source_x + 0.62 * (target_x - source_x)
    _draw_band(
        ax,
        start_x=source_x + node_width,
        end_x=reject_x,
        start_top=flow_top - retained_height,
        start_bottom=flow_top - source_height,
        end_top=reject_y + removed_height,
        end_bottom=reject_y,
        color="#AEB7C2",
        alpha=0.58,
        zorder=1,
    )
    ax.add_patch(Rectangle((reject_x, reject_y), node_width * 0.72, removed_height, color="#7D8793", zorder=3))
    ax.text(
        reject_x + node_width * 0.36,
        reject_y - 0.025,
        removed_label,
        ha="center",
        va="top",
        fontsize=9.6,
        color="#4B5563",
    )


def _draw_band(
    ax: plt.Axes,
    *,
    start_x: float,
    end_x: float,
    start_top: float,
    start_bottom: float,
    end_top: float,
    end_bottom: float,
    color: str,
    alpha: float,
    zorder: int,
) -> None:
    control = 0.46 * (end_x - start_x)
    path = MplPath(
        [
            (start_x, start_top),
            (start_x + control, start_top),
            (end_x - control, end_top),
            (end_x, end_top),
            (end_x, end_bottom),
            (end_x - control, end_bottom),
            (start_x + control, start_bottom),
            (start_x, start_bottom),
            (start_x, start_top),
        ],
        [
            MplPath.MOVETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ],
    )
    ax.add_patch(PathPatch(path, facecolor=color, edgecolor="none", alpha=alpha, zorder=zorder))
