"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/structural_screen.py

Landscape structural-screen figure for scientific communication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.constants import (
    STRONG_FOLD_MAX_WT_RUNTIME_CA_RMSD_ANGSTROM,
    STRONG_FOLD_MIN_MEAN_PLDDT,
    STRONG_FOLD_REVIEW_CLASS,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_DESIGNS_AND_FOLD_TRIAGE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.shared.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    TICK_SIZE,
    save_accessible_svg,
    style_open_axes,
)

from .catalog import COMMUNICATION_ROLE, STRUCTURAL_SCREEN_ID
from .style import POLICY_ORDER, policy_color, policy_label

_FILE_NAME = "structural_screen.svg"
_LOCAL_GEOMETRY_CUTOFF_ANGSTROM = 2.5
_ANGSTROM = "\N{LATIN CAPITAL LETTER A WITH RING ABOVE}"
_C_ALPHA = "C\N{GREEK SMALL LETTER ALPHA}"


def write_structural_screen(
    *,
    panel_root: Path,
    triage_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    triage_table_path: Path,
    selection_panel_path: Path,
) -> dict[str, Any]:
    """Plot model confidence against the local-geometry metric used by the screen."""

    path = panel_root / _FILE_NAME
    plotted_rows = [
        row
        for row in triage_rows
        if row.get("local_structure_max_gated_ca_rmsd_angstrom") is not None and row.get("mean_plddt") is not None
    ]
    if not plotted_rows:
        raise ValueError("Communication structural screen requires local RMSD and pLDDT values")
    selected_ids = {str(row.get("candidate_id") or "") for row in selected_rows}
    policy_rows_by_id = {
        policy_id: [row for row in plotted_rows if str(row.get("policy_id") or "") == policy_id]
        for policy_id in POLICY_ORDER
    }
    x_values = np.asarray(
        [float(row["local_structure_max_gated_ca_rmsd_angstrom"]) for row in plotted_rows],
        dtype=float,
    )
    y_values = np.asarray([float(row["mean_plddt"]) for row in plotted_rows], dtype=float)
    x_padding = max(float(np.ptp(x_values)) * 0.04, 0.04)
    y_padding = max(float(np.ptp(y_values)) * 0.05, 0.12)
    x_limits = (
        float(np.min(x_values)) - x_padding,
        max(float(np.max(x_values)) + x_padding, _LOCAL_GEOMETRY_CUTOFF_ANGSTROM + 0.12),
    )
    y_limits = (
        min(float(np.min(y_values)) - y_padding, STRONG_FOLD_MIN_MEAN_PLDDT - 0.12),
        max(float(np.max(y_values)) + y_padding, STRONG_FOLD_MIN_MEAN_PLDDT + 0.12),
    )

    fig = plt.figure(figsize=(9.6, 9.6))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(5.0, 1.25),
        height_ratios=(1.25, 5.0),
        hspace=0.04,
        wspace=0.04,
    )
    top_ax = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=top_ax)
    right_ax = fig.add_subplot(grid[1, 1], sharey=ax)
    legend_ax = fig.add_subplot(grid[0, 1])
    legend_ax.axis("off")

    histogram_x_values = []
    histogram_y_values = []
    histogram_colors = []
    for policy_id in POLICY_ORDER:
        policy_rows = policy_rows_by_id[policy_id]
        if not policy_rows:
            continue
        policy_x_values = [float(row["local_structure_max_gated_ca_rmsd_angstrom"]) for row in policy_rows]
        policy_y_values = [float(row["mean_plddt"]) for row in policy_rows]
        ax.scatter(
            policy_x_values,
            policy_y_values,
            s=34,
            alpha=0.52,
            color=policy_color(policy_id),
            edgecolors="none",
        )
        histogram_x_values.append(policy_x_values)
        histogram_y_values.append(policy_y_values)
        histogram_colors.append(policy_color(policy_id))

    x_bins = np.linspace(x_limits[0], x_limits[1], 22)
    y_bins = np.linspace(y_limits[0], y_limits[1], 22)
    top_ax.hist(
        histogram_x_values,
        bins=x_bins,
        color=histogram_colors,
        histtype="stepfilled",
        alpha=0.42,
        linewidth=1.0,
    )
    right_ax.hist(
        histogram_y_values,
        bins=y_bins,
        color=histogram_colors,
        histtype="stepfilled",
        orientation="horizontal",
        alpha=0.42,
        linewidth=1.0,
    )
    selected_plotted = [row for row in plotted_rows if str(row.get("candidate_id") or "") in selected_ids]
    if selected_plotted:
        ax.scatter(
            [float(row["local_structure_max_gated_ca_rmsd_angstrom"]) for row in selected_plotted],
            [float(row["mean_plddt"]) for row in selected_plotted],
            s=92,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.45,
            zorder=5,
        )
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.axvspan(_LOCAL_GEOMETRY_CUTOFF_ANGSTROM, x_limits[1], color="#6E7781", alpha=0.08)
    ax.axhspan(y_limits[0], STRONG_FOLD_MIN_MEAN_PLDDT, color="#6E7781", alpha=0.05)
    ax.axvline(
        _LOCAL_GEOMETRY_CUTOFF_ANGSTROM,
        color="#57606A",
        linestyle="--",
        linewidth=1.25,
    )
    ax.axhline(
        STRONG_FOLD_MIN_MEAN_PLDDT,
        color="#57606A",
        linestyle="--",
        linewidth=1.25,
    )
    ax.set_xlabel(f"Maximum local {_C_ALPHA} RMSD ({_ANGSTROM})", fontsize=LABEL_SIZE + 2.0)
    ax.set_ylabel("Mean ColabFold pLDDT", fontsize=LABEL_SIZE + 1.7)
    style_open_axes(ax)

    top_ax.set_ylabel("Sequence count", fontsize=LABEL_SIZE + 1.2)
    top_ax.tick_params(axis="x", bottom=False, labelbottom=False)
    top_ax.tick_params(axis="y", labelsize=TICK_SIZE + 1.0)
    top_ax.spines[["top", "right"]].set_visible(False)
    top_ax.grid(False)
    right_ax.set_xlabel("Sequence count", fontsize=LABEL_SIZE + 1.2)
    right_ax.tick_params(axis="y", left=False, labelleft=False)
    right_ax.tick_params(axis="x", labelsize=TICK_SIZE + 1.0)
    right_ax.spines[["top", "right"]].set_visible(False)
    right_ax.grid(False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=policy_color(policy_id),
            markeredgecolor="none",
            markersize=8,
            label=policy_label(policy_id),
        )
        for policy_id in POLICY_ORDER
        if policy_rows_by_id[policy_id]
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#111111",
            markeredgewidth=1.3,
            markersize=8,
            label="Selected panel",
        )
    )
    fig.suptitle("Predicted confidence and local geometry define the structural screen", fontsize=19, y=0.985)
    fig.text(
        0.51,
        0.944,
        (
            f"Active gates: mean pLDDT ≥ {STRONG_FOLD_MIN_MEAN_PLDDT:.1f}; "
            f"WT-runtime {_C_ALPHA} RMSD ≤ {STRONG_FOLD_MAX_WT_RUNTIME_CA_RMSD_ANGSTROM:.2f} {_ANGSTROM}; "
            f"maximum local {_C_ALPHA} RMSD ≤ {_LOCAL_GEOMETRY_CUTOFF_ANGSTROM:.1f} {_ANGSTROM}"
        ),
        ha="center",
        va="center",
        fontsize=LEGEND_SIZE + 0.6,
        color="#3F464D",
    )
    legend_ax.legend(
        handles=handles,
        frameon=False,
        fontsize=LEGEND_SIZE + 1.5,
        title="Design group",
        title_fontsize=LABEL_SIZE + 1.3,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(0.02, 0.48),
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.10, top=0.90)
    _assert_marginal_axes_aligned(main_ax=ax, top_ax=top_ax, right_ax=right_ax)

    local_pass_count = sum(
        1
        for row in plotted_rows
        if float(row["local_structure_max_gated_ca_rmsd_angstrom"]) <= _LOCAL_GEOMETRY_CUTOFF_ANGSTROM
    )
    strong_count = sum(1 for row in plotted_rows if str(row.get("fold_review_class") or "") == STRONG_FOLD_REVIEW_CLASS)
    joint_pass_count = sum(
        1
        for row in plotted_rows
        if str(row.get("fold_review_class") or "") == STRONG_FOLD_REVIEW_CLASS
        and float(row["local_structure_max_gated_ca_rmsd_angstrom"]) <= _LOCAL_GEOMETRY_CUTOFF_ANGSTROM
    )
    alt_text = (
        f"Landscape scatter plot of {len(plotted_rows)} complete ProteinMPNN sequences. The x-axis is maximum "
        "local C-alpha RMSD across protected review regions and the y-axis is mean ColabFold pLDDT. Marginal "
        "histograms show each distribution by generation policy. Dashed lines mark the active mean-pLDDT threshold "
        f"of {STRONG_FOLD_MIN_MEAN_PLDDT:.1f} and maximum-local-RMSD threshold of "
        f"{_LOCAL_GEOMETRY_CUTOFF_ANGSTROM:.1f} A. The strong fold class also requires WT-runtime C-alpha RMSD at "
        f"or below {STRONG_FOLD_MAX_WT_RUNTIME_CA_RMSD_ANGSTROM:.2f} A. {strong_count} sequences meet the strong "
        f"fold definition, {local_pass_count} meet the local-geometry threshold, and {joint_pass_count} meet both. "
        "Selected panel rows have black outlines."
    )
    save_accessible_svg(
        fig,
        path,
        title="Predicted confidence and local geometry define the structural screen",
        description=alt_text,
    )
    return make_deliverable_row(
        deliverable_id=STRUCTURAL_SCREEN_ID,
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=[
            "generation_policies_v3/selection/candidate_triage_table.parquet",
            "generation_policies_v3/selection/candidate_selection_panel.parquet",
        ],
        input_hashes=file_hashes(
            {"candidate_triage_table": triage_table_path, "candidate_selection_panel": selection_panel_path}
        ),
        alt_text=alt_text,
        description=(
            "Shows the active mean-pLDDT and local-RMSD thresholds. The strong fold class also requires WT-runtime "
            "C-alpha RMSD at or below 1.25 A, which is recorded but not plotted on the local-RMSD x-axis. Marginal "
            "histograms expose policy-specific distributions without covering the scatter."
        ),
        interpretation_limit=(
            "pLDDT and local RMSD assess predicted structural plausibility. They do not measure RT activity, "
            "processivity, or strand displacement."
        ),
        title="Predicted confidence and local geometry summarize the structural screen",
        role=COMMUNICATION_ROLE,
        render_mode="standard_visual",
        method_summary=(
            "Each ColabFold model is aligned once to the mapped reference. Eligibility requires the strong fold "
            "class (mean pLDDT at least 91.5 and WT-runtime C-alpha RMSD at most 1.25 A) and maximum residual local "
            "C-alpha RMSD at most 2.5 A across non-distal review regions."
        ),
        evidence_summary={
            "candidate_count": len(plotted_rows),
            "strong_fold_count": strong_count,
            "local_geometry_pass_count": local_pass_count,
            "joint_structural_pass_count": joint_pass_count,
            "selected_panel_count": len(selected_plotted),
        },
    )


def _assert_marginal_axes_aligned(*, main_ax: Any, top_ax: Any, right_ax: Any, tolerance: float = 1e-8) -> None:
    """Fail if marginal distributions no longer share the main plot bounds."""

    main_bounds = main_ax.get_position()
    top_bounds = top_ax.get_position()
    right_bounds = right_ax.get_position()
    differences = (
        abs(main_bounds.x0 - top_bounds.x0),
        abs(main_bounds.x1 - top_bounds.x1),
        abs(main_bounds.y0 - right_bounds.y0),
        abs(main_bounds.y1 - right_bounds.y1),
    )
    if any(value > tolerance for value in differences):
        raise RuntimeError(f"Structural-screen marginal axes are misaligned: {differences}")
