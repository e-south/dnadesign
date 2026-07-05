"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/proteinmpnn_fold_validation.py

Expanded fold-validation panel for Eco1 ProteinMPNN candidates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_DESIGNS_AND_FOLD_TRIAGE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    TITLE_SIZE,
    save_accessible_svg,
    style_open_axes,
)

from .proteinmpnn_fold_validation_support import (
    add_expanded_fold_legend,
    annotate_selected_rows,
    design_class_color,
    design_class_order,
    histogram_bins,
    join_expanded_fold_rows,
    temperature_marker,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_expanded_design_class_fold_validation(
    *,
    panel_root: Path,
    candidate_pool_path: Path,
    foldcheck_ranking_path: Path,
    selection_panel_table_path: Path,
) -> dict[str, Any]:
    """Render the expanded design-class ProteinMPNN fold-validation panel."""

    title = "Expanded designs preserve the RT fold"
    source_tables = [
        "design_classes/candidate_pool.parquet",
        "design_classes/foldcheck_review/foldcheck_candidate_ranking.parquet",
        "design_classes/selection/candidate_selection_panel.parquet",
    ]
    missing_inputs = [
        path for path in (candidate_pool_path, foldcheck_ranking_path, selection_panel_table_path) if not path.exists()
    ]
    if missing_inputs:
        return make_deliverable_row(
            deliverable_id="expanded_proteinmpnn_fold_validation",
            section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
            artifact_kind="svg",
            status="skipped_missing_input",
            path=panel_root / "expanded_proteinmpnn_fold_validation.svg",
            source_tables=source_tables,
            input_hashes=file_hashes(
                {
                    "candidate_pool": candidate_pool_path,
                    "foldcheck_candidate_ranking": foldcheck_ranking_path,
                    "selection_panel": selection_panel_table_path,
                }
            ),
            alt_text="Expanded design-class fold-validation plot was not rendered because an input table is missing.",
            description=(
                "The expanded design-class fold plot is skipped until candidate pool, fold metrics, "
                "and panel selection tables exist."
            ),
            interpretation_limit="Missing fold-review metrics cannot support structural triage.",
            title=title,
            skip_reason="Missing input table(s): " + ", ".join(str(path) for path in missing_inputs),
        )
    joined_rows = join_expanded_fold_rows(
        candidate_pool_path=candidate_pool_path,
        foldcheck_ranking_path=foldcheck_ranking_path,
        selection_panel_table_path=selection_panel_table_path,
    )
    if not joined_rows:
        raise ValueError("No expanded candidate rows could be joined to design-class fold-review rows")

    rmsd_values = np.array([row["wt_runtime_ca_rmsd"] for row in joined_rows], dtype=float)
    plddt_values = np.array([row["plddt"] for row in joined_rows], dtype=float)
    bins_x = histogram_bins(rmsd_values)
    bins_y = histogram_bins(plddt_values)
    class_order = design_class_order(joined_rows)
    temperatures = sorted({row["temperature"] for row in joined_rows})

    fig = plt.figure(figsize=(8.6, 8.6))
    ax, ax_hist_x, ax_hist_y = _add_joint_fold_axes(fig)

    for class_id in class_order:
        color = design_class_color(class_id)
        class_rows = [row for row in joined_rows if row["design_class_id"] == class_id]
        ax_hist_x.hist(
            [row["wt_runtime_ca_rmsd"] for row in class_rows],
            bins=bins_x,
            color=color,
            alpha=0.48,
            edgecolor="#ffffff",
            linewidth=0.35,
        )
        ax_hist_y.hist(
            [row["plddt"] for row in class_rows],
            bins=bins_y,
            orientation="horizontal",
            color=color,
            alpha=0.48,
            edgecolor="#ffffff",
            linewidth=0.35,
        )
        for temperature in temperatures:
            rows_for_temp = [row for row in class_rows if row["temperature"] == temperature]
            if not rows_for_temp:
                continue
            ax.scatter(
                [row["wt_runtime_ca_rmsd"] for row in rows_for_temp],
                [row["plddt"] for row in rows_for_temp],
                marker=temperature_marker(temperature, temperatures),
                c=color,
                s=46,
                edgecolors="#ffffff",
                linewidths=0.35,
                alpha=0.88,
            )

    selected_rows = [row for row in joined_rows if row["selected_for_panel"]]
    if selected_rows:
        ax.scatter(
            [row["wt_runtime_ca_rmsd"] for row in selected_rows],
            [row["plddt"] for row in selected_rows],
            marker="o",
            s=135,
            facecolors="none",
            edgecolors="#111111",
            linewidths=1.45,
            zorder=5,
        )
        annotate_selected_rows(ax, selected_rows)

    ax.set_xlabel("WT-runtime C-alpha RMSD (A)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Mean pLDDT", fontsize=LABEL_SIZE)
    ax_hist_x.set_ylabel("Count", fontsize=LABEL_SIZE)
    ax_hist_y.set_xlabel("Count", fontsize=LABEL_SIZE)
    ax_hist_x.set_ylim(bottom=0)
    ax_hist_y.set_xlim(left=0)
    ax_hist_x.tick_params(labelbottom=False, labelsize=LEGEND_SIZE)
    ax_hist_y.tick_params(labelleft=False, labelsize=LEGEND_SIZE)
    for plot_ax in (ax, ax_hist_x, ax_hist_y):
        style_open_axes(plot_ax)
    ax_hist_x.spines["bottom"].set_visible(True)
    ax_hist_y.spines["left"].set_visible(True)
    fig.suptitle(title, fontsize=TITLE_SIZE, y=0.972)
    add_expanded_fold_legend(fig, class_order=class_order, temperatures=temperatures)

    path = panel_root / "expanded_proteinmpnn_fold_validation.svg"
    alt = (
        f"Expanded design-class fold-validation plot for {len(joined_rows)} ProteinMPNN candidates. "
        "Color encodes design class, marker shape encodes sampling temperature, and selected panel rows "
        "are outlined."
    )
    save_accessible_svg(fig, path, title=title, description=alt, dpi=300)
    return make_deliverable_row(
        deliverable_id="expanded_proteinmpnn_fold_validation",
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=source_tables,
        input_hashes=file_hashes(
            {
                "candidate_pool": candidate_pool_path,
                "foldcheck_candidate_ranking": foldcheck_ranking_path,
                "selection_panel": selection_panel_table_path,
            }
        ),
        alt_text=alt,
        description=(
            "Shows the expanded ProteinMPNN candidate pool after ColabFold fold checks, with design class as "
            "color, sampling temperature as marker shape, and selected panel candidates outlined."
        ),
        interpretation_limit=(
            "RMSD and pLDDT are fold-review filters. They do not measure activity, processivity, strand "
            "displacement, or downstream construct acceptance."
        ),
        title=title,
        evidence_summary={
            "candidate_count": len(joined_rows),
            "selected_panel_candidate_count": len(selected_rows),
            "design_class_count": len(class_order),
        },
    )


def _add_joint_fold_axes(fig: Any) -> tuple[Any, Any, Any]:
    main_left = 0.145
    main_bottom = 0.285
    main_size = 0.51
    gap = 0.025
    marginal_height = 0.135
    side_width = 0.17
    ax = fig.add_axes([main_left, main_bottom, main_size, main_size])
    ax_hist_x = fig.add_axes(
        [main_left, main_bottom + main_size + gap, main_size, marginal_height],
        sharex=ax,
    )
    ax_hist_y = fig.add_axes(
        [main_left + main_size + gap, main_bottom, side_width, main_size],
        sharey=ax,
    )
    return ax, ax_hist_x, ax_hist_y
