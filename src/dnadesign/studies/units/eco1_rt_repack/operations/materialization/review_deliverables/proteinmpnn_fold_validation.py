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
import pyarrow.parquet as pq

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.design_classes.specs import ALL_SPECS
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
    OKABE_ITO,
    TITLE_SIZE,
    save_accessible_svg,
    style_open_axes,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.selection_readiness.plot_support import (
    class_label,
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

_REVIEW_CLASS_ORDER = (
    "strong_fold_preserved",
    "good_fold_preserved",
    "review_band",
    "low_confidence",
    "structural_outlier",
    "metric_missing",
)
_REVIEW_CLASS_COLORS = {
    "strong_fold_preserved": OKABE_ITO["green"],
    "good_fold_preserved": OKABE_ITO["sky"],
    "review_band": OKABE_ITO["blue"],
    "low_confidence": OKABE_ITO["yellow"],
    "structural_outlier": OKABE_ITO["vermillion"],
    "metric_missing": OKABE_ITO["gray"],
}


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


def write_design_class_fold_bin_counts(
    *,
    panel_root: Path,
    candidate_pool_path: Path,
    foldcheck_ranking_path: Path,
) -> dict[str, Any]:
    """Render fold-review class counts separately for each fixed-mask design class."""

    title = "Each fixed mask keeps foldable candidates"
    source_tables = [
        "design_classes/candidate_pool.parquet",
        "design_classes/foldcheck_review/foldcheck_candidate_ranking.parquet",
    ]
    missing_inputs = [path for path in (candidate_pool_path, foldcheck_ranking_path) if not path.exists()]
    path = panel_root / "design_class_fold_bin_counts.svg"
    if missing_inputs:
        return make_deliverable_row(
            deliverable_id="foldcheck_review_review_class_counts",
            section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
            artifact_kind="svg",
            status="skipped_missing_input",
            path=path,
            source_tables=source_tables,
            input_hashes=file_hashes(
                {
                    "candidate_pool": candidate_pool_path,
                    "foldcheck_candidate_ranking": foldcheck_ranking_path,
                }
            ),
            alt_text="Design-class fold-bin plot was not rendered because an input table is missing.",
            description="Fold-bin counts are skipped until the expanded candidate pool and foldcheck table exist.",
            interpretation_limit="Missing fold-review classes cannot support structural triage.",
            title=title,
            skip_reason="Missing input table(s): " + ", ".join(str(input_path) for input_path in missing_inputs),
        )

    joined_rows = _join_design_class_review_rows(
        candidate_pool_path=candidate_pool_path,
        foldcheck_ranking_path=foldcheck_ranking_path,
    )
    if not joined_rows:
        raise ValueError("No expanded candidate rows could be joined to design-class fold-review rows")

    class_ids = [spec.design_class_id for spec in ALL_SPECS]
    counts_by_class = {class_id: {review_class: 0 for review_class in _REVIEW_CLASS_ORDER} for class_id in class_ids}
    for row in joined_rows:
        class_id = str(row["design_class_id"])
        review_class = str(row.get("review_class") or "metric_missing")
        if class_id not in counts_by_class:
            counts_by_class[class_id] = {review_class: 0 for review_class in _REVIEW_CLASS_ORDER}
            class_ids.append(class_id)
        if review_class not in counts_by_class[class_id]:
            counts_by_class[class_id][review_class] = 0
        counts_by_class[class_id][review_class] += 1

    max_count = max((count for counts in counts_by_class.values() for count in counts.values()), default=0)
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 9.2), sharex=True, sharey=True)
    axes_list = list(axes.flatten())
    y_positions = np.arange(len(_REVIEW_CLASS_ORDER))
    for axis_index, (ax, class_id) in enumerate(zip(axes_list, class_ids, strict=False)):
        counts = counts_by_class[class_id]
        values = [counts.get(review_class, 0) for review_class in _REVIEW_CLASS_ORDER]
        bars = ax.barh(
            y_positions,
            values,
            color=[_REVIEW_CLASS_COLORS[review_class] for review_class in _REVIEW_CLASS_ORDER],
            edgecolor="#ffffff",
            linewidth=0.45,
        )
        ax.bar_label(bars, labels=[str(value) if value else "" for value in values], padding=2, fontsize=8.5)
        ax.set_title(class_label(class_id), fontsize=LEGEND_SIZE, pad=5)
        ax.set_xlim(0, max(1, max_count) * 1.14)
        ax.set_yticks(
            y_positions,
            [_review_class_axis_label(review_class) for review_class in _REVIEW_CLASS_ORDER],
            fontsize=8.8,
        )
        ax.invert_yaxis()
        style_open_axes(ax)
        ax.grid(axis="x", alpha=0.24)
        ax.grid(axis="y", visible=False)
        if axis_index % 2 != 0:
            ax.tick_params(axis="y", labelleft=False)
        if axis_index >= 4:
            ax.set_xlabel("Candidate count", fontsize=LEGEND_SIZE)
    for ax in axes_list[len(class_ids) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=TITLE_SIZE, y=0.972)
    fig.subplots_adjust(left=0.27, right=0.985, top=0.91, bottom=0.085, hspace=0.42, wspace=0.11)

    alt_parts = []
    for class_id in class_ids:
        counts = counts_by_class[class_id]
        visible_counts = [
            f"{_review_class_plain_label(review_class)} {counts.get(review_class, 0)}"
            for review_class in _REVIEW_CLASS_ORDER
            if counts.get(review_class, 0)
        ]
        alt_parts.append(f"{class_label(class_id)}: {', '.join(visible_counts) if visible_counts else 'no rows'}")
    alt = (
        f"Six-panel bar chart of expanded Eco1 fold-review classes for {len(joined_rows)} candidates. "
        + "; ".join(alt_parts)
        + "."
    )
    save_accessible_svg(fig, path, title=title, description=alt, dpi=220)
    return make_deliverable_row(
        deliverable_id="foldcheck_review_review_class_counts",
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=source_tables,
        input_hashes=file_hashes(
            {
                "candidate_pool": candidate_pool_path,
                "foldcheck_candidate_ranking": foldcheck_ranking_path,
            }
        ),
        alt_text=alt,
        description=(
            "Counts fold-review bins within each fixed-mask design class. The plot separates the six mask "
            "classes so aggregate fold preservation does not hide class-specific failures."
        ),
        interpretation_limit=(
            "Fold-review bins summarize ColabFold structural triage. They do not measure RT activity, "
            "processivity, strand displacement, or construct acceptance."
        ),
        title=title,
        evidence_summary={
            "candidate_count": len(joined_rows),
            "design_class_count": len(class_ids),
            "review_class_counts_by_design_class": counts_by_class,
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


def _join_design_class_review_rows(*, candidate_pool_path: Path, foldcheck_ranking_path: Path) -> list[dict[str, Any]]:
    candidate_rows = pq.read_table(candidate_pool_path, columns=["candidate_id", "design_class_id"]).to_pylist()
    ranking_rows = pq.read_table(foldcheck_ranking_path, columns=["candidate_id", "review_class"]).to_pylist()
    candidate_by_id = {str(row["candidate_id"]): str(row["design_class_id"]) for row in candidate_rows}
    joined_rows: list[dict[str, Any]] = []
    for ranking in ranking_rows:
        candidate_id = str(ranking["candidate_id"])
        class_id = candidate_by_id.get(candidate_id)
        if class_id is None:
            continue
        joined_rows.append(
            {
                "candidate_id": candidate_id,
                "design_class_id": class_id,
                "review_class": str(ranking.get("review_class") or "metric_missing"),
            }
        )
    return joined_rows


def _review_class_axis_label(review_class: str) -> str:
    labels = {
        "strong_fold_preserved": "Strong fold",
        "good_fold_preserved": "Good fold",
        "review_band": "Review band",
        "low_confidence": "Low pLDDT",
        "structural_outlier": "High RMSD",
        "metric_missing": "Metric missing",
    }
    return labels.get(review_class, review_class.replace("_", " "))


def _review_class_plain_label(review_class: str) -> str:
    labels = {
        "strong_fold_preserved": "strong fold",
        "good_fold_preserved": "good fold",
        "review_band": "review band",
        "low_confidence": "low pLDDT",
        "structural_outlier": "high RMSD",
        "metric_missing": "metric missing",
    }
    return labels.get(review_class, review_class.replace("_", " "))
