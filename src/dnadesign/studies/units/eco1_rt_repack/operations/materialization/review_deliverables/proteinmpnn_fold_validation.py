"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/proteinmpnn_fold_validation.py

Tao-style fold-validation panel for Eco1 ProteinMPNN candidates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pyarrow.parquet as pq
from matplotlib import gridspec
from matplotlib.lines import Line2D

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_TEMPERATURE_COLORS = {
    0.1: OKABE_ITO["green"],
    0.3: OKABE_ITO["orange"],
}


def write_tao_style_fold_validation(
    panel_root: Path,
    candidate_rows: list[dict[str, Any]],
    candidate_table_path: Path,
    foldcheck_ranking_path: Path,
) -> dict[str, Any]:
    """Render a joint fold-metric plot analogous to Tao-style AF2 filtering."""

    title = "Baseline ProteinMPNN designs cluster by ColabFold RMSD and pLDDT"
    source_tables = ["candidate_table.parquet", "foldcheck_review/foldcheck_candidate_ranking.parquet"]
    if not foldcheck_ranking_path.exists():
        return _skipped_fold_validation_row(
            panel_root=panel_root,
            candidate_table_path=candidate_table_path,
            foldcheck_ranking_path=foldcheck_ranking_path,
            source_tables=source_tables,
            title=title,
        )
    joined_rows = _join_candidate_fold_rows(candidate_rows, foldcheck_ranking_path)
    if not joined_rows:
        raise ValueError("No candidate rows could be joined to fold-review ranking rows")

    fig = plt.figure(figsize=(7.1, 7.4))
    grid = gridspec.GridSpec(2, 2, width_ratios=[4.0, 1.02], height_ratios=[1.0, 4.0], hspace=0.06, wspace=0.06)
    ax_hist_x = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[1, 0], sharex=ax_hist_x)
    ax_hist_y = fig.add_subplot(grid[1, 1], sharey=ax)

    rmsd_values = np.array([row["wt_runtime_ca_rmsd"] for row in joined_rows], dtype=float)
    plddt_values = np.array([row["plddt"] for row in joined_rows], dtype=float)
    temperatures = sorted({row["temperature"] for row in joined_rows})
    bins_x = _histogram_bins(rmsd_values)
    bins_y = _histogram_bins(plddt_values)
    for temperature in temperatures:
        rows_for_temp = [row for row in joined_rows if row["temperature"] == temperature]
        color = _temperature_color(temperature)
        x = [row["wt_runtime_ca_rmsd"] for row in rows_for_temp]
        y = [row["plddt"] for row in rows_for_temp]
        ax.scatter(x, y, c=color, s=42, edgecolors="#ffffff", linewidths=0.35, alpha=0.9)
        ax_hist_x.hist(x, bins=bins_x, color=color, alpha=0.72, edgecolor="#ffffff", linewidth=0.35)
        ax_hist_y.hist(
            y,
            bins=bins_y,
            orientation="horizontal",
            color=color,
            alpha=0.72,
            edgecolor="#ffffff",
            linewidth=0.35,
        )

    ax.set_xlabel("WT-runtime C-alpha RMSD (A)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Mean pLDDT", fontsize=LABEL_SIZE)
    _add_temperature_legend(fig, temperatures)
    ax_hist_x.set_ylabel("Count", fontsize=LABEL_SIZE)
    ax_hist_y.set_xlabel("Count", fontsize=LABEL_SIZE)
    ax_hist_x.tick_params(labelbottom=False, labelsize=LEGEND_SIZE)
    ax_hist_y.tick_params(labelleft=False, labelsize=LEGEND_SIZE)
    for plot_ax in (ax, ax_hist_x, ax_hist_y):
        style_open_axes(plot_ax)
    ax_hist_x.spines["bottom"].set_visible(True)
    ax_hist_y.spines["left"].set_visible(True)
    ax.set_box_aspect(1)
    fig.suptitle(title, fontsize=TITLE_SIZE, y=0.965)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.16, top=0.89)

    path = panel_root / "proteinmpnn_tao_style_fold_validation.svg"
    alt = (
        f"Baseline joint plot for {len(joined_rows)} accepted ProteinMPNN designs, "
        "showing WT-runtime C-alpha RMSD against mean pLDDT with marginal histograms."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return make_deliverable_row(
        deliverable_id="proteinmpnn_tao_style_fold_validation",
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=source_tables,
        input_hashes=file_hashes(
            {
                "candidate_table": candidate_table_path,
                "foldcheck_candidate_ranking": foldcheck_ranking_path,
            }
        ),
        alt_text=alt,
        description=(
            "Shows ColabFold confidence and WT-runtime RMSD for baseline ProteinMPNN designs under the "
            "current Eco1 mask. Expanded design-class fold selection is shown in the panel selection section."
        ),
        interpretation_limit=(
            "The plot uses one single active mask policy, not multiple distance-threshold "
            "redesign sets. RMSD and pLDDT do not measure activity, processivity, strand "
            "displacement, or hairpin readthrough."
        ),
        title=title,
        role="review_only",
    )


def _skipped_fold_validation_row(
    *,
    panel_root: Path,
    candidate_table_path: Path,
    foldcheck_ranking_path: Path,
    source_tables: list[str],
    title: str,
) -> dict[str, Any]:
    return make_deliverable_row(
        deliverable_id="proteinmpnn_tao_style_fold_validation",
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="svg",
        status="skipped_missing_input",
        path=panel_root / "proteinmpnn_tao_style_fold_validation.svg",
        source_tables=source_tables,
        input_hashes=file_hashes({"candidate_table": candidate_table_path}),
        alt_text="Tao-style fold-validation plot was not rendered because the fold-review ranking table is missing.",
        description="The joint fold-validation plot is skipped until fold-review metrics are available.",
        interpretation_limit="Missing fold-review metrics cannot support structural triage.",
        title=title,
        role="review_only",
        skip_reason=f"Missing input table: {foldcheck_ranking_path}",
    )


def _join_candidate_fold_rows(
    candidate_rows: list[dict[str, Any]],
    foldcheck_ranking_path: Path,
) -> list[dict[str, Any]]:
    ranking_rows = pq.read_table(
        foldcheck_ranking_path,
        columns=["candidate_id", "wt_runtime_ca_rmsd", "plddt"],
    ).to_pylist()
    candidate_by_id = {str(row["candidate_id"]): row for row in candidate_rows}
    joined_rows: list[dict[str, Any]] = []
    for ranking in ranking_rows:
        candidate = candidate_by_id.get(str(ranking.get("candidate_id")))
        if candidate is None:
            continue
        joined_rows.append(
            {
                "candidate_id": str(ranking["candidate_id"]),
                "wt_runtime_ca_rmsd": float(ranking["wt_runtime_ca_rmsd"]),
                "plddt": float(ranking["plddt"]),
                "temperature": float(candidate["temperature"]),
                "seed": int(candidate["seed"]),
            }
        )
    return joined_rows


def _add_temperature_legend(fig: Any, temperatures: list[float]) -> None:
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_temperature_color(temperature),
            markeredgecolor="#ffffff",
            label=f"Temperature {temperature:g}",
        )
        for temperature in temperatures
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=LEGEND_SIZE,
        title="Sampling temperature",
        title_fontsize=LEGEND_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.028),
        ncol=len(legend_handles),
    )


def _histogram_bins(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        center = float(values[0]) if values.size else 0.0
        return np.array([center - 0.5, center + 0.5], dtype=float)
    return np.histogram_bin_edges(values, bins="auto")


def _temperature_color(temperature: float) -> str:
    return _TEMPERATURE_COLORS.get(round(float(temperature), 3), "#5b7fa6")
