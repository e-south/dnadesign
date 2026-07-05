"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/proteinmpnn_diversity.py

ProteinMPNN candidate-diversity panels for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pyarrow.parquet as pq
from matplotlib.lines import Line2D

from .constants import SECTION_DESIGNS_AND_FOLD_TRIAGE
from .manifest import (
    file_hashes,
    make_deliverable_row,
)
from .proteinmpnn_residue_frequency import write_residue_frequency_heatmap
from .rendering import (
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


def write_proteinmpnn_diversity_panels(
    *,
    panel_root: Path,
    candidate_table_path: Path,
    candidate_pool_path: Path,
    design_classes_root: Path,
    mask_set_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Render compact ProteinMPNN diversity panels."""

    rows = pq.read_table(
        candidate_table_path,
        columns=[
            "candidate_id",
            "score",
            "global_score",
            "seq_recovery",
            "seed",
            "temperature",
            "mutation_count",
            "canonical_mutations",
            "sequence",
            "status",
            "rank",
        ],
    ).to_pylist()
    accepted_rows = [row for row in rows if str(row.get("status")) == "accepted"]
    if not accepted_rows:
        raise ValueError(f"No candidate_table rows with status=accepted found in {candidate_table_path}")
    if mask_set_path is None:
        raise ValueError("ProteinMPNN diversity panels require mask_set_path for design-class residue frequencies")
    deliverables = [
        _write_score_mutation_burden(panel_root, accepted_rows, candidate_table_path),
        write_residue_frequency_heatmap(
            panel_root=panel_root,
            candidate_pool_path=candidate_pool_path,
            baseline_mask_set_path=mask_set_path,
            design_classes_root=design_classes_root,
        ),
    ]
    return deliverables


def _write_score_mutation_burden(
    panel_root: Path,
    rows: list[dict[str, Any]],
    candidate_table_path: Path,
) -> dict[str, Any]:
    title = "ProteinMPNN scores describe proposal spread"
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.55))
    temperatures = [float(row["temperature"]) for row in rows]
    colors = [_temperature_color(temperature) for temperature in temperatures]
    axes[0].scatter(
        [int(row["mutation_count"]) for row in rows],
        [float(row["score"]) for row in rows],
        c=colors,
        s=40,
        edgecolors="#ffffff",
        linewidths=0.35,
    )
    axes[0].set_xlabel("Mutation count", fontsize=LABEL_SIZE)
    axes[0].set_ylabel("ProteinMPNN score", fontsize=LABEL_SIZE)
    axes[0].set_title("Score versus mutation burden", fontsize=LABEL_SIZE)
    axes[1].scatter(
        [float(row["seq_recovery"]) * 100.0 for row in rows],
        [float(row["global_score"]) for row in rows],
        c=colors,
        s=40,
        edgecolors="#ffffff",
        linewidths=0.35,
    )
    axes[1].set_xlabel("Sequence identity to Ec86 WT (%)", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("Global score", fontsize=LABEL_SIZE)
    axes[1].set_title("Global score versus WT identity", fontsize=LABEL_SIZE)
    for ax in axes:
        style_open_axes(ax)
        ax.set_box_aspect(1)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=_temperature_color(temperature),
            markeredgecolor="#ffffff",
            label=f"Temperature {temperature:g}",
        )
        for temperature in sorted(set(temperatures))
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=LEGEND_SIZE,
        title="Sampling temperature",
        title_fontsize=LEGEND_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=len(handles),
    )
    fig.suptitle(title, fontsize=TITLE_SIZE, y=0.965)
    fig.subplots_adjust(left=0.125, right=0.985, top=0.84, bottom=0.24, wspace=0.24)

    path = panel_root / "proteinmpnn_score_mutation_burden.svg"
    alt = (
        f"ProteinMPNN diversity panel for {len(rows)} candidate-table rows with status=accepted, showing "
        "mutation count versus score and WT sequence recovery versus global score."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return make_deliverable_row(
        deliverable_id="proteinmpnn_score_mutation_burden",
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=["candidate_table.parquet"],
        input_hashes=file_hashes({"candidate_table": candidate_table_path}),
        alt_text=alt,
        description=(
            "Summarizes ProteinMPNN proposal scores, global scores, mutation burden, and sequence identity "
            "for the original accepted candidate table. The expanded six-class fold panel is shown as core "
            "fold-triage evidence."
        ),
        interpretation_limit=(
            "Sequence recovery and ProteinMPNN scores are descriptive proposal metrics. "
            "They are not fold, synthesis, or activity acceptance criteria."
        ),
        title=title,
        role="review_only",
    )


def _temperature_color(temperature: float) -> str:
    return _TEMPERATURE_COLORS.get(round(float(temperature), 3), "#5b7fa6")
