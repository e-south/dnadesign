"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/biohub_esmc_sequence_preference_plot.py

Biohub ESMC candidate-preference plot rendering for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    OKABE_ITO,
    TICK_SIZE,
    TITLE_SIZE,
    save_accessible_svg,
    style_open_axes,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def render_candidate_preference_plot(path: Path, rows: list[dict[str, object]], *, title: str) -> None:
    """Render a ranked additive ESMC LLR bar plot."""

    ordered = list(rows)
    values = [float(row["llr_total"]) for row in ordered]
    y_positions = list(range(len(ordered)))
    labels = [_rank_label(index, str(row["candidate_id"])) for index, row in enumerate(ordered, start=1)]
    fig_width = max(8.8, min(11.2, 0.025 * len(ordered) + 8.8))
    fig_height = max(5.8, min(30.0, 0.29 * len(ordered) + 3.2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    colors = [_review_class_color(str(row.get("review_class") or "")) for row in ordered]
    ax.barh(y_positions, values, color=colors, edgecolor="#ffffff", linewidth=0.45, height=0.72)
    ax.axvline(0.0, color="#24292f", linewidth=1.0)
    y_tick_size = 8.4 if len(ordered) > 42 else TICK_SIZE
    ax.set_yticks(y_positions, labels, fontsize=y_tick_size)
    ax.invert_yaxis()
    ax.set_xlabel("WT-context single-substitution LLR sum", fontsize=LABEL_SIZE + 0.8, labelpad=8)
    ax.set_ylabel("ProteinMPNN candidate rank by additive ESMC LLR", fontsize=LABEL_SIZE + 0.8, labelpad=9)
    ax.set_title(title, fontsize=TITLE_SIZE + 1.0, pad=12)
    style_open_axes(ax, grid=True)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="x", labelsize=TICK_SIZE + 0.5)
    ax.tick_params(axis="y", pad=3)
    legend_handles = _legend_handles(ordered)
    bottom_margin = max(0.1, min(0.22, 1.45 / fig_height))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=((0.2 + 0.985) / 2.0, 0.018),
        ncol=min(5, max(1, len(legend_handles))),
        frameon=False,
        fontsize=LEGEND_SIZE - 1.0,
        handletextpad=0.45,
        columnspacing=0.95,
    )
    fig.subplots_adjust(left=0.2, right=0.985, top=0.93, bottom=bottom_margin)
    save_accessible_svg(
        fig,
        path,
        title=title,
        description=(
            f"Ranked bar plot of additive WT-context ESMC LLR sums for {len(ordered)} ProteinMPNN "
            "candidate sequences. Positive values indicate that the ESMC masked-marginal grid assigns "
            "higher probability to the candidate substitutions than to the WT residues at those positions."
        ),
    )


def render_model_stability_plot(
    path: Path,
    rows: list[dict[str, object]],
    *,
    title: str,
    left_label: str,
    right_label: str,
) -> None:
    """Render a two-model additive LLR comparison plot."""

    x_values = [float(row["left_llr_total"]) for row in rows]
    y_values = [float(row["right_llr_total"]) for row in rows]
    sign_changes = [bool(row.get("sign_change")) for row in rows]
    colors = [OKABE_ITO["vermillion"] if changed else OKABE_ITO["blue"] for changed in sign_changes]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(x_values, y_values, s=68, c=colors, edgecolor="#ffffff", linewidth=0.75, alpha=0.94)
    lower = min(x_values + y_values + [0.0])
    upper = max(x_values + y_values + [0.0])
    padding = max(0.5, 0.08 * (upper - lower if upper > lower else 1.0))
    ax.plot([lower - padding, upper + padding], [lower - padding, upper + padding], color="#5f6b7a", linewidth=1.0)
    ax.axhline(0.0, color="#24292f", linewidth=0.8, alpha=0.75)
    ax.axvline(0.0, color="#24292f", linewidth=0.8, alpha=0.75)
    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(lower - padding, upper + padding)
    ax.set_xlabel(f"{left_label} additive WT-context LLR", fontsize=LABEL_SIZE + 0.5, labelpad=8)
    ax.set_ylabel(f"{right_label} additive WT-context LLR", fontsize=LABEL_SIZE + 0.5, labelpad=8)
    ax.set_title(title, fontsize=TITLE_SIZE + 0.8, pad=13)
    style_open_axes(ax, grid=True)
    ax.tick_params(axis="both", labelsize=TICK_SIZE + 0.3)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=OKABE_ITO["blue"],
            markeredgecolor="#ffffff",
            markersize=8,
            label="Same LLR sign",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=OKABE_ITO["vermillion"],
            markeredgecolor="#ffffff",
            markersize=8,
            label="LLR sign changes",
        ),
        Line2D([0], [0], color="#5f6b7a", linewidth=1.0, label="Equal LLR"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=LEGEND_SIZE, handletextpad=0.55)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.91, bottom=0.14)
    save_accessible_svg(
        fig,
        path,
        title=title,
        description=(
            f"Scatter plot comparing additive WT-context ESMC LLR totals for {len(rows)} candidates between "
            f"{left_label} and {right_label}. Points above the diagonal score higher under the right model."
        ),
    )


def _rank_label(index: int, candidate_id: str) -> str:
    return f"V{index:03d} {candidate_id.removeprefix('thread_candidate_')[:8]}"


def _review_class_color(review_class: str) -> str:
    palette = {
        "strong_fold_preserved": OKABE_ITO["blue"],
        "good_fold_preserved": OKABE_ITO["green"],
        "review_band": OKABE_ITO["orange"],
        "fold_watch": OKABE_ITO["purple"],
        "low_confidence": OKABE_ITO["purple"],
        "structural_outlier": OKABE_ITO["vermillion"],
        "metric_missing": OKABE_ITO["gray"],
        "": OKABE_ITO["gray"],
    }
    return palette.get(review_class, OKABE_ITO["green"])


def _legend_handles(rows: list[dict[str, object]]) -> list[Patch]:
    classes = sorted({str(row.get("review_class") or "") for row in rows})
    if "" in classes:
        classes.remove("")
        classes.append("")
    return [
        Patch(
            facecolor=_review_class_color(review_class),
            edgecolor="none",
            label=_review_class_label(review_class),
        )
        for review_class in classes
    ]


def _review_class_label(review_class: str) -> str:
    labels = {
        "strong_fold_preserved": "RMSD <=1.25 A; pLDDT >=91.5",
        "good_fold_preserved": "RMSD <=2.0 A; pLDDT >=90",
        "review_band": "Intermediate",
        "low_confidence": "pLDDT < 90",
        "structural_outlier": "RMSD >5.0 A",
        "metric_missing": "Metrics missing",
        "": "Fold class unavailable",
    }
    return labels.get(review_class, review_class.replace("_", " "))
