"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/esmc_model_constraint.py

WT ESMC masked-marginal model-constraint review panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib
import pyarrow.parquet as pq
from matplotlib.patches import Patch

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    OKABE_ITO,
    TICK_SIZE,
    TITLE_SIZE,
    save_accessible_svg,
    style_open_axes,
)

from .esmc_model_constraint_metadata import (
    INTERPRETATION_LIMIT,
    METHOD_SUMMARY,
    SECTION,
    SOURCE_TABLES,
    missing_model_constraint_row,
    mutation_scoring_evidence_summary,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_MOTIF_ANCHOR_CLASS = "Motif anchors: NAxxH/YADD/VTG"
_CLASS_COLORS = {
    _MOTIF_ANCHOR_CLASS: OKABE_ITO["purple"],
    "Retained DNA/RNA <=5 A": OKABE_ITO["blue"],
    "Clade 9 >=25% WT plurality": OKABE_ITO["green"],
    "Other fixed-mask residues": OKABE_ITO["orange"],
    "ProteinMPNN-designable residues": "#9aa1a8",
}


def write_esmc_model_constraint_audit_panels(
    *,
    panel_root: Path,
    mutation_scoring_root: Path,
) -> list[dict[str, Any]]:
    """Link WT ESMC DMS-style plots and render MSA-vs-ESMC congruence panels."""

    mask_join_path = mutation_scoring_root / "wt_mutation_scoring_mask_join.parquet"
    if not mask_join_path.exists():
        return [missing_model_constraint_row(panel_root, mutation_scoring_root)]

    rows = pq.read_table(mask_join_path).to_pylist()
    accepted_rows = sorted(
        [row for row in rows if str(row.get("status") or "") == "accepted"],
        key=lambda row: int(row["canonical_position"]),
    )
    if not accepted_rows:
        raise ValueError(f"No accepted WT mutation-scoring rows found in {mask_join_path}")

    linked_rows = _linked_permuter_plot_rows(mutation_scoring_root)
    generated_rows = [
        _write_plurality_entropy_scatter(panel_root, accepted_rows, mask_join_path),
        _write_plurality_best_alt_scatter(panel_root, accepted_rows, mask_join_path),
        _write_constraint_tracks(panel_root, accepted_rows, mask_join_path),
    ]
    return [*linked_rows, *generated_rows]


def _linked_permuter_plot_rows(mutation_scoring_root: Path) -> list[dict[str, Any]]:
    plot_root = mutation_scoring_root / "plots"
    manifest_path = mutation_scoring_root / "wt_mutation_scoring_manifest.yaml"
    evidence_summary = mutation_scoring_evidence_summary(mutation_scoring_root)
    specs = [
        (
            "wt_esmc_entropy_by_position",
            "ESMC masked-position entropy varies across WT residues",
            plot_root / "wt_entropy_by_position.svg",
            "ESMC entropy across WT residues.",
            "Bar plot of ESMC masked-position entropy across the WT Ec86 sequence.",
            "Shows where the model is more or less uncertain about the WT residue identity.",
        ),
        (
            "wt_esmc_fraction_negative_alternate_llr",
            "Lower-LLR alternate fraction varies across WT residues",
            plot_root / "wt_fraction_negative_alternate_llr_by_position.svg",
            "Fraction of non-WT alternates with lower ESMC LLR than WT.",
            "Scatter plot of the fraction of alternate residues with negative LLR at each WT position.",
            "Shows positions where most single-residue alternatives score worse than the WT residue.",
        ),
        (
            "wt_esmc_substitution_llr_heatmap",
            "ESMC masked-marginal scores form a WT substitution matrix",
            plot_root / "wt_substitution_llr_heatmap.svg",
            "ESMC substitution LLR matrix.",
            "Permuter-style heatmap of ESMC masked-marginal LLR values for WT single substitutions.",
            "Shows DMS-shaped model scores for the WT sequence only.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for deliverable_id, title, path, alt_text, description, plain_role in specs:
        if path.exists():
            status = "linked_existing"
            skip_reason = ""
        else:
            status = "skipped_missing_input"
            skip_reason = f"Missing linked ESMC plot: {path}"
        rows.append(
            make_deliverable_row(
                deliverable_id=deliverable_id,
                section=SECTION,
                artifact_kind="linked_visual",
                status=status,
                path=path,
                source_tables=SOURCE_TABLES,
                input_hashes=file_hashes({"plot": path, "mutation_scoring_manifest": manifest_path}),
                alt_text=alt_text,
                description=f"{description} {plain_role}",
                interpretation_limit=INTERPRETATION_LIMIT,
                title=title,
                method_summary=METHOD_SUMMARY,
                evidence_summary=evidence_summary,
                role="review_only",
                render_mode=(
                    "wide_visual" if deliverable_id == "wt_esmc_substitution_llr_heatmap" else "standard_visual"
                ),
                skip_reason=skip_reason,
            )
        )
    return rows


def _write_plurality_entropy_scatter(
    panel_root: Path,
    rows: list[dict[str, Any]],
    mask_join_path: Path,
) -> dict[str, Any]:
    title = "Clade 9 plurality is inversely related to ESMC entropy"
    plurality_values = [float(row["wt_plurality_frequency"]) for row in rows]
    entropy_values = [float(row["canonical_entropy_bits"]) for row in rows]
    pearson_r = _pearson_r(plurality_values, entropy_values)
    r2 = pearson_r * pearson_r if math.isfinite(pearson_r) else float("nan")
    fig, ax = plt.subplots(figsize=(6.9, 7.3))
    for label in _CLASS_COLORS:
        class_rows = [row for row in rows if _constraint_class(row) == label]
        if not class_rows:
            continue
        ax.scatter(
            [float(row["wt_plurality_frequency"]) for row in class_rows],
            [float(row["canonical_entropy_bits"]) for row in class_rows],
            color=_CLASS_COLORS[label],
            label=label,
            s=26,
            alpha=0.86,
            edgecolors="#ffffff",
            linewidths=0.35,
        )
    annotation_box = {"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82}
    fit = _linear_fit(plurality_values, entropy_values)
    if fit is not None:
        x_min, x_max, slope, intercept = fit
        ax.plot(
            [x_min, x_max],
            [slope * x_min + intercept, slope * x_max + intercept],
            color="#24292f",
            linewidth=1.35,
            label=f"Linear fit, R2 = {r2:.2f}",
            zorder=4,
        )
    ax.set_xlabel("Clade 9 WT plurality frequency", fontsize=LABEL_SIZE)
    ax.set_ylabel("ESMC masked-position entropy (bits)", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
    _style_scatter_axes(ax)
    ax.text(
        0.96,
        0.96,
        f"Pearson r = {pearson_r:.2f}\nR2 = {r2:.2f}",
        transform=ax.transAxes,
        fontsize=TICK_SIZE,
        ha="right",
        va="top",
        bbox=annotation_box,
    )
    _add_class_legend_below(fig, ax, ncol=2)
    fig.tight_layout(rect=(0, 0.15, 1, 0.98))

    path = panel_root / "msa_plurality_vs_esmc_entropy.svg"
    alt = (
        "Scatter plot comparing clade 9 WT plurality frequency against ESMC masked-position entropy "
        f"for {len(rows)} WT Ec86 positions."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return make_deliverable_row(
        deliverable_id="msa_plurality_vs_esmc_entropy",
        section=SECTION,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=SOURCE_TABLES,
        input_hashes=file_hashes({"mask_join": mask_join_path}),
        alt_text=alt,
        description=(
            "Compares a phylogenetic constraint signal to an ESMC masked-marginal constraint signal "
            "at the same WT residue coordinates."
        ),
        interpretation_limit=INTERPRETATION_LIMIT,
        title=title,
        method_summary=METHOD_SUMMARY,
        evidence_summary={"mask_join_rows": len(rows), "pearson_r": round(pearson_r, 4), "r_squared": round(r2, 4)},
        role="review_only",
    )


def _write_plurality_best_alt_scatter(
    panel_root: Path,
    rows: list[dict[str, Any]],
    mask_join_path: Path,
) -> dict[str, Any]:
    title = "Best alternate LLR highlights where model and MSA disagree"
    fig, ax = plt.subplots(figsize=(6.9, 7.3))
    for label in _CLASS_COLORS:
        class_rows = [row for row in rows if _constraint_class(row) == label]
        if not class_rows:
            continue
        ax.scatter(
            [float(row["wt_plurality_frequency"]) for row in class_rows],
            [float(row["best_alt_llr"]) for row in class_rows],
            color=_CLASS_COLORS[label],
            label=label,
            s=26,
            alpha=0.86,
            edgecolors="#ffffff",
            linewidths=0.35,
        )
    ax.axhline(0.0, color="#6e7781", linestyle=":", linewidth=1.2)
    annotation_box = {"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.82}
    ax.text(
        0.02,
        0.05,
        "Best alternate equals WT at LLR = 0",
        transform=ax.transAxes,
        fontsize=TICK_SIZE,
        bbox=annotation_box,
    )
    ax.set_xlabel("Clade 9 WT plurality frequency", fontsize=LABEL_SIZE)
    ax.set_ylabel("Best alternate ESMC LLR vs WT", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
    _style_scatter_axes(ax)
    _add_class_legend_below(fig, ax, ncol=3)
    fig.tight_layout(rect=(0, 0.15, 1, 0.98))

    path = panel_root / "msa_plurality_vs_best_alt_llr.svg"
    alt = (
        "Scatter plot comparing clade 9 WT plurality frequency against the best ESMC alternate-residue "
        f"LLR for {len(rows)} WT Ec86 positions."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return make_deliverable_row(
        deliverable_id="msa_plurality_vs_best_alt_llr",
        section=SECTION,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=SOURCE_TABLES,
        input_hashes=file_hashes({"mask_join": mask_join_path}),
        alt_text=alt,
        description=(
            "Highlights positions where natural plurality and the model's most favorable single alternate "
            "tell different stories."
        ),
        interpretation_limit=INTERPRETATION_LIMIT,
        title=title,
        method_summary=METHOD_SUMMARY,
        evidence_summary={"mask_join_rows": len(rows)},
        role="review_only",
    )


def _write_constraint_tracks(
    panel_root: Path,
    rows: list[dict[str, Any]],
    mask_join_path: Path,
) -> dict[str, Any]:
    title = "Plurality, ESMC scores, and mask classes align along Ec86"
    positions = [int(row["canonical_position"]) for row in rows]
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11.4, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1, 0.28], "hspace": 0.18},
    )
    series = [
        (
            "WT plurality",
            [float(row["wt_plurality_frequency"]) for row in rows],
            OKABE_ITO["green"],
            (0.0, 1.05),
        ),
        ("ESMC entropy\n(bits)", [float(row["canonical_entropy_bits"]) for row in rows], OKABE_ITO["blue"], None),
        (
            "Lower-LLR\nalternate fraction",
            [float(row["fraction_negative_alternate_llr"]) for row in rows],
            OKABE_ITO["purple"],
            (0.0, 1.05),
        ),
    ]
    for ax, (label, values, color, ylim) in zip(axes[:3], series, strict=True):
        _draw_mask_background(ax, rows)
        ax.plot(positions, values, color=color, linewidth=1.4)
        ax.scatter(positions, values, color=color, s=8)
        ax.set_ylabel(label, fontsize=TICK_SIZE)
        if ylim is not None:
            ax.set_ylim(*ylim)
        style_open_axes(ax)
    _draw_mask_track(axes[3], rows)
    axes[3].set_xlabel("Ec86 canonical residue position", fontsize=LABEL_SIZE, labelpad=8)
    axes[0].set_title(title, fontsize=TITLE_SIZE, pad=10)
    fig.legend(
        handles=_observed_class_handles(rows),
        frameon=False,
        fontsize=LEGEND_SIZE,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.018),
    )
    fig.subplots_adjust(left=0.11, right=0.995, top=0.93, bottom=0.19, hspace=0.28)

    path = panel_root / "msa_esmc_constraint_tracks.svg"
    alt = (
        "Stacked residue-position tracks showing clade 9 WT plurality, ESMC entropy, fraction negative "
        "alternate LLR, and current Eco1 protected-position classes."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return make_deliverable_row(
        deliverable_id="msa_esmc_constraint_tracks",
        section=SECTION,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=SOURCE_TABLES,
        input_hashes=file_hashes({"mask_join": mask_join_path}),
        alt_text=alt,
        description=(
            "Shows where MSA plurality, ESMC masked-marginal entropy, the fraction of non-WT alternates "
            "with lower LLR than WT, and the current mask align along the WT sequence."
        ),
        interpretation_limit=INTERPRETATION_LIMIT,
        title=title,
        method_summary=METHOD_SUMMARY,
        evidence_summary={"mask_join_rows": len(rows)},
        role="review_only",
    )


def _draw_mask_track(ax: Any, rows: list[dict[str, Any]]) -> None:
    positions = [int(row["canonical_position"]) for row in rows]
    for row in rows:
        position = int(row["canonical_position"])
        label = _constraint_class(row)
        ax.axvspan(position - 0.5, position + 0.5, color=_CLASS_COLORS[label], linewidth=0)
    ax.set_xlim(min(positions) - 0.5, max(positions) + 10.0)
    ax.set_yticks([])
    ax.set_ylabel("Mask\nclass", fontsize=LABEL_SIZE)
    ax.spines[["top", "right", "left"]].set_visible(False)


def _draw_mask_background(ax: Any, rows: list[dict[str, Any]]) -> None:
    for start, end, label in _constraint_segments(rows):
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color=_CLASS_COLORS[label],
            alpha=0.055 if label != _MOTIF_ANCHOR_CLASS else 0.10,
            linewidth=0,
            zorder=0,
        )


def _constraint_segments(rows: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda row: int(row["canonical_position"]))
    segments: list[tuple[int, int, str]] = []
    start = int(sorted_rows[0]["canonical_position"])
    previous = start
    current_label = _constraint_class(sorted_rows[0])
    for row in sorted_rows[1:]:
        position = int(row["canonical_position"])
        label = _constraint_class(row)
        if label == current_label and position == previous + 1:
            previous = position
            continue
        segments.append((start, previous, current_label))
        start = previous = position
        current_label = label
    segments.append((start, previous, current_label))
    return segments


def _observed_class_handles(rows: list[dict[str, Any]]) -> list[Patch]:
    observed = {_constraint_class(row) for row in rows}
    return [Patch(facecolor=color, label=label) for label, color in _CLASS_COLORS.items() if label in observed]


def _constraint_class(row: dict[str, Any]) -> str:
    if bool(row.get("motif_protected")):
        return _MOTIF_ANCHOR_CLASS
    if bool(row.get("direct_retained_dna_rna_contact_5a")):
        return "Retained DNA/RNA <=5 A"
    if bool(row.get("evolutionarily_conserved_clade9_25pct_plurality")):
        return "Clade 9 >=25% WT plurality"
    if bool(row.get("protected")):
        return "Other fixed-mask residues"
    return "ProteinMPNN-designable residues"


def _style_scatter_axes(ax: Any) -> None:
    ax.set_xlim(-0.02, 1.02)
    style_open_axes(ax)
    ax.set_box_aspect(1)


def _add_class_legend_below(fig: Any, ax: Any, *, ncol: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        fig.legend(
            handles,
            labels,
            frameon=False,
            fontsize=LEGEND_SIZE,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=ncol,
        )


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float, float] | None:
    finite_pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if math.isfinite(x) and math.isfinite(y)]
    if len(finite_pairs) < 2:
        return None
    x_values = [pair[0] for pair in finite_pairs]
    y_values = [pair[1] for pair in finite_pairs]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    if denominator == 0.0:
        return None
    slope = sum((x - x_mean) * (y - y_mean) for x, y in finite_pairs) / denominator
    intercept = y_mean - slope * x_mean
    return min(x_values), max(x_values), slope, intercept


def _pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    finite_pairs = [(x, y) for x, y in zip(xs, ys, strict=True) if math.isfinite(x) and math.isfinite(y)]
    if len(finite_pairs) < 2:
        return float("nan")
    x_values = [pair[0] for pair in finite_pairs]
    y_values = [pair[1] for pair in finite_pairs]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    denominator = x_denom * y_denom
    return numerator / denominator if denominator else float("nan")
