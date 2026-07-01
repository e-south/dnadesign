"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/proteinmpnn_diversity.py

ProteinMPNN candidate-diversity panels for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import pyarrow.parquet as pq
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .constants import SECTION_DESIGNS_AND_FOLD_TRIAGE
from .manifest import (
    file_hashes,
    make_deliverable_row,
)
from .proteinmpnn_fold_validation import write_tao_style_fold_validation
from .proteinmpnn_variant_similarity import write_variant_similarity_heatmap
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

_MUTATION_RE = re.compile(r"^[A-Z*](?P<position>[0-9]+)[A-Z*]$")
_TEMPERATURE_COLORS = {
    0.1: OKABE_ITO["green"],
    0.3: OKABE_ITO["orange"],
}
_MOTIF_LABELS = {
    "retron_x_naxxh": "NAxxH",
    "catalytic_yadd": "YADD",
    "retron_y_vtg": "VTG",
}
_MUTATION_CONTEXT_LAYERS = (
    (
        "rt_interval_review_label",
        "#6e7781",
        0.075,
        "RT1-RT7 annotation intervals",
    ),
    (
        "direct_retained_dna_rna_contact_5a",
        OKABE_ITO["orange"],
        0.13,
        "Retained DNA/RNA <=5 A",
    ),
    (
        "evolutionarily_conserved_clade9_25pct_plurality",
        OKABE_ITO["blue"],
        0.11,
        "Clade 9 >=25% WT plurality",
    ),
    (
        "wang_ec86_direct_contact_prior",
        OKABE_ITO["purple"],
        0.16,
        "Wang/Ec86 direct-contact priors",
    ),
    (
        "motif_protected",
        OKABE_ITO["vermillion"],
        0.30,
        "Motif anchors: NAxxH/YADD/VTG",
    ),
)


def write_proteinmpnn_diversity_panels(
    *,
    panel_root: Path,
    candidate_table_path: Path,
    foldcheck_ranking_path: Path | None = None,
    foldcheck_fasta_path: Path | None = None,
    mask_set_path: Path | None = None,
    mask_residues: list[dict[str, Any]] | None = None,
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
    deliverables = [
        _write_score_mutation_burden(panel_root, accepted_rows, candidate_table_path),
        _write_mutation_density(panel_root, accepted_rows, candidate_table_path, mask_residues=mask_residues or []),
        write_variant_similarity_heatmap(
            panel_root,
            accepted_rows,
            candidate_table_path,
            foldcheck_ranking_path=foldcheck_ranking_path,
            foldcheck_fasta_path=foldcheck_fasta_path,
            mask_set_path=mask_set_path,
            mask_residues=mask_residues or [],
        ),
    ]
    if foldcheck_ranking_path is not None:
        deliverables.append(
            write_tao_style_fold_validation(
                panel_root,
                accepted_rows,
                candidate_table_path,
                foldcheck_ranking_path,
            )
        )
    return deliverables


def _write_score_mutation_burden(
    panel_root: Path,
    rows: list[dict[str, Any]],
    candidate_table_path: Path,
) -> dict[str, Any]:
    title = "ProteinMPNN proposal scores and mutation burden"
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
            "Summarizes ProteinMPNN proposal scores, global scores, mutation burden, "
            "and sequence identity for candidate-table rows with status=accepted."
        ),
        interpretation_limit=(
            "Sequence recovery and ProteinMPNN scores are descriptive proposal metrics. "
            "They are not fold, synthesis, or activity acceptance criteria."
        ),
        title=title,
    )


def _write_mutation_density(
    panel_root: Path,
    rows: list[dict[str, Any]],
    candidate_table_path: Path,
    *,
    mask_residues: list[dict[str, Any]],
) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    for row in rows:
        for mutation in row.get("canonical_mutations") or []:
            match = _MUTATION_RE.match(str(mutation))
            if match:
                counts[int(match.group("position"))] += 1
    positions = sorted(counts)
    title = "ProteinMPNN mutation density across allowed residues"
    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    _draw_residue_context_spans(ax, mask_residues)
    ax.bar(
        positions,
        [counts[position] for position in positions],
        width=0.92,
        color=OKABE_ITO["blue"],
        alpha=0.82,
        zorder=3,
    )
    ax.set_xlabel("Ec86 canonical residue position", fontsize=LABEL_SIZE)
    ax.set_ylabel("Candidate mutation count", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
    style_open_axes(ax)
    _add_context_legend(fig, mask_residues)
    bottom = 0.22 if mask_residues else 0.12
    fig.tight_layout(rect=(0, bottom, 1, 0.96))

    path = panel_root / "proteinmpnn_mutation_density.svg"
    alt = (
        f"Bar chart of mutation density across Ec86 residue positions for {len(rows)} accepted ProteinMPNN candidates."
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return make_deliverable_row(
        deliverable_id="proteinmpnn_mutation_density",
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=["candidate_table.parquet"],
        input_hashes=file_hashes({"candidate_table": candidate_table_path}),
        alt_text=alt,
        description="Shows where ProteinMPNN sampled mutations under the current mask.",
        interpretation_limit=(
            "Mutation density describes sampled sequence variation under the current mask. "
            "It does not imply residue importance or biochemical effect."
        ),
        title=title,
    )


def _temperature_color(temperature: float) -> str:
    return _TEMPERATURE_COLORS.get(round(float(temperature), 3), "#5b7fa6")


def _draw_residue_context_spans(ax: Any, mask_residues: list[dict[str, Any]]) -> None:
    for field, color, alpha, _label in _MUTATION_CONTEXT_LAYERS:
        if field == "rt_interval_review_label":
            segments = [(start, end) for start, end, _label in _rt_interval_segments(mask_residues)]
        elif field == "motif_protected":
            segments = [(start, end) for start, end, _label in _motif_segments(mask_residues)]
        else:
            segments = _boolean_segments(mask_residues, field)
        for start, end in segments:
            ax.axvspan(start - 0.5, end + 0.5, color=color, alpha=alpha, linewidth=0, zorder=0)


def _add_context_legend(fig: Any, mask_residues: list[dict[str, Any]]) -> None:
    if not mask_residues:
        return
    handles = [
        Patch(facecolor=color, alpha=alpha, label=label) for _field, color, alpha, label in _MUTATION_CONTEXT_LAYERS
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        columnspacing=1.0,
        handletextpad=0.42,
    )


def _rt_interval_segments(mask_residues: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    by_label: dict[str, list[int]] = {}
    for row in mask_residues:
        label = str(row.get("rt_interval_review_label") or "")
        if not label.startswith("RT"):
            continue
        by_label.setdefault(label, []).append(int(row["canonical_position"]))
    segments: list[tuple[int, int, str]] = []
    for label in sorted(by_label, key=_span_label_sort_key):
        segments.extend((start, end, label) for start, end in _segments(by_label[label]))
    return segments


def _motif_segments(mask_residues: list[dict[str, Any]]) -> list[tuple[int, int, str]]:
    by_reason: dict[str, list[int]] = {}
    for row in mask_residues:
        if not bool(row.get("motif_protected")):
            continue
        reason = str(row.get("manual_mask_reason") or "motif_protected")
        by_reason.setdefault(reason, []).append(int(row["canonical_position"]))
    segments: list[tuple[int, int, str]] = []
    for reason in sorted(by_reason):
        label = _MOTIF_LABELS.get(reason, reason.replace("_", " "))
        segments.extend((start, end, label) for start, end in _segments(by_reason[reason]))
    return segments


def _boolean_segments(mask_residues: list[dict[str, Any]], field: str) -> list[tuple[int, int]]:
    return _segments([int(row["canonical_position"]) for row in mask_residues if bool(row.get(field))])


def _segments(positions: list[int]) -> list[tuple[int, int]]:
    if not positions:
        return []
    sorted_positions = sorted(set(positions))
    segments: list[tuple[int, int]] = []
    start = sorted_positions[0]
    previous = start
    for position in sorted_positions[1:]:
        if position == previous + 1:
            previous = position
            continue
        segments.append((start, previous))
        start = previous = position
    segments.append((start, previous))
    return segments


def _span_label_sort_key(label: str) -> tuple[int, str]:
    number = "".join(character for character in label if character.isdigit())
    return (int(number) if number else 999, label)
