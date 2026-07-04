"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/msa_panel.py

MSA plurality and mask-context panel for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import pyarrow.parquet as pq
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    CONSERVATION_CLADE9_PROFILE_ID,
    CONSERVATION_SUBTYPE_PROFILE_ID,
    SECTION_CONSTRAINT_EVIDENCE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.mask_rows import (
    read_mask_residues,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.msa_panel_data import (
    alignment_matrix,
    format_msa_row_label,
    order_selected_records,
    read_fasta,
    select_display_records,
    source_manifest_accessions,
    source_record_accessions,
    source_record_labels,
    subtype_row_segments,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    OKABE_ITO,
    TICK_SIZE,
    TITLE_SIZE,
    save_accessible_svg,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class MsaPanelProfile:
    """Display contract for one Eco1 conservation MSA panel."""

    profile_id: str
    scope_label: str
    row_label_prefix: str
    deliverable_id: str
    file_name: str
    aligned_fasta_source_table: str
    source_manifest_table: str
    current_mask_denominator: bool


CLADE9_MSA_PANEL = MsaPanelProfile(
    profile_id=CONSERVATION_CLADE9_PROFILE_ID,
    scope_label="clade 9",
    row_label_prefix="C9",
    deliverable_id="msa_plurality_mask_panel",
    file_name="msa_plurality_mask_panel.svg",
    aligned_fasta_source_table=f"conservation_alignments/{CONSERVATION_CLADE9_PROFILE_ID}.aligned.fasta",
    source_manifest_table=f"conservation_sources/{CONSERVATION_CLADE9_PROFILE_ID}.source_manifest.yaml",
    current_mask_denominator=True,
)
SUBTYPE_MSA_PANEL = MsaPanelProfile(
    profile_id=CONSERVATION_SUBTYPE_PROFILE_ID,
    scope_label="Eco1 subtype II-A3/42_1",
    row_label_prefix="II-A3",
    deliverable_id="msa_subtype_plurality_panel",
    file_name="msa_subtype_plurality_panel.svg",
    aligned_fasta_source_table=f"conservation_alignments/{CONSERVATION_SUBTYPE_PROFILE_ID}.aligned.fasta",
    source_manifest_table=f"conservation_sources/{CONSERVATION_SUBTYPE_PROFILE_ID}.source_manifest.yaml",
    current_mask_denominator=False,
)


def write_msa_plurality_mask_panel(
    *,
    panel_root: Path,
    panel_profile: MsaPanelProfile = CLADE9_MSA_PANEL,
    aligned_fasta_path: Path,
    source_manifest_path: Path,
    conservation_profile_path: Path,
    mask_set_path: Path,
    mask_residues: list[dict[str, Any]] | None = None,
    max_display_rows: int | None = None,
    subtype_source_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Render a canonical-coordinate MSA plurality panel."""

    records = read_fasta(aligned_fasta_path)
    if not records:
        raise ValueError(f"No FASTA records found in {aligned_fasta_path}")
    source_labels = source_record_labels(source_manifest_path, row_label_prefix=panel_profile.row_label_prefix)
    source_accessions = source_record_accessions(source_manifest_path)
    subtype_accessions = (
        source_manifest_accessions(subtype_source_manifest_path) if subtype_source_manifest_path else set()
    )
    profile_rows = _read_profile_rows(conservation_profile_path, profile_id=panel_profile.profile_id)
    residues = mask_residues if mask_residues is not None else read_mask_residues(mask_set_path)
    positions = [int(row["canonical_position"]) for row in profile_rows]
    conserved_25 = {int(row["canonical_position"]) for row in profile_rows if bool(row["passes_conservation_mask"])}
    conserved_50 = {
        int(row["canonical_position"])
        for row in profile_rows
        if bool(row.get("wt_is_plurality")) and float(row.get("wt_frequency") or 0.0) >= 0.50
    }
    protected = {int(row["canonical_position"]) for row in residues if bool(row.get("protected"))}
    selected_records = select_display_records(
        records,
        profile_rows,
        protected_positions=protected,
        conserved_positions=conserved_25,
        max_display_rows=max_display_rows,
    )
    selected_records = order_selected_records(
        selected_records,
        source_accessions=source_accessions,
        subtype_accessions=subtype_accessions,
    )
    target_id, _target_sequence = records[0]
    matrix = alignment_matrix(selected_records, profile_rows)

    title = _panel_title(record_count=len(records), panel_profile=panel_profile)
    fig_width, fig_height = _figure_size(len(selected_records))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(
        matrix,
        aspect="auto",
        interpolation="none",
        cmap=ListedColormap(["#fffdf7", "#c9c9c9", OKABE_ITO["green"]]),
    )
    for index, position in enumerate(positions):
        if position in conserved_50:
            ax.add_patch(
                Rectangle(
                    (index - 0.5, -1.58),
                    1.0,
                    0.22,
                    facecolor=OKABE_ITO["purple"],
                    edgecolor="none",
                    clip_on=False,
                )
            )
        if position in conserved_25:
            ax.add_patch(
                Rectangle(
                    (index - 0.5, -1.25),
                    1.0,
                    0.25,
                    facecolor=OKABE_ITO["orange"],
                    edgecolor="none",
                    clip_on=False,
                )
            )
        if position in protected:
            ax.add_patch(
                Rectangle(
                    (index - 0.5, -0.9),
                    1.0,
                    0.25,
                    facecolor=OKABE_ITO["blue"],
                    edgecolor="none",
                    clip_on=False,
                )
            )
    for start, count in subtype_row_segments(
        selected_records,
        source_accessions=source_accessions,
        subtype_accessions=subtype_accessions,
    ):
        ax.add_patch(
            Rectangle(
                (-0.5, start - 0.5),
                len(positions),
                count,
                facecolor=OKABE_ITO["sky"],
                edgecolor=OKABE_ITO["blue"],
                alpha=0.12,
                linewidth=0.8,
                clip_on=False,
            )
        )
    ax.set_yticks(
        range(len(selected_records)),
        [
            format_msa_row_label(
                record_id,
                source_labels=source_labels,
                profile_id=panel_profile.profile_id,
                row_label_prefix=panel_profile.row_label_prefix,
            )
            for record_id, _seq in selected_records
        ],
        fontsize=_row_label_size(len(selected_records)),
    )
    ax.tick_params(axis="y", pad=2)
    tick_indexes = _position_tick_indexes(positions)
    ax.set_xticks(tick_indexes, [str(positions[index]) for index in tick_indexes], fontsize=TICK_SIZE)
    top_axis = ax.secondary_xaxis("top")
    top_axis.set_xticks(
        list(range(len(positions))),
        [str(row["wt_aa"]) for row in profile_rows],
        fontsize=_residue_label_size(len(positions), len(selected_records)),
    )
    top_axis.tick_params(length=0, pad=2)
    ax.set_xlabel("Ec86 canonical residue position", fontsize=LABEL_SIZE)
    ax.set_ylabel("Accepted alignment rows", fontsize=LABEL_SIZE)
    ax.set_ylim(len(selected_records) - 0.5, -1.78)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=24)
    ax.spines[["top", "right"]].set_visible(False)
    margins = _figure_margins(len(selected_records), fig_height=fig_height)
    legend_center_x = _axes_center_x(margins)
    fig.legend(
        handles=[
            Patch(facecolor=OKABE_ITO["green"], label="Matches Ec86"),
            Patch(facecolor="#c9c9c9", label="Differs from Ec86"),
            Patch(facecolor="#fbfaf7", edgecolor="#888888", label="Gap"),
            Patch(facecolor=OKABE_ITO["orange"], label=f"WT plurality >=25% ({panel_profile.scope_label})"),
            Patch(facecolor=OKABE_ITO["purple"], label="WT plurality >=50% (design-class threshold)"),
            Patch(
                facecolor=OKABE_ITO["sky"],
                edgecolor=OKABE_ITO["blue"],
                alpha=0.35,
                label="Subtype II-A3/42_1 rows",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=OKABE_ITO["blue"],
                markeredgecolor=OKABE_ITO["blue"],
                markersize=6,
                label="Mask-protected",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(legend_center_x, 0.012),
        ncol=4,
        frameon=False,
        fontsize=LEGEND_SIZE - 0.8,
        columnspacing=1.35,
        handletextpad=0.5,
        borderaxespad=0.2,
    )
    fig.subplots_adjust(**margins)

    path = panel_root / panel_profile.file_name
    alt = (
        f"Canonical-coordinate MSA panel for all {len(selected_records)} accepted alignment rows from the "
        f"{len(records)}-record {panel_profile.profile_id} alignment. The first row is {target_id}; "
        "the remaining rows use source-manifest labels with clade row or node identifiers and provider "
        f"accessions. Vertical markings show columns passing the 25 percent WT-plurality rule in the "
        f"{panel_profile.scope_label} profile, the 50 percent design-class threshold cue, and current "
        "protected mask positions. The clade 9 view also marks rows that belong to the narrower "
        "II-A3/42_1 subtype set when that source set is available."
    )
    source_tables = [
        panel_profile.aligned_fasta_source_table,
        panel_profile.source_manifest_table,
        "conservation_profile.parquet",
        "mask_set.yaml",
    ]
    input_paths = {
        "aligned_fasta": aligned_fasta_path,
        "source_manifest": source_manifest_path,
        "conservation_profile": conservation_profile_path,
        "mask_set": mask_set_path,
    }
    if subtype_source_manifest_path is not None:
        source_tables.append(f"conservation_sources/{CONSERVATION_SUBTYPE_PROFILE_ID}.source_manifest.yaml")
        input_paths["subtype_source_manifest"] = subtype_source_manifest_path
    save_accessible_svg(fig, path, title=title, description=alt, dpi=320)
    return make_deliverable_row(
        deliverable_id=panel_profile.deliverable_id,
        section=SECTION_CONSTRAINT_EVIDENCE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=source_tables,
        input_hashes=file_hashes(input_paths),
        alt_text=alt,
        description=_panel_description(panel_profile),
        interpretation_limit=_panel_interpretation_limit(panel_profile),
        title=title,
        evidence_summary={
            "profile_id": panel_profile.profile_id,
            "accepted_alignment_rows": len(records),
            "source_records": max(0, len(records) - 1),
            "current_mask_denominator": panel_profile.current_mask_denominator,
            "marked_subtype_rows": len(
                [
                    record_id
                    for record_id, _sequence in selected_records
                    if source_accessions.get(record_id) in subtype_accessions
                ]
            ),
        },
    )


def _panel_title(*, record_count: int, panel_profile: MsaPanelProfile) -> str:
    if panel_profile.current_mask_denominator:
        return f"The {record_count}-record clade 9 MSA shows the active 25% WT-plurality mask denominator"
    return f"The {record_count}-record Eco1 subtype II-A3/42_1 MSA shows the narrower subtype conservation context"


def _panel_description(panel_profile: MsaPanelProfile) -> str:
    if panel_profile.current_mask_denominator:
        return (
            "Shows the Eco1/Ec86 anchor row against all accepted clade 9 alignment rows with source-manifest "
            "row labels. The current conservation mask uses this clade 9 denominator; subtype rows are marked "
            "when they are present in the clade 9 source set."
        )
    return (
        "Shows the Eco1/Ec86 anchor row against all accepted II-A3/42_1 subtype alignment rows with "
        "source-manifest row labels. This is an independent subtype-context plot; it does not replace the "
        "clade 9 denominator used by the current mask policy."
    )


def _panel_interpretation_limit(panel_profile: MsaPanelProfile) -> str:
    if panel_profile.current_mask_denominator:
        return (
            "This panel explains the active conservation denominator and mask context. It does not rank "
            "ProteinMPNN candidates or establish biochemical function."
        )
    return (
        "This subtype panel is a narrower family context view. It is not the active conservation mask "
        "denominator unless the mask policy is explicitly changed."
    )


def _read_profile_rows(path: Path, *, profile_id: str) -> list[dict[str, Any]]:
    rows = pq.read_table(
        path,
        columns=[
            "canonical_position",
            "profile_id",
            "wt_aa",
            "msa_column",
            "passes_conservation_mask",
            "wt_frequency",
            "wt_is_plurality",
        ],
    ).to_pylist()
    selected = [row for row in rows if row["profile_id"] == profile_id]
    if not selected:
        raise ValueError(f"No rows found for profile {profile_id}")
    return sorted(selected, key=lambda row: int(row["canonical_position"]))


def _figure_size(row_count: int) -> tuple[float, float]:
    return _figure_width(row_count), _figure_height(row_count)


def _figure_width(row_count: int) -> float:
    if row_count <= 80:
        return 18.4
    return 22.4


def _figure_height(row_count: int) -> float:
    if row_count <= 60:
        return max(5.8, row_count * 0.22 + 1.75)
    if row_count <= 140:
        return row_count * 0.14 + 2.15
    return row_count * 0.096 + 2.35


def _figure_margins(row_count: int, *, fig_height: float) -> dict[str, float]:
    bottom = max(0.04, min(0.12, 0.86 / fig_height))
    top = 1.0 - max(0.036, min(0.12, 0.78 / fig_height))
    left = 0.205 if row_count <= 80 else 0.112
    return {
        "left": left,
        "right": 0.995,
        "bottom": bottom,
        "top": top,
    }


def _axes_center_x(margins: dict[str, float]) -> float:
    return (float(margins["left"]) + float(margins["right"])) / 2.0


def _row_label_size(row_count: int) -> float:
    if row_count <= 60:
        return float(TICK_SIZE)
    if row_count <= 140:
        return 5.2
    return 3.4


def _position_tick_indexes(positions: list[int]) -> list[int]:
    return [index for index, position in enumerate(positions) if position == 1 or position % 40 == 0]


def _residue_label_size(position_count: int, row_count: int) -> float:
    if row_count <= 80:
        return 7.4 if position_count <= 180 else 6.4
    return 6.0
