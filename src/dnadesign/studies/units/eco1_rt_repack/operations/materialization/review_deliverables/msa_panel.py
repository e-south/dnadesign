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
import yaml
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
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    LABEL_SIZE,
    LEGEND_SIZE,
    OKABE_ITO,
    TICK_SIZE,
    TITLE_SIZE,
    save_accessible_svg,
    shorten_label,
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
) -> dict[str, Any]:
    """Render a canonical-coordinate MSA plurality panel."""

    records = _read_fasta(aligned_fasta_path)
    if not records:
        raise ValueError(f"No FASTA records found in {aligned_fasta_path}")
    source_labels = _source_record_labels(source_manifest_path, row_label_prefix=panel_profile.row_label_prefix)
    profile_rows = _read_profile_rows(conservation_profile_path, profile_id=panel_profile.profile_id)
    residues = mask_residues if mask_residues is not None else read_mask_residues(mask_set_path)
    positions = [int(row["canonical_position"]) for row in profile_rows]
    conserved = {int(row["canonical_position"]) for row in profile_rows if bool(row["passes_conservation_mask"])}
    protected = {int(row["canonical_position"]) for row in residues if bool(row.get("protected"))}
    selected_records = _select_display_records(
        records,
        profile_rows,
        protected_positions=protected,
        conserved_positions=conserved,
        max_display_rows=max_display_rows,
    )
    target_id, _target_sequence = records[0]
    matrix = _alignment_matrix(selected_records, profile_rows)

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
        if position in conserved:
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
    ax.set_yticks(
        range(len(selected_records)),
        [
            _format_msa_row_label(
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
    ax.set_ylim(len(selected_records) - 0.5, -1.42)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=24)
    ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=[
            Patch(facecolor=OKABE_ITO["green"], label="Matches Ec86"),
            Patch(facecolor="#c9c9c9", label="Differs from Ec86"),
            Patch(facecolor="#fbfaf7", edgecolor="#888888", label="Gap"),
            Patch(facecolor=OKABE_ITO["orange"], label=f"WT plurality >=25% ({panel_profile.scope_label})"),
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
        bbox_to_anchor=(0.5, 0.012),
        ncol=5,
        frameon=False,
        fontsize=LEGEND_SIZE - 0.8,
        columnspacing=1.35,
        handletextpad=0.5,
        borderaxespad=0.2,
    )
    margins = _figure_margins(len(selected_records), fig_height=fig_height)
    fig.subplots_adjust(**margins)

    path = panel_root / panel_profile.file_name
    alt = (
        f"Canonical-coordinate MSA panel for all {len(selected_records)} accepted alignment rows from the "
        f"{len(records)}-record {panel_profile.profile_id} alignment. The first row is {target_id}; "
        "the remaining rows use source-manifest labels with clade row or node identifiers and provider "
        f"accessions. Vertical markings show columns passing the 25 percent WT-plurality rule in the "
        f"{panel_profile.scope_label} profile and current protected mask positions."
    )
    save_accessible_svg(fig, path, title=title, description=alt, dpi=320)
    return make_deliverable_row(
        deliverable_id=panel_profile.deliverable_id,
        section=SECTION_CONSTRAINT_EVIDENCE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=[
            panel_profile.aligned_fasta_source_table,
            panel_profile.source_manifest_table,
            "conservation_profile.parquet",
            "mask_set.yaml",
        ],
        input_hashes=file_hashes(
            {
                "aligned_fasta": aligned_fasta_path,
                "source_manifest": source_manifest_path,
                "conservation_profile": conservation_profile_path,
                "mask_set": mask_set_path,
            }
        ),
        alt_text=alt,
        description=_panel_description(panel_profile),
        interpretation_limit=_panel_interpretation_limit(panel_profile),
        title=title,
        evidence_summary={
            "profile_id": panel_profile.profile_id,
            "accepted_alignment_rows": len(records),
            "source_records": max(0, len(records) - 1),
            "current_mask_denominator": panel_profile.current_mask_denominator,
        },
    )


def _panel_title(*, record_count: int, panel_profile: MsaPanelProfile) -> str:
    if panel_profile.current_mask_denominator:
        return f"{record_count}-record clade 9 MSA: 25% plurality mask"
    return f"{record_count}-record II-A3/42_1 Eco1 subtype MSA"


def _panel_description(panel_profile: MsaPanelProfile) -> str:
    if panel_profile.current_mask_denominator:
        return (
            "Shows the Eco1/Ec86 anchor row against all accepted clade 9 alignment rows with source-manifest "
            "row labels. The current conservation mask uses this clade 9 denominator."
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
        ],
    ).to_pylist()
    selected = [row for row in rows if row["profile_id"] == profile_id]
    if not selected:
        raise ValueError(f"No rows found for profile {profile_id}")
    return sorted(selected, key=lambda row: int(row["canonical_position"]))


def _read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    current_id = ""
    chunks: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if current_id:
                records.append((current_id, "".join(chunks)))
            current_id = line[1:].strip()
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if current_id:
        records.append((current_id, "".join(chunks)))
    return records


def source_manifest_accessions(path: Path) -> set[str]:
    """Return accession identifiers from a conservation source manifest."""

    return {
        str(row.get("accession") or "").strip()
        for row in _source_manifest_records(path)
        if str(row.get("accession") or "").strip()
    }


def _source_manifest_records(path: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"source manifest must be a YAML mapping: {path}")
    raw_records = payload.get("included_records")
    if not isinstance(raw_records, list):
        raise ValueError(f"source manifest must declare included_records as a list: {path}")
    records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"source manifest included_records[{index}] must be a mapping")
        records.append(raw_record)
    return records


def _source_record_labels(path: Path, *, row_label_prefix: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    for index, raw_record in enumerate(_source_manifest_records(path)):
        record_id = str(raw_record.get("record_id") or "").strip()
        if not record_id:
            raise ValueError(f"source manifest included_records[{index}].record_id is required")
        accession = str(raw_record.get("accession") or "").strip()
        node = _record_node(record_id)
        label = " ".join(part for part in (row_label_prefix, node, accession) if part)
        labels[record_id] = shorten_label(label or record_id, max_length=52)
    return labels


def _record_node(record_id: str) -> str:
    if record_id.startswith("clade9_neighbor_"):
        return record_id.rsplit("_", maxsplit=1)[-1]
    parts = record_id.rsplit("__", 2)
    if len(parts) == 3:
        return parts[1]
    return ""


def _format_msa_row_label(
    record_id: str,
    *,
    source_labels: dict[str, str],
    profile_id: str,
    row_label_prefix: str,
) -> str:
    if record_id == "eco1_rt_ec86kit_reference":
        return "Ec86 reference"
    if record_id in source_labels:
        return source_labels[record_id]
    prefix = f"{profile_id}__"
    if record_id.startswith(prefix):
        parts = [part for part in record_id[len(prefix) :].split("__") if part]
        if parts:
            terminal = parts[-1]
            return f"{row_label_prefix} row {shorten_label(terminal, max_length=14)}"
    if record_id.startswith("clade9_neighbor_"):
        return f"{row_label_prefix} row {record_id.rsplit('_', maxsplit=1)[-1]}"
    return shorten_label(record_id, max_length=28)


def _select_display_records(
    records: list[tuple[str, str]],
    profile_rows: list[dict[str, Any]],
    *,
    protected_positions: set[int],
    conserved_positions: set[int],
    max_display_rows: int | None,
) -> list[tuple[str, str]]:
    if max_display_rows is None or len(records) <= max_display_rows:
        return records
    target_record = records[0]
    scored = [
        (
            _display_difference_score(sequence, profile_rows, protected_positions, conserved_positions),
            record_id,
            sequence,
        )
        for record_id, sequence in records[1:]
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [target_record, *[(record_id, sequence) for _score, record_id, sequence in scored[: max_display_rows - 1]]]


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


def _display_difference_score(
    sequence: str,
    profile_rows: list[dict[str, Any]],
    protected_positions: set[int],
    conserved_positions: set[int],
) -> int:
    score = 0
    for row in profile_rows:
        position = int(row["canonical_position"])
        column = int(row["msa_column"]) - 1
        residue = sequence[column] if 0 <= column < len(sequence) else "-"
        if residue == str(row["wt_aa"]):
            continue
        score += 1
        if position in conserved_positions:
            score += 3
        if position in protected_positions:
            score += 2
    return score


def _alignment_matrix(records: list[tuple[str, str]], profile_rows: list[dict[str, Any]]) -> list[list[int]]:
    matrix: list[list[int]] = []
    wt_by_index = [str(row["wt_aa"]) for row in profile_rows]
    columns = [int(row["msa_column"]) - 1 for row in profile_rows]
    for _record_id, sequence in records:
        row_values: list[int] = []
        for index, column in enumerate(columns):
            residue = sequence[column] if 0 <= column < len(sequence) else "-"
            if residue == "-":
                row_values.append(0)
            elif residue == wt_by_index[index]:
                row_values.append(2)
            else:
                row_values.append(1)
        matrix.append(row_values)
    return matrix
