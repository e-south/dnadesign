"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/msa_panel.py

MSA plurality and mask-context panel for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pyarrow.parquet as pq
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    CONSERVATION_PROFILE_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.mask_rows import (
    read_mask_residues,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rendering import (
    save_accessible_svg,
    shorten_label,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def write_msa_plurality_mask_panel(
    *,
    panel_root: Path,
    aligned_fasta_path: Path,
    conservation_profile_path: Path,
    mask_set_path: Path,
    mask_residues: list[dict[str, Any]] | None = None,
    max_display_rows: int = 46,
) -> dict[str, Any]:
    """Render a compact canonical-coordinate MSA plurality panel."""

    records = _read_fasta(aligned_fasta_path)
    if not records:
        raise ValueError(f"No FASTA records found in {aligned_fasta_path}")
    profile_rows = _read_profile_rows(conservation_profile_path)
    residues = mask_residues if mask_residues is not None else read_mask_residues(mask_set_path)
    target_id, _target_sequence = records[0]
    selected_records = [records[0], *records[1:max_display_rows]]
    positions = [int(row["canonical_position"]) for row in profile_rows]
    matrix = _alignment_matrix(selected_records, profile_rows)
    conserved = {int(row["canonical_position"]) for row in profile_rows if bool(row["passes_conservation_mask"])}
    protected = {int(row["canonical_position"]) for row in residues if bool(row.get("protected"))}

    fig, ax = plt.subplots(figsize=(12.6, max(5.2, len(selected_records) * 0.19 + 1.8)))
    ax.imshow(matrix, aspect="auto", interpolation="none", cmap=ListedColormap(["#fbfaf7", "#d3d6d0", "#386c55"]))
    for index, position in enumerate(positions):
        if position in conserved:
            ax.axvspan(index - 0.5, index + 0.5, color="#efc94c", alpha=0.28, linewidth=0)
        if position in protected:
            ax.scatter(index, -0.65, marker="s", s=10, color="#547aa5", clip_on=False)
    ax.set_yticks(
        range(len(selected_records)),
        [_format_msa_row_label(record_id) for record_id, _seq in selected_records],
        fontsize=9,
    )
    tick_indexes = [index for index, position in enumerate(positions) if position == 1 or position % 40 == 0]
    ax.set_xticks(tick_indexes, [str(positions[index]) for index in tick_indexes], fontsize=10)
    ax.set_xlabel("Ec86 canonical residue position", fontsize=11)
    ax.set_ylabel("Clade 9 record", fontsize=11)
    ax.set_title("Clade 9 alignment shows which Ec86 positions were protected.", fontsize=13, pad=28)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor="#386c55", label="Matches Ec86"),
            Patch(facecolor="#d3d6d0", label="Differs from Ec86"),
            Patch(facecolor="#fbfaf7", edgecolor="#888888", label="Gap"),
            Patch(facecolor="#efc94c", alpha=0.45, label="25% WT plurality"),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="#547aa5",
                markeredgecolor="#547aa5",
                markersize=6,
                label="Protected position",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=5,
        frameon=False,
        fontsize=10.5,
    )
    fig.tight_layout()

    path = panel_root / "msa_plurality_mask_panel.svg"
    alt = (
        f"Canonical-coordinate MSA panel for {len(selected_records)} displayed rows from "
        f"{CONSERVATION_PROFILE_ID}. The first row is {target_id}; vertical markings show "
        "positions passing the 25 percent WT-plurality rule and protected mask positions."
    )
    save_accessible_svg(fig, path, title="Clade 9 alignment protected-position context", description=alt)
    return make_deliverable_row(
        deliverable_id="msa_plurality_mask_panel",
        section="scaffold_and_mask",
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=[
            "conservation_alignments/ec86_clade9_conservation_v1.aligned.fasta",
            "conservation_profile.parquet",
            "mask_set.yaml",
        ],
        input_hashes=file_hashes(
            {
                "aligned_fasta": aligned_fasta_path,
                "conservation_profile": conservation_profile_path,
                "mask_set": mask_set_path,
            }
        ),
        alt_text=alt,
        description=(
            "Shows the Eco1/Ec86 anchor row against a readable subset of clade 9 neighbors, "
            "using canonical residue coordinates so plurality and mask decisions line up with "
            "the design surface."
        ),
        interpretation_limit=(
            "This panel explains the conservation and mask context. It does not rank "
            "ProteinMPNN candidates or establish biochemical function."
        ),
    )


def _read_profile_rows(path: Path) -> list[dict[str, Any]]:
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
    selected = [row for row in rows if row["profile_id"] == CONSERVATION_PROFILE_ID]
    if not selected:
        raise ValueError(f"No rows found for profile {CONSERVATION_PROFILE_ID}")
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


def _format_msa_row_label(record_id: str) -> str:
    if record_id == "eco1_rt_ec86kit_reference":
        return "Ec86 reference"
    prefix = f"{CONSERVATION_PROFILE_ID}__"
    if record_id.startswith(prefix):
        parts = [part for part in record_id[len(prefix) :].split("__") if part]
        if parts:
            return f"C9-{shorten_label(parts[-1], max_length=20)}"
    return shorten_label(record_id, max_length=28)


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
