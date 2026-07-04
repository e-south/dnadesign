"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/proteinmpnn_variant_similarity.py

ProteinMPNN WT/variant sequence-similarity heatmap for Eco1 review deliverables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib
import pyarrow.parquet as pq
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from .constants import SECTION_DESIGNS_AND_FOLD_TRIAGE
from .manifest import file_hashes, make_deliverable_row
from .rendering import LABEL_SIZE, LEGEND_SIZE, OKABE_ITO, TITLE_SIZE, save_accessible_svg

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_MUTATION_RE = re.compile(r"^[A-Z*](?P<position>[0-9]+)[A-Z*]$")
_SIMILARITY_CMAP = ListedColormap(("#d8e8f3", OKABE_ITO["vermillion"], "#8c959f"))
_SIMILARITY_NORM = BoundaryNorm((-0.5, 0.5, 1.5, 2.5), _SIMILARITY_CMAP.N)
_SIMILARITY_LEGEND = (
    ("Same as WT", "#d8e8f3"),
    ("Different from WT", OKABE_ITO["vermillion"]),
    ("Missing backbone context", "#8c959f"),
)


def write_variant_similarity_heatmap(
    panel_root: Path,
    rows: list[dict[str, Any]],
    candidate_table_path: Path,
    *,
    foldcheck_ranking_path: Path | None,
    foldcheck_fasta_path: Path | None,
    mask_set_path: Path | None,
    mask_residues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Render a categorical WT/variant sequence-similarity heatmap."""

    panel_root.mkdir(parents=True, exist_ok=True)
    title = "Baseline ProteinMPNN variants are mapped against the Ec86 WT sequence"
    fasta_sequences = (
        _read_fasta(foldcheck_fasta_path) if foldcheck_fasta_path and foldcheck_fasta_path.exists() else {}
    )
    wt_sequence = fasta_sequences.get("wild_type") or _wt_sequence_from_mask(mask_residues)
    if not wt_sequence:
        raise ValueError("ProteinMPNN similarity heatmap requires a WT sequence from fold-check FASTA or mask residues")
    ordered_rows = _ordered_candidate_rows(rows, foldcheck_ranking_path)
    candidate_sequences = [
        (
            str(row["candidate_id"]),
            _sequence_for_candidate(row, wt_sequence=wt_sequence, fasta_sequences=fasta_sequences),
        )
        for row in ordered_rows
    ]
    missing_positions = _missing_backbone_positions(mask_residues)
    row_ids = ["wild_type", *[candidate_id for candidate_id, _sequence in candidate_sequences]]
    matrix = _similarity_matrix(
        wt_sequence=wt_sequence,
        candidate_sequences=[("wild_type", wt_sequence), *candidate_sequences],
        missing_positions=missing_positions,
    )
    sequence_length = len(wt_sequence)
    row_count = len(row_ids)
    fig, ax = plt.subplots(figsize=_figure_size(sequence_length=sequence_length, row_count=row_count))
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=_SIMILARITY_CMAP, norm=_SIMILARITY_NORM)
    _style_similarity_axes(ax, row_ids=row_ids, sequence_length=sequence_length, title=title)
    _add_similarity_legend(fig)
    path = panel_root / "proteinmpnn_variant_similarity_heatmap.svg"
    alt = (
        f"Categorical heatmap comparing WT Ec86 and {len(candidate_sequences)} baseline ProteinMPNN variants "
        f"over {sequence_length} canonical positions. Cells show whether each residue matches WT, differs "
        "from WT, or lacks mapped backbone context."
    )
    source_tables, input_paths = _source_tables_and_hash_paths(
        candidate_table_path=candidate_table_path,
        mask_set_path=mask_set_path,
        foldcheck_fasta_path=foldcheck_fasta_path,
        foldcheck_ranking_path=foldcheck_ranking_path,
    )
    save_accessible_svg(fig, path, title=title, description=alt)
    return make_deliverable_row(
        deliverable_id="proteinmpnn_variant_similarity_heatmap",
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="svg",
        status="rendered",
        path=path,
        source_tables=source_tables,
        input_hashes=file_hashes(input_paths),
        alt_text=alt,
        description=(
            "Maps WT and baseline accepted ProteinMPNN variants against Ec86 canonical positions using the "
            "fold-check FASTA when present. The plot separates residue identity differences from missing "
            "backbone context; expanded design-class selection is shown in the selection panel."
        ),
        interpretation_limit=(
            "This is a descriptive sequence-similarity view. It does not measure fold quality, ESMC likelihood, "
            "or RT activity."
        ),
        title=title,
        role="review_only",
        evidence_summary={
            "sequence_count": row_count,
            "variant_count": len(candidate_sequences),
            "sequence_length": sequence_length,
            "sequence_source": "foldcheck_request/input_sequences.fasta"
            if fasta_sequences
            else "candidate_table_or_mask",
            "missing_backbone_position_count": len(missing_positions),
        },
    )


def _similarity_matrix(
    *,
    wt_sequence: str,
    candidate_sequences: list[tuple[str, str]],
    missing_positions: set[int],
) -> list[list[int]]:
    matrix: list[list[int]] = []
    for _candidate_id, sequence in candidate_sequences:
        values = []
        for position, wt_residue in enumerate(wt_sequence, start=1):
            if position in missing_positions:
                values.append(2)
                continue
            candidate_residue = sequence[position - 1] if position <= len(sequence) else ""
            values.append(0 if candidate_residue == wt_residue else 1)
        matrix.append(values)
    return matrix


def _figure_size(*, sequence_length: int, row_count: int) -> tuple[float, float]:
    return (
        max(9.5, min(16.0, 5.4 + sequence_length * 0.035)),
        max(4.2, min(19.5, 1.8 + row_count * 0.155)),
    )


def _style_similarity_axes(ax: Any, *, row_ids: list[str], sequence_length: int, title: str) -> None:
    row_count = len(row_ids)
    ax.set_title(title, fontsize=TITLE_SIZE, pad=10)
    ax.set_xlabel("Ec86 canonical residue position", fontsize=LABEL_SIZE)
    ax.set_ylabel("WT and ProteinMPNN variant", fontsize=LABEL_SIZE)
    ax.set_yticks(
        list(range(row_count)),
        [_similarity_row_label(candidate_id, row_index) for row_index, candidate_id in enumerate(row_ids)],
        fontsize=max(5.2, min(8.0, 84.0 / max(row_count, 1))),
    )
    tick_positions = _similarity_x_tick_positions(sequence_length)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [str(position + 1) for position in tick_positions],
        fontsize=7.4 if sequence_length > 80 else LABEL_SIZE,
    )
    ax.tick_params(axis="both", length=2.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_similarity_legend(fig: Any) -> None:
    handles = [Patch(facecolor=color, label=label) for label, color in _SIMILARITY_LEGEND]
    fig.legend(
        handles=handles,
        frameon=False,
        fontsize=LEGEND_SIZE,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(handles),
        columnspacing=1.1,
        handlelength=1.3,
        handletextpad=0.42,
    )
    fig.subplots_adjust(left=0.235, right=0.99, top=0.91, bottom=0.16)


def _source_tables_and_hash_paths(
    *,
    candidate_table_path: Path,
    mask_set_path: Path | None,
    foldcheck_fasta_path: Path | None,
    foldcheck_ranking_path: Path | None,
) -> tuple[list[str], dict[str, Path]]:
    source_tables = ["candidate_table.parquet"]
    input_paths = {"candidate_table": candidate_table_path}
    if mask_set_path is not None:
        source_tables.append("mask_set.yaml")
        input_paths["mask_set"] = mask_set_path
    if foldcheck_fasta_path is not None and foldcheck_fasta_path.exists():
        source_tables.append("foldcheck_request/input_sequences.fasta")
        input_paths["foldcheck_input_fasta"] = foldcheck_fasta_path
    if foldcheck_ranking_path is not None and foldcheck_ranking_path.exists():
        source_tables.append("foldcheck_review/foldcheck_candidate_ranking.parquet")
        input_paths["foldcheck_candidate_ranking"] = foldcheck_ranking_path
    return source_tables, input_paths


def _read_fasta(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    sequences: dict[str, list[str]] = {}
    current_id = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            current_id = stripped[1:].split(None, 1)[0]
            if not current_id:
                raise ValueError(f"FASTA header without sequence id in {path}")
            sequences.setdefault(current_id, [])
            continue
        if not current_id:
            raise ValueError(f"FASTA sequence line appears before a header in {path}")
        sequences[current_id].append(stripped)
    return {sequence_id: "".join(parts).upper() for sequence_id, parts in sequences.items()}


def _wt_sequence_from_mask(mask_residues: list[dict[str, Any]]) -> str:
    rows = sorted(mask_residues, key=lambda row: int(row["canonical_position"]))
    return "".join(str(row.get("wt_aa") or "X")[:1].upper() for row in rows)


def _ordered_candidate_rows(
    rows: list[dict[str, Any]],
    foldcheck_ranking_path: Path | None,
) -> list[dict[str, Any]]:
    rank_by_candidate = _rank_by_candidate(foldcheck_ranking_path)
    return sorted(
        rows,
        key=lambda row: (
            rank_by_candidate.get(str(row["candidate_id"]), int(row.get("rank") or 1_000_000)),
            int(row.get("mutation_count") or 0),
            str(row["candidate_id"]),
        ),
    )


def _rank_by_candidate(foldcheck_ranking_path: Path | None) -> dict[str, int]:
    if foldcheck_ranking_path is None or not foldcheck_ranking_path.exists():
        return {}
    rows = pq.read_table(foldcheck_ranking_path).to_pylist()
    ranked: dict[str, int] = {}
    for fallback_rank, row in enumerate(rows, start=1):
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id:
            ranked[candidate_id] = int(row.get("review_rank") or fallback_rank)
    return ranked


def _sequence_for_candidate(
    row: dict[str, Any],
    *,
    wt_sequence: str,
    fasta_sequences: dict[str, str],
) -> str:
    candidate_id = str(row["candidate_id"])
    fasta_sequence = fasta_sequences.get(candidate_id, "")
    if len(fasta_sequence) == len(wt_sequence):
        return fasta_sequence
    candidate_sequence = str(row.get("sequence") or "").upper()
    if len(candidate_sequence) == len(wt_sequence):
        return candidate_sequence
    return _apply_canonical_mutations(wt_sequence, row.get("canonical_mutations") or [], candidate_id=candidate_id)


def _apply_canonical_mutations(wt_sequence: str, mutations: list[Any], *, candidate_id: str) -> str:
    sequence = list(wt_sequence)
    for mutation in mutations:
        mutation_text = str(mutation)
        match = _MUTATION_RE.match(mutation_text)
        if not match:
            raise ValueError(f"Unrecognized canonical mutation for {candidate_id}: {mutation_text}")
        position = int(match.group("position"))
        if not 1 <= position <= len(sequence):
            raise ValueError(f"Canonical mutation outside WT sequence for {candidate_id}: {mutation_text}")
        expected_residue = mutation_text[0]
        observed_residue = sequence[position - 1]
        if expected_residue != observed_residue:
            raise ValueError(
                f"Canonical mutation WT residue mismatch for {candidate_id}: "
                f"{mutation_text} expected {expected_residue} at {position}, found {observed_residue}"
            )
        sequence[position - 1] = mutation_text[-1]
    return "".join(sequence)


def _missing_backbone_positions(mask_residues: list[dict[str, Any]]) -> set[int]:
    missing = set()
    for row in mask_residues:
        position = int(row["canonical_position"])
        mapping_status = str(row.get("mapping_status") or "")
        if bool(row.get("non_fixed_missing_backbone")):
            missing.add(position)
        elif mapping_status and mapping_status != "mapped":
            missing.add(position)
        elif row.get("has_backbone_coordinates") is False:
            missing.add(position)
    return missing


def _similarity_row_label(candidate_id: str, row_index: int) -> str:
    if candidate_id == "wild_type":
        return "WT Ec86"
    compact = candidate_id.removeprefix("thread_candidate_")
    return f"V{row_index:03d} {compact[:12]}"


def _similarity_x_tick_positions(sequence_length: int) -> list[int]:
    if sequence_length <= 40:
        return list(range(sequence_length))
    step = 20 if sequence_length <= 360 else max(25, round(sequence_length / 18))
    positions = list(range(0, sequence_length, step))
    final = sequence_length - 1
    if positions[-1] != final:
        positions.append(final)
    return positions
