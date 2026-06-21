"""QC calculations for generic MSA visualization sidecars."""

from __future__ import annotations

import hashlib
from pathlib import Path

from dnadesign.aligner.msa.visualization.contracts.models import (
    PositionQc,
    ProfileQc,
)


def build_profile_qc(
    *,
    profile_id: str,
    aligned_fasta: Path,
    records: dict[str, str],
    target_aligned: str,
    target_hash: str,
    output_root: Path,
) -> ProfileQc:
    """Build per-profile visualization metadata."""

    return ProfileQc(
        profile_id=profile_id,
        aligned_fasta_path=aligned_fasta,
        record_count=len(records),
        alignment_length=len(target_aligned),
        canonical_position_count=len(target_aligned.replace("-", "")),
        inserted_column_count=target_aligned.count("-"),
        target_ungapped_sha256=target_hash,
        profile_qc_path=output_root / f"{profile_id}.msa_qc.yaml",
        profile_svg_path=output_root / f"{profile_id}.position_qc.svg",
        profile_exemplar_svg_path=output_root / f"{profile_id}.exemplar_windows.svg",
        profile_alignment_overview_svg_path=output_root / f"{profile_id}.alignment_overview.svg",
        profile_consensus_histogram_svg_path=output_root / f"{profile_id}.consensus_histogram.svg",
    )


def build_position_qc(
    *,
    profile_id: str,
    records: dict[str, str],
    target_row_id: str,
) -> list[PositionQc]:
    """Build target-position QC rows for one aligned FASTA profile."""

    target_aligned = records[target_row_id]
    rows: list[PositionQc] = []
    canonical_position = 0
    for column_index, target_aa in enumerate(target_aligned, start=1):
        if target_aa == "-":
            continue
        canonical_position += 1
        residues = [sequence[column_index - 1] for sequence in records.values()]
        gap_count = residues.count("-")
        non_gap_residues = [residue for residue in residues if residue != "-"]
        non_gap_count = len(non_gap_residues)
        plurality_aa, plurality_count = _plurality(non_gap_residues)
        rows.append(
            PositionQc(
                profile_id=profile_id,
                canonical_position=canonical_position,
                alignment_column=column_index,
                target_aa=target_aa,
                non_gap_count=non_gap_count,
                gap_count=gap_count,
                gap_fraction=gap_count / len(residues),
                plurality_aa=plurality_aa,
                plurality_count=plurality_count,
                plurality_frequency=(plurality_count / non_gap_count if non_gap_count else 0.0),
            )
        )
    return rows


def sha256_text(value: str) -> str:
    """Return the repository-standard SHA-256 text digest."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plurality(residues: list[str]) -> tuple[str, int]:
    if not residues:
        return "", 0
    counts: dict[str, int] = {}
    for residue in residues:
        counts[residue] = counts.get(residue, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
