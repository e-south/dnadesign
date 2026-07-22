"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/sequence/evidence.py

Sequence and folding evidence checks for Retron review-output packages.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ...compiler.exceptions import RetronMsdCompilerError
from .index import SequenceReviewFrame


@dataclass(frozen=True)
class SequenceEvidenceSummary:
    folding_status_ok_count: int
    native_structure_png_verified_count: int
    reverse_complement_verified_count: int

    def as_manifest(self) -> dict[str, int]:
        return {
            "folding_status_ok_count": self.folding_status_ok_count,
            "native_structure_png_verified_count": self.native_structure_png_verified_count,
            "reverse_complement_verified_count": self.reverse_complement_verified_count,
        }


def verify_sequence_evidence(
    frames: Sequence[SequenceReviewFrame],
    *,
    materialized_root: Path,
) -> SequenceEvidenceSummary:
    folding_ok = 0
    native_png_ok = 0
    reverse_complement_ok = 0
    for frame in frames:
        if frame.row.get("folding_status") == "ok":
            folding_ok += 1
        if Path(frame.row["secondary_structure_native_png"]).suffix == ".png":
            native_png_ok += 1
        forward = _read_fasta_sequence(materialized_root / frame.row["forward_fasta"])
        observed_reverse = _read_fasta_sequence(materialized_root / frame.row["reverse_complement_fasta"])
        expected_reverse = _reverse_complement(forward)
        if observed_reverse != expected_reverse:
            raise RetronMsdCompilerError(
                f"Retron review row {frame.order} reverse_complement_fasta does not match "
                f"forward_fasta reverse complement: {materialized_root / frame.row['reverse_complement_fasta']}"
            )
        reverse_complement_ok += 1
    return SequenceEvidenceSummary(
        folding_status_ok_count=folding_ok,
        native_structure_png_verified_count=native_png_ok,
        reverse_complement_verified_count=reverse_complement_ok,
    )


def _read_fasta_sequence(path: Path) -> str:
    if not path.is_file():
        raise RetronMsdCompilerError(f"Retron review FASTA evidence file not found: {path}")
    sequence = "".join(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(">")
    ).upper()
    if not sequence:
        raise RetronMsdCompilerError(f"Retron review FASTA evidence file has no sequence: {path}")
    invalid = sorted(set(sequence) - set("ACGTN"))
    if invalid:
        raise RetronMsdCompilerError(f"Retron review FASTA evidence file has invalid DNA symbols {invalid}: {path}")
    return sequence


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1].upper()


__all__ = ["SequenceEvidenceSummary", "verify_sequence_evidence"]
