"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/validation.py

Validation helpers for FASTA and aligned FASTA records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

_PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX")
_DNA_ALPHABET = set("ACGTN")


def validate_fasta_records(
    records: Mapping[str, str],
    *,
    alphabet: str = "protein",
    allow_gaps: bool = False,
) -> None:
    """Validate record ids and sequence alphabets."""

    if not records:
        raise ValueError("FASTA is empty")
    allowed = _alphabet_for(alphabet)
    if allow_gaps:
        allowed = allowed | {"-"}
    for record_id, sequence in records.items():
        if not record_id:
            raise ValueError("FASTA record id must be non-empty")
        if not sequence:
            raise ValueError(f"FASTA record {record_id!r} has an empty sequence")
        for character in sequence.upper():
            if character not in allowed:
                label = "protein" if alphabet == "protein" else "DNA"
                raise ValueError(f"Invalid {label} character {character!r} in FASTA record {record_id!r}")


def validate_aligned_fasta_records(
    records: Mapping[str, str],
    *,
    target_row_id: str | None = None,
    alphabet: str = "protein",
) -> None:
    """Validate aligned FASTA shape and optional target row presence."""

    validate_fasta_records(records, alphabet=alphabet, allow_gaps=True)
    alignment_lengths = {len(sequence) for sequence in records.values()}
    if len(alignment_lengths) != 1:
        raise ValueError("aligned FASTA records must have one alignment length")
    if target_row_id is not None and target_row_id not in records:
        raise ValueError(f"aligned FASTA is missing target row {target_row_id!r}")


def _alphabet_for(alphabet: str) -> set[str]:
    if alphabet == "protein":
        return set(_PROTEIN_ALPHABET)
    if alphabet == "dna":
        return set(_DNA_ALPHABET)
    raise ValueError(f"unsupported FASTA alphabet: {alphabet!r}")
