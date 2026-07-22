"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/realization/sequences.py

DNA sequence normalization contracts for construct realization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.usr import normalize_sequence

from ..contracts.errors import ValidationError


def alphabet_for_sequence(sequence: str) -> str:
    return "dna_5" if "N" in sequence.upper() else "dna_4"


def ensure_dna_text(text: str, *, label: str) -> str:
    seq = str(text or "").strip().upper()
    if not seq:
        raise ValidationError(f"{label} cannot be empty.")
    try:
        alphabet = alphabet_for_sequence(seq)
        normalize_sequence(seq, "dna", alphabet)
    except ValueError as exc:
        raise ValidationError(f"{label} must be valid DNA (ACGT or ACGTN).") from exc
    return seq
