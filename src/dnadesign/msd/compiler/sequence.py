"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/msd/compiler/sequence.py

Validate concrete DNA inputs for MSD compilation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


class MsdSequenceInputError(ValueError):
    """Raised when a Retron MSD sequence input is not concrete DNA."""


def validate_dna_sequence(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise MsdSequenceInputError(f"{label} cannot be empty.")
    invalid = sorted(set(text.upper()) - {"A", "C", "G", "T"})
    if invalid:
        raise MsdSequenceInputError(f"{label} contains non-DNA bases: {''.join(invalid)}.")
    return text


__all__ = ["MsdSequenceInputError", "validate_dna_sequence"]
