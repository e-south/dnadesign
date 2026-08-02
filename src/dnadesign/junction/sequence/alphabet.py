"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/junction/sequence/alphabet.py

Strict DNA alphabet validation and strand-orientation helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

DNA_ALPHABET = frozenset("ACGT")
_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def validate_dna(sequence: str, *, name: str = "sequence") -> str:
    """Return *sequence* after validating strict, unambiguous uppercase DNA.

    Empty sequences are valid so distance and substring primitives retain their
    conventional boundary behavior.
    """

    if not isinstance(sequence, str):
        raise TypeError(f"{name} must be a string, got {type(sequence).__name__}")

    invalid = [(index, base) for index, base in enumerate(sequence) if base not in DNA_ALPHABET]
    if invalid:
        details = ", ".join(f"position {index}: {base!r}" for index, base in invalid)
        raise ValueError(
            f"{name} must contain only uppercase A/C/G/T; invalid character(s) at {details}. "
            "Provide unambiguous uppercase DNA without whitespace."
        )
    return sequence


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a 5-prime-to-3-prime DNA sequence."""

    return validate_dna(sequence).translate(_COMPLEMENT)[::-1]


__all__ = ["DNA_ALPHABET", "reverse_complement", "validate_dna"]
