"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/pairwise/validation.py

Nucleotide sequence validation for pairwise scoring.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any


def validate_sequence(seq: Any) -> str:
    """Validate and normalize a nucleotide sequence."""

    if not isinstance(seq, str) or not seq.strip():
        raise ValueError(f"Invalid sequence: must be a non-empty string, got: {seq!r}")
    normalized = seq.upper()
    allowed = set("ACGTN")
    for base in normalized:
        if base not in allowed:
            raise ValueError(f"Invalid character {base!r} in sequence: {normalized}")
    return normalized


def extract_sequence(item: Any, key: str = "sequence") -> str:
    """Extract and validate a nucleotide sequence from a string or mapping."""

    if isinstance(item, str):
        return validate_sequence(item)
    if isinstance(item, dict):
        seq = item.get(key)
        if seq is None:
            raise ValueError(f"Dictionary does not contain the key {key!r}.")
        return validate_sequence(seq)
    raise ValueError("Item must be a string or a dict with a sequence key.")
