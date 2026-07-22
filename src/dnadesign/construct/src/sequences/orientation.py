"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/sequences/orientation.py

Orientation helpers for emitted Construct products.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

_IUPAC_DNA_COMPLEMENT = str.maketrans(
    "ACGTRYSWKMBDHVNacgtryswkmbdhvn",  # pragma: allowlist secret
    "TGCAYRSWMKVHDBNtgcayrswmkvhdbn",  # pragma: allowlist secret
)


def complement(sequence: str) -> str:
    return sequence.translate(_IUPAC_DNA_COMPLEMENT)


def reverse_complement(sequence: str) -> str:
    return complement(sequence)[::-1]


def reverse_complement_anchor_bounds(
    *,
    sequence_length: int,
    anchor_start_0: int,
    anchor_end_0: int,
) -> tuple[int, int]:
    return sequence_length - anchor_end_0, sequence_length - anchor_start_0
