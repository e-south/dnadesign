"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/utils/sequence_utils.py

Sequence utilities for DenseGen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations


def gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    g = seq.count("G")
    c = seq.count("C")
    return (g + c) / len(seq)
