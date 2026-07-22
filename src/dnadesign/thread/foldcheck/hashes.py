"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/thread/foldcheck/hashes.py

Hash helpers for generic fold-check artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib


def sequence_hash(sequence: str) -> str:
    """Return the canonical SHA-256 URI for one amino-acid sequence."""

    normalized = "".join(sequence.split()).upper()
    return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()
