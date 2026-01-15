"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/canonical.py

Canonical sequence normalization + ID computation.
Aligned with dnadesign.usr.src.normalize (sha1(bio_type|sequence_norm)).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import re


def normalize_sequence(seq: str, bio_type: str, alphabet: str) -> str:
    """
    Trim only. Preserve case (lowercase may carry information).
    Enforce alphabet constraints without coercing case.
    """
    s = (seq or "").strip()
    if bio_type == "dna" and alphabet == "dna_4":
        if re.search(r"[^ACGTacgt]", s):
            raise ValueError(f"Non-ACGT character in sequence: {s}")
    return s


def compute_id(bio_type: str, sequence_norm: str) -> str:
    """
    Deterministic id: sha1(bio_type|sequence_norm).
    Case is preserved (no uppercasing).
    """
    h = hashlib.sha1()
    h.update(f"{bio_type}|{sequence_norm}".encode("utf-8"))
    return h.hexdigest()
