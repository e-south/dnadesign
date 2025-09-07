"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/usr/src/normalize.py

Case-preserving normalization and deterministic ID computation.

- `normalize_sequence`: trims whitespace; preserves case; enforces dna_4 alphabet
- `compute_id`: sha1(bio_type|sequence_norm) on the **trimmed, original** string

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import hashlib
import re


def normalize_sequence(seq: str, bio_type: str, alphabet: str) -> str:
    """
    Trim only. Preserve case (lowercase may carry information).
    Enforce alphabet constraints without coercing case.
    """
    s = (seq or "").strip()
    if bio_type == "dna" and alphabet == "dna_4":
        # allow only A/C/G/T in either case
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
