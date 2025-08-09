"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/protocols/scan_dna.py

Implements a “scan” protocol that introduces every possible
single-nucleotide mutation at each position (full saturation)
or within specified subsequence ranges.

Each variant records:
  - its new sequence
  - which nucleotide was changed (e.g. “A5T”)

Usage:
  variants = generate_variants(ref_entry, params, regions, lookup_tables)

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Dict, List

# The four canonical DNA bases
_ALPHABET = "ACGT"


def generate_variants(
    ref_entry: Dict,
    params: Dict,
    regions: List | None = None,
    lookup_tables: List[str] | None = None,
) -> List[Dict]:
    """
    Create all single-nucleotide variants from the reference sequence.
    """
    raw_seq = ref_entry.get("sequence")
    name = ref_entry.get("ref_name", "<unknown>")

    if not isinstance(raw_seq, str):
        raise ValueError(
            f"[{name}] sequence must be a string, got {type(raw_seq).__name__}"
        )
    if not re.fullmatch(r"[ACGTacgt]+", raw_seq):
        raise ValueError(f"[{name}] invalid characters in sequence: {raw_seq!r}")

    seq = raw_seq.upper()
    regions = regions or [[0, len(seq)]]

    variants: List[Dict] = []
    for start, end in regions:
        if not (0 <= start < end <= len(seq)):
            raise ValueError(
                f"[{name}] invalid region bounds: [{start},{end}) for length {len(seq)}"
            )
        for idx, wt_nt in enumerate(seq[start:end], start=start):
            for alt_nt in _ALPHABET:
                if alt_nt == wt_nt:
                    continue
                mutated_seq = seq[:idx] + alt_nt + seq[idx + 1 :]
                variants.append(
                    {
                        "sequence": mutated_seq,
                        "modifications": [f"{wt_nt}{idx+1}{alt_nt}"],
                    }
                )

    return variants
