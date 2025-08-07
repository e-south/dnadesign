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
  - an auto-generated UUID as its id

Usage:
  variants = generate_variants(ref_entry, params, regions, lookup_tables)

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import uuid
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

    1) Validate that the input sequence is a simple string of A/C/G/T.
    2) Determine which positions to mutate:
         - if `regions` omitted or empty → every position
         - otherwise → only positions in the given [[start,end), …] slices
    3) For each position, substitute each alternative base ≠ wild‐type.
    4) Record each mutation as a dict:
         {
           "id": unique UUID,
           "ref_name": original reference name,
           "protocol": "scan_dna",
           "sequence": mutated_seq,
           "modifications": [ e.g. "A5T" ],
         }

    Args:
      ref_entry: dict containing at least “sequence” and “ref_name”
      params:  unused here but kept for API consistency
      regions: list of [start,end) index pairs in 0-based, half-open form
      lookup_tables: unused in DNA scan (placeholder for other protocols)

    Returns:
      List of variant dictionaries ready for evaluation & selection.
    """
    raw_seq = ref_entry.get("sequence")
    name = ref_entry.get("ref_name", "<unknown>")

    # --- 1) Sequence sanity checks ---
    if not isinstance(raw_seq, str):
        raise ValueError(
            f"[{name}] sequence must be a string, got {type(raw_seq).__name__}"
        )
    if not re.fullmatch(r"[ACGTacgt]+", raw_seq):
        raise ValueError(f"[{name}] invalid characters in sequence: {raw_seq!r}")

    seq = raw_seq.upper()
    # Default to full-length if no regions provided
    regions = regions or [[0, len(seq)]]

    variants: List[Dict] = []
    # --- 2) Iterate through each region and each position ---
    for start, end in regions:
        # Validate region bounds
        if not (0 <= start < end <= len(seq)):
            raise ValueError(
                f"[{name}] invalid region bounds: [{start},{end}) for length {len(seq)}"
            )
        for idx, wt_nt in enumerate(seq[start:end], start=start):
            # 3) For each alt base, skip wild‐type and build a variant
            for alt_nt in _ALPHABET:
                if alt_nt == wt_nt:
                    continue
                mutated_seq = seq[:idx] + alt_nt + seq[idx + 1 :]
                variants.append(
                    {
                        "id": str(uuid.uuid4()),  # universally unique
                        "ref_name": name,  # track original reference
                        "protocol": "scan_dna",  # protocol label
                        "sequence": mutated_seq,  # new mutated sequence
                        "modifications": [f"{wt_nt}{idx+1}{alt_nt}"],  # e.g. "A5T"
                    }
                )

    return variants
