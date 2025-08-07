"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/protocols/scan_codon.py

Codon-level saturation - for each codon in the reference, replace it
with the most-frequent codon of every other amino acid, based on
a lookup CSV with columns: codon, amino_acid, fraction, frequency.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

# number of nucleotides per codon
_TRIPLE = 3


def _load_codon_table(path: str | Path) -> Dict[str, List[str]]:
    """
    Read a tabular CSV with headers:
       codon, amino_acid, fraction, frequency

    Builds a mapping amino_acid → list of codons sorted by descending frequency.
    """
    aa2entries: Dict[str, List[Tuple[str, float]]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=",")
        # check required columns
        expected = {"codon", "amino_acid", "frequency"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Codon table must include columns {expected}, got {reader.fieldnames}"
            )
        for row in reader:
            aa = row["amino_acid"]
            codon = row["codon"].upper()
            try:
                freq = float(row["frequency"])
            except ValueError:
                raise ValueError(f"Invalid frequency value: {row['frequency']!r}")
            aa2entries.setdefault(aa, []).append((codon, freq))

    # sort each AA’s codons by frequency descending, then extract codon lists
    aa2codons: Dict[str, List[str]] = {}
    for aa, entries in aa2entries.items():
        # sort by freq descending
        sorted_codons = sorted(entries, key=lambda cf: cf[1], reverse=True)
        aa2codons[aa] = [codon for codon, _ in sorted_codons]

    return aa2codons


def generate_variants(
    ref_entry: Dict,
    params: Dict,
    regions: List[Tuple[int, int]] | None = None,
    lookup_tables: List[str] | None = None,
) -> List[Dict]:
    """
    For each codon in ref_entry["sequence"] (or within specified codon-index regions),
    substitute it with the most-frequent codon of every other amino acid.

    Args:
      ref_entry: {
        "ref_name": str,
        "sequence": str,         # length % 3 == 0
        "modifications": list,   # ignored here
        "round": int             # ignored here
      }
      params: unused
      regions: optional list of (codon_start, codon_end) pairs
      lookup_tables: must contain at least one path to a codon CSV

    Returns:
      List of variant dicts, each with:
        - id: UUID
        - ref_name: same as input
        - protocol: "scan_codon"
        - sequence: mutated full‐length DNA
        - modifications: ["WT_CODON@<codon_index>→NEW_CODON(<AA>)"]
    """
    if not lookup_tables:
        raise ValueError("scan_codon requires at least one codon lookup table")
    # load and sort codon usage
    aa2codons = _load_codon_table(lookup_tables[0])

    # prepare reference sequence
    seq = ref_entry["sequence"].upper()
    if len(seq) % _TRIPLE != 0:
        raise ValueError("Sequence length must be divisible by 3 for codon scanning")
    codons = [seq[i : i + _TRIPLE] for i in range(0, len(seq), _TRIPLE)]

    # determine which codon indices to scan
    total = len(codons)
    if regions:
        # assume single region for simplicity
        start, end = regions[0]
        scan_ix = range(start, min(end, total))
    else:
        scan_ix = range(total)

    variants: List[Dict] = []
    ref_name = ref_entry["ref_name"]

    # scan each codon index
    for ci in scan_ix:
        wt = codons[ci]
        # find the WT amino acid by membership in our table
        wt_aa = next((aa for aa, clist in aa2codons.items() if wt in clist), None)
        if wt_aa is None:
            # skip codons not found in table
            continue

        # substitute every *other* amino acid by its top codon
        for aa, codon_list in aa2codons.items():
            if aa == wt_aa or not codon_list:
                continue
            new_codon = codon_list[0]  # most frequent for this AA
            mutated = codons.copy()
            mutated[ci] = new_codon
            variants.append(
                {
                    "id": str(uuid.uuid4()),
                    "ref_name": ref_name,
                    "protocol": "scan_codon",
                    "sequence": "".join(mutated),
                    "modifications": [f"{wt}@{ci}→{new_codon}({aa})"],
                }
            )

    return variants
