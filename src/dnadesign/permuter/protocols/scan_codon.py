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
from pathlib import Path
from typing import Dict, List, Tuple

_TRIPLE = 3


def _load_codon_table(path: str | Path) -> Dict[str, List[str]]:
    aa2entries: Dict[str, List[Tuple[str, float]]] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=",")
        expected = {"codon", "amino_acid", "frequency"}
        if not expected.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Codon table must include columns {expected}, got {reader.fieldnames}"
            )
        for row in reader:
            aa = row["amino_acid"]
            codon = row["codon"].upper()
            freq = float(row["frequency"])
            aa2entries.setdefault(aa, []).append((codon, freq))
    aa2codons: Dict[str, List[str]] = {}
    for aa, entries in aa2entries.items():
        sorted_codons = sorted(entries, key=lambda cf: cf[1], reverse=True)
        aa2codons[aa] = [codon for codon, _ in sorted_codons]
    return aa2codons


def generate_variants(
    ref_entry: Dict,
    params: Dict,
    regions: List[Tuple[int, int]] | None = None,
    lookup_tables: List[str] | None = None,
) -> List[Dict]:
    if not lookup_tables:
        raise ValueError("scan_codon requires at least one codon lookup table")
    aa2codons = _load_codon_table(lookup_tables[0])

    seq = ref_entry["sequence"].upper()
    if len(seq) % _TRIPLE != 0:
        raise ValueError("Sequence length must be divisible by 3 for codon scanning")
    codons = [seq[i : i + _TRIPLE] for i in range(0, len(seq), _TRIPLE)]

    total = len(codons)
    if regions:
        start, end = regions[0]
        scan_ix = range(start, min(end, total))
    else:
        scan_ix = range(total)

    variants: List[Dict] = []
    ref_name = ref_entry["ref_name"]

    for ci in scan_ix:
        wt = codons[ci]
        wt_aa = next((aa for aa, clist in aa2codons.items() if wt in clist), None)
        if wt_aa is None:
            continue
        for aa, codon_list in aa2codons.items():
            if aa == wt_aa or not codon_list:
                continue
            new_codon = codon_list[0]
            mutated = codons.copy()
            mutated[ci] = new_codon
            variants.append(
                {
                    "ref_name": ref_name,
                    "protocol": "scan_codon",
                    "sequence": "".join(mutated),
                    "modifications": [f"{wt}@{ci}→{new_codon}({aa})"],
                }
            )

    return variants
