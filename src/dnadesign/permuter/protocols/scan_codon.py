"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/protocols/scan_codon.py

Codon-level scanning for protein-coding sequences.
- replaces each codon by all its synonymous alternatives
  as defined in the provided lookup_table(s).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import csv
from typing import Dict, List


def _load_codon_table(path: str) -> Dict[str, List[str]]:
    # returns aa -> list of codons
    aa2codons: Dict[str, List[str]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            aa = row["amino_acid"]
            codon = row["codon"]
            aa2codons.setdefault(aa, []).append(codon)
    return aa2codons


def generate_variants(
    ref_entry: Dict,
    params: Dict,
    regions: List[List[int]] = None,
    lookup_tables: List[str] = None,
) -> List[Dict]:
    """
    - regions: list of [codon_index_start, codon_index_end), empty=all codons
    - lookup_tables[0]: path to codon table CSV (see spec above)
    """
    seq = ref_entry["sequence"]
    assert len(seq) % 3 == 0, "Sequence length must be multiple of 3"
    codon_count = len(seq) // 3

    # load map aa->codons
    assert lookup_tables and len(lookup_tables) == 1, "Provide one codon table"
    aa2codons = _load_codon_table(lookup_tables[0])

    # decide which codon indices to scan
    if not regions:
        codon_idxs = range(codon_count)
    else:
        codon_idxs = []
        for start, end in regions:
            assert 0 <= start < end <= codon_count
            codon_idxs.extend(range(start, end))

    variants = []
    for ci in codon_idxs:
        pos = 3 * ci
        orig = seq[pos : pos + 3]
        # figure out its amino acid
        aa = None
        for a, codons in aa2codons.items():
            if orig in codons:
                aa = a
                break
        assert aa, f"Original codon {orig} not in lookup table"

        for alt in aa2codons[aa]:
            if alt == orig:
                continue
            mutant = seq[:pos] + alt + seq[pos + 3 :]
            variants.append(
                {
                    "id": f"{ref_entry['id']}_codon{ci}_{orig}>{alt}",
                    "sequence": mutant,
                    "modifications": [{"codon_idx": ci, "from": orig, "to": alt}],
                }
            )

    return variants
