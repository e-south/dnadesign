"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/permuter/protocols/scan_dna.py

Permutes biological sequences by single-nucleotide mutations.
  - if `regions` is empty or None → full-length single-nt saturation
  - otherwise → only positions in provided [[start,end),…] ranges

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from typing import Dict, List


def generate_variants(
    ref_entry: Dict,
    params: Dict,
    regions: List[List[int]] = None,
    lookup_tables: List[str] = None,
) -> List[Dict]:
    """
    Produce single-nucleotide variants over DNA (or other) alphabets.
    - params["alphabet"] : "DNA" → implicitly A/C/G/T; can be extended
    - regions: list of [start,end) zero-based slices; [] or None = full sequence
    - lookup_tables: currently unused (reserved for codon-based protocols)
    """
    seq = ref_entry["sequence"]
    # choose alphabet
    alph = params.get("alphabet", "DNA")
    if alph.upper() == "DNA":
        letters = "ACGT"
    else:
        # fallback: treat `alphabet` as iterable of symbols
        letters = list(alph)

    # figure out which positions to mutate
    if not regions:
        positions = range(len(seq))
    else:
        positions = []
        for start, end in regions:
            assert (
                0 <= start < end <= len(seq)
            ), f"region [{start},{end}) out of bounds for length {len(seq)}"
            positions.extend(range(start, end))

    variants = []
    for i in positions:
        wild = seq[i]
        for alt in letters:
            if alt == wild:
                continue
            mutated = seq[:i] + alt + seq[i + 1 :]
            variants.append(
                {
                    "id": f"{ref_entry['id']}_{i}_{alt}",
                    "sequence": mutated,
                    "modifications": [{"pos": i, "from": wild, "to": alt}],
                }
            )

    return variants
