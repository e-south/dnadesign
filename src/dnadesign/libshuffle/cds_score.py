"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/cds_score.py

Provides a wrapper function to compute the Composite Diversity Score (CDS)
from a subsample of sequence entries. Each sequence is expected to contain a 
meta_nmf dictionary with a "program_composition" key (a row-normalized vector).

The CDS is computed using the function compute_composite_diversity_score from
dnadesign.nmf.diagnostics.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import numpy as np
from dnadesign.nmf.diagnostics import compute_composite_diversity_score

def compute_cds_from_sequences(subsample: list, alpha: float = 0.5) -> dict:
    """
    Extracts program_composition vectors from each sequence in the subsample,
    validates that they all have the same length, and computes the CDS.
    
    Returns a dictionary with keys:
      - "cds_score": Composite Diversity Score (float)
      - "dominance": Dominance component (float)
      - "program_diversity": Program Diversity component (float)
    """
    vectors = []
    for seq in subsample:
        meta = seq.get("meta_nmf")
        if meta is None or "program_composition" not in meta:
            raise ValueError("Each sequence must contain 'meta_nmf.program_composition'")
        vec = meta["program_composition"]
        total = sum(vec)
        if total == 0:
            # Instead of raising an error, log a warning and skip this sequence.
            print(f"Warning: Skipping sequence {seq.get('id')} because its program_composition sums to zero.")
            continue
        normalized_vec = [x / total for x in vec]
        vectors.append(normalized_vec)
    
    if not vectors:
        raise ValueError("No valid program_composition vectors found in the subsample.")
    
    lengths = set(len(v) for v in vectors)
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent program_composition lengths in subsample: {lengths}")
    
    W = np.array(vectors, dtype=np.float64)
    cds, dominance, program_diversity = compute_composite_diversity_score(W, alpha)
    
    return {
        "cds_score": float(cds),
        "dominance": float(dominance),
        "program_diversity": float(program_diversity)
    }