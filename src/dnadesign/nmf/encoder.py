"""
--------------------------------------------------------------------------------
<dnadesign project>
nmf/encoder.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import numpy as np
from typing import List, Dict, Tuple
from .parser import robust_parse_tfbs
import logging

logger = logging.getLogger(__name__)

def encode_sequence_matrix(sequences: List[Dict], config: dict) -> Tuple[np.ndarray, List[str]]:
    """
    Encode sequence dictionaries into a feature matrix X using motif-centric occupancy encoding.
    Each sequence (row) is represented by motif occurrence counts derived from meta_tfbs_parts_in_array,
    with counts capped at max_motif_occupancy (default = 5). Fixed motifs (from fixed_elements) are ignored.
    
    Returns:
      X: numpy array of shape (n_sequences, n_features)
      feature_names: list of motif strings (columns)
    """
    # Only motif_occupancy mode is supported.
    encoding_mode = config["nmf"].get("encoding_mode", "motif_occupancy")
    if encoding_mode != "motif_occupancy":
        raise ValueError("Only 'motif_occupancy' encoding is supported in the updated pipeline.")
    
    max_occ = config["nmf"].get("max_motif_occupancy", 5)
    
    def extract_fixed_motifs(seq):
        fixed = set()
        if "fixed_elements" in seq:
            constraints = seq["fixed_elements"].get("promoter_constraints", [])
            for constraint in constraints:
                if "upstream" in constraint:
                    fixed.add(constraint["upstream"].upper())
                if "downstream" in constraint:
                    fixed.add(constraint["downstream"].upper())
        return fixed

    n_sequences = len(sequences)
    global_motif_set = set()
    sequence_motif_info = []  # list of dict: motif -> count
    
    for seq in sequences:
        fixed_motifs = extract_fixed_motifs(seq)
        motif_counts = {}
        tfbs_array = seq.get("meta_tfbs_parts_in_array", [])
        for motif in tfbs_array:
            motif_up = motif.upper()
            if motif_up in fixed_motifs:
                continue
            motif_counts[motif_up] = motif_counts.get(motif_up, 0) + 1
        # Cap counts
        for motif in motif_counts:
            motif_counts[motif] = min(motif_counts[motif], max_occ)
        global_motif_set.update(motif_counts.keys())
        sequence_motif_info.append(motif_counts)
    
    global_motif_list = sorted(list(global_motif_set))
    feature_names = global_motif_list[:]  # features are motif strings
    n_features = len(feature_names)
    X = np.zeros((n_sequences, n_features), dtype=float)
    for i, motif_counts in enumerate(sequence_motif_info):
        for motif, count in motif_counts.items():
            if motif in feature_names:
                col = feature_names.index(motif)
                X[i, col] = count
    return X, feature_names