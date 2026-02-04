"""
--------------------------------------------------------------------------------
<dnadesign project>
nmf/encoder.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

from .utils import build_motif2tf_map, reverse_complement

logger = logging.getLogger(__name__)


def find_all_occurrences(seq: str, motif: str) -> List[int]:
    """
    Find all 0-based start indices of motif occurrences in seq.
    """
    positions = []
    start = 0
    while True:
        pos = seq.find(motif, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1  # continue searching after this match
    return positions


def encode_with_positional_bins(sequences: List[Dict], config: dict) -> Tuple[np.ndarray, List[str]]:
    """
    Encode sequences into a feature matrix using positional binning.
    For each sequence, for each motif in meta_tfbs_parts_in_array (which contains
    just the motif strings), the function:
      - Retrieves the transcription factor name using the motif-to-TF map (from utils.build_motif2tf_map).
      - Determines whether the motif occurs on the forward or reverse strand.
      - Finds all occurrences (start positions) of the motif.
      - Determines which bins (of equal width over the sequence) the motif occupies.
      - Increments the corresponding feature(s) in the output matrix.

    The final feature columns are named as "tf_fwd_bin{b}" and "tf_rev_bin{b}".
    """
    num_bins = config["nmf"]["positional_binning"]["num_bins"]

    # Build a mapping from motif to TF using meta_tfbs_parts.
    motif2tf = build_motif2tf_map(sequences)

    # Build the global TF list by iterating over meta_tfbs_parts_in_array and looking up the TF.
    global_tfs = set()
    for seq in sequences:
        for motif_str in seq.get("meta_tfbs_parts_in_array", []):
            motif_candidate = motif_str.upper()
            if motif_candidate in motif2tf:
                global_tfs.add(motif2tf[motif_candidate])
            else:
                logger.warning(
                    f"No TF mapping found for motif '{motif_candidate}' in sequence {seq.get('id', 'unknown')}."
                )
    sorted_tfs = sorted(global_tfs)

    # Create feature names: for each TF, for each strand (fwd, rev), for each bin.
    feature_names = []
    for tf in sorted_tfs:
        for strand in ["fwd", "rev"]:
            for b in range(num_bins):
                feature_names.append(f"{tf}_{strand}_bin{b}")

    n_sequences = len(sequences)
    n_features = len(feature_names)
    X = np.zeros((n_sequences, n_features), dtype=float)

    # Process each sequence.
    for seq_idx, seq_dict in enumerate(sequences):
        seq_str = seq_dict.get("sequence", "").upper()
        if not seq_str:
            continue
        L = len(seq_str)

        for motif_str in seq_dict.get("meta_tfbs_parts_in_array", []):
            motif_candidate = motif_str.upper()
            # Retrieve the TF name using the mapping.
            if motif_candidate not in motif2tf:
                logger.error(
                    f"Motif '{motif_candidate}' has no corresponding TF mapping in sequence "
                    f"{seq_dict.get('id', 'unknown')}."
                )
                continue
            tf_name = motif2tf[motif_candidate]

            # Detect occurrences on the forward strand.
            starts_fwd = find_all_occurrences(seq_str, motif_candidate)
            if starts_fwd:
                strand = "fwd"
                starts = starts_fwd
            else:
                # Try the reverse complement.
                rev_candidate = reverse_complement(motif_candidate)
                starts_rev = find_all_occurrences(seq_str, rev_candidate)
                if not starts_rev:
                    logger.error(f"Motif '{motif_candidate}' not found in either strand in sequence {seq_idx}.")
                    continue
                strand = "rev"
                starts = starts_rev

            motif_len = len(motif_candidate)
            for s_start in starts:
                s_end = s_start + motif_len
                # Determine bins spanned by the motif: from the bin containing s_start to the bin containing s_end-1.
                binA = (s_start * num_bins) // L
                binB = ((s_end - 1) * num_bins) // L

                # Determine the column offset for the given TF.
                try:
                    tf_index = sorted_tfs.index(tf_name)
                except ValueError:
                    logger.error(f"TF '{tf_name}' not found in global TF list for sequence {seq_idx}.")
                    continue
                tf_col_offset = tf_index * (2 * num_bins)
                if strand == "rev":
                    tf_col_offset += num_bins

                # Increment all bins that overlap with the motif occurrence.
                for bin_idx in range(binA, binB + 1):
                    col_idx = tf_col_offset + bin_idx
                    X[seq_idx, col_idx] += 1.0
    return X, feature_names


def encode_sequence_matrix(sequences: List[Dict], config: dict) -> Tuple[np.ndarray, List[str]]:
    """
    Encode sequence dictionaries into a feature matrix X.

    Supports two encoding modes:
      - "motif_occupancy": existing occupancy-based encoding.
      - "positional_bins": new positional and strand-aware binning encoding.

    Depending on the encoding_mode in the configuration, the appropriate method is called.
    """
    encoding_mode = config["nmf"].get("encoding_mode", "motif_occupancy")

    if encoding_mode == "positional_bins":
        if config["nmf"].get("positional_binning", {}).get("enable", False):
            return encode_with_positional_bins(sequences, config)
        else:
            raise ValueError("Positional bins encoding mode selected but positional_binning.enable is False in config.")

    elif encoding_mode == "motif_occupancy":
        # Legacy motif occupancy encoding.
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
        sequence_motif_info = []  # List of dict: motif -> count

        for seq in sequences:
            fixed_motifs = extract_fixed_motifs(seq)
            motif_counts = {}
            tfbs_array = seq.get("meta_tfbs_parts_in_array", [])
            for motif in tfbs_array:
                motif_up = motif.upper()
                if motif_up in fixed_motifs:
                    continue
                motif_counts[motif_up] = motif_counts.get(motif_up, 0) + 1
            # Cap counts.
            for motif in motif_counts:
                motif_counts[motif] = min(motif_counts[motif], max_occ)
            global_motif_set.update(motif_counts.keys())
            sequence_motif_info.append(motif_counts)

        global_motif_list = sorted(list(global_motif_set))
        feature_names = global_motif_list[:]  # features are motif strings.
        n_features = len(feature_names)
        X = np.zeros((n_sequences, n_features), dtype=float)
        for i, motif_counts in enumerate(sequence_motif_info):
            for motif, count in motif_counts.items():
                if motif in feature_names:
                    col = feature_names.index(motif)
                    X[i, col] = count
        return X, feature_names
    else:
        raise ValueError(f"Unsupported encoding mode: {encoding_mode}")
