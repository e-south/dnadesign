"""
--------------------------------------------------------------------------------
<dnadesign project>
billboard/core.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import torch
import numpy as np
import warnings
from scipy.stats import entropy as scipy_entropy

def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(complement)[::-1]

def load_pt_files(pt_paths):
    """Load sequence entries from the provided .pt file paths."""
    sequences = []
    for path in pt_paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, list):
            raise ValueError(f"Data in '{path}' is not a list of sequences.")
        sequences.extend(data)
    return sequences

def validate_sequence(seq):
    """Ensure a sequence dictionary contains required keys."""
    required_keys = ["sequence", "meta_tfbs_parts", "meta_tfbs_parts_in_array"]
    for key in required_keys:
        if key not in seq:
            raise ValueError(f"Sequence entry (id: {seq.get('id', 'unknown')}) is missing required key: {key}")

def robust_parse_tfbs(part, seq_id="unknown"):
    """
    Parse a meta_tfbs_parts string robustly.
    Supports two formats:
      - Colon-delimited: "tf:motif"
      - Underscore-delimited (legacy): "idx_{index}_{TF_name}_{motif}"
    Returns a tuple (tf_name, motif).
    """
    if ":" in part:
        tf_name, motif = part.split(":", 1)
        tf_name = tf_name.lower()
        if not motif.isalpha() or not all(c in "ATCGatcg" for c in motif):
            raise ValueError(f"Motif candidate '{motif}' is not a valid nucleotide sequence in sequence (id: {seq_id})")
        return tf_name, motif
    if part.startswith("idx_"):
        s = part[4:]
    else:
        s = part
    parts = s.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid meta_tfbs_parts format in sequence (id: {seq_id}): {part}")
    motif_candidate = parts[-1]
    if not motif_candidate.isalpha() or not all(c in "ATCGatcg" for c in motif_candidate):
        raise ValueError(f"Motif candidate '{motif_candidate}' is not a valid nucleotide sequence in sequence (id: {seq_id})")
    tf_name = "_".join(parts[1:-1]).lower() if len(parts) > 2 else parts[0].lower()
    return tf_name, motif_candidate

def process_sequences(pt_paths, config):
    """
    Load, validate, and process sequences from one or more .pt files.
    Computes occupancy matrices, TF frequency counts, per-sequence TF rosters,
    motif info, and additional statistics.
    
    For each sequence, for every motif in meta_tfbs_parts_in_array, the function finds
    the actual position of the motif (or its reverse complement) and updates the occupancy matrices.
    """
    sequences = load_pt_files(pt_paths)
    if not sequences:
        raise ValueError("No sequences loaded from provided pt files.")
    
    seq_lengths = []
    for seq in sequences:
        validate_sequence(seq)
        seq_lengths.append(len(seq["sequence"]))
    unique_lengths = set(seq_lengths)
    if len(unique_lengths) > 1:
        msg = f"Inconsistent sequence lengths found: {unique_lengths}"
        if not config.get("allow_variable_sequence_length", False):
            raise ValueError(msg)
        else:
            warnings.warn(msg)
    sequence_length = max(unique_lengths)
    
    tf_frequency = {}
    occupancy_forward_dict = {}
    occupancy_reverse_dict = {}
    per_sequence_combos = []
    motif_info = []
    motif_stats = []
    mapping_failures = []
    total_tfbs_instances = 0
    
    for seq in sequences:
        seq_str = seq["sequence"]
        tf_names_in_seq = []
        tfbs_parts = seq["meta_tfbs_parts"]
        motif_to_tf = {}
        for part in tfbs_parts:
            try:
                tf_name, motif = robust_parse_tfbs(part, seq_id=seq.get("id", "unknown"))
                motif_to_tf[motif] = tf_name
            except ValueError as e:
                mapping_failures.append({
                    "sequence_id": seq.get("id", "unknown"),
                    "tf": "",
                    "motif": "",
                    "reason": str(e)
                })
                continue
        for motif in seq.get("meta_tfbs_parts_in_array", []):
            if motif not in motif_to_tf:
                mapping_failures.append({
                    "sequence_id": seq.get("id", "unknown"),
                    "tf": "",
                    "motif": motif,
                    "reason": "Motif not found in meta_tfbs_parts mapping"
                })
                continue
            tf_name = motif_to_tf[motif]
            tf_names_in_seq.append(tf_name)
            tf_frequency[tf_name] = tf_frequency.get(tf_name, 0) + 1
            total_tfbs_instances += 1
            motif_info.append({
                "tf": tf_name,
                "motif": motif,
                "status": "pass"
            })
            motif_length = len(motif)
            index_fwd = seq_str.find(motif)
            if index_fwd != -1:
                if tf_name not in occupancy_forward_dict:
                    occupancy_forward_dict[tf_name] = np.zeros(sequence_length, dtype=int)
                occupancy_forward_dict[tf_name][index_fwd:index_fwd+motif_length] += 1
            else:
                rc_motif = reverse_complement(motif)
                index_rev = seq_str.find(rc_motif)
                if index_rev != -1:
                    if tf_name not in occupancy_reverse_dict:
                        occupancy_reverse_dict[tf_name] = np.zeros(sequence_length, dtype=int)
                    occupancy_reverse_dict[tf_name][index_rev:index_rev+motif_length] += 1
        unique_roster = set(tf_names_in_seq)
        per_sequence_combos.append(sorted(unique_roster))
    
    sorted_tf_list = sorted(list(set(occupancy_forward_dict.keys()) | set(occupancy_reverse_dict.keys())))
    full_forward_matrix = []
    full_reverse_matrix = []
    for tf in sorted_tf_list:
        forward = occupancy_forward_dict.get(tf, np.zeros(sequence_length, dtype=int))
        reverse = occupancy_reverse_dict.get(tf, np.zeros(sequence_length, dtype=int))
        full_forward_matrix.append(forward)
        full_reverse_matrix.append(reverse)
    full_forward_matrix = np.array(full_forward_matrix)
    full_reverse_matrix = np.array(full_reverse_matrix)
    
    occupancy_tf_list = [tf for tf in sorted_tf_list if not (tf.endswith("_upstream") or tf.endswith("_downstream"))]
    occupancy_forward_matrix = np.array([occupancy_forward_dict.get(tf, np.zeros(sequence_length, dtype=int)) for tf in occupancy_tf_list])
    occupancy_reverse_matrix = np.array([occupancy_reverse_dict.get(tf, np.zeros(sequence_length, dtype=int)) for tf in occupancy_tf_list])
    
    def compute_entropy(matrix):
        pos_totals = matrix.sum(axis=0)
        if pos_totals.sum() == 0:
            return 0.0
        prob_dist = pos_totals / pos_totals.sum()
        return float(scipy_entropy(prob_dist))
    
    positional_entropy_forward = compute_entropy(full_forward_matrix)
    positional_entropy_reverse = compute_entropy(full_reverse_matrix)
    global_positional_entropy = (positional_entropy_forward + positional_entropy_reverse) / 2.0
    pos_norm_factor = np.log2(sequence_length) if sequence_length > 1 else 1
    global_positional_entropy_norm = global_positional_entropy / pos_norm_factor if pos_norm_factor > 0 else 0.0
    
    per_tf_entropy = []
    for tf in sorted_tf_list:
        forward_array = occupancy_forward_dict.get(tf, np.zeros(sequence_length, dtype=int))
        reverse_array = occupancy_reverse_dict.get(tf, np.zeros(sequence_length, dtype=int))
        def safe_entropy(arr):
            return 0.0 if arr.sum() == 0 else float(scipy_entropy(arr / arr.sum()))
        avg_entropy = (safe_entropy(forward_array) + safe_entropy(reverse_array)) / 2.0
        per_tf_entropy.append({
            "tf": tf,
            "entropy_forward": safe_entropy(forward_array),
            "entropy_reverse": safe_entropy(reverse_array),
            "avg_entropy": avg_entropy
        })
    
    # Aggregate the per-TF positional entropy
    # If config has key "positional_entropy_method" with value "median", then we compute the median;
    # otherwise, we compute the mean.
    if "positional_entropy_method" in config:
        method = config["positional_entropy_method"]
    else:
        method = "median"  # default
    tf_entropies = [entry["avg_entropy"] for entry in per_tf_entropy if entry["avg_entropy"] is not None]
    if tf_entropies:
        if method == "median":
            aggregated_pos_entropy = float(np.median(tf_entropies))
        else:
            aggregated_pos_entropy = float(np.mean(tf_entropies))
    else:
        aggregated_pos_entropy = 0.0

    num_sequences = len(sequences)
    average_tfbs_per_sequence = total_tfbs_instances / num_sequences if num_sequences > 0 else 0
    
    results = {
        "sequences": sequences,
        "num_sequences": num_sequences,
        "sequence_length": sequence_length,
        "unique_tfs": sorted_tf_list,
        "occupancy_tf_list": occupancy_tf_list,
        "total_tfbs_instances": total_tfbs_instances,
        "average_tfbs_per_sequence": average_tfbs_per_sequence,
        "failed_mappings": mapping_failures,
        "full_coverage_forward_matrix": full_forward_matrix,
        "full_coverage_reverse_matrix": full_reverse_matrix,
        "occupancy_forward_matrix": occupancy_forward_matrix,
        "occupancy_reverse_matrix": occupancy_reverse_matrix,
        "tf_frequency": tf_frequency,
        "positional_entropy_forward": positional_entropy_forward,
        "positional_entropy_reverse": positional_entropy_reverse,
        "per_tf_entropy": per_tf_entropy,
        "aggregated_positional_entropy": aggregated_pos_entropy,
        "per_sequence_combos": per_sequence_combos,
        "combo_counts": { "|".join(combo): count for combo, count in zip(per_sequence_combos, [1]*len(per_sequence_combos) )},  # simplified
        "global_positional_entropy": global_positional_entropy,
        "global_positional_entropy_norm": global_positional_entropy_norm,
        "motif_info": motif_info,
        "motif_stats": motif_stats
    }
    return results

def compute_tf_richness(sequences, config):
    """Compute the number of unique TFs across all sequences."""
    unique_tfs = set()
    for seq in sequences:
        roster = set()
        for part in seq.get("meta_tfbs_parts", []):
            if ":" in part:
                tf_name, _ = part.split(":", 1)
                roster.add(tf_name.lower().strip())
        unique_tfs.update(roster)
    return len(unique_tfs)

def compute_gini(tf_frequency):
    """Compute the Gini coefficient for TF frequency counts and return 1 - Gini."""
    values = np.array(list(tf_frequency.values()), dtype=float)
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = values.size
    cumulative = np.cumsum(sorted_vals)
    if cumulative[-1] == 0:
        return 0.0
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return 1 - gini

def compute_jaccard_index(set_a, set_b):
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return intersection / union

def compute_mean_jaccard(sequences, config):
    """Compute the mean pairwise Jaccard dissimilarity (1 - Jaccard index) of TF rosters."""
    tf_rosters = []
    for seq in sequences:
        roster = set()
        for part in seq.get("meta_tfbs_parts", []):
            if ":" in part:
                tf_name, _ = part.split(":", 1)
                roster.add(tf_name.lower().strip())
        tf_rosters.append(roster)
    n = len(tf_rosters)
    if n < 2:
        return 0.0
    dissimilarities = []
    for i in range(n):
        for j in range(i+1, n):
            jaccard = compute_jaccard_index(tf_rosters[i], tf_rosters[j])
            dissimilarities.append(1 - jaccard)
    return np.mean(dissimilarities) if dissimilarities else 0.0

def compute_core_metrics(results, config):
    """
    Compute core diversity metrics:
      - tf_richness
      - 1_minus_gini
      - mean_jaccard
      - median_tf_entropy (or mean_tf_entropy if configured)
    """
    sequences = results["sequences"]
    tf_richness = compute_tf_richness(sequences, config)
    
    # Recompute TF frequency from the sequences.
    tf_frequency = {}
    for seq in sequences:
        for part in seq.get("meta_tfbs_parts", []):
            if ":" in part:
                tf_name, _ = part.split(":", 1)
                tf_name = tf_name.lower().strip()
                tf_frequency[tf_name] = tf_frequency.get(tf_name, 0) + 1
    
    one_minus_gini = compute_gini(tf_frequency)
    mean_jaccard = compute_mean_jaccard(sequences, config)
    
    # Use the per_tf_entropy computed in process_sequences.
    if "per_tf_entropy" in results:
        tf_entropies = [ (entry["entropy_forward"] + entry["entropy_reverse"]) / 2.0 for entry in results["per_tf_entropy"] ]
    else:
        tf_entropies = []
    # Determine aggregation method from config: "median" or "mean" (default to median).
    method = config.get("positional_entropy_method", "median")
    if tf_entropies:
        if method == "median":
            aggregated_pos_entropy = float(np.median(tf_entropies))
        else:
            aggregated_pos_entropy = float(np.mean(tf_entropies))
    else:
        aggregated_pos_entropy = 0.0
    
    return {
        "tf_richness": tf_richness,
        "1_minus_gini": one_minus_gini,
        "mean_jaccard": mean_jaccard,
        "median_tf_entropy": aggregated_pos_entropy
    }

def compute_composite_metrics(core_metrics, config):
    """
    Compute composite metrics from the core metrics using a weighted sum.
    Since Billboard processes a single PT file (one observation),
    only the weighted sum composite is computed.
    
    Expected keys in core_metrics: "tf_richness", "1_minus_gini", "mean_jaccard", "median_tf_entropy".
    """
    weights = config.get("composite_weights", {})
    weighted_sum = 0.0
    total_weight = 0.0
    for metric, value in core_metrics.items():
        w = weights.get(metric, 0)
        weighted_sum += w * value
        total_weight += w
    if total_weight > 0:
        weighted_sum /= total_weight
    else:
        weighted_sum = 0.0

    return {
        "billboard_weighted_sum": weighted_sum
    }